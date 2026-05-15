use std::{collections::BTreeMap, path::Path};

use crate::{
    compare::ActivationRow,
    timing::{TimeGroup, TimeRow},
};

pub(crate) fn forward_key(path: &str) -> (u8, usize, u8) {
    let trace = path
        .strip_suffix(".safetensors")
        .or_else(|| path.strip_suffix(".time.json"))
        .unwrap_or(path);
    match trace {
        "embedding/token_ids" => return (0, 0, 0),
        "embedding/embedded_context" => return (0, 0, 1),
        "layer_norm0/embedded_context" => return (0, 0, 2),
        "lm_head/embedded_context" => return (2, 0, 0),
        "lm_head/logits" => return (2, 0, 1),
        _ => {}
    }

    let Some(rest) = trace.strip_prefix("cells/cell_") else {
        return (3, 0, 0);
    };
    let Some((cell, name)) = rest.split_once('/') else {
        return (3, 0, 0);
    };
    let Ok(cell) = cell.parse::<usize>() else {
        return (3, 0, 0);
    };
    let step = match name {
        "time_mixer/value_from_first_cell" => 0,
        "time_mixer/embedded_context" => 1,
        "embedded_context_after_time_mixer" => 2,
        "channel_mixer/embedded_context" => 3,
        "embedded_context_after_channel_mixer" => 4,
        _ => return (3, 0, 0),
    };
    (1, cell, step)
}

pub(crate) fn print_rows(
    rows: &[ActivationRow],
    compared: usize,
    missing: usize,
    extra: usize,
    color: bool,
) {
    let header = [
        "status", "path", "dtype", "shape", "count", "max_abs", "mean_abs", "max_rel", "mean_rel",
        "cosine", "reason",
    ];
    let table: Vec<[String; 11]> = rows
        .iter()
        .map(|row| {
            [
                row.status.into(),
                row.path.clone(),
                row.dtype.clone(),
                row.shape.clone(),
                row.count.map_or("-".into(), |v| v.to_string()),
                num(row.max_abs),
                num(row.mean_abs),
                num(row.max_rel),
                num(row.mean_rel),
                row.cosine.map_or("-".into(), |v| format!("{v:.8}")),
                row.reason.clone(),
            ]
        })
        .collect();
    let mut widths = header.map(str::len);
    for row in &table {
        for (i, value) in row.iter().enumerate() {
            widths[i] = widths[i].max(value.len());
        }
    }

    println!("activation_comparison");
    // Trace paths are often longer than 64 chars; dynamic widths keep later columns readable.
    println!("{}", line(&header.map(str::to_owned), &widths));
    for row in &table {
        let text = line(row, &widths);
        let status = &row[0];
        if color && status == "PASS" {
            println!("\x1b[32m{text}\x1b[0m");
        } else if color && matches!(status.as_str(), "FAIL" | "MISSING" | "EXTRA") {
            println!("\x1b[31m{text}\x1b[0m");
        } else {
            println!("{text}");
        }
    }

    let passed = rows.iter().filter(|r| r.status == "PASS").count();
    let failed = rows.len() - passed;
    let worst_abs = rows.iter().filter_map(|r| r.max_abs).max_by(f64::total_cmp);
    let worst_rel = rows.iter().filter_map(|r| r.max_rel).max_by(f64::total_cmp);
    let worst_cos = rows
        .iter()
        .filter_map(|r| r.cosine)
        .min_by(f64::total_cmp)
        .map_or("-".into(), |v| format!("{v:.8}"));
    println!(
        "activation_summary compared={compared} passed={passed} failed={failed} missing={missing} extra={extra} worst_abs={} worst_rel={} worst_cosine={worst_cos}",
        num(worst_abs),
        num(worst_rel),
    );
}

pub(crate) fn print_timing_rows(
    rows: &[TimeRow],
    compared: usize,
    missing: usize,
    extra: usize,
    ignored: usize,
    color: bool,
) {
    print_timing_groups(rows);

    let header = [
        "status",
        "path",
        "actual_ms",
        "baseline_ms",
        "delta_ms",
        "speedup",
        "reason",
    ];
    let table: Vec<[String; 7]> = rows
        .iter()
        .map(|row| {
            [
                row.status.into(),
                row.path.clone(),
                ms(row.actual_ns),
                ms(row.baseline_ns),
                delta_ms(row.actual_ns, row.baseline_ns),
                row.speedup.map_or("-".into(), |v| format!("{v:.2}x")),
                row.reason.clone(),
            ]
        })
        .collect();
    let mut widths = header.map(str::len);
    for row in &table {
        for (i, value) in row.iter().enumerate() {
            widths[i] = widths[i].max(value.len());
        }
    }

    println!();
    println!("timing_scope=canonical_compute ignored={ignored}");
    println!("module_timing_comparison");
    println!("{}", timing_line(&header.map(str::to_owned), &widths));
    for row in &table {
        let text = timing_line(row, &widths);
        let status = &row[0];
        if color && status == "PASS" {
            println!("\x1b[32m{text}\x1b[0m");
        } else if color && matches!(status.as_str(), "FAIL" | "MISSING" | "EXTRA") {
            println!("\x1b[31m{text}\x1b[0m");
        } else {
            println!("{text}");
        }
    }

    let passed = rows.iter().filter(|r| r.status == "PASS").count();
    let failed = rows.len() - passed;
    let actual_total = rows.iter().filter_map(|r| r.actual_ns).sum::<u64>();
    let baseline_total = rows.iter().filter_map(|r| r.baseline_ns).sum::<u64>();
    let all_speedups_reported = rows
        .iter()
        .filter(|row| row.status == "PASS")
        .all(|row| row.speedup.is_some());
    let speedup = if actual_total == 0 || !all_speedups_reported {
        "-".into()
    } else {
        format!("{:.2}x", baseline_total as f64 / actual_total as f64)
    };
    println!(
        "timing_summary compared={compared} passed={passed} failed={failed} missing={missing} extra={extra} ignored={ignored} actual_total_ms={} baseline_total_ms={} speedup={speedup}",
        ns_ms(actual_total),
        ns_ms(baseline_total),
    );
}

pub(crate) fn timing_groups(rows: &[TimeRow]) -> Vec<TimeGroup> {
    let mut groups = BTreeMap::<String, TimeGroup>::new();
    for row in rows {
        let (Some(actual_ns), Some(baseline_ns)) = (row.actual_ns, row.baseline_ns) else {
            continue;
        };
        let module = module_key(&row.path);
        let group = groups.entry(module.clone()).or_insert_with(|| TimeGroup {
            module,
            count: 0,
            actual_ns: 0,
            baseline_ns: 0,
        });
        group.count += 1;
        group.actual_ns += actual_ns;
        group.baseline_ns += baseline_ns;
    }
    let mut groups = groups.into_values().collect::<Vec<_>>();
    groups.sort_by(|a, b| {
        forward_key(&a.module)
            .cmp(&forward_key(&b.module))
            .then_with(|| a.module.cmp(&b.module))
    });
    groups
}

fn print_timing_groups(rows: &[TimeRow]) {
    let groups = timing_groups(rows);
    if groups.is_empty() {
        return;
    }

    let header = [
        "module",
        "count",
        "actual_total_ms",
        "baseline_total_ms",
        "speedup",
    ];
    let table: Vec<[String; 5]> = groups
        .iter()
        .map(|group| {
            [
                group.module.clone(),
                group.count.to_string(),
                ns_ms(group.actual_ns),
                ns_ms(group.baseline_ns),
                timing_speedup(group.actual_ns, group.baseline_ns),
            ]
        })
        .collect();
    let mut widths = header.map(str::len);
    for row in &table {
        for (i, value) in row.iter().enumerate() {
            widths[i] = widths[i].max(value.len());
        }
    }

    println!();
    println!("timing_by_module");
    println!("{}", timing_group_line(&header.map(str::to_owned), &widths));
    for row in &table {
        println!("{}", timing_group_line(row, &widths));
    }
}

fn module_key(path: &str) -> String {
    let module = path
        .strip_prefix("timing/")
        .unwrap_or(path)
        .strip_suffix(".time.json")
        .unwrap_or(path);
    let Some(rest) = module.strip_prefix("cells/cell_") else {
        return module.to_owned();
    };
    let Some((_, name)) = rest.split_once('/') else {
        return module.to_owned();
    };
    format!("cells/*/{name}")
}

fn line(row: &[String; 11], widths: &[usize; 11]) -> String {
    let mut line = String::new();
    for (i, value) in row.iter().enumerate() {
        if i > 0 {
            line.push_str("  ");
        }
        if (4..=9).contains(&i) {
            line.push_str(&format!("{value:>width$}", width = widths[i]));
        } else {
            line.push_str(&format!("{value:<width$}", width = widths[i]));
        }
    }
    line
}

fn timing_group_line(row: &[String; 5], widths: &[usize; 5]) -> String {
    let mut line = String::new();
    for (i, value) in row.iter().enumerate() {
        if i > 0 {
            line.push_str("  ");
        }
        if (1..=4).contains(&i) {
            line.push_str(&format!("{value:>width$}", width = widths[i]));
        } else {
            line.push_str(&format!("{value:<width$}", width = widths[i]));
        }
    }
    line
}

fn timing_line(row: &[String; 7], widths: &[usize; 7]) -> String {
    let mut line = String::new();
    for (i, value) in row.iter().enumerate() {
        if i > 0 {
            line.push_str("  ");
        }
        if (2..=5).contains(&i) {
            line.push_str(&format!("{value:>width$}", width = widths[i]));
        } else {
            line.push_str(&format!("{value:<width$}", width = widths[i]));
        }
    }
    line
}

fn num(value: Option<f64>) -> String {
    value.map_or("-".into(), |v| format!("{v:.6e}"))
}

fn ms(value: Option<u64>) -> String {
    value.map_or("-".into(), ns_ms)
}

fn delta_ms(actual: Option<u64>, baseline: Option<u64>) -> String {
    match (actual, baseline) {
        (Some(actual), Some(baseline)) => format!("{:.3}", (actual as f64 - baseline as f64) / 1e6),
        _ => "-".into(),
    }
}

fn timing_speedup(actual_ns: u64, baseline_ns: u64) -> String {
    if actual_ns == 0 {
        "-".into()
    } else {
        format!("{:.2}x", baseline_ns as f64 / actual_ns as f64)
    }
}

fn ns_ms(value: u64) -> String {
    format!("{:.3}", value as f64 / 1e6)
}

pub(crate) fn shape(shape: &[usize]) -> String {
    format!(
        "[{}]",
        shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

pub(crate) fn show(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}
