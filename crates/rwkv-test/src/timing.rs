use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::display::show;

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct TimeRecord {
    module: String,
    elapsed_ns: u64,
    repeat: usize,
    warmup: usize,
    samples_ns: Vec<u64>,
}

pub(crate) struct TimeRow {
    pub(crate) status: &'static str,
    pub(crate) path: String,
    pub(crate) actual_ns: Option<u64>,
    pub(crate) baseline_ns: Option<u64>,
    pub(crate) speedup: Option<f64>,
    pub(crate) reason: String,
}

#[derive(Clone, Debug)]
pub(crate) struct TimeFile {
    path: PathBuf,
    record: TimeRecord,
}

pub(crate) struct TimeGroup {
    pub(crate) module: String,
    pub(crate) count: usize,
    pub(crate) actual_ns: u64,
    pub(crate) baseline_ns: u64,
}

pub(crate) struct TimingComparison {
    pub(crate) rows: Vec<TimeRow>,
    pub(crate) compared: usize,
    pub(crate) missing: usize,
    pub(crate) extra: usize,
    pub(crate) ignored: usize,
}

pub(crate) fn scan_timing(root: &Path) -> Result<BTreeMap<String, TimeFile>> {
    fn walk(timing_root: &Path, dir: &Path, out: &mut BTreeMap<String, TimeFile>) -> Result<()> {
        for entry in
            fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            let ty = entry.file_type()?;
            if ty.is_dir() {
                walk(timing_root, &path, out)?;
            } else if ty.is_file() && show(&path).ends_with(".time.json") {
                let content = fs::read_to_string(&path)
                    .with_context(|| format!("failed to read {}", path.display()))?;
                let record = sonic_rs::from_str::<TimeRecord>(&content)
                    .with_context(|| format!("failed to parse timing JSON {}", path.display()))?;
                let rel = path.strip_prefix(timing_root)?;
                let module_from_path = show(rel)
                    .strip_suffix(".time.json")
                    .expect("timing file has .time.json suffix")
                    .to_owned();
                if record.module != module_from_path {
                    bail!(
                        "{} module mismatch: JSON has {}, path has {}",
                        path.display(),
                        record.module,
                        module_from_path
                    );
                }
                validate_time_record(&record)
                    .with_context(|| format!("invalid timing JSON {}", path.display()))?;
                if out
                    .insert(
                        record.module.clone(),
                        TimeFile {
                            path: timing_rel(&record.module),
                            record,
                        },
                    )
                    .is_some()
                {
                    bail!("duplicate timing module in {}", path.display());
                }
            }
        }
        Ok(())
    }

    let mut out = BTreeMap::new();
    let timing_root = root.join("timing");
    if timing_root.exists() {
        walk(&timing_root, &timing_root, &mut out)?;
    }
    Ok(out)
}

pub(crate) fn compare_timing(
    actual: &BTreeMap<String, TimeFile>,
    baseline: &BTreeMap<String, TimeFile>,
) -> Result<TimingComparison> {
    let (mut rows, mut compared, mut missing, mut extra) = (Vec::new(), 0, 0, 0);
    let mut ignored = 0;
    let mut comparable = BTreeSet::new();

    for module in actual
        .keys()
        .chain(baseline.keys())
        .collect::<BTreeSet<_>>()
    {
        match timing_module_kind(module) {
            TimingModuleKind::Required => {
                comparable.insert(module.clone());
            }
            TimingModuleKind::Optional => {
                if actual.contains_key(module) && baseline.contains_key(module) {
                    comparable.insert(module.clone());
                } else {
                    ignored += 1;
                }
            }
            TimingModuleKind::Ignored => {
                ignored += 1;
            }
        }
    }

    let actual = actual
        .iter()
        .filter(|(module, _)| comparable.contains(*module))
        .collect::<BTreeMap<_, _>>();
    let baseline = baseline
        .iter()
        .filter(|(module, _)| comparable.contains(*module))
        .collect::<BTreeMap<_, _>>();

    for (module, actual_file) in &actual {
        if let Some(baseline_file) = baseline.get(module) {
            compared += 1;
            rows.push(compare_time_one(actual_file, baseline_file));
        } else {
            missing += 1;
            rows.push(time_issue(
                "MISSING",
                &actual_file.path,
                None,
                None,
                "baseline timing is missing",
            ));
        }
    }
    for (module, baseline_file) in baseline {
        if actual.contains_key(module) {
            continue;
        }
        extra += 1;
        rows.push(time_issue(
            "EXTRA",
            &baseline_file.path,
            None,
            None,
            "canonical timing has no actual match",
        ));
    }

    Ok(TimingComparison {
        rows,
        compared,
        missing,
        extra,
        ignored,
    })
}

fn validate_time_record(record: &TimeRecord) -> Result<()> {
    if record.module.is_empty() {
        bail!("module must not be empty");
    }
    if record.elapsed_ns == 0 {
        bail!("elapsed_ns must be positive");
    }
    if record.repeat == 0 {
        bail!("repeat must be positive");
    }
    if record.samples_ns.len() != record.repeat {
        bail!(
            "samples_ns length {} must equal repeat {}",
            record.samples_ns.len(),
            record.repeat
        );
    }
    if record.samples_ns.iter().any(|sample| *sample == 0) {
        bail!("samples_ns entries must be positive");
    }
    let average = record.samples_ns.iter().sum::<u64>() as f64 / record.samples_ns.len() as f64;
    if record.elapsed_ns != average.round() as u64 {
        bail!("elapsed_ns must equal rounded average of samples_ns");
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TimingModuleKind {
    Required,
    Optional,
    Ignored,
}

fn timing_module_kind(module: &str) -> TimingModuleKind {
    match module {
        "embedding"
        | "layer_norm0"
        | "lm_head"
        | "loss/l2wrap_cross_entropy"
        | "loss/head_l2wrap_cross_entropy" => return TimingModuleKind::Required,
        "lm_head/projection" => return TimingModuleKind::Optional,
        _ => {}
    }

    let Some(rest) = module.strip_prefix("cells/cell_") else {
        return TimingModuleKind::Ignored;
    };
    let Some((cell, name)) = rest.split_once('/') else {
        return TimingModuleKind::Ignored;
    };
    if cell.len() != 4 || !cell.bytes().all(|b| b.is_ascii_digit()) {
        return TimingModuleKind::Ignored;
    }
    if matches!(
        name,
        "pre_layer_norm_for_time_mix"
            | "time_mixer"
            | "embedded_context_after_time_mixer"
            | "pre_layer_norm_for_channel_mix"
            | "channel_mixer"
            | "embedded_context_after_channel_mixer"
    ) {
        TimingModuleKind::Required
    } else {
        TimingModuleKind::Ignored
    }
}

fn compare_time_one(actual: &TimeFile, baseline: &TimeFile) -> TimeRow {
    let actual_ns = actual.record.elapsed_ns;
    let baseline_ns = baseline.record.elapsed_ns;
    let (status, speedup, reason) = if actual.record.repeat != baseline.record.repeat
        || actual.record.warmup != baseline.record.warmup
    {
        (
            "FAIL",
            None,
            format!(
                "timing profile mismatch: actual repeat={} warmup={} baseline repeat={} warmup={}",
                actual.record.repeat,
                actual.record.warmup,
                baseline.record.repeat,
                baseline.record.warmup
            ),
        )
    } else if actual.record.warmup == 0 {
        (
            "PASS",
            None,
            "timing profile is cold/debug; speedup is not reported".to_owned(),
        )
    } else {
        let speedup = baseline_ns as f64 / actual_ns as f64;
        if speedup < 1.0 {
            (
                "FAIL",
                Some(speedup),
                "actual timing is slower than baseline".to_owned(),
            )
        } else {
            ("PASS", Some(speedup), String::new())
        }
    };

    TimeRow {
        status,
        path: show(&actual.path),
        actual_ns: Some(actual_ns),
        baseline_ns: Some(baseline_ns),
        speedup,
        reason,
    }
}

fn time_issue(
    status: &'static str,
    path: &Path,
    actual_ns: Option<u64>,
    baseline_ns: Option<u64>,
    reason: impl Into<String>,
) -> TimeRow {
    TimeRow {
        status,
        path: show(path),
        actual_ns,
        baseline_ns,
        speedup: None,
        reason: reason.into(),
    }
}

pub(crate) fn timing_rel(module: &str) -> PathBuf {
    PathBuf::from(format!("timing/{module}.time.json"))
}

#[cfg(test)]
mod tests;
