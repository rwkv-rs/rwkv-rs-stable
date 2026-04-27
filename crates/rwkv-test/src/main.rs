#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
#![warn(rustdoc::invalid_html_tags)]

//! Command line tools for validating RWKV trace outputs.

use std::{
    collections::BTreeSet,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use half::{bf16, f16};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};

#[derive(Parser)]
#[command(name = "rwkv-test", about = "RWKV trace comparison utilities")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Compare(Args),
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    actual: PathBuf,
    #[arg(long)]
    baseline: PathBuf,
    #[arg(long, default_value_t = 0.0)]
    atol: f64,
    #[arg(long, default_value_t = 0.0)]
    rtol: f64,
    #[arg(long, default_value_t = 1.0)]
    cos_min: f64,
    #[arg(long, default_value_t = 1e-12)]
    rel_eps: f64,
}

struct Tensor {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
    values: Vec<f64>,
    exact: bool,
}

struct Row {
    status: &'static str,
    path: String,
    dtype: String,
    shape: String,
    count: Option<usize>,
    max_abs: Option<f64>,
    mean_abs: Option<f64>,
    max_rel: Option<f64>,
    mean_rel: Option<f64>,
    cosine: Option<f64>,
    reason: String,
}

fn main() -> ExitCode {
    let Command::Compare(args) = Cli::parse().command;
    match compare(args) {
        Ok(true) => ExitCode::from(1),
        Ok(false) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::from(2)
        }
    }
}

fn compare(args: Args) -> Result<bool> {
    check_args(&args)?;
    let actual = scan(&args.actual)?;
    let baseline = scan(&args.baseline)?;
    let (mut rows, mut compared, mut missing, mut extra) = (Vec::new(), 0, 0, 0);

    for rel in &actual {
        if baseline.contains(rel) {
            compared += 1;
            rows.push(compare_one(&args, rel)?);
        } else {
            missing += 1;
            rows.push(issue("MISSING", rel, "-", "-", "baseline file is missing"));
        }
    }
    for rel in baseline.difference(&actual) {
        extra += 1;
        rows.push(issue(
            "EXTRA",
            rel,
            "-",
            "-",
            "baseline file has no actual match",
        ));
    }

    rows.sort_by(|a, b| {
        (b.status != "PASS")
            .cmp(&(a.status != "PASS"))
            .then_with(|| {
                b.max_abs
                    .unwrap_or(f64::INFINITY)
                    .total_cmp(&a.max_abs.unwrap_or(f64::INFINITY))
            })
            .then_with(|| a.path.cmp(&b.path))
    });
    print_rows(&rows, compared, missing, extra);
    Ok(rows.iter().any(|row| row.status != "PASS"))
}

fn check_args(args: &Args) -> Result<()> {
    if !args.actual.is_dir() {
        bail!("actual path is not a directory: {}", args.actual.display());
    }
    if !args.baseline.is_dir() {
        bail!(
            "baseline path is not a directory: {}",
            args.baseline.display()
        );
    }
    for (name, value) in [
        ("atol", args.atol),
        ("rtol", args.rtol),
        ("cos-min", args.cos_min),
    ] {
        if !value.is_finite() || value < 0.0 {
            bail!("{name} must be finite and non-negative");
        }
    }
    if !args.rel_eps.is_finite() || args.rel_eps <= 0.0 {
        bail!("rel-eps must be finite and greater than 0");
    }
    Ok(())
}

fn compare_one(args: &Args, rel: &Path) -> Result<Row> {
    let actual = match load(&args.actual.join(rel))? {
        Ok(tensor) => tensor,
        Err(reason) => return Ok(issue("FAIL", rel, "-", "-", format!("actual: {reason}"))),
    };
    let baseline = match load(&args.baseline.join(rel))? {
        Ok(tensor) => tensor,
        Err(reason) => return Ok(issue("FAIL", rel, "-", "-", format!("baseline: {reason}"))),
    };
    if actual.dtype != baseline.dtype {
        return Ok(issue(
            "FAIL",
            rel,
            format!("actual={} baseline={}", actual.dtype, baseline.dtype),
            "-",
            "dtype mismatch",
        ));
    }
    if actual.shape != baseline.shape {
        return Ok(issue(
            "FAIL",
            rel,
            actual.dtype.to_string(),
            format!(
                "actual={} baseline={}",
                shape(&actual.shape),
                shape(&baseline.shape)
            ),
            "shape mismatch",
        ));
    }

    let (count, max_abs, mean_abs, max_rel, mean_rel, cosine) =
        stats(&actual.values, &baseline.values, args.rel_eps);
    let mut reasons = Vec::new();
    if actual.exact && actual.bytes != baseline.bytes {
        reasons.push("exact tensor differs".to_owned());
    }
    if !actual.exact && max_abs > args.atol && max_rel > args.rtol {
        reasons.push(format!(
            "max_abs {max_abs:.6e} > atol {:.6e} and max_rel {max_rel:.6e} > rtol {:.6e}",
            args.atol, args.rtol
        ));
    }
    if !actual.exact && cosine < args.cos_min {
        reasons.push(format!("cosine {cosine:.8} < cos_min {:.8}", args.cos_min));
    }

    Ok(Row {
        status: if reasons.is_empty() { "PASS" } else { "FAIL" },
        path: show(rel),
        dtype: actual.dtype.to_string(),
        shape: shape(&actual.shape),
        count: Some(count),
        max_abs: Some(max_abs),
        mean_abs: Some(mean_abs),
        max_rel: Some(max_rel),
        mean_rel: Some(mean_rel),
        cosine: Some(cosine),
        reason: reasons.join("; "),
    })
}

fn scan(root: &Path) -> Result<BTreeSet<PathBuf>> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeSet<PathBuf>) -> Result<()> {
        for entry in
            fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            let ty = entry.file_type()?;
            if ty.is_dir() {
                walk(root, &path, out)?;
            } else if ty.is_file() && path.extension() == Some(OsStr::new("safetensors")) {
                out.insert(path.strip_prefix(root)?.to_owned());
            }
        }
        Ok(())
    }
    let mut out = BTreeSet::new();
    walk(root, root, &mut out)?;
    Ok(out)
}

fn load(path: &Path) -> Result<std::result::Result<Tensor, String>> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("failed to parse safetensors {}", path.display()))?;
    let tensors = st.tensors();
    if tensors.len() != 1 {
        return Ok(Err(format!(
            "expected exactly one tensor, found {}",
            tensors.len()
        )));
    }
    let (name, view) = tensors.into_iter().next().unwrap();
    let Some(stem) = path.file_stem().and_then(OsStr::to_str) else {
        return Ok(Err("file stem is not valid UTF-8".into()));
    };
    if name != stem {
        return Ok(Err(format!(
            "tensor name `{name}` does not match file stem `{stem}`"
        )));
    }
    Ok(decode(view))
}

fn decode(view: TensorView<'_>) -> std::result::Result<Tensor, String> {
    let (raw, dtype, dims) = (view.data(), view.dtype(), view.shape().to_vec());
    let values = match dtype {
        Dtype::F64 => chunks(raw, 8, |b| f64::from_le_bytes(arr(b))),
        Dtype::F32 => chunks(raw, 4, |b| f32::from_le_bytes(arr(b)) as f64),
        Dtype::F16 => chunks(raw, 2, |b| {
            f16::from_bits(u16::from_le_bytes(arr(b))).to_f64()
        }),
        Dtype::BF16 => chunks(raw, 2, |b| {
            bf16::from_bits(u16::from_le_bytes(arr(b))).to_f64()
        }),
        Dtype::I64 => chunks(raw, 8, |b| i64::from_le_bytes(arr(b)) as f64),
        Dtype::I32 => chunks(raw, 4, |b| i32::from_le_bytes(arr(b)) as f64),
        Dtype::I16 => chunks(raw, 2, |b| i16::from_le_bytes(arr(b)) as f64),
        Dtype::I8 => raw.iter().map(|v| *v as i8 as f64).collect(),
        Dtype::U64 => chunks(raw, 8, |b| u64::from_le_bytes(arr(b)) as f64),
        Dtype::U32 => chunks(raw, 4, |b| u32::from_le_bytes(arr(b)) as f64),
        Dtype::U16 => chunks(raw, 2, |b| u16::from_le_bytes(arr(b)) as f64),
        Dtype::U8 => raw.iter().map(|v| f64::from(*v)).collect(),
        Dtype::BOOL => raw
            .iter()
            .map(|v| if *v == 0 { 0.0 } else { 1.0 })
            .collect(),
        Dtype::F4
        | Dtype::F6_E2M3
        | Dtype::F6_E3M2
        | Dtype::F8_E5M2
        | Dtype::F8_E4M3
        | Dtype::F8_E8M0
        | Dtype::C64 => return Err(format!("unsupported dtype {dtype}")),
        _ => return Err(format!("unsupported dtype {dtype}")),
    };

    let count = dims
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or_else(|| format!("shape {} overflows usize", shape(&dims)))?;
    if count != values.len() {
        return Err(format!(
            "shape {} expects {count} elements, decoded {}",
            shape(&dims),
            values.len()
        ));
    }

    // Integer and bool tensors usually carry ids or masks, so tolerance-based comparison is wrong.
    Ok(Tensor {
        dtype,
        shape: dims,
        bytes: raw.to_vec(),
        values,
        exact: matches!(
            dtype,
            Dtype::I64
                | Dtype::I32
                | Dtype::I16
                | Dtype::I8
                | Dtype::U64
                | Dtype::U32
                | Dtype::U16
                | Dtype::U8
                | Dtype::BOOL
        ),
    })
}

fn chunks(raw: &[u8], size: usize, f: impl Fn(&[u8]) -> f64) -> Vec<f64> {
    raw.chunks_exact(size).map(f).collect()
}

fn arr<const N: usize>(bytes: &[u8]) -> [u8; N] {
    bytes.try_into().unwrap()
}

fn stats(a: &[f64], b: &[f64], eps: f64) -> (usize, f64, f64, f64, f64, f64) {
    let (mut max_abs, mut sum_abs, mut max_rel, mut sum_rel, mut dot, mut aa, mut bb) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (&a, &b) in a.iter().zip(b) {
        let abs = (a - b).abs();
        let rel = abs / b.abs().max(eps);
        max_abs = f64::max(max_abs, abs);
        max_rel = f64::max(max_rel, rel);
        sum_abs += abs;
        sum_rel += rel;
        dot += a * b;
        aa += a * a;
        bb += b * b;
    }

    // Exact matches should not fail because cosine rounds to a value just below 1.0.
    let n = a.len();
    let cosine = if n == 0 || max_abs == 0.0 {
        1.0
    } else if aa == 0.0 || bb == 0.0 {
        0.0
    } else {
        dot / (aa.sqrt() * bb.sqrt())
    };
    (
        n,
        max_abs,
        if n == 0 { 0.0 } else { sum_abs / n as f64 },
        max_rel,
        if n == 0 { 0.0 } else { sum_rel / n as f64 },
        cosine,
    )
}

fn issue(
    status: &'static str,
    path: &Path,
    dtype: impl Into<String>,
    shape: impl Into<String>,
    reason: impl Into<String>,
) -> Row {
    Row {
        status,
        path: show(path),
        dtype: dtype.into(),
        shape: shape.into(),
        count: None,
        max_abs: None,
        mean_abs: None,
        max_rel: None,
        mean_rel: None,
        cosine: None,
        reason: reason.into(),
    }
}

fn print_rows(rows: &[Row], compared: usize, missing: usize, extra: usize) {
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

    // Trace paths are often longer than 64 chars; dynamic widths keep later columns readable.
    print_line(&header.map(str::to_owned), &widths);
    for row in &table {
        print_line(row, &widths);
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
        "summary compared={compared} passed={passed} failed={failed} missing={missing} extra={extra} worst_abs={} worst_rel={} worst_cosine={worst_cos}",
        num(worst_abs),
        num(worst_rel),
    );
}

fn print_line(row: &[String; 11], widths: &[usize; 11]) {
    for (i, value) in row.iter().enumerate() {
        if i > 0 {
            print!("  ");
        }
        if (4..=9).contains(&i) {
            print!("{value:>width$}", width = widths[i]);
        } else {
            print!("{value:<width$}", width = widths[i]);
        }
    }
    println!();
}

fn num(value: Option<f64>) -> String {
    value.map_or("-".into(), |v| format!("{v:.6e}"))
}

fn shape(shape: &[usize]) -> String {
    format!(
        "[{}]",
        shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn show(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}
