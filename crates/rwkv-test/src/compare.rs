use std::{
    collections::BTreeSet,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};

use crate::{
    cli::{CompareArgs, CompareRwkvNnArgs},
    display::{forward_key, print_rows, print_timing_rows, shape, show},
    numeric::numeric_stats,
    safetensor::load,
    timing::{compare_timing, scan_timing},
};

pub(crate) struct ActivationRow {
    pub(crate) status: &'static str,
    pub(crate) path: String,
    pub(crate) dtype: String,
    pub(crate) shape: String,
    pub(crate) count: Option<usize>,
    pub(crate) max_abs: Option<f64>,
    pub(crate) mean_abs: Option<f64>,
    pub(crate) max_rel: Option<f64>,
    pub(crate) mean_rel: Option<f64>,
    pub(crate) cosine: Option<f64>,
    pub(crate) reason: String,
}

/// Runs `rwkv-nn`, writes an actual trace, and compares it against the baseline.
pub fn compare_rwkv_nn(args: CompareRwkvNnArgs) -> Result<bool> {
    #[cfg(feature = "cuda")]
    {
        crate::rwkv_nn_trace::generate(&args)?;
        println!(
            "rwkv_nn_timing_profile profile=train-forward-steady-state scope=canonical_compute backend=cuda-bf16 warmup={} repeat={}",
            args.warmup, args.repeat
        );
        compare(CompareArgs {
            actual: args.actual,
            baseline: args.baseline,
            atol: args.atol,
            rtol: args.rtol,
            cos_min: args.cos_min,
            rel_eps: args.rel_eps,
            color: args.color,
            allow_extra_baseline: true,
        })
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = args;
        bail!("compare-rwkv-nn requires the rwkv-test cuda feature");
    }
}

/// Compares two RWKV trace case directories and prints a CLI table.
pub fn compare(args: CompareArgs) -> Result<bool> {
    check_args(&args)?;
    let actual = scan(&args.actual)?;
    let baseline = scan(&args.baseline)?;
    let actual_timing = scan_timing(&args.actual)?;
    let baseline_timing = scan_timing(&args.baseline)?;
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
    if !args.allow_extra_baseline {
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
    }

    rows.sort_by(|a, b| {
        forward_key(&a.path)
            .cmp(&forward_key(&b.path))
            .then_with(|| a.path.cmp(&b.path))
    });
    let mut timing = compare_timing(&actual_timing, &baseline_timing)?;
    timing.rows.sort_by(|a, b| {
        forward_key(&a.path)
            .cmp(&forward_key(&b.path))
            .then_with(|| a.path.cmp(&b.path))
    });
    print_rows(&rows, compared, missing, extra, args.color.use_color());
    print_timing_rows(
        &timing.rows,
        timing.compared,
        timing.missing,
        timing.extra,
        timing.ignored,
        args.color.use_color(),
    );
    Ok(rows.iter().any(|row| row.status != "PASS")
        || timing.rows.iter().any(|row| row.status != "PASS"))
}

fn check_args(args: &CompareArgs) -> Result<()> {
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

fn compare_one(args: &CompareArgs, rel: &Path) -> Result<ActivationRow> {
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

    let stats = numeric_stats(&actual.values, &baseline.values, args.rel_eps);
    let mut reasons = Vec::new();
    if actual.exact && actual.bytes != baseline.bytes {
        reasons.push("exact tensor differs".to_owned());
    }
    if !actual.exact && stats.max_abs > args.atol && stats.max_rel > args.rtol {
        reasons.push(format!(
            "max_abs {:.6e} > atol {:.6e} and max_rel {:.6e} > rtol {:.6e}",
            stats.max_abs, args.atol, stats.max_rel, args.rtol
        ));
    }
    if !actual.exact && stats.cosine < args.cos_min {
        reasons.push(format!(
            "cosine {:.8} < cos_min {:.8}",
            stats.cosine, args.cos_min
        ));
    }

    Ok(ActivationRow {
        status: if reasons.is_empty() { "PASS" } else { "FAIL" },
        path: show(rel),
        dtype: actual.dtype.to_string(),
        shape: shape(&actual.shape),
        count: Some(stats.count),
        max_abs: Some(stats.max_abs),
        mean_abs: Some(stats.mean_abs),
        max_rel: Some(stats.max_rel),
        mean_rel: Some(stats.mean_rel),
        cosine: Some(stats.cosine),
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

fn issue(
    status: &'static str,
    path: &Path,
    dtype: impl Into<String>,
    shape: impl Into<String>,
    reason: impl Into<String>,
) -> ActivationRow {
    ActivationRow {
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
