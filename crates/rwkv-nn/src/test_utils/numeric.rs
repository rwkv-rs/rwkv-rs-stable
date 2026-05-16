//! Numeric assertion helpers for trace-backed unit tests.

use burn::{prelude::Backend, tensor::Tensor};

/// Tolerances used by tensor comparison assertions.
#[derive(Clone, Copy, Debug)]
pub struct NumericTolerance {
    /// Absolute-error threshold.
    pub atol: f64,
    /// Relative-error threshold.
    pub rtol: f64,
    /// Minimum cosine similarity.
    pub cos_min: f64,
    /// Denominator floor for relative error.
    pub rel_eps: f64,
}

impl NumericTolerance {
    /// Creates a tolerance from absolute error, relative error, and cosine bounds.
    pub const fn new(atol: f64, rtol: f64, cos_min: f64) -> Self {
        Self {
            atol,
            rtol,
            cos_min,
            rel_eps: 1.0e-12,
        }
    }

    /// Exact comparison in `rwkv-test` terms.
    pub const fn exact() -> Self {
        Self::new(0.0, 0.0, 1.0)
    }
}

/// Numeric statistics reported when a comparison fails.
#[derive(Clone, Copy, Debug)]
pub struct NumericStats {
    /// Number of elements compared.
    pub count: usize,
    /// Maximum absolute error.
    pub max_abs: f64,
    /// Mean absolute error.
    pub mean_abs: f64,
    /// Maximum relative error.
    pub max_rel: f64,
    /// Mean relative error.
    pub mean_rel: f64,
    /// Cosine similarity.
    pub cosine: f64,
}

/// Asserts two floating-point tensors match using the same statistics as `rwkv-test compare`.
pub fn assert_tensor_close<B: Backend, const D: usize>(
    label: &str,
    actual: Tensor<B, D>,
    expected: Tensor<B, D>,
    tolerance: NumericTolerance,
) {
    let actual = actual.into_data().convert::<f32>();
    let expected = expected.into_data().convert::<f32>();
    assert_eq!(
        actual.shape, expected.shape,
        "{label}: shape mismatch actual={:?} expected={:?}",
        actual.shape, expected.shape
    );

    let actual_values = actual.iter::<f32>().map(f64::from).collect::<Vec<_>>();
    let expected_values = expected.iter::<f32>().map(f64::from).collect::<Vec<_>>();
    let stats = stats(&actual_values, &expected_values, tolerance.rel_eps);

    assert!(
        stats.max_abs <= tolerance.atol || stats.max_rel <= tolerance.rtol,
        "{label}: max_abs {:.6e} > atol {:.6e} and max_rel {:.6e} > rtol {:.6e}; \
         mean_abs {:.6e}, mean_rel {:.6e}, cosine {:.8}, count {}",
        stats.max_abs,
        tolerance.atol,
        stats.max_rel,
        tolerance.rtol,
        stats.mean_abs,
        stats.mean_rel,
        stats.cosine,
        stats.count
    );
    assert!(
        stats.cosine >= tolerance.cos_min,
        "{label}: cosine {:.8} < cos_min {:.8}; max_abs {:.6e}, max_rel {:.6e}, count {}",
        stats.cosine,
        tolerance.cos_min,
        stats.max_abs,
        stats.max_rel,
        stats.count
    );
}

fn stats(actual: &[f64], expected: &[f64], eps: f64) -> NumericStats {
    let (mut max_abs, mut sum_abs, mut max_rel, mut sum_rel, mut dot, mut aa, mut bb) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (&actual, &expected) in actual.iter().zip(expected) {
        let abs = (actual - expected).abs();
        let rel = abs / expected.abs().max(eps);
        max_abs = f64::max(max_abs, abs);
        max_rel = f64::max(max_rel, rel);
        sum_abs += abs;
        sum_rel += rel;
        dot += actual * expected;
        aa += actual * actual;
        bb += expected * expected;
    }

    let count = actual.len();
    let cosine = if count == 0 || max_abs == 0.0 {
        1.0
    } else if aa == 0.0 || bb == 0.0 {
        0.0
    } else {
        dot / (aa.sqrt() * bb.sqrt())
    };

    NumericStats {
        count,
        max_abs,
        mean_abs: if count == 0 {
            0.0
        } else {
            sum_abs / count as f64
        },
        max_rel,
        mean_rel: if count == 0 {
            0.0
        } else {
            sum_rel / count as f64
        },
        cosine,
    }
}
