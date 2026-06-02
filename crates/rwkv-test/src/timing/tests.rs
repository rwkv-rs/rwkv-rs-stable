use std::{fs, path::Path};

use tempfile::tempdir;

use super::*;

#[test]
fn compare_timing_uses_only_canonical_compute_rows() {
    let temp = tempdir().unwrap();
    let actual = temp.path().join("actual");
    let baseline = temp.path().join("baseline");

    write_time(&actual, "cells/cell_0000/time_mixer", 10);
    write_time(&baseline, "cells/cell_0000/time_mixer", 40);
    write_time(&actual, "loss/l2wrap_cross_entropy", 5);
    write_time(&baseline, "loss/l2wrap_cross_entropy", 10);

    write_time(&actual, "embedding/token_ids", 1);
    write_time(&baseline, "embedding/token_ids", 1);
    write_time(&baseline, "loss/l2wrap_cross_entropy/lse", 300);
    write_time(&actual, "lm_head/projection", 50);

    let comparison = compare_timing(
        &scan_timing(&actual).unwrap(),
        &scan_timing(&baseline).unwrap(),
    )
    .unwrap();

    assert_eq!(comparison.compared, 2);
    assert_eq!(comparison.missing, 0);
    assert_eq!(comparison.extra, 0);
    assert_eq!(comparison.ignored, 3);
    assert_eq!(
        comparison
            .rows
            .iter()
            .map(|row| row.path.as_str())
            .collect::<Vec<_>>(),
        [
            "timing/cells/cell_0000/time_mixer.time.json",
            "timing/loss/l2wrap_cross_entropy.time.json",
        ],
    );
    assert_eq!(
        crate::display::timing_groups(&comparison.rows)
            .into_iter()
            .map(|group| (group.module, group.actual_ns, group.baseline_ns))
            .collect::<Vec<_>>(),
        [
            ("cells/*/time_mixer".to_owned(), 10, 40,),
            ("loss/l2wrap_cross_entropy".to_owned(), 5, 10),
        ],
    );
}

#[test]
fn compare_timing_compares_optional_projection_when_both_sides_have_it() {
    let temp = tempdir().unwrap();
    let actual = temp.path().join("actual");
    let baseline = temp.path().join("baseline");

    write_time(&actual, "lm_head", 10);
    write_time(&baseline, "lm_head", 20);
    write_time(&actual, "lm_head/projection", 50);
    write_time(&baseline, "lm_head/projection", 100);

    let comparison = compare_timing(
        &scan_timing(&actual).unwrap(),
        &scan_timing(&baseline).unwrap(),
    )
    .unwrap();

    assert_eq!(comparison.compared, 2);
    assert_eq!(comparison.missing, 0);
    assert_eq!(comparison.extra, 0);
    assert_eq!(comparison.ignored, 0);
    assert_eq!(
        comparison
            .rows
            .iter()
            .map(|row| (row.path.as_str(), row.status, row.speedup))
            .collect::<Vec<_>>(),
        [
            ("timing/lm_head.time.json", "PASS", Some(2.0)),
            ("timing/lm_head/projection.time.json", "PASS", Some(2.0)),
        ],
    );
}

#[test]
fn compare_timing_fails_baseline_only_canonical_rows() {
    let temp = tempdir().unwrap();
    let actual = temp.path().join("actual");
    let baseline = temp.path().join("baseline");

    write_time(&actual, "embedding", 5);
    write_time(&baseline, "embedding", 10);
    write_time(&baseline, "layer_norm0", 20);
    write_time(&baseline, "loss/l2wrap_cross_entropy/lse", 30);

    let comparison = compare_timing(
        &scan_timing(&actual).unwrap(),
        &scan_timing(&baseline).unwrap(),
    )
    .unwrap();

    assert_eq!(comparison.compared, 1);
    assert_eq!(comparison.missing, 0);
    assert_eq!(comparison.extra, 1);
    assert_eq!(comparison.ignored, 1);
    assert_eq!(comparison.rows.len(), 2);
    assert_eq!(comparison.rows[1].status, "EXTRA");
    assert_eq!(comparison.rows[1].path, "timing/layer_norm0.time.json");
    assert_eq!(
        comparison.rows[1].reason,
        "canonical timing has no actual match"
    );
}

#[test]
fn scan_timing_rejects_invalid_timing_schema() {
    let temp = tempdir().unwrap();
    let root = temp.path().join("actual");

    write_time_raw(
        &root,
        "embedding",
        r#"{"module":"embedding","elapsed_ns":0,"repeat":1,"warmup":0,"samples_ns":[0]}"#,
    );
    let error = scan_timing(&root).unwrap_err().to_string();
    assert!(error.contains("invalid timing JSON"));

    fs::remove_dir_all(&root).unwrap();
    write_time_raw(
        &root,
        "embedding",
        r#"{"module":"embedding","elapsed_ns":11,"repeat":2,"warmup":1,"samples_ns":[10,10]}"#,
    );
    let error = scan_timing(&root).unwrap_err().to_string();
    assert!(error.contains("invalid timing JSON"));
}

#[test]
fn compare_timing_does_not_report_speedup_for_cold_profile() {
    let temp = tempdir().unwrap();
    let actual = temp.path().join("actual");
    let baseline = temp.path().join("baseline");

    write_time_with_profile(&actual, "embedding", 5, 1, 0, &[5]);
    write_time_with_profile(&baseline, "embedding", 10, 1, 0, &[10]);

    let comparison = compare_timing(
        &scan_timing(&actual).unwrap(),
        &scan_timing(&baseline).unwrap(),
    )
    .unwrap();

    assert_eq!(comparison.rows[0].status, "PASS");
    assert_eq!(comparison.rows[0].speedup, None);
    assert!(comparison.rows[0].reason.contains("cold/debug"));
}

#[test]
fn compare_timing_fails_when_actual_is_slower_than_baseline() {
    let temp = tempdir().unwrap();
    let actual = temp.path().join("actual");
    let baseline = temp.path().join("baseline");

    write_time(&actual, "embedding", 20);
    write_time(&baseline, "embedding", 10);

    let comparison = compare_timing(
        &scan_timing(&actual).unwrap(),
        &scan_timing(&baseline).unwrap(),
    )
    .unwrap();

    assert_eq!(comparison.rows[0].status, "FAIL");
    assert_eq!(comparison.rows[0].speedup, Some(0.5));
    assert_eq!(
        comparison.rows[0].reason,
        "actual timing is slower than baseline"
    );
}

#[test]
fn compare_timing_fails_profile_mismatch() {
    let temp = tempdir().unwrap();
    let actual = temp.path().join("actual");
    let baseline = temp.path().join("baseline");

    write_time_with_profile(&actual, "embedding", 5, 3, 1, &[5, 5, 5]);
    write_time_with_profile(&baseline, "embedding", 10, 1, 1, &[10]);

    let comparison = compare_timing(
        &scan_timing(&actual).unwrap(),
        &scan_timing(&baseline).unwrap(),
    )
    .unwrap();

    assert_eq!(comparison.rows[0].status, "FAIL");
    assert!(
        comparison.rows[0]
            .reason
            .contains("timing profile mismatch")
    );
}

fn write_time(root: &Path, module: &str, elapsed_ns: u64) {
    write_time_with_profile(
        root,
        module,
        elapsed_ns,
        3,
        1,
        &[elapsed_ns, elapsed_ns, elapsed_ns],
    );
}

fn write_time_with_profile(
    root: &Path,
    module: &str,
    elapsed_ns: u64,
    repeat: usize,
    warmup: usize,
    samples_ns: &[u64],
) {
    let samples = samples_ns
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    write_time_raw(
        root,
        module,
        &format!(
            "{{\"module\":\"{module}\",\"elapsed_ns\":{elapsed_ns},\"repeat\":{repeat},\"warmup\":{warmup},\"samples_ns\":[{samples}]}}\n"
        ),
    );
}

fn write_time_raw(root: &Path, module: &str, content: &str) {
    let path = root.join(timing_rel(module));
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, content).unwrap();
}
