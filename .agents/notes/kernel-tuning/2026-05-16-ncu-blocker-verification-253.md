# NCU Blocker Verification 253

- Date: 2026-05-16.
- Branch/worktree: `kernel-tuning-ncu-blocker-verification-253-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout contains broad unrelated workspace edits and previous kernel-tuning notes. This attempt owns only this note unless a later entry explicitly expands scope.
- User constraint: use `10.100.1.253` first and do not use local GPU timing.
- Scope: verify the remote Nsight Compute permission blocker without running a GPU counter profile, then update the project kernel-tuning skill with the current blocker details. This branch must not repeat targeted `ncu` counter commands that already failed with `ERR_NVGPUCTRPERM`.
- Prior evidence:
  - `.agents/skills/kernel-tuning/SKILL.md` records that `10.100.1.253` reports `RmProfilingAdminOnly: 1`, `ncu` is installed at `/usr/local/cuda-13.0/bin/ncu`, and previous normal-user counter runs failed with `ERR_NVGPUCTRPERM`.
  - `2026-05-16-privileged-ncu-command-plan-gb10.md` records the privileged runbook and admin unblock path.
  - `2026-05-16-completion-audit-253-current.md` marks remote `ncu` achieved occupancy / warp efficiency / memory throughput as incomplete and environment-blocked.
- Machine/GPU: remote `caizus@10.100.1.253` / `spark-35ac` / `NVIDIA GB10`.
- Commands to run:
  - `test -x /usr/local/cuda-13.0/bin/ncu && /usr/local/cuda-13.0/bin/ncu --version`
  - `grep RmProfilingAdminOnly /proc/driver/nvidia/params`
  - `sudo -n true`
  - `/usr/local/cuda-13.0/bin/ncu --list-sets | head`
- Expected decision boundary:
  - If `ncu` is present and list-sets works but `RmProfilingAdminOnly` remains `1` and passwordless sudo is unavailable, keep the blocker as current and do not run counter profiles as normal user.
  - If the restriction has changed to `0` or sudo works, open a new profiler branch and run the existing privileged runbook filters.

## Results

- Command host: `spark-35ac`.
- `/usr/local/cuda-13.0/bin/ncu` exists and is executable.
- `ncu --version` succeeded:
  - `NVIDIA (R) Nsight Compute Command Line Profiler`
  - `Version 2025.3.1.0 (build 36398880) (public-release)`
- Driver parameter:
  - `RmProfilingAdminOnly: 1`
- Passwordless sudo:
  - `sudo -n true` failed with `sudo: a password is required`, status `1`.
- `ncu --list-sets` succeeded and shows the expected metric sets:
  - `basic`: `LaunchStats, Occupancy, SpeedOfLight, WorkloadDistribution`
  - `detailed`: includes `ComputeWorkloadAnalysis`, `MemoryWorkloadAnalysis`, `Occupancy`, `SchedulerStats`, `SpeedOfLight`, `WarpStateStats`, and related charts/tables.
  - `full`: includes the detailed workload, scheduler, source, speed-of-light, roofline, warp-state, NVLink, and PM sampling sections.

## Decision

- The remote `ncu` blocker is current and confirmed:
  - tool installation is fine;
  - metric sets are available;
  - normal user cannot bypass the driver restriction with passwordless sudo;
  - driver policy still restricts performance counters to admin users.
- Do not run target `ncu` counter profiles as `caizus` in this permission boundary. The next valid counter run requires an admin to either run the command under a privileged context or change the NVIDIA profiling policy so `RmProfilingAdminOnly` becomes `0`.
- Next edit: update `.agents/skills/kernel-tuning/SKILL.md` so future kernel-tuning attempts know `ncu` is installed and metric sets are available, but normal-user counter collection remains blocked by driver policy and lack of passwordless sudo.
- Skill update result:
  - Updated `.agents/skills/kernel-tuning/SKILL.md` known GB10 profiling blocker with `ncu 2025.3.1.0`, successful `ncu --list-sets`, `RmProfilingAdminOnly: 1`, and `sudo -n true` failure.
  - The skill now distinguishes tool installation from permission blocking: future work should not diagnose this as missing `ncu`, and should not rerun counter profiles as normal `caizus`.
- Next command: run `git diff --check` for the note and skill, then sync both files to `10.100.1.253`.
- Follow-up validation command: run `/usr/local/cuda-13.0/bin/ncu --list-sections` on `10.100.1.253` and grep for the section names used in the privileged runbook: `SpeedOfLight`, `Occupancy`, `SchedulerStats`, `WarpStateStats`, and `MemoryWorkloadAnalysis`. This validates command syntax without running a target counter profile.
- Section-name validation result:
  - `MemoryWorkloadAnalysis` exists.
  - `Occupancy` exists.
  - `SchedulerStats` exists.
  - `SpeedOfLight` exists.
  - `WarpStateStats` exists.
  - Therefore the privileged runbook's `--section` names are valid for Nsight Compute `2025.3.1`; the remaining blocker is permission, not runbook syntax.
