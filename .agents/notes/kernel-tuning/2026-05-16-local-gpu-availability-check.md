# Local GPU Availability Check

- Date: 2026-05-16
- Branch/worktree: `kernel-tuning-local-gpu-availability-check-20260516` in the existing dirty local checkout.
- Dirty-tree constraint: this checkout inherits broad unrelated workspace changes and the current remote-clean tuning tree. This branch is read-only except for this note.
- Prior-note/source search command:
  - `rg -n "local GPU|本机|local compare|speedup=|nvidia-smi|local timing|GPU busy|不要用本机|10\\.100\\.1\\.253" .agents/notes/kernel-tuning /root/.codex/memories/MEMORY.md -S`
- Matched prior evidence:
  - User constraint: debug first on `10.100.1.253`; local GPU may be busy and timing may be inaccurate.
  - `2026-05-16-current-goal-audit-remote-clean.md` says the active goal is not complete because local trace-backed speedup `>1.0` has not been achieved or freshly verified on an idle local GPU.
  - Prior local compares remain below `1.0`; the current remote clean gate is `2.10x`.
- Changed boundary: this is only a lightweight availability check, not a local benchmark or profiler run.
- Machine/GPU: local machine, CUDA GPU if visible.
- Shape/dtype: not applicable; no model workload is launched.
- Candidate parameters: none.
- Expected keep/revert boundary: do not run `compare-rwkv-nn`, `ncu`, or `nsys` locally unless the GPU is clearly idle and the user explicitly allows local timing. Record GPU process/memory utilization as availability evidence only.
- Next command: run `nvidia-smi` query for GPU name, utilization, memory, and active compute processes.

## 2026-05-16 Availability Result

- GPU query result: `NVIDIA GeForce RTX 5090`, compute capability `12.0`, utilization `5%`, memory `3022 / 32607 MiB`.
- Compute process query result: no active compute apps reported by `nvidia-smi --query-compute-apps`.
- Interpretation: the local GPU appears idle at this instant, but the user previously said the local GPU is reserved for other work and local timing may be inaccurate. This branch does not launch a benchmark.
- Decision: do not run local `compare-rwkv-nn` until the user explicitly allows local timing or confirms the machine is idle enough for a final acceptance run.
