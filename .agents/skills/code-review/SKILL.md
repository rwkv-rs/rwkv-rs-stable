---
name: code-review
description: Code review constraints and reviewer behavior for this Rust workspace. Use for code review, PR review, pre-merge checks, quality inspection, reviewing comments/tests, or deciding whether changes should block merge.
---

# Code Review Constraints

## Goals

- The primary goal of code review is continuous improvement of repository-wide code health, not achieving local "perfection" before merging.
- Reviewers should prioritize correctness, regression risk, maintainability, architectural consistency, and test effectiveness. Do not spend primary effort on formatting or pure personal preference.
- Mark pure suggestions as non-blocking. Block merge for issues that would reduce code health.

## Default Process

- Whenever the task is code review, PR review, pre-merge check, code quality inspection, or "look at whether this code has issues", use `$pragmatic-clean-code-reviewer` by default.
- If `$pragmatic-clean-code-reviewer` is unavailable, use `$skill-installer` to install `https://github.com/Zhen-Bo/pragmatic-clean-code-reviewer` as `pragmatic-clean-code-reviewer` before continuing the review.
- If the user does not provide level calibration, use that skill's default `L3 Team` rules.
- During review, inspect the diff and direct context first, then read related modules as needed. Do not expand review scope without bounds.
- Review output should lead with findings, sorted by severity. State must-fix issues before should-fix issues.

## Repository Baseline

- Before applying higher-level clean-code rules, first check whether the change satisfies `$coding-conventions`.
- If naming, `unsafe` usage, `Result` boundaries, configuration loading, or comment style violates hard repository constraints, raise it directly as a review finding.
- Judge Rust idioms according to this repository's existing patterns, workspace structure, and module responsibilities. Do not mechanically apply object-oriented-language rules.
- Issues reliably caught by `rustfmt`, `clippy`, the compiler, or existing automation should not be the main review content unless they hide deeper logic or design problems.

## Comment Review

- Comments should explain design intent, background constraints, invariants, compatibility reasons, or why a more direct implementation cannot be used.
- Treat the following comments as dead comments by default unless they carry additional context:
  - Comments that literally restate surface code behavior.
  - Comments that rewrite variable names, branch names, or function calls into natural language.
  - Comments inconsistent with the current implementation.
  - Comments that try to "teach" an obscure implementation without making the code or interface itself clearer.
- If a reviewer asks the author to explain hard-to-understand logic, the expected outcome is usually rewriting the code, extracting better-named structure, or adding necessary comments for future readers. It should not stop at an explanation in the review conversation.

## Test Review

- Tests must prove behavior, contracts, or regression risks that could be broken. They should not only prove "the code compiles" or "the current implementation executes once".
- Treat the following tests as dead tests by default and point them out in review:
  - Tests that only verify type assembly, construction, or getter/setter flows that are guaranteed by compilation and have no behavioral branch, boundary condition, or contract constraint.
  - Tests whose assertions are isomorphic to the implementation and just copy the tested logic into the test to compute it again.
  - Tests with no observable behavior constraints, only happy path coverage, and no failure under common regressions.
  - Tests that depend on heavy mocks or fixed scaffolding without verifying real business invariants.
- Good tests should hit at least one of: boundary conditions, error propagation, state transitions, external contracts, historical regressions, or cross-module integration risk.
- If a new test only improves coverage appearance without improving regression capture, do not treat it as sufficient testing.

## Reviewer Behavior

- Comment on code, not authors. Keep wording direct, clear, and respectful.
- When raising issues, explain why they affect code health, correctness, or maintainability when possible.
- Balance "pointing out the problem" and "giving the solution": reviewers do not need to write the implementation, but should provide a sufficiently concrete repair direction.
- Clearly label purely educational or non-blocking suggestions as `Nit`, `Optional`, or `FYI` so authors do not interpret every comment as required.
- If an author proves with facts, data, or stable engineering principles that another implementation is also reasonable, accept the author's approach. Do not package personal preference as a hard rule.

## Conclusion Standard

- If the change clearly improves overall system code health, reviewers should lean toward approval even if it is still imperfect.
- Changes that reduce overall code health should not be approved, even if the issues are scattered and "each one is small".
- When the review conclusion depends on risk judgment, priority order is: security > correctness > regression risk > design consistency > style and readability details.

## References

- `$pragmatic-clean-code-reviewer`: Use as the default review skill. Install from `https://github.com/Zhen-Bo/pragmatic-clean-code-reviewer` with `$skill-installer` if it is unavailable.
- Google Engineering Practices, Reviewer Guide: https://google.github.io/eng-practices/review/reviewer/
