---
name: commit-and-pr
description: Commit message and pull request conventions for this Rust workspace. Use when preparing commits, writing PR descriptions, choosing Conventional Commit types/scopes, or documenting validation commands and affected areas.
---

# Commit and PR Conventions

## Commit Messages

Commit history uses Conventional Commits style. Recommended format:

```text
<type>(<scope>): <summary>
```

Examples:
- `fix(rwkv-infer): skip guided decoding during prefill`
- `feat(rwkv-nn/kernels): add guided token mask kernel`
- `chore(deps): bump workspace dependencies`

## Conventional Commit Types

- `feat`: Add a feature.
- `fix`: Fix a bug.
- `docs`: Documentation change.
- `style`: Code formatting change that does not affect functionality.
- `refactor`: Code refactor.
- `perf`: Improve performance.
- `test`: Test change.
- `build`: Change project build or external dependencies.
- `ci`: Change continuous integration configuration or script commands.
- `chore`: Change build flow or helper tools.
- `revert`: Revert code.

## PR Requirements

- Explain the change purpose and impact scope.
- List affected crates, examples, or deployment flows.
- State the verification commands actually run.
- Clearly mark any extra configuration, token, hardware, or driver environment requirements.
- Screenshots are needed only for interface or documentation presentation changes.
