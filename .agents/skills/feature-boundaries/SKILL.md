---
name: feature-boundaries
description: Conditional compilation and Cargo feature boundary rules for this Rust workspace. Use when adding or reviewing features, optional dependencies, dependency feature forwarding, cfg gates, default features, public API gating, or crate facade behavior.
---

# Feature Boundaries

## Goal

Keep conditional compilation control at the crate that owns the affected code and direct
dependency. Parent crates may compose public child-crate features, but must not reach through
children to control their private implementation dependencies.

## Ownership Checks

- Put a feature in the crate that contains the `#[cfg]` or `cfg_attr` code it controls.
- Put dependency feature mappings in the crate that directly depends on that dependency.
- When a crate uses a direct dependency's gated API, declare the exact dependency feature in that
  crate's manifest instead of relying on another crate to activate it.
- If a dependency type appears in public API, treat that dependency as this crate's public contract.
- Put provider-crate contract features under the crate that directly consumes that contract.
- A pure facade must not select provider-crate feature families on behalf of downstream crates.
- Do not define feature rules for modules or crates whose responsibility is not implemented or stable.

## Layering Rules

- For `A -> B -> C`, `A` may enable `B/some-feature`; `A` must not enable `C/some-feature`
  to satisfy `B`'s implementation.
- A facade feature may forward a direct child feature only when the facade's own public API or
  implementation genuinely owns that capability.
- Keep implementation-only dependencies behind the owning crate. Do not duplicate a dependency's
  feature matrix across multiple parent crates.
- Use `default-features = false` for workspace child dependencies unless there is an explicit,
  documented reason to inherit their defaults.
- Keep dependency versions in root `workspace.dependencies`; put crate-specific feature choices in
  the consuming crate unless the feature is intentionally workspace-wide.

## Naming and Defaults

- Prefer capability names over dependency names.
- Use dependency, backend, or technology names only when that choice is intentionally public.
- Keep `default` lightweight, portable, and low-surprise.
- Prefer the smallest named feature set that matches the source code's actual imports and API use.
- Exclude heavy, platform-specific, experimental, networked, hardware-specific, benchmark, or
  observability-only behavior from `default`.
- Avoid broad convenience features such as `full`, backend bundles, runtime bundles, macros, or
  observability features when narrower feature names satisfy the code.

## Public API Gates

- Feature-gated public API must compile cleanly when the feature is disabled.
- Disabled features must not leave public types, trait bounds, re-exports, or docs that require
  the disabled dependency.
- If a facade re-exports another crate, its feature should normally enable or forward that crate,
  not reconstruct that crate's internal dependency choices.

## Verification

- After changing dependency features, run a scope-matched check such as `cargo check -p <crate>`.
- Inspect actual activation with `cargo tree -p <crate> -e features` and confirm that broad features
  like runtime `full`, backend bundles, or macros were not introduced accidentally.
- If a necessary direct dependency feature pulls extra transitive features, record the source first;
  optimize further only when doing so does not break ownership, API, or implementation boundaries.

## Review Checklist

- Reject transitive forwarding, such as a parent crate enabling a grandchild dependency feature for a
  child crate's implementation.
- Reject hidden default expansion that changes compile time, platform support, or runtime
  requirements without an explicit user feature.
- Reject feature fixes that only work because a sibling, facade, or test target happens to activate
  the dependency feature first.
- Reject speculative feature names for future crates or modules before their real responsibility
  and direct dependencies are clear.
- Reject stale comments or docs that describe old feature ownership.
- When unsure, choose the smaller public feature surface and keep implementation choices local.
