---
name: project-structure
description: Repository structure and module organization guidance for this Rust workspace. Use when deciding where code belongs, understanding crate responsibilities, adding dependencies, or mapping features to crates, examples, assets, and generated build output.
---

# Project Structure and Module Organization

## Repository Positioning

This project aims to build a complete Rust ecosystem around RWKV models. Many modules are still under construction. Some descriptions represent planned direction and should not be assumed to be current implementation.

## Directory Overview

- `crates/`: Core library code.
- `examples/`: End-to-end examples, experiment entrypoints, and integration validation scripts.
- `.github/workflows/`: Deployment workflows.
- `target/`: Build artifacts. Do not treat this as source code to modify.

## Main Crate Responsibilities

- `rwkv`: Unified export entrypoint and feature facade layer.
- `rwkv-config`, `rwkv-derive`: Configuration contracts, derive macros, and configuration loading.
- `rwkv-data`: Data cleaning pipelines, mmap dataset read/write, and sampling logic.
- `rwkv-nn`: Model structures, modules, layers, functions, and high-performance operator implementations.
- `rwkv-train`: Pre-training, post-training, PEFT, StateTuning, and other training capabilities.
- `rwkv-infer`: High-performance inference engine, IPC/HTTP interfaces, and backend integration.
- `rwkv-eval`: LLM benchmark evaluation.
- `rwkv-export`: Model weight mapping and export.
- `rwkv-prompt`: Prompt and StateTuning effect research.
- `rwkv-trace`: Interpretability and training/inference process analysis.
- `rwkv-agent`: Event loop, tool calling, memory, and interactive agent capabilities.
- `rwkv-bench`: Performance benchmarking and bottleneck analysis.

## Examples and Resources

- `examples/text-data-clean-pipeline`: Data cleaning pipeline example.
- `examples/paired-dna-mmap-dataset-converter`: mmap data conversion example.
- If large directories such as `data/` or `weights/` exist in the future, treat them as assets or weights. Avoid scattering implementation logic there.

## Structure Constraints

Manage dependency versions for all child crates through the workspace. When adding dependencies, prefer declaring them in root `Cargo.toml` under `workspace.dependencies`, then reference them from the child crate.
