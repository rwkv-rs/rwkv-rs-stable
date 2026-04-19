<div align="center">

# rwkv-rs

[![中文版本](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-red)](./README-zh.md)
![Rust](https://img.shields.io/badge/Rust-Systems%20Grade-black)
![RWKV](https://img.shields.io/badge/RWKV-Not%20Another%20Demo-blue)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)

**RWKV needs infrastructure, not more scripts.**

`rwkv-rs` is a Rust project for people who are done pretending that research
prototypes are enough.

</div>

---

## The Problem

Most RWKV code is written to prove a point.

It trains once. Benchmarks once. Demos once. Then reality starts asking harder
questions: deployment, maintainability, regression tracking, service quality,
performance evidence, system boundaries.

That is where most projects run out of answers.

Fast experiments are useful. They are not infrastructure.

---

## Our Position

`rwkv-rs` rejects the idea that RWKV should live inside a pile of temporary
glue.

We are not interested in shipping another disposable training script, another
thin inference wrapper, or another project that becomes harder to trust the
moment it grows beyond one machine and one author.

We want a serious RWKV stack.

One that can be read. One that can be extended. One that can be deployed. One
that does not collapse the moment real engineering standards show up.

---

## Why Rust

Because systems that are expected to last should be built like systems that are
expected to last.

RWKV does not need more runtime ambiguity, softer boundaries, or another layer
of convenience that turns into technical debt under load.

Rust gives `rwkv-rs` the right kind of pressure: explicit boundaries, stronger
concurrency guarantees, predictable performance, and fewer excuses for chaos.

---

## Why Burn

Because discipline matters more than vanity.

`rwkv-rs` is built on [Burn](https://github.com/tracel-ai/burn) for the same
reason serious projects reuse strong foundations: the goal is not to rebuild a
deep learning framework from zero just to say we did.

The goal is to spend effort where it counts: RWKV itself, and the software
stack around it.

---

## Not For Everyone

If you want the shortest path to a throwaway demo, this is probably not the
project you are looking for.

If you want RWKV code that survives contact with real engineering constraints,
it is.

`rwkv-rs` is for researchers, teams, and developers who care about more than
making a screenshot look convincing.

---

## Read More

- Project book: [rwkv-rs-book](./rwkv-rs-book)
- Introduction: [rwkv-rs-book/src/introduction.md](./rwkv-rs-book/src/introduction.md)
- API docs: <https://docs.rs/rwkv>

---

## License

Apache-2.0
