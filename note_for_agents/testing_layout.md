# 测试与基准目录规范

## 核心规则
- 单元测试跟随被测函数所在文件: 函数在哪里定义, 对应单元测试就写在该源码文件的 `#[cfg(test)] mod tests` 中.
- `src/test_utils/**` 用于测试工具和共用逻辑, 例如 backend 类型, fixture, 随机输入构造和断言辅助.
- 跨模块, 集成, 端到端或需要独立测试 crate 边界的测试可放在 `tests/**`.
- 性能基准放在 `benches/**`, 目录结构应与被测源码结构对应.
- 参考数据放在 `testdata/`, 参考数据生成脚本放在 `testgen/`.

## 路径映射
- 源码: `crates/<crate>/src/foo/bar.rs`
- 单元测试: `crates/<crate>/src/foo/bar.rs` 中的 `#[cfg(test)] mod tests`
- 集成测试: `crates/<crate>/tests/foo/bar.rs`
- 基准: `crates/<crate>/benches/foo/bar.rs`

示例:
- `crates/rwkv-nn/src/kernels/addcmul/mod.rs`
  - 单元测试函数: `forward`, `backward`
- `crates/rwkv-nn/benches/kernels/addcmul/forward.rs`
  - 单一算子基准测试函数: `forward`
  - 多算子文件中的基准测试函数: `<kernel_name>_forward`
- `crates/rwkv-nn/benches/kernels/addcmul/backward.rs`
  - 单一算子基准测试函数: `backward`
  - 多算子文件中的基准测试函数: `<kernel_name>_backward`
- `crates/rwkv-eval/src/cores/datasets/maths/gsm8k.rs`
  - 集成或数据集测试可放在 `crates/rwkv-eval/tests/cores/datasets/maths/gsm8k.rs`
- `examples/rwkv-lm/src/inferring.rs`
  - 基准可放在 `examples/rwkv-lm/benches/inferring.rs`

## 模块组织
- 单元测试使用被测源码文件内的 `#[cfg(test)] mod tests`.
- `tests/**` 和 `benches/**` 下的目录结构应与源码结构对齐, 便于从源码路径反查测试或基准.
- 共享测试工具放在 `src/test_utils/**`; 共享基准工具放在对应 `benches/**/mod.rs`.
- 基准入口使用 `#[path = "../mod.rs"] mod common;` 引入同层共享工具.
- 若某个测试或基准覆盖多个输入规模, 场景或 baseline/custom 对比, 应在同一个测试函数内部组织.

## 命名约定
- “测试函数”包括单元测试函数和基准测试函数.
- 普通测试函数命名为 `<被测函数>`.
- 一个测试函数对应一个被测函数; 测试函数内部可以多次调用被测函数, 覆盖不同输入, 边界, 预期输出或 benchmark 输入规模.
- 自定义规则: 所有神经网络 `Module` 和 `Kernel` 的测试函数统一命名为 `forward` / `backward`, 对应前向数值准确性/性能和反向梯度准确性/性能.
- 自定义算子基准若一个 kernel module 只暴露一个算子, 函数命名为 `forward` / `backward`; 若一个 kernel module 暴露多个算子, 函数命名为 `<kernel_name>_forward` / `<kernel_name>_backward`.
- benchmark group, Criterion `BenchmarkId` 和测试体内部变量可以表达场景, 规模, baseline/custom; 测试函数名本身不承载场景.
- 测试函数名使用可直接定位被测对象的语义名称.

## 执行入口
- 单元测试通过 `cargo test -p <crate> <测试函数名>` 或 `cargo nextest run -p <crate> <测试函数名>` 执行.
- `rwkv-eval` 的 benchmark 数据集测试统一从 `cargo nextest run -p rwkv-eval --test benchmark_datasets ...` 进入.
- 单个 crate 的 Criterion 基准通过 `cargo bench -p <crate> --bench <name>` 执行; 需要接入 CI benchmark 历史时, 额外追加 `-- --output-format bencher`.

## 约束
- Python 不参与正式测试编排, 只允许在 `testgen/` 中生成 `testdata/`.
- 改动源码时, 应同步维护与被测函数同文件的单元测试; 命中热点或性能敏感路径时, 同步维护对应 `benches/**` 基准.
- `src/test_utils/**` 只提供测试辅助能力.
