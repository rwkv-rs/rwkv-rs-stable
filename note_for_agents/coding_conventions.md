# 代码风格与实现约束

## 基础风格
- 统一使用半角字符.
- 以 `rustfmt.toml` 为准: 最大行宽 `100`, 4 空格缩进, 分组导入, 不自动重排 import.
- 所有 `use` 尽量放在文件开头, 避免在代码中写绝对路径引用.

## 命名约定
- 模块与函数: `snake_case`
- 类型与 trait: `PascalCase`
- 常量: `SCREAMING_SNAKE_CASE`
- 数量前缀: `num_*`
- 索引命名: `*_index`
- 可训练参数: `params_*`
- 布尔状态: `is_*`
- 布尔控制: `need_*`

## 实现偏好
- 先确认任务目标与边界, 再决定实现手段.
- 若用户或上下文要求遵循官方或社区常见做法, 默认先选择 Rust 官方文档, 编译器 lint, `rustdoc`, `clippy`, 标准库与主流 crate 已提供的能力, 不要自立门户.
- 设计新机制前, 先验证它是否与仓库当前基线兼容, 是否真的解决目标问题, 以及维护成本是否值得.
- 优先避免 `unsafe`, 只有在确有必要且收益明确时才使用.
- 优先拒绝使用 `Result`; 在引入 `Result` 前, 必须先深入思考: 为什么这个错误需要由调用方处理, 而不是直接 `panic!`, `expect` 或 `unwrap`.
- 判断异常边界时, 优先考虑用户心智负担: 若让调用方处理该异常只会增加使用复杂度, 而不是带来真实可恢复性, 就不要把它设计成 `Result`.
- 若核心目标是让系统在正确使用前提下稳定运行, 则不应把内部逻辑错误, 状态不变量破坏, 调度契约破坏, shape 契约不成立或实现缺陷包装成对外 `Err`.
- `Result` 只应用于用户输入异常, 外部 I/O, 配置缺失, 网络/存储依赖失败等确实需要由接口侧或调用方决策的异常路径.
- 对代码逻辑问题导致的 `Result::Err`, 默认结论应是实现有缺陷需要修复, 而不是继续把异常向上传递到接口侧.
- 若补充注释, 优先解释设计意图, 背景约束和实现原因, 而不是复述代码表面行为.

## rwkv-nn 自定义算子
- backend trait 只表达公开 primitive 能力, 例如 forward 的 `fused_<kernel>`; 不要为了单个自定义反向路径新增 `*BackwardBackend` trait.
- autodiff 反向是算子内部实现细节. 在 `backward.rs` 中把 Burn `Backward` 节点放在 `Autodiff<...>` backend impl 附近, 梯度逻辑直接写进 `Backward::backward`.
- 若反向可以用普通 Burn tensor ops 表达, 按 template 风格直接写 tensor ops.
- 若反向需要 CubeCL kernel 或 autotune, 让 `Autodiff<CubeBackend<...>, C>` impl 具体到能直接调用 kernel; autotune 与 launch 逻辑保留在对应 `Backward::backward` 局部.
- 不要创建只转发参数或只命名单个调用点的 helper, 例如 `*_backward`, `*_autodiff`, `*_variant`. 只有当 helper 具备真实复用价值或显著降低复杂度时才抽出.
- tuner candidate 如果只服务一个 backward 节点, 直接内联在 `Tunable::new(...).ok()` 的 closure 中; 具体 kernel launch 的安全注释跟随对应 `unsafe` 块.
- 所有 `use` 仍放文件头部; 可以通过代码顺序表达 “autodiff impl 优先, kernel 细节靠后”, 不要把 import 放到文件中段.

## 配置约束
本项目优先使用 `rwkv-config` 从 TOML 加载配置; 不要默认引入 `.env`, 环境变量或命令行参数式配置作为通用方案, 除非任务明确要求并已与现有设计保持一致.
