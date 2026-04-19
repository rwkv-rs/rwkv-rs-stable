<div align="center">

# rwkv-rs

[![English README](https://img.shields.io/badge/README-English-blue)](./README.md)
![Rust](https://img.shields.io/badge/Rust-%E7%B3%BB%E7%BB%9F%E7%BA%A7-black)
![RWKV](https://img.shields.io/badge/RWKV-%E4%B8%8D%E6%98%AF%E5%8F%88%E4%B8%80%E4%B8%AA%20Demo-red)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)

**RWKV 需要的是基础设施, 不是更多脚本。**

`rwkv-rs` 是写给那些已经不再把研究原型当成终点的人看的 Rust 项目。

</div>

---

## 问题在哪

今天大多数 RWKV 代码, 本质上都是为了证明一个点子成立。

它能训一次, 能跑一次 benchmark, 能做一次演示. 然后现实开始提出更难的问题:
部署怎么办, 维护怎么办, 回归怎么追踪, 服务质量怎么保证, 性能证据在哪里,
系统边界在哪里。

很多项目走到这里, 就没有答案了。

快速实验当然有价值. 但快速实验不是基础设施。

---

## 我们的立场

`rwkv-rs` 不接受 RWKV 长期停留在一堆临时胶水代码里的状态。

我们不想再做一个一次性的训练脚本, 不想再做一个只包一层接口的推理薄封装,
也不想再做一个只要项目规模稍微变大, 就立刻变得难以信任的仓库。

我们要的是一套严肃的 RWKV 工程栈。

它应该能读, 能扩展, 能部署, 能在真正的工程标准面前站得住, 而不是一碰到
真实场景就开始塌。

---

## 为什么是 Rust

因为一个想长期存在的系统, 就应该用对待长期系统的方式来构建。

RWKV 不需要更多运行时含糊空间, 不需要更软的边界, 也不需要另一层看似省事,
最后却在压力下全部变成技术债的便利抽象。

Rust 给 `rwkv-rs` 的不是噱头, 而是正确的约束: 更明确的边界, 更可靠的并发
保证, 更可预测的性能, 以及更少把混乱合理化的借口。

---

## 为什么基于 Burn

因为纪律比虚荣重要。

`rwkv-rs` 基于 [Burn](https://github.com/tracel-ai/burn), 原因很简单: 严肃项目
会复用足够强的地基, 而不是为了证明自己什么都能造, 就从零重复搭一遍深度学习
框架。

真正该花力气的地方, 是 RWKV 本身, 以及围绕它建立起来的整套软件栈。

---

## 它本来就不打算讨好所有人

如果你要的是最快跑通一个一次性 demo 的路径, 这个项目大概率不适合你。

如果你要的是一套在真实工程约束面前还能站住的 RWKV 代码, 那它就是给你准备的。

`rwkv-rs` 面向的是研究者, 团队, 和那些不满足于“截图看起来能跑”这件事的开发者。

---

## 继续阅读

- 项目手册: [rwkv-rs-book](./rwkv-rs-book)
- 项目简介: [rwkv-rs-book/src/introduction.md](./rwkv-rs-book/src/introduction.md)
- API 文档: <https://docs.rs/rwkv>

---

## License

Apache-2.0
