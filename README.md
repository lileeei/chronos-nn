# Chronos-NN

`Chronos-NN` 是一个使用 Rust 从零开始实现循环神经网络（RNN）及其现代变体的项目。本项目的目标是遵循 RNN 模型演进的历史轨迹，深入理解其核心原理与架构变迁，最终构建一个功能丰富、模块化的神经网络库。

## 项目状态

当前项目处于初始规划阶段。我们已经定义了详细的实现蓝图和迭代计划。

*   **高级实现计划:** [rnn_implementation_plan.md](rnn_implementation_plan.md)
*   **详细迭代计划:** [iteration_plan.md](iteration_plan.md)

## 核心技术栈

*   **语言:** [Rust](https://www.rust-lang.org/)
*   **核心计算库:** [ndarray](https://github.com/rust-ndarray/ndarray)

## 构建与测试

你可以使用标准的 Cargo 命令来构建和测试本项目。

```bash
# 构建项目 (优化版本)
cargo build --release

# 运行所有测试
cargo test
``` 