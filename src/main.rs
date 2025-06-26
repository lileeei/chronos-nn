use chronos_nn::benchmarks::{
    performance_tests::PerformanceBenchmark,
    memory_profiling::MemoryProfiler,
    optimization_utils::run_optimization_analysis,
};

fn main() {
    println!("🚀 Chronos-NN 性能基准测试套件");
    println!("{}", "=".repeat(50));

    // 运行性能基准测试
    let mut benchmark = PerformanceBenchmark::new();
    benchmark.run_all_benchmarks();

    // 导出结果
    if let Err(e) = benchmark.export_csv("benchmark_results.csv") {
        println!("❌ 导出CSV失败: {}", e);
    }

    println!("\n");

    // 运行内存分析
    MemoryProfiler::run_memory_analysis();

    println!("\n");

    // 运行优化分析
    run_optimization_analysis();

    println!("\n🎉 所有基准测试完成！");
}
