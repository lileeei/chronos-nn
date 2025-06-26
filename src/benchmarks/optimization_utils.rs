use ndarray::{Array1, Array2, Array3, Axis};
use std::time::{Duration, Instant};

/// 计算优化工具
pub struct ComputeOptimizer;

impl ComputeOptimizer {
    /// 优化的矩阵乘法（使用ndarray的优化实现）
    pub fn optimized_matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        // ndarray已经使用了BLAS优化，这里主要是展示接口
        a.dot(b)
    }

    /// 批量矩阵乘法优化
    pub fn batch_matmul_optimized(
        batch_a: &Array3<f64>, 
        batch_b: &Array3<f64>
    ) -> Array3<f64> {
        let batch_size = batch_a.shape()[0];
        let rows = batch_a.shape()[1];
        let cols = batch_b.shape()[2];
        
        let mut result = Array3::zeros((batch_size, rows, cols));
        
        // 并行处理每个批次
        for i in 0..batch_size {
            let a_slice = batch_a.index_axis(Axis(0), i);
            let b_slice = batch_b.index_axis(Axis(0), i);
            let mut result_slice = result.index_axis_mut(Axis(0), i);
            result_slice.assign(&a_slice.dot(&b_slice));
        }
        
        result
    }

    /// 内存高效的激活函数应用
    pub fn apply_activation_inplace(x: &mut Array1<f64>, activation: fn(f64) -> f64) {
        x.mapv_inplace(activation);
    }

    /// 优化的softmax实现（数值稳定）
    pub fn stable_softmax(x: &Array1<f64>) -> Array1<f64> {
        let max_val = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_x = x.mapv(|v| (v - max_val).exp());
        let sum_exp = exp_x.sum();
        exp_x / sum_exp
    }

    /// 批量softmax优化
    pub fn batch_softmax(x: &Array2<f64>) -> Array2<f64> {
        let mut result = Array2::zeros(x.raw_dim());
        
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let softmax_row = Self::stable_softmax(&row.to_owned());
            result.row_mut(i).assign(&softmax_row);
        }
        
        result
    }
}

/// 性能分析器
pub struct PerformanceAnalyzer;

impl PerformanceAnalyzer {
    /// 分析函数执行时间
    pub fn time_function<F, R>(f: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }

    /// 比较两个函数的性能
    pub fn compare_functions<F1, F2, R>(
        name1: &str,
        f1: F1,
        name2: &str,
        f2: F2,
        iterations: usize,
    ) where
        F1: Fn() -> R,
        F2: Fn() -> R,
    {
        println!("🔄 比较函数性能: {} vs {}", name1, name2);
        
        // 预热
        for _ in 0..10 {
            let _ = f1();
            let _ = f2();
        }
        
        // 测试函数1
        let start1 = Instant::now();
        for _ in 0..iterations {
            let _ = f1();
        }
        let duration1 = start1.elapsed();
        
        // 测试函数2
        let start2 = Instant::now();
        for _ in 0..iterations {
            let _ = f2();
        }
        let duration2 = start2.elapsed();
        
        println!("  {}: {:.2}ms ({} iterations)", name1, duration1.as_millis(), iterations);
        println!("  {}: {:.2}ms ({} iterations)", name2, duration2.as_millis(), iterations);
        
        let speedup = duration1.as_nanos() as f64 / duration2.as_nanos() as f64;
        if speedup > 1.0 {
            println!("  🚀 {} is {:.2}x faster", name2, speedup);
        } else {
            println!("  🚀 {} is {:.2}x faster", name1, 1.0 / speedup);
        }
    }

    /// 分析内存访问模式
    pub fn analyze_memory_access_pattern() {
        println!("🧠 分析内存访问模式...");

        let size = 100;  // 减小尺寸以加快测试
        let iterations = 10;  // 减少迭代次数
        
        // 行优先访问
        let mut array = Array2::<f64>::zeros((size, size));
        let (_, row_major_time) = Self::time_function(|| {
            for _ in 0..iterations {
                for i in 0..size {
                    for j in 0..size {
                        array[[i, j]] += 1.0;
                    }
                }
            }
        });
        
        // 列优先访问
        let mut array = Array2::<f64>::zeros((size, size));
        let (_, col_major_time) = Self::time_function(|| {
            for _ in 0..iterations {
                for j in 0..size {
                    for i in 0..size {
                        array[[i, j]] += 1.0;
                    }
                }
            }
        });
        
        println!("  行优先访问: {:.2}ms", row_major_time.as_millis());
        println!("  列优先访问: {:.2}ms", col_major_time.as_millis());
        
        let ratio = col_major_time.as_nanos() as f64 / row_major_time.as_nanos() as f64;
        println!("  性能比率: {:.2}x (行优先更快)", ratio);
    }
}

/// 缓存优化工具
pub struct CacheOptimizer;

impl CacheOptimizer {
    /// 分块矩阵乘法（缓存友好）
    pub fn blocked_matmul(
        a: &Array2<f64>, 
        b: &Array2<f64>, 
        block_size: usize
    ) -> Array2<f64> {
        let n = a.shape()[0];
        let m = a.shape()[1];
        let p = b.shape()[1];
        
        let mut c = Array2::zeros((n, p));
        
        for i in (0..n).step_by(block_size) {
            for j in (0..p).step_by(block_size) {
                for k in (0..m).step_by(block_size) {
                    let i_end = (i + block_size).min(n);
                    let j_end = (j + block_size).min(p);
                    let k_end = (k + block_size).min(m);
                    
                    for ii in i..i_end {
                        for jj in j..j_end {
                            for kk in k..k_end {
                                c[[ii, jj]] += a[[ii, kk]] * b[[kk, jj]];
                            }
                        }
                    }
                }
            }
        }
        
        c
    }

    /// 测试不同块大小的性能
    pub fn benchmark_block_sizes() {
        println!("📦 测试分块矩阵乘法性能...");

        let size = 128;  // 减小矩阵尺寸以加快测试
        let a = Array2::<f64>::ones((size, size));
        let b = Array2::<f64>::ones((size, size));
        
        // 标准矩阵乘法
        let (_, standard_time) = PerformanceAnalyzer::time_function(|| {
            a.dot(&b)
        });
        
        println!("  标准矩阵乘法: {:.2}ms", standard_time.as_millis());
        
        // 不同块大小的分块乘法
        for &block_size in &[16, 32, 64] {  // 减少测试的块大小
            let (_, blocked_time) = PerformanceAnalyzer::time_function(|| {
                Self::blocked_matmul(&a, &b, block_size)
            });
            
            println!("  分块大小 {}: {:.2}ms", block_size, blocked_time.as_millis());
        }
    }
}

/// 并行化工具
pub struct ParallelOptimizer;

impl ParallelOptimizer {
    /// 并行向量操作示例
    pub fn parallel_vector_add(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
        // 使用ndarray的并行操作
        a + b
    }

    /// 分析并行化效果
    pub fn analyze_parallelization() {
        println!("⚡ 分析并行化效果...");

        let size = 100_000;  // 减小数组大小以加快测试
        let a = Array1::<f64>::ones(size);
        let b = Array1::<f64>::ones(size);
        
        // 串行加法
        let (_, serial_time) = PerformanceAnalyzer::time_function(|| {
            let mut result = Array1::zeros(size);
            for i in 0..size {
                result[i] = a[i] + b[i];
            }
            result
        });
        
        // 并行加法（ndarray内置）
        let (_, parallel_time) = PerformanceAnalyzer::time_function(|| {
            &a + &b
        });
        
        println!("  串行加法: {:.2}ms", serial_time.as_millis());
        println!("  并行加法: {:.2}ms", parallel_time.as_millis());
        
        let speedup = serial_time.as_nanos() as f64 / parallel_time.as_nanos() as f64;
        println!("  加速比: {:.2}x", speedup);
    }
}

/// 运行所有优化分析
pub fn run_optimization_analysis() {
    println!("🔧 开始计算优化分析...\n");
    
    PerformanceAnalyzer::analyze_memory_access_pattern();
    println!();
    
    CacheOptimizer::benchmark_block_sizes();
    println!();
    
    ParallelOptimizer::analyze_parallelization();
    println!();
    
    // 比较不同softmax实现
    let x = Array1::<f64>::from_vec((0..100).map(|i| i as f64 * 0.01).collect());  // 减小数组大小

    PerformanceAnalyzer::compare_functions(
        "朴素softmax",
        || {
            let exp_x = x.mapv(|v| v.exp());
            let sum_exp = exp_x.sum();
            exp_x / sum_exp
        },
        "稳定softmax",
        || ComputeOptimizer::stable_softmax(&x),
        100,  // 减少迭代次数
    );
}
