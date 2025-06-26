use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};


/// 简单的内存使用跟踪器
pub struct MemoryTracker {
    allocated: AtomicUsize,
    peak_usage: AtomicUsize,
}

impl MemoryTracker {
    pub const fn new() -> Self {
        Self {
            allocated: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
        }
    }

    pub fn allocate(&self, size: usize) {
        let current = self.allocated.fetch_add(size, Ordering::Relaxed) + size;
        let mut peak = self.peak_usage.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_usage.compare_exchange_weak(peak, current, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }
    }

    pub fn deallocate(&self, size: usize) {
        self.allocated.fetch_sub(size, Ordering::Relaxed);
    }

    pub fn current_usage(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }

    pub fn peak_usage(&self) -> usize {
        self.peak_usage.load(Ordering::Relaxed)
    }

    pub fn reset(&self) {
        self.allocated.store(0, Ordering::Relaxed);
        self.peak_usage.store(0, Ordering::Relaxed);
    }
}

/// 全局内存跟踪器实例
static MEMORY_TRACKER: MemoryTracker = MemoryTracker::new();

/// 自定义分配器，用于跟踪内存使用
pub struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            MEMORY_TRACKER.allocate(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        MEMORY_TRACKER.deallocate(layout.size());
    }
}

/// 内存使用分析结果
#[derive(Debug, Clone)]
pub struct MemoryAnalysis {
    pub test_name: String,
    pub initial_memory: usize,
    pub peak_memory: usize,
    pub final_memory: usize,
    pub memory_leaked: usize,
}

impl MemoryAnalysis {
    pub fn memory_increase(&self) -> usize {
        self.peak_memory.saturating_sub(self.initial_memory)
    }

    pub fn print_report(&self) {
        println!("📊 内存分析报告: {}", self.test_name);
        println!("  初始内存: {} bytes", self.initial_memory);
        println!("  峰值内存: {} bytes", self.peak_memory);
        println!("  最终内存: {} bytes", self.final_memory);
        println!("  内存增长: {} bytes", self.memory_increase());
        if self.memory_leaked > 0 {
            println!("  ⚠️  内存泄漏: {} bytes", self.memory_leaked);
        }
    }
}

/// 内存分析器
pub struct MemoryProfiler;

impl MemoryProfiler {
    /// 分析数组内存使用（简化版本，使用估算）
    pub fn analyze_array_memory() -> MemoryAnalysis {
        let f64_size = std::mem::size_of::<f64>();

        // 估算不同大小数组的内存使用
        let small_array_size = 1000 * f64_size;
        let medium_array_size = 100 * 100 * f64_size;
        let large_array_size = 10 * 100 * 100 * f64_size;

        let total_estimated = small_array_size + medium_array_size + large_array_size;

        MemoryAnalysis {
            test_name: "Array Memory Usage".to_string(),
            initial_memory: 0,
            peak_memory: total_estimated,
            final_memory: 0,
            memory_leaked: 0,
        }
    }

    /// 分析RNN层的内存使用（估算版本）
    pub fn analyze_rnn_memory() -> MemoryAnalysis {
        let f64_size = std::mem::size_of::<f64>();

        // 估算RNN层的内存使用
        let input_size = 100;
        let hidden_size = 128;
        let batch_size = 32;
        let seq_len = 50;

        // 权重矩阵内存
        let weight_memory = (hidden_size * hidden_size + hidden_size * input_size + hidden_size) * f64_size;
        // 激活值内存
        let activation_memory = batch_size * seq_len * hidden_size * f64_size;
        // 输入数据内存
        let input_memory = batch_size * seq_len * input_size * f64_size;

        let total_estimated = weight_memory + activation_memory + input_memory;

        MemoryAnalysis {
            test_name: "RNN Layer Memory".to_string(),
            initial_memory: 0,
            peak_memory: total_estimated,
            final_memory: 0,
            memory_leaked: 0,
        }
    }

    /// 分析LSTM层的内存使用（估算版本）
    pub fn analyze_lstm_memory() -> MemoryAnalysis {
        let f64_size = std::mem::size_of::<f64>();

        // 估算LSTM层的内存使用
        let input_size = 100;
        let hidden_size = 128;
        let batch_size = 32;
        let seq_len = 50;

        // LSTM有4个门，每个门都有权重矩阵
        let weight_memory = 4 * (hidden_size * (hidden_size + input_size) + hidden_size) * f64_size;
        // 激活值内存（隐藏状态 + 细胞状态）
        let activation_memory = batch_size * seq_len * hidden_size * 2 * f64_size;
        // 输入数据内存
        let input_memory = batch_size * seq_len * input_size * f64_size;

        let total_estimated = weight_memory + activation_memory + input_memory;

        MemoryAnalysis {
            test_name: "LSTM Layer Memory".to_string(),
            initial_memory: 0,
            peak_memory: total_estimated,
            final_memory: 0,
            memory_leaked: 0,
        }
    }

    /// 运行完整的内存分析
    pub fn run_memory_analysis() {
        println!("🔍 开始内存使用分析...\n");

        let array_analysis = Self::analyze_array_memory();
        array_analysis.print_report();
        println!();

        let rnn_analysis = Self::analyze_rnn_memory();
        rnn_analysis.print_report();
        println!();

        let lstm_analysis = Self::analyze_lstm_memory();
        lstm_analysis.print_report();
        println!();

        // 比较不同层的内存效率
        println!("📈 内存效率比较:");
        println!("  RNN 内存增长: {} bytes", rnn_analysis.memory_increase());
        println!("  LSTM 内存增长: {} bytes", lstm_analysis.memory_increase());
        
        let efficiency_ratio = if lstm_analysis.memory_increase() > 0 {
            rnn_analysis.memory_increase() as f64 / lstm_analysis.memory_increase() as f64
        } else {
            0.0
        };
        println!("  RNN/LSTM 内存比率: {:.2}", efficiency_ratio);

        println!();
        MemoryOptimizer::print_optimization_report();
    }
}

/// 内存使用优化建议
pub struct MemoryOptimizer;

impl MemoryOptimizer {
    /// 估算模型内存需求
    pub fn estimate_model_memory(
        input_size: usize,
        hidden_size: usize,
        seq_len: usize,
        batch_size: usize,
        model_type: &str,
    ) -> usize {
        let f64_size = std::mem::size_of::<f64>();
        
        match model_type {
            "RNN" => {
                // 权重矩阵 + 偏置 + 激活值
                let weights = (hidden_size * hidden_size + hidden_size * input_size + hidden_size) * f64_size;
                let activations = batch_size * seq_len * hidden_size * f64_size;
                weights + activations
            },
            "LSTM" => {
                // 4个门的权重 + 偏置 + 激活值 + 细胞状态
                let weights = 4 * (hidden_size * (hidden_size + input_size) + hidden_size) * f64_size;
                let activations = batch_size * seq_len * hidden_size * 2 * f64_size; // h + c
                weights + activations
            },
            "GRU" => {
                // 3个门的权重 + 偏置 + 激活值
                let weights = 3 * (hidden_size * (hidden_size + input_size) + hidden_size) * f64_size;
                let activations = batch_size * seq_len * hidden_size * f64_size;
                weights + activations
            },
            _ => 0,
        }
    }

    /// 提供内存优化建议
    pub fn suggest_optimizations(memory_usage: usize) -> Vec<String> {
        let mut suggestions = Vec::new();

        if memory_usage > 1_000_000_000 { // > 1GB
            suggestions.push("考虑减少批次大小以降低内存使用".to_string());
            suggestions.push("使用梯度累积代替大批次训练".to_string());
            suggestions.push("考虑使用模型并行化".to_string());
        }

        if memory_usage > 100_000_000 { // > 100MB
            suggestions.push("考虑使用更小的隐藏层大小".to_string());
            suggestions.push("使用截断的反向传播减少序列长度".to_string());
        }

        suggestions.push("使用in-place操作减少临时数组分配".to_string());
        suggestions.push("考虑使用f32而不是f64以减少内存使用".to_string());

        suggestions
    }

    /// 打印内存优化报告
    pub fn print_optimization_report() {
        println!("🛠️  内存优化建议");
        println!("{}", "=".repeat(50));

        // 估算不同配置的内存使用
        let configs = [
            ("小型RNN", "RNN", 50, 64, 25, 16),
            ("中型LSTM", "LSTM", 100, 128, 50, 32),
            ("大型LSTM", "LSTM", 200, 256, 100, 64),
        ];

        for (name, model_type, input_size, hidden_size, seq_len, batch_size) in configs {
            let memory = Self::estimate_model_memory(input_size, hidden_size, seq_len, batch_size, model_type);
            println!("{}: ~{:.1} MB", name, memory as f64 / 1_000_000.0);
            
            if memory > 100_000_000 {
                let suggestions = Self::suggest_optimizations(memory);
                for suggestion in suggestions.iter().take(2) {
                    println!("  💡 {}", suggestion);
                }
            }
        }
    }
}
