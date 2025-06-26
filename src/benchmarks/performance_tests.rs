use std::time::{Duration, Instant};
use ndarray::Array3;
use crate::layers::{
    rnn_layer::RnnLayer,
    lstm_layer::LstmLayer,
    gru_layer::GruLayer,
    multi_layer_rnn::MultiLayerLstm,
};
use crate::attention::encoder_decoder::AttentionEncoderDecoder;


/// 性能测试结果
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub test_name: String,
    pub duration: Duration,
    pub throughput: f64,  // samples per second
    pub memory_usage: Option<usize>,  // bytes
}

impl BenchmarkResult {
    pub fn new(test_name: String, duration: Duration, samples: usize) -> Self {
        let throughput = samples as f64 / duration.as_secs_f64();
        Self {
            test_name,
            duration,
            throughput,
            memory_usage: None,
        }
    }

    pub fn with_memory(mut self, memory_bytes: usize) -> Self {
        self.memory_usage = Some(memory_bytes);
        self
    }
}

/// 性能基准测试套件
pub struct PerformanceBenchmark {
    pub results: Vec<BenchmarkResult>,
}

impl PerformanceBenchmark {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// 运行所有基准测试
    pub fn run_all_benchmarks(&mut self) {
        println!("🚀 开始运行性能基准测试...\n");

        self.benchmark_rnn_forward();
        self.benchmark_lstm_forward();
        self.benchmark_gru_forward();
        self.benchmark_multi_layer_lstm();
        self.benchmark_attention_encoder_decoder();
        self.benchmark_batch_processing();
        self.benchmark_sequence_lengths();

        println!("\n📊 基准测试完成！");
        self.print_summary();
    }

    /// 基准测试 RNN 前向传播
    pub fn benchmark_rnn_forward(&mut self) {
        let input_size = 50;
        let hidden_size = 64;
        let seq_len = 25;
        let batch_size = 16;
        let iterations = 10;

        let rnn = RnnLayer::new(input_size, hidden_size);
        let input = Array3::<f64>::ones((batch_size, seq_len, input_size));

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = rnn.forward_batch(&input);
        }
        let duration = start.elapsed();

        let result = BenchmarkResult::new(
            "RNN Forward Pass".to_string(),
            duration,
            iterations * batch_size,
        );
        
        println!("✅ RNN Forward: {:.2}ms, {:.0} samples/sec", 
                 duration.as_millis(), result.throughput);
        self.results.push(result);
    }

    /// 基准测试 LSTM 前向传播
    pub fn benchmark_lstm_forward(&mut self) {
        let input_size = 50;
        let hidden_size = 64;
        let seq_len = 25;
        let batch_size = 16;
        let iterations = 10;

        let lstm = LstmLayer::new(input_size, hidden_size);
        let input = Array3::<f64>::ones((batch_size, seq_len, input_size));

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = lstm.forward_batch(&input);
        }
        let duration = start.elapsed();

        let result = BenchmarkResult::new(
            "LSTM Forward Pass".to_string(),
            duration,
            iterations * batch_size,
        );
        
        println!("✅ LSTM Forward: {:.2}ms, {:.0} samples/sec", 
                 duration.as_millis(), result.throughput);
        self.results.push(result);
    }

    /// 基准测试 GRU 前向传播
    pub fn benchmark_gru_forward(&mut self) {
        let input_size = 50;
        let hidden_size = 64;
        let seq_len = 25;
        let batch_size = 16;
        let iterations = 10;

        let gru = GruLayer::new(input_size, hidden_size);
        let input = Array3::<f64>::ones((batch_size, seq_len, input_size));

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = gru.forward_batch(&input);
        }
        let duration = start.elapsed();

        let result = BenchmarkResult::new(
            "GRU Forward Pass".to_string(),
            duration,
            iterations * batch_size,
        );
        
        println!("✅ GRU Forward: {:.2}ms, {:.0} samples/sec", 
                 duration.as_millis(), result.throughput);
        self.results.push(result);
    }

    /// 基准测试多层 LSTM
    pub fn benchmark_multi_layer_lstm(&mut self) {
        let input_size = 50;
        let hidden_size = 64;
        let seq_len = 25;
        let batch_size = 8;
        let iterations = 5;

        let layer1 = LstmLayer::new(input_size, hidden_size);
        let layer2 = LstmLayer::new(hidden_size, hidden_size);
        let layer3 = LstmLayer::new(hidden_size, hidden_size);
        
        let multi_lstm = MultiLayerLstm::new(
            vec![layer1, layer2, layer3],
            0.1,
            false,
            hidden_size,
        );

        let input = Array3::<f64>::ones((batch_size, seq_len, input_size));

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = multi_lstm.forward(&input);
        }
        let duration = start.elapsed();

        let result = BenchmarkResult::new(
            "Multi-Layer LSTM".to_string(),
            duration,
            iterations * batch_size,
        );
        
        println!("✅ Multi-Layer LSTM: {:.2}ms, {:.0} samples/sec", 
                 duration.as_millis(), result.throughput);
        self.results.push(result);
    }

    /// 基准测试注意力编码器-解码器
    pub fn benchmark_attention_encoder_decoder(&mut self) {
        let input_size = 25;
        let encoder_hidden = 32;
        let decoder_hidden = 32;
        let output_size = 25;
        let seq_len = 10;
        let batch_size = 4;
        let iterations = 5;

        let model = AttentionEncoderDecoder::new(
            input_size,
            encoder_hidden,
            decoder_hidden,
            output_size,
        );

        let input_seq = Array3::<f64>::ones((batch_size, seq_len, input_size));
        let target_seq = Array3::<f64>::ones((batch_size, seq_len, input_size));

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = model.forward(&input_seq, &target_seq);
        }
        let duration = start.elapsed();

        let result = BenchmarkResult::new(
            "Attention Encoder-Decoder".to_string(),
            duration,
            iterations * batch_size,
        );
        
        println!("✅ Attention Encoder-Decoder: {:.2}ms, {:.0} samples/sec", 
                 duration.as_millis(), result.throughput);
        self.results.push(result);
    }

    /// 基准测试不同批次大小的性能
    pub fn benchmark_batch_processing(&mut self) {
        let input_size = 50;
        let hidden_size = 64;
        let seq_len = 25;
        let iterations = 5;

        let lstm = LstmLayer::new(input_size, hidden_size);

        for &batch_size in &[1, 4, 8, 16] {
            let input = Array3::<f64>::ones((batch_size, seq_len, input_size));

            let start = Instant::now();
            for _ in 0..iterations {
                let _ = lstm.forward_batch(&input);
            }
            let duration = start.elapsed();

            let result = BenchmarkResult::new(
                format!("LSTM Batch Size {}", batch_size),
                duration,
                iterations * batch_size,
            );
            
            println!("✅ LSTM Batch {}: {:.2}ms, {:.0} samples/sec", 
                     batch_size, duration.as_millis(), result.throughput);
            self.results.push(result);
        }
    }

    /// 基准测试不同序列长度的性能
    pub fn benchmark_sequence_lengths(&mut self) {
        let input_size = 50;
        let hidden_size = 64;
        let batch_size = 8;
        let iterations = 5;

        let lstm = LstmLayer::new(input_size, hidden_size);

        for &seq_len in &[5, 10, 20, 40] {
            let input = Array3::<f64>::ones((batch_size, seq_len, input_size));

            let start = Instant::now();
            for _ in 0..iterations {
                let _ = lstm.forward_batch(&input);
            }
            let duration = start.elapsed();

            let result = BenchmarkResult::new(
                format!("LSTM Seq Length {}", seq_len),
                duration,
                iterations * batch_size,
            );
            
            println!("✅ LSTM Seq {}: {:.2}ms, {:.0} samples/sec", 
                     seq_len, duration.as_millis(), result.throughput);
            self.results.push(result);
        }
    }

    /// 打印性能总结
    pub fn print_summary(&self) {
        println!("\n📈 性能基准测试总结");
        println!("{}", "=".repeat(60));
        
        for result in &self.results {
            println!("{:<30} | {:>8.2}ms | {:>10.0} samples/sec", 
                     result.test_name, 
                     result.duration.as_millis(),
                     result.throughput);
        }
        
        println!("{}", "=".repeat(60));
        
        // 找出最快和最慢的测试
        if let (Some(fastest), Some(slowest)) = (
            self.results.iter().max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap()),
            self.results.iter().min_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap())
        ) {
            println!("🏆 最快: {} ({:.0} samples/sec)", fastest.test_name, fastest.throughput);
            println!("🐌 最慢: {} ({:.0} samples/sec)", slowest.test_name, slowest.throughput);
        }
    }

    /// 导出结果为CSV格式
    pub fn export_csv(&self, filename: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(filename)?;
        writeln!(file, "Test Name,Duration (ms),Throughput (samples/sec),Memory (bytes)")?;
        
        for result in &self.results {
            writeln!(file, "{},{},{},{}", 
                     result.test_name,
                     result.duration.as_millis(),
                     result.throughput,
                     result.memory_usage.unwrap_or(0))?;
        }
        
        println!("📄 结果已导出到: {}", filename);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_result_creation() {
        let result = BenchmarkResult::new(
            "Test".to_string(),
            Duration::from_millis(100),
            1000,
        );

        assert_eq!(result.test_name, "Test");
        assert_eq!(result.duration, Duration::from_millis(100));
        assert_eq!(result.throughput, 10000.0); // 1000 samples / 0.1 seconds = 10000 samples/sec
    }

    #[test]
    fn test_performance_benchmark_creation() {
        let benchmark = PerformanceBenchmark::new();
        assert_eq!(benchmark.results.len(), 0);
    }

    #[test]
    fn test_benchmark_rnn_forward() {
        // 简化的测试版本
        let input_size = 10;
        let hidden_size = 16;
        let seq_len = 5;
        let batch_size = 2;
        let iterations = 1;

        let rnn = RnnLayer::new(input_size, hidden_size);
        let input = Array3::<f64>::ones((batch_size, seq_len, input_size));

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = rnn.forward_batch(&input);
        }
        let duration = start.elapsed();

        assert!(duration.as_nanos() > 0);
    }

    #[test]
    fn test_benchmark_lstm_forward() {
        // 简化的测试版本
        let input_size = 10;
        let hidden_size = 16;
        let seq_len = 5;
        let batch_size = 2;
        let iterations = 1;

        let lstm = LstmLayer::new(input_size, hidden_size);
        let input = Array3::<f64>::ones((batch_size, seq_len, input_size));

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = lstm.forward_batch(&input);
        }
        let duration = start.elapsed();

        assert!(duration.as_nanos() > 0);
    }
}
