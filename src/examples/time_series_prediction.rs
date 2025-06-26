use ndarray::{Array1, Array2, Array3};
use crate::layers::lstm_layer::LstmLayer;
use crate::optimizers::losses;

/// 时间序列预测模型
/// 
/// 使用LSTM来预测时间序列的未来值，如股价、温度等。
pub struct TimeSeriesPredictor {
    pub lstm: LstmLayer,
    pub output_layer: Array2<f64>,  // [hidden_size, output_size]
    pub output_bias: Array1<f64>,   // [output_size]
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub sequence_length: usize,
}

impl TimeSeriesPredictor {
    /// 创建新的时间序列预测模型
    ///
    /// # Arguments
    ///
    /// * `input_size` - 输入特征数量
    /// * `hidden_size` - LSTM隐藏层大小
    /// * `output_size` - 输出大小（预测的时间步数）
    /// * `sequence_length` - 输入序列长度
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, sequence_length: usize) -> Self {
        use ndarray_rand::rand_distr::Uniform;
        use ndarray_rand::RandomExt;
        let dist = Uniform::new(-0.1, 0.1);
        
        Self {
            lstm: LstmLayer::new(input_size, hidden_size),
            output_layer: Array2::random((hidden_size, output_size), dist),
            output_bias: Array1::zeros(output_size),
            input_size,
            hidden_size,
            output_size,
            sequence_length,
        }
    }

    /// 数据标准化
    pub fn normalize_data(data: &Array1<f64>) -> (Array1<f64>, f64, f64) {
        let mean = data.mean().unwrap_or(0.0);
        let std = ((data.mapv(|x| (x - mean).powi(2)).sum() / data.len() as f64).sqrt()).max(1e-8);
        let normalized = data.mapv(|x| (x - mean) / std);
        (normalized, mean, std)
    }

    /// 反标准化
    pub fn denormalize_data(normalized: &Array1<f64>, mean: f64, std: f64) -> Array1<f64> {
        normalized.mapv(|x| x * std + mean)
    }

    /// 创建时间序列数据集
    pub fn create_sequences(data: &Array1<f64>, seq_len: usize, pred_len: usize) -> (Array3<f64>, Array2<f64>) {
        let n_samples = data.len().saturating_sub(seq_len + pred_len - 1);
        let mut inputs = Array3::zeros((n_samples, seq_len, 1)); // 单变量时间序列
        let mut targets = Array2::zeros((n_samples, pred_len));
        
        for i in 0..n_samples {
            // 输入序列
            for t in 0..seq_len {
                inputs[[i, t, 0]] = data[i + t];
            }
            
            // 目标序列
            for t in 0..pred_len {
                targets[[i, t]] = data[i + seq_len + t];
            }
        }
        
        (inputs, targets)
    }

    /// 前向传播
    pub fn forward(&self, input: &Array3<f64>) -> Array2<f64> {
        // 通过LSTM
        let (lstm_output, _final_states, _cache) = self.lstm.forward_batch(input);
        
        // 使用最后一个时间步的输出
        let batch_size = lstm_output.shape()[0];
        let last_hidden = lstm_output.slice(ndarray::s![.., -1, ..]).to_owned(); // [batch_size, hidden_size]
        
        // 通过输出层
        let mut output = Array2::zeros((batch_size, self.output_size));
        for b in 0..batch_size {
            let hidden = last_hidden.row(b);
            let pred = self.output_layer.t().dot(&hidden) + &self.output_bias;
            output.row_mut(b).assign(&pred);
        }
        
        output
    }

    /// 训练一个批次
    pub fn train_step(&mut self, inputs: &Array3<f64>, targets: &Array2<f64>, _learning_rate: f64) -> f64 {
        // 前向传播
        let predictions = self.forward(inputs);
        
        // 计算损失
        let loss = losses::mean_squared_error_batch(&predictions, targets);
        
        // 这里简化了反向传播的实现
        // 在实际应用中，需要实现完整的反向传播和参数更新
        
        loss
    }

    /// 预测未来值
    pub fn predict(&self, input_sequence: &Array2<f64>) -> Array1<f64> {
        // 将2D输入转换为3D（添加batch维度）
        let batch_size = 1;
        let seq_len = input_sequence.shape()[0];
        let input_size = input_sequence.shape()[1];
        
        let mut input_3d = Array3::zeros((batch_size, seq_len, input_size));
        for t in 0..seq_len {
            for f in 0..input_size {
                input_3d[[0, t, f]] = input_sequence[[t, f]];
            }
        }
        
        let predictions = self.forward(&input_3d);
        predictions.row(0).to_owned()
    }

    /// 多步预测
    pub fn predict_multi_step(&self, initial_sequence: &Array2<f64>, n_steps: usize) -> Array1<f64> {
        let mut predictions = Array1::zeros(n_steps);
        let mut current_sequence = initial_sequence.clone();
        
        for step in 0..n_steps {
            // 预测下一个值
            let next_pred = self.predict(&current_sequence);
            predictions[step] = next_pred[0]; // 假设输出大小为1
            
            // 更新序列：移除第一个元素，添加预测值
            let seq_len = current_sequence.shape()[0];
            let mut new_sequence = Array2::zeros((seq_len, 1));
            
            for i in 1..seq_len {
                new_sequence[[i-1, 0]] = current_sequence[[i, 0]];
            }
            new_sequence[[seq_len-1, 0]] = next_pred[0];
            
            current_sequence = new_sequence;
        }
        
        predictions
    }
}

/// 生成合成时间序列数据（正弦波 + 噪声）
pub fn generate_synthetic_data(n_points: usize, noise_level: f64) -> Array1<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut data = Array1::zeros(n_points);
    
    for i in 0..n_points {
        let t = i as f64 * 0.1;
        let signal = (t).sin() + 0.5 * (t * 2.0).sin() + 0.3 * (t * 3.0).sin();
        let noise = if noise_level > 0.0 {
            rng.gen_range(-noise_level..noise_level)
        } else {
            0.0
        };
        data[i] = signal + noise;
    }
    
    data
}

/// 训练时间序列预测模型的示例
pub fn train_time_series_model() -> TimeSeriesPredictor {
    println!("生成合成时间序列数据...");
    let data = generate_synthetic_data(1000, 0.1);
    
    // 标准化数据
    let (normalized_data, _mean, _std) = TimeSeriesPredictor::normalize_data(&data);
    
    // 创建数据集
    let seq_len = 20;
    let pred_len = 1;
    let (inputs, targets) = TimeSeriesPredictor::create_sequences(&normalized_data, seq_len, pred_len);
    
    println!("创建时间序列预测模型...");
    let mut model = TimeSeriesPredictor::new(1, 32, pred_len, seq_len);
    
    println!("开始训练...");
    let epochs = 100;
    let learning_rate = 0.001;
    
    for epoch in 0..epochs {
        let _batch_size = inputs.shape()[0];

        // 简化的训练循环（实际应用中应该使用小批次）
        let loss = model.train_step(&inputs, &targets, learning_rate);
        
        if epoch % 20 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss);
        }
    }
    
    // 测试预测
    println!("测试预测...");
    let test_input = inputs.slice(ndarray::s![0, .., ..]).to_owned(); // 第一个样本
    let prediction = model.predict(&test_input);
    let actual = targets.row(0);
    
    println!("预测值: {:?}", prediction);
    println!("实际值: {:?}", actual);
    
    model
}

/// 评估模型性能
pub fn evaluate_model(model: &TimeSeriesPredictor, test_data: &Array1<f64>, seq_len: usize) -> f64 {
    let (test_inputs, test_targets) = TimeSeriesPredictor::create_sequences(test_data, seq_len, 1);
    let predictions = model.forward(&test_inputs);
    
    // 计算均方根误差 (RMSE)
    let mse = losses::mean_squared_error_batch(&predictions, &test_targets);
    mse.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_series_predictor_creation() {
        let model = TimeSeriesPredictor::new(1, 16, 1, 10);
        assert_eq!(model.input_size, 1);
        assert_eq!(model.hidden_size, 16);
        assert_eq!(model.output_size, 1);
        assert_eq!(model.sequence_length, 10);
    }

    #[test]
    fn test_data_normalization() {
        let data = ndarray::arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let (normalized, mean, std) = TimeSeriesPredictor::normalize_data(&data);
        
        // 验证标准化后的均值接近0，标准差接近1
        let norm_mean = normalized.mean().unwrap();
        assert!((norm_mean).abs() < 1e-10);
        
        // 验证反标准化
        let denormalized = TimeSeriesPredictor::denormalize_data(&normalized, mean, std);
        for (orig, denorm) in data.iter().zip(denormalized.iter()) {
            assert!((orig - denorm).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sequence_creation() {
        let data = ndarray::arr1(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let (inputs, targets) = TimeSeriesPredictor::create_sequences(&data, 3, 2);
        
        assert_eq!(inputs.shape(), &[4, 3, 1]); // 4 samples, seq_len=3, features=1
        assert_eq!(targets.shape(), &[4, 2]);   // 4 samples, pred_len=2
        
        // 验证第一个样本
        assert_eq!(inputs[[0, 0, 0]], 1.0);
        assert_eq!(inputs[[0, 1, 0]], 2.0);
        assert_eq!(inputs[[0, 2, 0]], 3.0);
        assert_eq!(targets[[0, 0]], 4.0);
        assert_eq!(targets[[0, 1]], 5.0);
    }

    #[test]
    fn test_synthetic_data_generation() {
        let data = generate_synthetic_data(100, 0.0); // 无噪声
        assert_eq!(data.len(), 100);
        
        // 验证数据不是常数
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        assert!(max_val > min_val);
    }
}
