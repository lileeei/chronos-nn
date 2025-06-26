use std::collections::HashMap;
use ndarray::{Array1, Array2, Array3};
use crate::layers::lstm_layer::LstmLayer;
use crate::optimizers::losses;

/// 字符级语言模型
/// 
/// 这个模型可以学习文本的字符级模式，并生成新的文本。
/// 它使用LSTM来捕获长期依赖关系。
pub struct CharLevelLanguageModel {
    pub lstm: LstmLayer,
    pub output_layer: Array2<f64>,  // [hidden_size, vocab_size]
    pub output_bias: Array1<f64>,   // [vocab_size]
    pub char_to_idx: HashMap<char, usize>,
    pub idx_to_char: HashMap<usize, char>,
    pub vocab_size: usize,
    pub hidden_size: usize,
}

impl CharLevelLanguageModel {
    /// 创建新的字符级语言模型
    ///
    /// # Arguments
    ///
    /// * `text` - 训练文本，用于构建词汇表
    /// * `hidden_size` - LSTM隐藏层大小
    pub fn new(text: &str, hidden_size: usize) -> Self {
        // 构建字符词汇表
        let unique_chars: std::collections::HashSet<char> = text.chars().collect();
        let mut char_to_idx = HashMap::new();
        let mut idx_to_char = HashMap::new();
        
        for (i, &ch) in unique_chars.iter().enumerate() {
            char_to_idx.insert(ch, i);
            idx_to_char.insert(i, ch);
        }
        
        let vocab_size = unique_chars.len();
        
        // 初始化网络层
        use ndarray_rand::rand_distr::Uniform;
        use ndarray_rand::RandomExt;
        let dist = Uniform::new(-0.1, 0.1);
        
        Self {
            lstm: LstmLayer::new(vocab_size, hidden_size),
            output_layer: Array2::random((hidden_size, vocab_size), dist),
            output_bias: Array1::zeros(vocab_size),
            char_to_idx,
            idx_to_char,
            vocab_size,
            hidden_size,
        }
    }

    /// 将文本转换为one-hot编码
    fn text_to_onehot(&self, text: &str) -> Array3<f64> {
        let chars: Vec<char> = text.chars().collect();
        let seq_len = chars.len();
        let mut onehot = Array3::zeros((1, seq_len, self.vocab_size)); // batch_size=1
        
        for (t, &ch) in chars.iter().enumerate() {
            if let Some(&idx) = self.char_to_idx.get(&ch) {
                onehot[[0, t, idx]] = 1.0;
            }
        }
        
        onehot
    }

    /// 将one-hot编码转换回文本
    fn onehot_to_text(&self, onehot: &Array3<f64>) -> String {
        let mut text = String::new();
        let seq_len = onehot.shape()[1];
        
        for t in 0..seq_len {
            let mut max_idx = 0;
            let mut max_val = onehot[[0, t, 0]];
            
            for i in 1..self.vocab_size {
                if onehot[[0, t, i]] > max_val {
                    max_val = onehot[[0, t, i]];
                    max_idx = i;
                }
            }
            
            if let Some(&ch) = self.idx_to_char.get(&max_idx) {
                text.push(ch);
            }
        }
        
        text
    }

    /// Softmax函数
    fn softmax(&self, x: &Array1<f64>) -> Array1<f64> {
        let max_val = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_x = x.mapv(|v| (v - max_val).exp());
        let sum_exp = exp_x.sum();
        exp_x / sum_exp
    }

    /// 前向传播
    pub fn forward(&self, input: &Array3<f64>) -> Array3<f64> {
        // 通过LSTM
        let (lstm_output, _final_states, _cache) = self.lstm.forward_batch(input);
        
        // 通过输出层
        let batch_size = lstm_output.shape()[0];
        let seq_len = lstm_output.shape()[1];
        let mut output = Array3::zeros((batch_size, seq_len, self.vocab_size));
        
        for b in 0..batch_size {
            for t in 0..seq_len {
                let hidden = lstm_output.slice(ndarray::s![b, t, ..]).to_owned();
                let logits = self.output_layer.t().dot(&hidden) + &self.output_bias;
                let probs = self.softmax(&logits);
                
                for i in 0..self.vocab_size {
                    output[[b, t, i]] = probs[i];
                }
            }
        }
        
        output
    }

    /// 训练一个批次
    pub fn train_step(&mut self, input_text: &str, target_text: &str, _learning_rate: f64) -> f64 {
        // 准备数据
        let input_onehot = self.text_to_onehot(input_text);
        let target_onehot = self.text_to_onehot(target_text);
        
        // 前向传播
        let output = self.forward(&input_onehot);
        
        // 计算损失（交叉熵）
        let seq_len = output.shape()[1];
        let mut total_loss = 0.0;
        
        for t in 0..seq_len {
            let pred = output.slice(ndarray::s![0, t, ..]).to_owned();
            let target = target_onehot.slice(ndarray::s![0, t, ..]).to_owned();
            total_loss += losses::binary_cross_entropy(&pred, &target);
        }
        
        total_loss / seq_len as f64
    }

    /// 生成文本
    pub fn generate(&self, seed_text: &str, length: usize, temperature: f64) -> String {
        let mut generated = seed_text.to_string();
        let mut current_input = self.text_to_onehot(seed_text);
        
        for _ in 0..length {
            // 前向传播
            let output = self.forward(&current_input);
            
            // 获取最后一个时间步的输出
            let last_output = output.slice(ndarray::s![0, -1, ..]).to_owned();
            
            // 应用温度采样
            let scaled_logits = last_output.mapv(|x| x / temperature);
            let probs = self.softmax(&scaled_logits);
            
            // 采样下一个字符
            let next_char_idx = self.sample_from_probs(&probs);
            
            if let Some(&next_char) = self.idx_to_char.get(&next_char_idx) {
                generated.push(next_char);
                
                // 准备下一次输入（使用最后一个字符）
                let mut next_input = Array3::zeros((1, 1, self.vocab_size));
                next_input[[0, 0, next_char_idx]] = 1.0;
                current_input = next_input;
            } else {
                break;
            }
        }
        
        generated
    }

    /// 从概率分布中采样
    fn sample_from_probs(&self, probs: &Array1<f64>) -> usize {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let random_val: f64 = rng.gen_range(0.0..1.0);
        
        let mut cumsum = 0.0;
        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if random_val <= cumsum {
                return i;
            }
        }
        
        // 如果没有采样到，返回最后一个索引
        probs.len() - 1
    }

    /// 获取词汇表大小
    pub fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// 获取字符到索引的映射
    pub fn get_char_to_idx(&self) -> &HashMap<char, usize> {
        &self.char_to_idx
    }
}

/// 训练字符级语言模型的辅助函数
pub fn train_char_lm(text: &str, hidden_size: usize, epochs: usize, learning_rate: f64) -> CharLevelLanguageModel {
    let mut model = CharLevelLanguageModel::new(text, hidden_size);
    let chars: Vec<char> = text.chars().collect();
    
    println!("开始训练字符级语言模型...");
    println!("词汇表大小: {}", model.get_vocab_size());
    println!("隐藏层大小: {}", hidden_size);
    
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        
        // 使用滑动窗口创建训练样本
        let seq_len = 10; // 序列长度
        for i in 0..chars.len().saturating_sub(seq_len) {
            let input_chars: String = chars[i..i+seq_len].iter().collect();
            let target_chars: String = chars[i+1..i+seq_len+1].iter().collect();
            
            let loss = model.train_step(&input_chars, &target_chars, learning_rate);
            total_loss += loss;
            num_batches += 1;
        }
        
        let avg_loss = total_loss / num_batches as f64;
        if epoch % 10 == 0 {
            println!("Epoch {}: Average Loss = {:.4}", epoch, avg_loss);
            
            // 生成一些示例文本
            if !chars.is_empty() {
                let seed = chars[0].to_string();
                let generated = model.generate(&seed, 20, 1.0);
                println!("Generated: {}", generated);
            }
        }
    }
    
    model
}
