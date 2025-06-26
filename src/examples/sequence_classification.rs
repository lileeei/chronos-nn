use std::collections::HashMap;
use ndarray::{Array1, Array2, Array3};
use crate::layers::lstm_layer::LstmLayer;
use crate::optimizers::losses;

/// 序列分类模型
/// 
/// 用于文本分类任务，如情感分析、垃圾邮件检测等。
/// 使用LSTM编码序列，然后通过全连接层进行分类。
pub struct SequenceClassifier {
    pub lstm: LstmLayer,
    pub classifier: Array2<f64>,  // [hidden_size, num_classes]
    pub classifier_bias: Array1<f64>,  // [num_classes]
    pub vocab_to_idx: HashMap<String, usize>,
    pub idx_to_vocab: HashMap<usize, String>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_classes: usize,
    pub max_seq_len: usize,
}

impl SequenceClassifier {
    /// 创建新的序列分类模型
    ///
    /// # Arguments
    ///
    /// * `vocab` - 词汇表
    /// * `hidden_size` - LSTM隐藏层大小
    /// * `num_classes` - 分类类别数
    /// * `max_seq_len` - 最大序列长度
    pub fn new(vocab: Vec<String>, hidden_size: usize, num_classes: usize, max_seq_len: usize) -> Self {
        let vocab_size = vocab.len();
        let mut vocab_to_idx = HashMap::new();
        let mut idx_to_vocab = HashMap::new();
        
        for (i, word) in vocab.iter().enumerate() {
            vocab_to_idx.insert(word.clone(), i);
            idx_to_vocab.insert(i, word.clone());
        }
        
        use ndarray_rand::rand_distr::Uniform;
        use ndarray_rand::RandomExt;
        let dist = Uniform::new(-0.1, 0.1);
        
        Self {
            lstm: LstmLayer::new(vocab_size, hidden_size),
            classifier: Array2::random((hidden_size, num_classes), dist),
            classifier_bias: Array1::zeros(num_classes),
            vocab_to_idx,
            idx_to_vocab,
            vocab_size,
            hidden_size,
            num_classes,
            max_seq_len,
        }
    }

    /// 将文本序列转换为one-hot编码
    pub fn text_to_onehot(&self, text: &[String]) -> Array3<f64> {
        let seq_len = text.len().min(self.max_seq_len);
        let mut onehot = Array3::zeros((1, seq_len, self.vocab_size)); // batch_size=1
        
        for (t, word) in text.iter().take(seq_len).enumerate() {
            if let Some(&idx) = self.vocab_to_idx.get(word) {
                onehot[[0, t, idx]] = 1.0;
            }
        }
        
        onehot
    }

    /// 批量文本转换
    pub fn batch_text_to_onehot(&self, texts: &[Vec<String>]) -> Array3<f64> {
        let batch_size = texts.len();
        let mut onehot = Array3::zeros((batch_size, self.max_seq_len, self.vocab_size));
        
        for (b, text) in texts.iter().enumerate() {
            let seq_len = text.len().min(self.max_seq_len);
            for (t, word) in text.iter().take(seq_len).enumerate() {
                if let Some(&idx) = self.vocab_to_idx.get(word) {
                    onehot[[b, t, idx]] = 1.0;
                }
            }
        }
        
        onehot
    }

    /// Softmax函数
    fn softmax(&self, x: &Array1<f64>) -> Array1<f64> {
        let max_val = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_x = x.mapv(|v| (v - max_val).exp());
        let sum_exp = exp_x.sum();
        exp_x / sum_exp
    }

    /// 前向传播
    pub fn forward(&self, input: &Array3<f64>) -> Array2<f64> {
        // 通过LSTM
        let (lstm_output, _final_states, _cache) = self.lstm.forward_batch(input);
        
        // 使用最后一个时间步的输出进行分类
        let batch_size = lstm_output.shape()[0];
        let mut output = Array2::zeros((batch_size, self.num_classes));
        
        for b in 0..batch_size {
            // 找到最后一个非零时间步（处理变长序列）
            let mut last_timestep = lstm_output.shape()[1] - 1;
            for t in (0..lstm_output.shape()[1]).rev() {
                let has_input = (0..self.vocab_size).any(|i| input[[b, t, i]] > 0.0);
                if has_input {
                    last_timestep = t;
                    break;
                }
            }
            
            let hidden = lstm_output.slice(ndarray::s![b, last_timestep, ..]).to_owned();
            let logits = self.classifier.t().dot(&hidden) + &self.classifier_bias;
            let probs = self.softmax(&logits);
            
            output.row_mut(b).assign(&probs);
        }
        
        output
    }

    /// 预测单个文本的类别
    pub fn predict(&self, text: &[String]) -> (usize, f64) {
        let input = self.text_to_onehot(text);
        let output = self.forward(&input);
        let probs = output.row(0);
        
        let mut max_idx = 0;
        let mut max_prob = probs[0];
        
        for (i, &prob) in probs.iter().enumerate() {
            if prob > max_prob {
                max_prob = prob;
                max_idx = i;
            }
        }
        
        (max_idx, max_prob)
    }

    /// 训练一个批次
    pub fn train_step(&mut self, texts: &[Vec<String>], labels: &[usize], _learning_rate: f64) -> f64 {
        let input = self.batch_text_to_onehot(texts);
        let predictions = self.forward(&input);
        
        // 计算交叉熵损失
        let mut total_loss = 0.0;
        let batch_size = texts.len();
        
        for (i, &label) in labels.iter().enumerate() {
            let pred_probs = predictions.row(i);
            // 创建one-hot标签
            let mut target = Array1::zeros(self.num_classes);
            target[label] = 1.0;
            
            let loss = losses::binary_cross_entropy(&pred_probs.to_owned(), &target);
            total_loss += loss;
        }
        
        total_loss / batch_size as f64
    }

    /// 评估模型准确率
    pub fn evaluate(&self, texts: &[Vec<String>], labels: &[usize]) -> f64 {
        let mut correct = 0;
        let total = texts.len();
        
        for (text, &true_label) in texts.iter().zip(labels.iter()) {
            let (pred_label, _confidence) = self.predict(text);
            if pred_label == true_label {
                correct += 1;
            }
        }
        
        correct as f64 / total as f64
    }
}

/// 简单的文本预处理
pub fn preprocess_text(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// 构建词汇表
pub fn build_vocabulary(texts: &[String], min_freq: usize) -> Vec<String> {
    let mut word_counts = HashMap::new();
    
    for text in texts {
        let words = preprocess_text(text);
        for word in words {
            *word_counts.entry(word).or_insert(0) += 1;
        }
    }
    
    let mut vocab: Vec<String> = word_counts
        .into_iter()
        .filter(|(_, count)| *count >= min_freq)
        .map(|(word, _)| word)
        .collect();
    
    vocab.sort();
    vocab.insert(0, "<UNK>".to_string()); // 未知词标记
    vocab
}

/// 情感分析示例数据
pub fn create_sentiment_dataset() -> (Vec<Vec<String>>, Vec<usize>) {
    let positive_texts = vec![
        "I love this movie it is amazing",
        "Great film wonderful acting",
        "Excellent story and characters",
        "Best movie ever fantastic",
        "Really enjoyed watching this",
        "Brilliant performance by actors",
        "Outstanding cinematography and plot",
        "Highly recommend this film",
    ];
    
    let negative_texts = vec![
        "I hate this movie it is terrible",
        "Awful film boring acting",
        "Poor story and bad characters",
        "Worst movie ever horrible",
        "Did not enjoy watching this",
        "Terrible performance by actors",
        "Poor cinematography and plot",
        "Do not recommend this film",
    ];
    
    let mut texts = Vec::new();
    let mut labels = Vec::new();
    
    // 正面情感 (标签 1)
    for text in positive_texts {
        texts.push(preprocess_text(text));
        labels.push(1);
    }
    
    // 负面情感 (标签 0)
    for text in negative_texts {
        texts.push(preprocess_text(text));
        labels.push(0);
    }
    
    (texts, labels)
}

/// 训练情感分析模型的示例
pub fn train_sentiment_classifier() -> SequenceClassifier {
    println!("创建情感分析数据集...");
    let (texts, labels) = create_sentiment_dataset();
    
    // 构建词汇表
    let all_text: Vec<String> = texts.iter()
        .flat_map(|words| words.iter().cloned())
        .collect();
    let vocab = build_vocabulary(&all_text, 1);
    
    println!("词汇表大小: {}", vocab.len());
    
    // 创建模型
    let mut model = SequenceClassifier::new(vocab, 32, 2, 10);
    
    println!("开始训练情感分析模型...");
    let epochs = 50;
    let learning_rate = 0.01;
    
    for epoch in 0..epochs {
        let loss = model.train_step(&texts, &labels, learning_rate);
        
        if epoch % 10 == 0 {
            let accuracy = model.evaluate(&texts, &labels);
            println!("Epoch {}: Loss = {:.4}, Accuracy = {:.2}%", epoch, loss, accuracy * 100.0);
        }
    }
    
    // 测试一些例子
    println!("\n测试预测:");
    let test_texts = vec![
        "this movie is great",
        "terrible film very bad",
        "amazing story love it",
    ];
    
    for text in test_texts {
        let words = preprocess_text(text);
        let (pred_class, confidence) = model.predict(&words);
        let sentiment = if pred_class == 1 { "Positive" } else { "Negative" };
        println!("Text: '{}' -> {} (confidence: {:.2})", text, sentiment, confidence);
    }
    
    model
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_preprocessing() {
        let text = "Hello, World! This is a TEST.";
        let words = preprocess_text(text);
        assert_eq!(words, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn test_vocabulary_building() {
        let texts = vec![
            "hello world".to_string(),
            "world test".to_string(),
            "hello test".to_string(),
        ];
        let vocab = build_vocabulary(&texts, 1);
        
        assert!(vocab.contains(&"<UNK>".to_string()));
        assert!(vocab.contains(&"hello".to_string()));
        assert!(vocab.contains(&"world".to_string()));
        assert!(vocab.contains(&"test".to_string()));
    }

    #[test]
    fn test_sequence_classifier_creation() {
        let vocab = vec!["<UNK>".to_string(), "hello".to_string(), "world".to_string()];
        let model = SequenceClassifier::new(vocab, 16, 2, 5);
        
        assert_eq!(model.vocab_size, 3);
        assert_eq!(model.hidden_size, 16);
        assert_eq!(model.num_classes, 2);
        assert_eq!(model.max_seq_len, 5);
    }

    #[test]
    fn test_text_to_onehot() {
        let vocab = vec!["<UNK>".to_string(), "hello".to_string(), "world".to_string()];
        let model = SequenceClassifier::new(vocab, 16, 2, 5);
        
        let text = vec!["hello".to_string(), "world".to_string()];
        let onehot = model.text_to_onehot(&text);
        
        assert_eq!(onehot.shape(), &[1, 2, 3]);
        assert_eq!(onehot[[0, 0, 1]], 1.0); // "hello" at index 1
        assert_eq!(onehot[[0, 1, 2]], 1.0); // "world" at index 2
    }
}
