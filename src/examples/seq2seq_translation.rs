use std::collections::HashMap;
use ndarray::{Array1, Array3};
use crate::attention::encoder_decoder::AttentionEncoderDecoder;

/// 序列到序列翻译模型
pub struct Seq2SeqTranslator {
    pub model: AttentionEncoderDecoder,
    pub src_vocab_to_idx: HashMap<String, usize>,
    pub tgt_vocab_to_idx: HashMap<String, usize>,
    pub tgt_idx_to_vocab: HashMap<usize, String>,
    pub src_vocab_size: usize,
    pub tgt_vocab_size: usize,
}

impl Seq2SeqTranslator {
    /// 创建新的翻译模型
    pub fn new(
        src_vocab: Vec<String>,
        tgt_vocab: Vec<String>,
        encoder_hidden_size: usize,
        decoder_hidden_size: usize,
    ) -> Self {
        let src_vocab_size = src_vocab.len();
        let tgt_vocab_size = tgt_vocab.len();
        
        let mut src_vocab_to_idx = HashMap::new();
        for (i, word) in src_vocab.iter().enumerate() {
            src_vocab_to_idx.insert(word.clone(), i);
        }
        
        let mut tgt_vocab_to_idx = HashMap::new();
        let mut tgt_idx_to_vocab = HashMap::new();
        for (i, word) in tgt_vocab.iter().enumerate() {
            tgt_vocab_to_idx.insert(word.clone(), i);
            tgt_idx_to_vocab.insert(i, word.clone());
        }
        
        let model = AttentionEncoderDecoder::new(
            src_vocab_size,
            encoder_hidden_size,
            decoder_hidden_size,
            tgt_vocab_size,
        );
        
        Self {
            model,
            src_vocab_to_idx,
            tgt_vocab_to_idx,
            tgt_idx_to_vocab,
            src_vocab_size,
            tgt_vocab_size,
        }
    }

    /// 将源语言文本转换为one-hot编码
    pub fn encode_source(&self, text: &[String]) -> Array3<f64> {
        let seq_len = text.len();
        let mut onehot = Array3::zeros((1, seq_len, self.src_vocab_size));
        
        for (t, word) in text.iter().enumerate() {
            if let Some(&idx) = self.src_vocab_to_idx.get(word) {
                onehot[[0, t, idx]] = 1.0;
            }
        }
        
        onehot
    }

    /// 简化的翻译函数
    pub fn translate(&self, src_text: &[String]) -> Vec<String> {
        let src_input = self.encode_source(src_text);
        
        // 创建开始标记
        let start_token = {
            let mut token = Array1::zeros(self.tgt_vocab_size);
            token[0] = 1.0; // 假设索引0是开始标记
            token
        };
        
        // 使用模型生成翻译
        let (generated, _attention) = self.model.generate(&src_input, 10, &start_token);
        
        // 将生成的序列转换回文本
        let mut result = Vec::new();
        let seq_len = generated.shape()[1];
        
        for t in 0..seq_len {
            let mut max_idx = 0;
            let mut max_val = generated[[0, t, 0]];
            
            for i in 1..self.tgt_vocab_size {
                if generated[[0, t, i]] > max_val {
                    max_val = generated[[0, t, i]];
                    max_idx = i;
                }
            }
            
            if let Some(word) = self.tgt_idx_to_vocab.get(&max_idx) {
                if word != "EOS" {
                    result.push(word.clone());
                } else {
                    break;
                }
            }
        }

        result
    }
}

/// 创建简单的英法翻译数据集
pub fn create_translation_dataset() -> (Vec<Vec<String>>, Vec<Vec<String>>) {
    let english_sentences = vec![
        vec!["hello".to_string(), "world".to_string()],
        vec!["good".to_string(), "morning".to_string()],
        vec!["how".to_string(), "are".to_string(), "you".to_string()],
        vec!["thank".to_string(), "you".to_string()],
    ];

    let french_sentences = vec![
        vec!["bonjour".to_string(), "monde".to_string()],
        vec!["bon".to_string(), "matin".to_string()],
        vec!["comment".to_string(), "allez".to_string(), "vous".to_string()],
        vec!["merci".to_string()],
    ];

    (english_sentences, french_sentences)
}

/// 构建翻译词汇表
pub fn build_translation_vocab(sentences: &[Vec<String>]) -> Vec<String> {
    let mut vocab = std::collections::HashSet::new();
    vocab.insert("SOS".to_string()); // Start of sequence
    vocab.insert("EOS".to_string()); // End of sequence
    vocab.insert("UNK".to_string()); // Unknown word

    for sentence in sentences {
        for word in sentence {
            vocab.insert(word.clone());
        }
    }

    let mut vocab_vec: Vec<String> = vocab.into_iter().collect();
    vocab_vec.sort();
    vocab_vec
}

/// 训练简单翻译模型的示例
pub fn train_translation_model() -> Seq2SeqTranslator {
    println!("创建翻译数据集...");
    let (english_sentences, french_sentences) = create_translation_dataset();

    // 构建词汇表
    let english_vocab = build_translation_vocab(&english_sentences);
    let french_vocab = build_translation_vocab(&french_sentences);

    println!("英语词汇表大小: {}", english_vocab.len());
    println!("法语词汇表大小: {}", french_vocab.len());

    // 创建翻译模型
    let translator = Seq2SeqTranslator::new(
        english_vocab,
        french_vocab,
        32, // encoder hidden size
        32, // decoder hidden size
    );

    println!("翻译模型创建完成");

    // 测试翻译
    let test_sentence = vec!["hello".to_string(), "world".to_string()];
    let translation = translator.translate(&test_sentence);
    println!("翻译 '{:?}' -> '{:?}'", test_sentence, translation);

    translator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seq2seq_translator_creation() {
        let src_vocab = vec!["hello".to_string(), "world".to_string()];
        let tgt_vocab = vec!["bonjour".to_string(), "monde".to_string()];

        let translator = Seq2SeqTranslator::new(src_vocab, tgt_vocab, 16, 16);

        assert_eq!(translator.src_vocab_size, 2);
        assert_eq!(translator.tgt_vocab_size, 2);
    }

    #[test]
    fn test_translation_dataset() {
        let (english, french) = create_translation_dataset();
        assert_eq!(english.len(), french.len());
        assert!(!english.is_empty());
    }

    #[test]
    fn test_vocab_building() {
        let sentences = vec![
            vec!["hello".to_string(), "world".to_string()],
            vec!["good".to_string(), "morning".to_string()],
        ];

        let vocab = build_translation_vocab(&sentences);
        assert!(vocab.contains(&"SOS".to_string()));
        assert!(vocab.contains(&"EOS".to_string()));
        assert!(vocab.contains(&"hello".to_string()));
        assert!(vocab.contains(&"world".to_string()));
    }
}
