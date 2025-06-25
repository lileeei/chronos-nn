use ndarray::{Array1, Array2, Array3};
use crate::layers::lstm_layer::{LstmLayer, LstmBatchCache};
use crate::attention::attention_mechanisms::{AttentionMechanism, DotProductAttention};

/// 简单的Encoder-Decoder架构，带有注意力机制
pub struct AttentionEncoderDecoder {
    pub encoder: LstmLayer,
    pub decoder: LstmLayer,
    pub attention: Box<dyn AttentionMechanism>,
    pub output_projection: Array2<f64>,  // [decoder_hidden_size, output_vocab_size]
    pub output_bias: Array1<f64>,        // [output_vocab_size]
}

impl AttentionEncoderDecoder {
    /// 创建新的Encoder-Decoder模型
    ///
    /// # Arguments
    ///
    /// * `input_size` - 输入词汇表大小
    /// * `encoder_hidden_size` - 编码器隐藏层大小
    /// * `decoder_hidden_size` - 解码器隐藏层大小
    /// * `output_size` - 输出词汇表大小
    pub fn new(
        input_size: usize,
        encoder_hidden_size: usize,
        decoder_hidden_size: usize,
        output_size: usize,
    ) -> Self {
        use ndarray_rand::rand_distr::Uniform;
        use ndarray_rand::RandomExt;

        let dist = Uniform::new(-0.1, 0.1);

        Self {
            encoder: LstmLayer::new(input_size, encoder_hidden_size),
            decoder: LstmLayer::new(input_size + encoder_hidden_size, decoder_hidden_size), // 输入 + 上下文
            attention: Box::new(DotProductAttention),
            output_projection: Array2::random((decoder_hidden_size, output_size), dist),
            output_bias: Array1::zeros(output_size),
        }
    }

    /// 编码输入序列
    ///
    /// # Arguments
    ///
    /// * `input_sequence` - 输入序列 [batch_size, seq_len, input_size]
    ///
    /// # Returns
    ///
    /// * `(encoder_outputs, encoder_cache)` - 编码器输出和缓存
    pub fn encode(&self, input_sequence: &Array3<f64>) -> (Array3<f64>, LstmBatchCache) {
        let (encoder_outputs, _final_states, cache) = self.encoder.forward_batch(input_sequence);
        (encoder_outputs, cache)
    }

    /// 解码单个时间步
    ///
    /// # Arguments
    ///
    /// * `decoder_input` - 解码器输入 [batch_size, input_size]
    /// * `decoder_hidden` - 解码器隐藏状态 [batch_size, decoder_hidden_size]
    /// * `encoder_outputs` - 编码器输出 [batch_size, seq_len, encoder_hidden_size]
    ///
    /// # Returns
    ///
    /// * `(output_logits, new_hidden, attention_weights)` - 输出logits、新隐藏状态和注意力权重
    pub fn decode_step(
        &self,
        decoder_input: &Array2<f64>,
        decoder_hidden: &Array2<f64>,
        encoder_outputs: &Array3<f64>,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let batch_size = decoder_input.shape()[0];
        let _seq_len = encoder_outputs.shape()[1];
        let encoder_hidden_size = encoder_outputs.shape()[2];

        // 为了简化，我们使用线性投影来匹配维度
        // 在实际实现中，应该有专门的投影层
        let query_proj = if decoder_hidden.shape()[1] != encoder_hidden_size {
            // 简单的维度匹配：截断或填充
            let min_dim = decoder_hidden.shape()[1].min(encoder_hidden_size);
            let mut proj = Array2::zeros((batch_size, encoder_hidden_size));
            for b in 0..batch_size {
                for i in 0..min_dim {
                    proj[[b, i]] = decoder_hidden[[b, i]];
                }
            }
            proj
        } else {
            decoder_hidden.clone()
        };

        // 计算注意力上下文
        let (contexts, attention_weights) = self.attention.forward_batch(
            &query_proj,
            encoder_outputs,
            encoder_outputs, // 在这个简单实现中，keys和values相同
        );

        // 将解码器输入和注意力上下文连接
        let mut decoder_input_with_context = Array2::zeros((batch_size, decoder_input.shape()[1] + encoder_hidden_size));
        for b in 0..batch_size {
            // 复制原始输入
            for i in 0..decoder_input.shape()[1] {
                decoder_input_with_context[[b, i]] = decoder_input[[b, i]];
            }
            // 添加上下文
            for i in 0..encoder_hidden_size {
                decoder_input_with_context[[b, decoder_input.shape()[1] + i]] = contexts[[b, i]];
            }
        }

        // 通过解码器LSTM
        // 注意：这里简化了实现，实际应该维护解码器的状态
        let input_3d = decoder_input_with_context.insert_axis(ndarray::Axis(1)); // 添加seq_len=1维度
        let (decoder_output, _final_states, _cache) = self.decoder.forward_batch(&input_3d);
        let decoder_output_2d = decoder_output.index_axis(ndarray::Axis(1), 0).to_owned(); // 移除seq_len维度

        // 投影到输出词汇表
        let output_logits = decoder_output_2d.dot(&self.output_projection) + &self.output_bias;

        (output_logits, decoder_output_2d, attention_weights)
    }

    /// 完整的前向传播（用于训练）
    ///
    /// # Arguments
    ///
    /// * `input_sequence` - 输入序列 [batch_size, input_seq_len, input_size]
    /// * `target_sequence` - 目标序列 [batch_size, target_seq_len, input_size]
    ///
    /// # Returns
    ///
    /// * `(output_logits, attention_weights)` - 输出logits和注意力权重
    pub fn forward(
        &self,
        input_sequence: &Array3<f64>,
        target_sequence: &Array3<f64>,
    ) -> (Array3<f64>, Array3<f64>) {
        let batch_size = input_sequence.shape()[0];
        let target_seq_len = target_sequence.shape()[1];
        let output_size = self.output_projection.shape()[1];

        // 编码
        let (encoder_outputs, _encoder_cache) = self.encode(input_sequence);

        // 解码
        let mut all_outputs = Array3::zeros((batch_size, target_seq_len, output_size));
        let mut all_attention_weights = Array3::zeros((batch_size, target_seq_len, encoder_outputs.shape()[1]));
        
        // 初始化解码器隐藏状态（简化：使用零向量）
        let mut decoder_hidden = Array2::zeros((batch_size, self.decoder.cell.hidden_size));

        for t in 0..target_seq_len {
            let decoder_input = target_sequence.index_axis(ndarray::Axis(1), t).to_owned();
            
            let (output_logits, new_hidden, attention_weights) = self.decode_step(
                &decoder_input,
                &decoder_hidden,
                &encoder_outputs,
            );

            // 存储输出
            for b in 0..batch_size {
                for i in 0..output_size {
                    all_outputs[[b, t, i]] = output_logits[[b, i]];
                }
                for i in 0..encoder_outputs.shape()[1] {
                    all_attention_weights[[b, t, i]] = attention_weights[[b, i]];
                }
            }

            decoder_hidden = new_hidden;
        }

        (all_outputs, all_attention_weights)
    }

    /// 推理模式的解码（贪心解码）
    ///
    /// # Arguments
    ///
    /// * `input_sequence` - 输入序列 [batch_size, seq_len, input_size]
    /// * `max_length` - 最大输出长度
    /// * `start_token` - 开始标记 [input_size]
    ///
    /// # Returns
    ///
    /// * `(generated_sequence, attention_weights)` - 生成的序列和注意力权重
    pub fn generate(
        &self,
        input_sequence: &Array3<f64>,
        max_length: usize,
        start_token: &Array1<f64>,
    ) -> (Array3<f64>, Array3<f64>) {
        let batch_size = input_sequence.shape()[0];
        let input_size = start_token.len();

        // 编码
        let (encoder_outputs, _encoder_cache) = self.encode(input_sequence);

        // 初始化输出序列
        let mut generated_sequence = Array3::zeros((batch_size, max_length, input_size));
        let mut all_attention_weights = Array3::zeros((batch_size, max_length, encoder_outputs.shape()[1]));
        
        // 初始化解码器隐藏状态
        let mut decoder_hidden = Array2::zeros((batch_size, self.decoder.cell.hidden_size));
        
        // 设置起始输入
        let mut current_input = Array2::zeros((batch_size, input_size));
        for b in 0..batch_size {
            current_input.row_mut(b).assign(start_token);
        }

        for t in 0..max_length {
            let (output_logits, new_hidden, attention_weights) = self.decode_step(
                &current_input,
                &decoder_hidden,
                &encoder_outputs,
            );

            // 贪心选择（选择概率最高的token）
            // 这里简化为直接使用logits作为下一个输入
            current_input = output_logits.clone();

            // 存储结果
            for b in 0..batch_size {
                for i in 0..input_size {
                    generated_sequence[[b, t, i]] = current_input[[b, i]];
                }
                for i in 0..encoder_outputs.shape()[1] {
                    all_attention_weights[[b, t, i]] = attention_weights[[b, i]];
                }
            }

            decoder_hidden = new_hidden;
        }

        (generated_sequence, all_attention_weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, Array3};

    #[test]
    fn test_encoder_decoder_creation() {
        let model = AttentionEncoderDecoder::new(10, 20, 15, 8);

        assert_eq!(model.encoder.cell.input_size, 10);
        assert_eq!(model.encoder.cell.hidden_size, 20);
        assert_eq!(model.decoder.cell.input_size, 10 + 20); // input + context
        assert_eq!(model.decoder.cell.hidden_size, 15);
        assert_eq!(model.output_projection.shape(), &[15, 8]);
    }

    #[test]
    fn test_encoding() {
        let model = AttentionEncoderDecoder::new(5, 8, 6, 4);

        let input_sequence = Array3::<f64>::ones((2, 4, 5)); // [batch=2, seq_len=4, input_size=5]
        let (encoder_outputs, _cache) = model.encode(&input_sequence);

        assert_eq!(encoder_outputs.shape(), &[2, 4, 8]); // [batch, seq_len, hidden_size]
    }

    #[test]
    fn test_decode_step() {
        let model = AttentionEncoderDecoder::new(5, 8, 6, 4);

        let decoder_input = Array2::<f64>::ones((2, 5)); // [batch=2, input_size=5]
        let decoder_hidden = Array2::<f64>::ones((2, 6)); // [batch=2, decoder_hidden=6]
        let encoder_outputs = Array3::<f64>::ones((2, 3, 8)); // [batch=2, seq_len=3, encoder_hidden=8]

        let (output_logits, new_hidden, attention_weights) = model.decode_step(
            &decoder_input,
            &decoder_hidden,
            &encoder_outputs,
        );

        assert_eq!(output_logits.shape(), &[2, 4]); // [batch, output_size]
        assert_eq!(new_hidden.shape(), &[2, 6]); // [batch, decoder_hidden]
        assert_eq!(attention_weights.shape(), &[2, 3]); // [batch, encoder_seq_len]

        // 验证注意力权重和为1
        for b in 0..2 {
            let weight_sum: f64 = attention_weights.row(b).sum();
            assert!((weight_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_forward_pass() {
        let model = AttentionEncoderDecoder::new(3, 5, 4, 2);

        let input_sequence = Array3::<f64>::ones((1, 3, 3)); // [batch=1, input_seq=3, input_size=3]
        let target_sequence = Array3::<f64>::ones((1, 2, 3)); // [batch=1, target_seq=2, input_size=3]

        let (output_logits, attention_weights) = model.forward(&input_sequence, &target_sequence);

        assert_eq!(output_logits.shape(), &[1, 2, 2]); // [batch, target_seq, output_size]
        assert_eq!(attention_weights.shape(), &[1, 2, 3]); // [batch, target_seq, input_seq]

        // 验证每个时间步的注意力权重和为1
        for t in 0..2 {
            let weight_sum: f64 = attention_weights.slice(ndarray::s![0, t, ..]).sum();
            assert!((weight_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_generation() {
        let model = AttentionEncoderDecoder::new(3, 5, 4, 3);

        let input_sequence = Array3::<f64>::ones((1, 3, 3)); // [batch=1, seq_len=3, input_size=3]
        let start_token = arr1(&[1.0, 0.0, 0.0]); // [input_size=3]

        let (generated_sequence, attention_weights) = model.generate(&input_sequence, 2, &start_token);

        assert_eq!(generated_sequence.shape(), &[1, 2, 3]); // [batch, max_length, input_size]
        assert_eq!(attention_weights.shape(), &[1, 2, 3]); // [batch, max_length, input_seq]

        // 验证每个时间步的注意力权重和为1
        for t in 0..2 {
            let weight_sum: f64 = attention_weights.slice(ndarray::s![0, t, ..]).sum();
            assert!((weight_sum - 1.0).abs() < 1e-6);
        }
    }
}
