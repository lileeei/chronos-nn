use ndarray::Array3;
use crate::layers::lstm_layer::{LstmLayer, LstmBatchCache};
use crate::layers::lstm_cell::LstmCellGradient;

/// Dropout层，用于正则化
#[derive(Clone)]
pub struct Dropout {
    pub rate: f64,
    pub training: bool,
}

impl Dropout {
    pub fn new(rate: f64) -> Self {
        Self {
            rate,
            training: true,
        }
    }

    /// 设置训练模式
    pub fn train(&mut self) {
        self.training = true;
    }

    /// 设置评估模式
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// 应用dropout
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        if !self.training || self.rate == 0.0 {
            return x.clone();
        }

        // 简单的dropout实现：随机将一些元素设为0
        let scale = 1.0 / (1.0 - self.rate);
        x.mapv(|v| {
            if rand::random::<f64>() < self.rate {
                0.0
            } else {
                v * scale
            }
        })
    }
}

/// 多层LSTM架构，支持堆叠多个LSTM层
pub struct MultiLayerLstm {
    pub layers: Vec<LstmLayer>,
    pub dropouts: Vec<Dropout>,
    pub use_residual: bool,
    pub hidden_size: usize,
}

impl MultiLayerLstm {
    /// 创建一个新的多层LSTM
    ///
    /// # Arguments
    ///
    /// * `layers` - LSTM层的向量
    /// * `dropout_rate` - Dropout率
    /// * `use_residual` - 是否使用残差连接
    /// * `hidden_size` - 隐藏层大小（用于残差连接）
    pub fn new(layers: Vec<LstmLayer>, dropout_rate: f64, use_residual: bool, hidden_size: usize) -> Self {
        let num_layers = layers.len();
        let dropouts = (0..num_layers.saturating_sub(1))
            .map(|_| Dropout::new(dropout_rate))
            .collect();

        Self {
            layers,
            dropouts,
            use_residual,
            hidden_size,
        }
    }

    /// 设置训练模式
    pub fn train(&mut self) {
        for dropout in &mut self.dropouts {
            dropout.train();
        }
    }

    /// 设置评估模式
    pub fn eval(&mut self) {
        for dropout in &mut self.dropouts {
            dropout.eval();
        }
    }

    /// 前向传播
    pub fn forward(&self, xs: &Array3<f64>) -> (Array3<f64>, Vec<LstmBatchCache>) {
        let mut current_input = xs.clone();
        let mut caches = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let (layer_output, _cs, cache) = layer.forward_batch(&current_input);
            caches.push(cache);

            // 应用残差连接（如果启用且不是第一层）
            let mut next_input = if self.use_residual && i > 0 &&
                current_input.shape()[2] == layer_output.shape()[2] {
                // 残差连接：output = layer_output + input
                &layer_output + &current_input
            } else {
                layer_output
            };

            // 应用dropout（除了最后一层）
            if i < self.dropouts.len() {
                next_input = self.dropouts[i].forward(&next_input);
            }

            current_input = next_input;
        }

        (current_input, caches)
    }

    /// 反向传播
    pub fn backward(&self, d_output: &Array3<f64>, caches: &[LstmBatchCache]) -> Vec<LstmCellGradient> {
        let d_input = d_output.clone();
        let mut gradients = Vec::new();

        // 从最后一层开始反向传播
        for (i, (layer, cache)) in self.layers.iter().zip(caches.iter()).enumerate().rev() {
            // 如果使用残差连接，需要将梯度传递给跳跃连接
            let layer_d_input = if self.use_residual && i > 0 {
                // 残差连接的梯度：d_input 同时传递给当前层和跳跃连接
                d_input.clone()
            } else {
                d_input.clone()
            };

            let grad = layer.backward_batch(&layer_d_input, cache);
            gradients.push(grad);

            // 如果不是第一层，计算传递给下一层的梯度
            if i > 0 {
                // 这里需要根据具体的层类型来计算梯度传播
                // 简化实现：假设梯度直接传播
                if self.use_residual {
                    // 残差连接：梯度需要加上跳跃连接的梯度
                    // d_input 保持不变，因为残差连接会将梯度直接传递
                }
            }
        }

        // 反转梯度向量，使其与层的顺序一致
        gradients.reverse();
        gradients
    }

    /// 更新所有层的参数
    pub fn update_parameters(&mut self, gradients: &[LstmCellGradient], learning_rate: f64) {
        for (layer, grad) in self.layers.iter_mut().zip(gradients.iter()) {
            layer.cell.update(grad, learning_rate);
        }
    }

    /// 获取层数
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_dropout() {
        let mut dropout = Dropout::new(0.5);
        let input = Array3::<f64>::ones((2, 3, 4));

        // 训练模式下应该有一些元素被设为0
        dropout.train();
        let output_train = dropout.forward(&input);

        // 评估模式下应该保持原样
        dropout.eval();
        let output_eval = dropout.forward(&input);

        assert_eq!(output_eval, input);
        assert_eq!(output_train.shape(), input.shape());
    }

    #[test]
    fn test_multi_layer_lstm_creation() {
        let layer1 = LstmLayer::new(10, 20);
        let layer2 = LstmLayer::new(20, 20);
        let layer3 = LstmLayer::new(20, 15);

        let multi_lstm = MultiLayerLstm::new(
            vec![layer1, layer2, layer3],
            0.1,
            true,
            20
        );

        assert_eq!(multi_lstm.num_layers(), 3);
        assert_eq!(multi_lstm.dropouts.len(), 2); // n-1 dropout layers
        assert!(multi_lstm.use_residual);
    }

    #[test]
    fn test_multi_layer_lstm_forward() {
        let layer1 = LstmLayer::new(5, 8);
        let layer2 = LstmLayer::new(8, 8);

        let multi_lstm = MultiLayerLstm::new(
            vec![layer1, layer2],
            0.0, // 不使用dropout以便测试
            false, // 不使用残差连接
            8
        );

        let input = Array3::<f64>::ones((2, 4, 5)); // [batch, seq_len, input_dim]
        let (output, caches) = multi_lstm.forward(&input);

        assert_eq!(output.shape(), &[2, 4, 8]); // [batch, seq_len, hidden_size]
        assert_eq!(caches.len(), 2);
    }

    #[test]
    fn test_multi_layer_lstm_training() {
        use crate::optimizers::losses;

        let layer1 = LstmLayer::new(3, 5);
        let layer2 = LstmLayer::new(5, 5);

        let mut multi_lstm = MultiLayerLstm::new(
            vec![layer1, layer2],
            0.1,
            false,
            5
        );

        let input = Array3::<f64>::ones((2, 4, 3));
        let target = Array3::<f64>::zeros((2, 4, 5));

        // 前向传播
        let (output, caches) = multi_lstm.forward(&input);

        // 计算损失
        let output_2d = output.to_shape((8, 5)).unwrap().to_owned();
        let target_2d = target.to_shape((8, 5)).unwrap().to_owned();
        let loss = losses::mean_squared_error_batch(&output_2d, &target_2d);

        // 反向传播
        let d_output = Array3::<f64>::ones((2, 4, 5)) * 0.1;
        let gradients = multi_lstm.backward(&d_output, &caches);

        assert_eq!(gradients.len(), 2);
        assert!(loss >= 0.0);

        // 更新参数
        multi_lstm.update_parameters(&gradients, 0.01);
    }
}
