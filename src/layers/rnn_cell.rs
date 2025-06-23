use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use crate::activations::traits::Activation;

/// 存储 RnnCell 的梯度。
#[derive(Debug)]
pub struct RnnCellGradient {
    pub w_hh: Array2<f64>,
    pub w_ih: Array2<f64>,
    pub b_h: Array1<f64>,
}

impl RnnCellGradient {
    /// 创建一个零初始化的梯度实例。
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            w_hh: Array2::zeros((hidden_size, hidden_size)),
            w_ih: Array2::zeros((hidden_size, input_size)),
            b_h: Array1::zeros(hidden_size),
        }
    }
}

/// 表示一个基础的 RNN 单元 (Vanilla RNN Cell)。
pub struct RnnCell {
    /// 权重矩阵，用于连接上一时间步的隐藏状态 (h_{t-1})。
    pub w_hh: Array2<f64>,
    /// 权重矩阵，用于连接当前时间步的输入 (x_t)。
    pub w_ih: Array2<f64>,
    /// 隐藏层的偏置项。
    pub b_h: Array1<f64>,
    /// 此单元使用的激活函数。
    pub activation: Box<dyn Activation>,
}

impl RnnCell {
    /// 创建并初始化一个新的 RnnCell。
    ///
    /// # Arguments
    ///
    /// * `input_size` - 输入向量 `x_t` 的维度。
    /// * `hidden_size` - 隐藏状态 `h_t` 的维度。
    /// * `activation` - 一个实现了 Activation Trait 的激活函数实例。
    pub fn new(input_size: usize, hidden_size: usize, activation: Box<dyn Activation>) -> Self {
        // 使用均匀分布 (-0.1, 0.1) 来随机初始化权重
        let dist = Uniform::new(-0.1, 0.1);
        Self {
            w_hh: Array2::random((hidden_size, hidden_size), dist),
            w_ih: Array2::random((hidden_size, input_size), dist),
            b_h: Array1::zeros(hidden_size),
            activation,
        }
    }

    /// 执行 RnnCell 的前向传播计算。
    ///
    /// # Arguments
    ///
    /// * `x_t` - 当前时间步的输入向量。
    /// * `h_prev` - 上一时间步的隐藏状态。
    ///
    /// # Returns
    ///
    /// * `(h_next, z)` - 当前时间步计算出的新隐藏状态和线性变换结果。
    pub fn forward(&self, x_t: &Array1<f64>, h_prev: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        // z = W_hh * h_{t-1} + W_ih * x_t + b_h
        let z = self.w_hh.dot(h_prev) + self.w_ih.dot(x_t) + &self.b_h;
        // h_t = activation(z)
        let h_next = self.activation.forward(&z);
        (h_next, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_rnn_cell_forward_pass() {
        use crate::activations::functions::Tanh;
        let input_size = 10;
        let hidden_size = 20;
        let cell = RnnCell::new(input_size, hidden_size, Box::new(Tanh));

        // 创建一个虚拟的输入向量和上一个隐藏状态
        let x_t = Array::from_vec(vec![1.0; input_size]);
        let h_prev = Array::from_vec(vec![0.5; hidden_size]);

        // 执行前向传播
        let (h_next, _) = cell.forward(&x_t, &h_prev);

        // 验证输出的维度是否正确
        assert_eq!(h_next.len(), hidden_size);

        // 验证输出值在 tanh 的范围内 (-1, 1)
        for &val in h_next.iter() {
            assert!(val > -1.0 && val < 1.0);
        }
    }
} 