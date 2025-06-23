use crate::layers::rnn_cell::{RnnCell, RnnCellGradient};
use ndarray::Array1;

/// 表示一个完整的 RNN 层，它按时间步处理序列数据。
pub struct RnnLayer {
    pub cell: RnnCell,
    hidden_size: usize,
}

/// 缓存 RNN 在前向传播过程中的中间值，以便用于反向传播。
pub struct Cache {
    /// 每个时间步的输入 (x_t)
    inputs: Vec<Array1<f64>>,
    /// 每个时间步的隐藏状态 (h_t)
    hidden_states: Vec<Array1<f64>>,
}

impl RnnLayer {
    /// 创建一个新的 RnnLayer。
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            cell: RnnCell::new(input_size, hidden_size),
            hidden_size,
        }
    }

    /// 对整个输入序列执行前向传播。
    ///
    /// # Arguments
    ///
    /// * `xs` - 输入序列，每个元素是 `x_t`。
    ///
    /// # Returns
    ///
    /// * `(hidden_states, cache)` - 所有时间步的隐藏状态和用于反向传播的缓存。
    pub fn forward(&self, xs: &[Array1<f64>]) -> (Vec<Array1<f64>>, Cache) {
        let seq_len = xs.len();
        let mut hidden_states = Vec::with_capacity(seq_len + 1);
        let mut inputs_cache = Vec::with_capacity(seq_len);

        // 初始隐藏状态 h_0 为零向量
        let mut h_prev = Array1::zeros(self.hidden_size);
        hidden_states.push(h_prev.clone());

        for x_t in xs {
            let h_next = self.cell.forward(x_t, &h_prev);
            hidden_states.push(h_next.clone());
            inputs_cache.push(x_t.clone());
            h_prev = h_next;
        }

        let cache = Cache {
            inputs: inputs_cache,
            hidden_states: hidden_states.clone(),
        };

        // 移除初始的 h_0，只返回每个输入对应的隐藏状态
        (hidden_states[1..].to_vec(), cache)
    }

    /// 对整个序列执行反向传播 (BPTT)。
    ///
    /// # Arguments
    ///
    /// * `d_h_list` - 损失函数对每个时间步隐藏状态输出的梯度。
    /// * `cache` - 前向传播过程中生成的缓存。
    ///
    /// # Returns
    ///
    /// * `RnnCellGradient` - 计算出的包含所有参数梯度的结构体。
    pub fn backward(&self, d_h_list: &[Array1<f64>], cache: &Cache) -> RnnCellGradient {
        let input_size = self.cell.w_ih.shape()[1];
        let mut grads = RnnCellGradient::new(input_size, self.hidden_size);
        let mut dh_next = Array1::zeros(self.hidden_size);

        // 反向遍历时间步
        for t in (0..cache.inputs.len()).rev() {
            let x_t = &cache.inputs[t];
            let h_t = &cache.hidden_states[t + 1];
            let h_prev = &cache.hidden_states[t];

            let mut dh_t = d_h_list[t].clone();
            dh_t += &dh_next;

            // 通过 tanh 激活函数反向传播
            // tanh的导数是 1 - tanh(z)^2。因为 h_t = tanh(z)，所以导数是 1 - h_t^2。
            let dtanh = h_t.mapv(|v| 1.0 - v.powi(2));
            let da_t = dh_t * &dtanh;

            // 计算参数的梯度
            let da_t_col = da_t.view().into_shape_with_order((self.hidden_size, 1)).unwrap();
            let x_t_row = x_t.view().into_shape_with_order((1, input_size)).unwrap();
            grads.w_ih += &da_t_col.dot(&x_t_row);

            let h_prev_row = h_prev.view().into_shape_with_order((1, self.hidden_size)).unwrap();
            grads.w_hh += &da_t_col.dot(&h_prev_row);

            grads.b_h += &da_t;

            // 计算并传递到上一个时间步的梯度
            dh_next = self.cell.w_hh.t().dot(&da_t);
        }

        grads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::{losses, sgd::Sgd};
    use ndarray::arr1;

    #[test]
    fn test_rnn_layer_training_step() {
        let input_size = 3;
        let hidden_size = 4;
        let seq_len = 5;
        let learning_rate = 0.01;

        let mut layer = RnnLayer::new(input_size, hidden_size);
        let optimizer = Sgd::new(learning_rate);

        // 创建虚拟的输入和目标序列
        let xs: Vec<_> = (0..seq_len)
            .map(|_| arr1(&[1.0, 2.0, 3.0]))
            .collect();
        let ys: Vec<_> = (0..seq_len)
            .map(|_| arr1(&[0.5, -0.5, 0.2, -0.2]))
            .collect();

        let mut last_loss = f64::MAX;

        // 模拟一个简单的训练循环
        for i in 0..10 {
            // 1. 前向传播
            let (ps, cache) = layer.forward(&xs);

            // 2. 计算损失
            let mut total_loss = 0.0;
            let mut d_ps = Vec::new();
            for (p, y) in ps.iter().zip(ys.iter()) {
                total_loss += losses::mean_squared_error(p, y);
                d_ps.push(losses::mean_squared_error_derivative(p, y));
            }
            total_loss /= seq_len as f64;
            
            println!("Iteration {}: Loss = {}", i, total_loss);
            
            // 3. 反向传播
            let grads = layer.backward(&d_ps, &cache);

            // 4. 更新权重
            optimizer.update(&mut layer.cell, &grads);

            // 检查损失是否在减小
            if i > 0 {
                assert!(total_loss < last_loss, "Loss did not decrease at iteration {}", i);
            }
            last_loss = total_loss;
        }

        // 确认最终损失远小于初始损失
        assert!(last_loss < 1.0);
    }
} 