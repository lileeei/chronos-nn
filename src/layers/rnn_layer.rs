use crate::activations::functions::Tanh;
use crate::layers::rnn_cell::{RnnCell, RnnCellGradient};
use ndarray::{Array1, Array2, Array3, Axis, s};

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
    /// 每个时间步的线性变换结果 (z_t)
    linear_outputs: Vec<Array1<f64>>,
}

/// 批处理缓存结构，存储每个 batch 的中间状态。
pub struct BatchCache {
    pub inputs: Vec<Array2<f64>>,         // [seq_len] of [batch_size, input_dim]
    pub hidden_states: Vec<Array2<f64>>, // [seq_len+1] of [batch_size, hidden_size]
    pub linear_outputs: Vec<Array2<f64>>, // [seq_len] of [batch_size, hidden_size]
}

impl RnnLayer {
    /// 创建一个新的 RnnLayer。
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            cell: RnnCell::new(input_size, hidden_size, Box::new(Tanh {})),
            hidden_size,
        }
    }

    /// 对整个输入序列执行前向传播。
    pub fn forward(&self, xs: &[Array1<f64>]) -> (Vec<Array1<f64>>, Cache) {
        let seq_len = xs.len();
        let mut hidden_states = Vec::with_capacity(seq_len + 1);
        let mut inputs_cache = Vec::with_capacity(seq_len);
        let mut linear_outputs_cache = Vec::with_capacity(seq_len);

        let mut h_prev = Array1::zeros(self.hidden_size);
        hidden_states.push(h_prev.clone());

        for x_t in xs {
            let (h_next, z) = self.cell.forward(x_t, &h_prev);
            hidden_states.push(h_next.clone());
            inputs_cache.push(x_t.clone());
            linear_outputs_cache.push(z);
            h_prev = h_next;
        }

        let cache = Cache {
            inputs: inputs_cache,
            hidden_states: hidden_states.clone(),
            linear_outputs: linear_outputs_cache,
        };
        (hidden_states[1..].to_vec(), cache)
    }

    /// 对整个序列执行反向传播 (BPTT)。
    pub fn backward(&self, d_h_list: &[Array1<f64>], cache: &Cache) -> RnnCellGradient {
        let input_size = self.cell.w_ih.shape()[1];
        let mut grads = RnnCellGradient::new(input_size, self.hidden_size);
        let mut dh_next = Array1::zeros(self.hidden_size);

        for t in (0..cache.inputs.len()).rev() {
            let x_t = &cache.inputs[t];
            let h_prev = &cache.hidden_states[t];
            let z_t = &cache.linear_outputs[t];

            let mut dh_t = d_h_list[t].clone();
            dh_t += &dh_next;

            let da_t = dh_t * &self.cell.activation.derivative(z_t);

            let da_t_col = da_t.view().into_shape_with_order((self.hidden_size, 1)).unwrap();
            let x_t_row = x_t.view().into_shape_with_order((1, input_size)).unwrap();
            grads.w_ih += &da_t_col.dot(&x_t_row);

            let h_prev_row = h_prev.view().into_shape_with_order((1, self.hidden_size)).unwrap();
            grads.w_hh += &da_t_col.dot(&h_prev_row);

            grads.b_h += &da_t;

            dh_next = self.cell.w_hh.t().dot(&da_t);
        }
        grads
    }

    /// 批处理前向传播：输入 shape [batch_size, seq_len, input_dim]
    /// 返回所有时间步的隐藏状态 [batch_size, seq_len, hidden_size] 及缓存
    pub fn forward_batch(&self, xs: &Array3<f64>) -> (Array3<f64>, BatchCache) {
        let (batch_size, seq_len, _input_dim) = xs.dim();
        let mut hidden_states = Vec::with_capacity(seq_len + 1);
        let mut inputs_cache = Vec::with_capacity(seq_len);
        let mut linear_outputs_cache = Vec::with_capacity(seq_len);

        // 初始隐藏状态 [batch_size, hidden_size]
        let mut h_prev = Array2::zeros((batch_size, self.hidden_size));
        hidden_states.push(h_prev.clone());

        for t in 0..seq_len {
            let x_t = xs.slice(s![.., t, ..]).to_owned(); // [batch_size, input_dim]
            let (h_next, z) = self.cell.forward_batch(&x_t, &h_prev);
            hidden_states.push(h_next.clone());
            inputs_cache.push(x_t);
            linear_outputs_cache.push(z);
            h_prev = h_next;
        }
        // 拼接所有时间步的隐藏状态为 [batch_size, seq_len, hidden_size]
        let hs_stack = ndarray::stack(Axis(1), &hidden_states[1..].iter().map(|h| h.view()).collect::<Vec<_>>()).unwrap();
        let cache = BatchCache {
            inputs: inputs_cache,
            hidden_states,
            linear_outputs: linear_outputs_cache,
        };
        (hs_stack, cache)
    }

    /// 批处理反向传播：输入 d_h_list [batch_size, seq_len, hidden_size]，返回累加梯度
    pub fn backward_batch(&self, d_h_list: &Array3<f64>, cache: &BatchCache) -> RnnCellGradient {
        let (batch_size, seq_len, hidden_size) = d_h_list.dim();
        let input_size = self.cell.w_ih.shape()[1];
        let mut grads = RnnCellGradient::new(input_size, self.hidden_size);
        let mut dh_next = Array2::zeros((batch_size, hidden_size));

        for t in (0..seq_len).rev() {
            let x_t = &cache.inputs[t]; // [batch_size, input_size]
            let h_prev = &cache.hidden_states[t]; // [batch_size, hidden_size]
            let z_t = &cache.linear_outputs[t]; // [batch_size, hidden_size]

            let mut dh_t = d_h_list.slice(s![.., t, ..]).to_owned(); // [batch_size, hidden_size]
            dh_t += &dh_next;

            let da_t = self.cell.activation.derivative_batch(z_t) * &dh_t; // [batch_size, hidden_size]

            // 累加 batch 梯度
            grads.w_ih = &grads.w_ih + &(&da_t.t().dot(x_t) / batch_size as f64);
            grads.w_hh = &grads.w_hh + &(&da_t.t().dot(h_prev) / batch_size as f64);
            grads.b_h = &grads.b_h + &(da_t.sum_axis(Axis(0)) / batch_size as f64);

            dh_next = da_t.dot(&self.cell.w_hh);
        }
        grads
    }

    /// Many-to-one 前向传播：输入 shape [batch_size, seq_len, input_dim]
    /// 返回每个 batch 的最后一个隐藏状态 [batch_size, hidden_size] 及缓存
    pub fn forward_many_to_one(&self, xs: &Array3<f64>) -> (Array2<f64>, BatchCache) {
        let (hs_stack, cache) = self.forward_batch(xs);
        let seq_len = hs_stack.shape()[1];
        let last_h = hs_stack.index_axis(Axis(1), seq_len - 1).to_owned();
        (last_h, cache)
    }

    /// Many-to-many 前向传播：输入 shape [batch_size, seq_len, input_dim]
    /// 返回所有时间步的隐藏状态 [batch_size, seq_len, hidden_size] 及缓存
    pub fn forward_many_to_many(&self, xs: &Array3<f64>) -> (Array3<f64>, BatchCache) {
        self.forward_batch(xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::{losses, sgd::Sgd};
    use ndarray::{arr1, arr2, Array3};

    #[test]
    fn test_rnn_layer_training_step() {
        let input_size = 3;
        let hidden_size = 4;
        let seq_len = 5;
        let learning_rate = 0.01;

        let mut layer = RnnLayer::new(input_size, hidden_size);
        let optimizer = Sgd::new(learning_rate).with_gradient_clipping(1.0);
        
        let xs: Vec<_> = (0..seq_len)
            .map(|_| arr1(&[1.0, 2.0, 3.0]))
            .collect();
        let ys: Vec<_> = (0..seq_len)
            .map(|_| arr1(&[0.5, -0.5, 0.2, -0.2]))
            .collect();

        let mut last_loss = f64::MAX;

        for i in 0..10 {
            let (ps, cache) = layer.forward(&xs);

            let mut total_loss = 0.0;
            let mut d_ps = Vec::new();
            for (p, y) in ps.iter().zip(ys.iter()) {
                total_loss += losses::mean_squared_error(p, y);
                d_ps.push(losses::mean_squared_error_derivative(p, y));
            }
            total_loss /= seq_len as f64;
            
            println!("Iteration {}: Loss = {}", i, total_loss);
            
            let mut grads = layer.backward(&d_ps, &cache);

            optimizer.update(&mut layer.cell, &mut grads);

            if i > 0 {
                assert!(total_loss < last_loss, "Loss did not decrease at iteration {}", i);
            }
            last_loss = total_loss;
        }

        assert!(last_loss < 1.0);
    }

    #[test]
    fn test_rnn_layer_training_step_batch() {
        let input_size = 2;
        let hidden_size = 3;
        let seq_len = 3;
        let batch_size = 2;
        let learning_rate = 0.05;

        let mut layer = RnnLayer::new(input_size, hidden_size);
        let optimizer = Sgd::new(learning_rate).with_gradient_clipping(1.0);

        // 构造输入 [batch_size, seq_len, input_size]
        let mut xs = Array3::<f64>::zeros((batch_size, seq_len, input_size));
        xs.slice_mut(s![0, .., ..]).assign(&arr2(&[[1.0, 2.0], [2.0, 1.0], [0.5, 1.5]]));
        xs.slice_mut(s![1, .., ..]).assign(&arr2(&[[0.5, 1.0], [1.5, 0.5], [1.0, 1.0]]));

        // 构造目标输出 [batch_size, seq_len, hidden_size]
        let mut ys = Array3::<f64>::zeros((batch_size, seq_len, hidden_size));
        ys.slice_mut(s![0, .., ..]).assign(&arr2(&[[0.1, 0.2, 0.3], [0.2, 0.1, 0.4], [0.3, 0.2, 0.1]]));
        ys.slice_mut(s![1, .., ..]).assign(&arr2(&[[0.2, 0.1, 0.2], [0.1, 0.3, 0.2], [0.2, 0.2, 0.2]]));

        let mut last_loss = f64::MAX;
        for i in 0..20 {
            let (ps, cache) = layer.forward_batch(&xs);
            // 展平 [batch_size, seq_len, hidden_size] 为 [batch_size * seq_len, hidden_size]
            let ps_flat = ndarray::Array2::from_shape_vec(
                (batch_size * seq_len, hidden_size),
                ps.iter().copied().collect()
            ).unwrap();
            let ys_flat = ndarray::Array2::from_shape_vec(
                (batch_size * seq_len, hidden_size),
                ys.iter().copied().collect()
            ).unwrap();
            let loss = losses::mean_squared_error_batch(&ps_flat, &ys_flat);
            let d_ps = losses::mean_squared_error_derivative_batch(&ps_flat, &ys_flat)
                .into_shape_with_order((batch_size, seq_len, hidden_size)).unwrap();
            let mut grads = layer.backward_batch(&d_ps, &cache);
            optimizer.update(&mut layer.cell, &mut grads);
            if i > 0 {
                assert!(loss < last_loss + 1e-6, "Batch loss did not decrease at iter {i}");
            }
            last_loss = loss;
        }
        assert!(last_loss < 0.5);
    }

    #[test]
    fn test_rnn_layer_training_step_many_to_one() {
        let input_size = 2;
        let hidden_size = 3;
        let seq_len = 4;
        let batch_size = 2;
        let learning_rate = 0.05;

        let mut layer = RnnLayer::new(input_size, hidden_size);
        let optimizer = Sgd::new(learning_rate).with_gradient_clipping(1.0);

        // 构造输入 [batch_size, seq_len, input_size]
        let mut xs = Array3::<f64>::zeros((batch_size, seq_len, input_size));
        xs.slice_mut(s![0, .., ..]).assign(&arr2(&[[1.0, 2.0], [2.0, 1.0], [0.5, 1.5], [1.5, 0.5]]));
        xs.slice_mut(s![1, .., ..]).assign(&arr2(&[[0.5, 1.0], [1.5, 0.5], [1.0, 1.0], [0.0, 2.0]]));

        // 构造目标输出 [batch_size, hidden_size]，如 one-hot 分类
        let ys = arr2(&[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]);

        let mut last_loss = f64::MAX;
        for i in 0..20 {
            let (ps, cache) = layer.forward_many_to_one(&xs);
            let loss = losses::mean_squared_error_batch(&ps, &ys);
            let d_ps = losses::mean_squared_error_derivative_batch(&ps, &ys);
            // 只对最后一个时间步反向传播
            let mut d_h_list = Array3::<f64>::zeros((batch_size, seq_len, hidden_size));
            d_h_list.slice_mut(s![.., seq_len-1, ..]).assign(&d_ps);
            let mut grads = layer.backward_batch(&d_h_list, &cache);
            optimizer.update(&mut layer.cell, &mut grads);
            if i > 0 {
                assert!(loss < last_loss + 1e-6, "Many-to-one loss did not decrease at iter {i}");
            }
            last_loss = loss;
        }
        assert!(last_loss < 0.5);
    }

    #[test]
    fn test_rnn_layer_training_step_many_to_many() {
        let input_size = 2;
        let hidden_size = 2;
        let seq_len = 3;
        let batch_size = 2;
        let learning_rate = 0.05;

        let mut layer = RnnLayer::new(input_size, hidden_size);
        let optimizer = Sgd::new(learning_rate).with_gradient_clipping(1.0);

        // 构造输入 [batch_size, seq_len, input_size]
        let mut xs = Array3::<f64>::zeros((batch_size, seq_len, input_size));
        xs.slice_mut(s![0, .., ..]).assign(&arr2(&[[1.0, 2.0], [2.0, 1.0], [0.5, 1.5]]));
        xs.slice_mut(s![1, .., ..]).assign(&arr2(&[[0.5, 1.0], [1.5, 0.5], [1.0, 1.0]]));

        // 构造目标输出 [batch_size, seq_len, hidden_size]，如每步 one-hot 标签
        let mut ys = Array3::<f64>::zeros((batch_size, seq_len, hidden_size));
        ys.slice_mut(s![0, .., ..]).assign(&arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]));
        ys.slice_mut(s![1, .., ..]).assign(&arr2(&[[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]));

        let mut last_loss = f64::MAX;
        for i in 0..20 {
            let (ps, cache) = layer.forward_many_to_many(&xs);
            // 展平 [batch_size, seq_len, hidden_size] 为 [batch_size * seq_len, hidden_size]
            let ps_flat = ndarray::Array2::from_shape_vec(
                (batch_size * seq_len, hidden_size),
                ps.iter().copied().collect()
            ).unwrap();
            let ys_flat = ndarray::Array2::from_shape_vec(
                (batch_size * seq_len, hidden_size),
                ys.iter().copied().collect()
            ).unwrap();
            let loss = losses::mean_squared_error_batch(&ps_flat, &ys_flat);
            let d_ps = losses::mean_squared_error_derivative_batch(&ps_flat, &ys_flat)
                .into_shape_with_order((batch_size, seq_len, hidden_size)).unwrap();
            let mut grads = layer.backward_batch(&d_ps, &cache);
            optimizer.update(&mut layer.cell, &mut grads);
            if i > 0 {
                assert!(loss < last_loss + 1e-6, "Many-to-many loss did not decrease at iter {i}");
            }
            last_loss = loss;
        }
        assert!(last_loss < 0.5);
    }
} 