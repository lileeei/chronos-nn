use crate::activations::functions::Tanh;
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
    /// 每个时间步的线性变换结果 (z_t)
    linear_outputs: Vec<Array1<f64>>,
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
} 