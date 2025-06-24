use crate::layers::lstm_cell::{LstmCell, LstmCellGradient};
use ndarray::{Array2, Array3, Axis, s};
use ndarray::Array1;
use crate::optimizers::{losses, sgd::Sgd};
use ndarray::{arr2, arr1};

/// LSTM 层的批处理缓存
pub struct LstmBatchCache {
    pub h_states: Vec<Array2<f64>>, // [seq_len+1] of [batch, hidden_size]
    pub c_states: Vec<Array2<f64>>, // [seq_len+1] of [batch, hidden_size]
}

/// LSTM 层，支持批处理序列前向传播
pub struct LstmLayer {
    pub cell: LstmCell,
    pub hidden_size: usize,
}

impl LstmLayer {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            cell: LstmCell::new(input_size, hidden_size),
            hidden_size,
        }
    }

    /// 批处理序列前向传播
    /// 输入 xs: [batch, seq_len, input_dim]
    /// 返回 hs: [batch, seq_len, hidden_size], cs: [batch, seq_len, hidden_size]，以及缓存
    pub fn forward_batch(&self, xs: &Array3<f64>) -> (Array3<f64>, Array3<f64>, LstmBatchCache) {
        let (batch_size, seq_len, _input_dim) = xs.dim();
        let mut h_states = Vec::with_capacity(seq_len + 1);
        let mut c_states = Vec::with_capacity(seq_len + 1);
        let mut hs_out = Vec::with_capacity(seq_len);
        let mut cs_out = Vec::with_capacity(seq_len);
        let mut h_prev = Array2::zeros((batch_size, self.hidden_size));
        let mut c_prev = Array2::zeros((batch_size, self.hidden_size));
        h_states.push(h_prev.clone());
        c_states.push(c_prev.clone());
        for t in 0..seq_len {
            let x_t = xs.slice(s![.., t, ..]).to_owned();
            let (h_t, c_t) = self.cell.forward_batch(&x_t, &h_prev, &c_prev);
            hs_out.push(h_t.clone());
            cs_out.push(c_t.clone());
            h_states.push(h_t.clone());
            c_states.push(c_t.clone());
            h_prev = h_t;
            c_prev = c_t;
        }
        let hs_stack = ndarray::stack(Axis(1), &hs_out.iter().map(|h| h.view()).collect::<Vec<_>>()).unwrap();
        let cs_stack = ndarray::stack(Axis(1), &cs_out.iter().map(|c| c.view()).collect::<Vec<_>>()).unwrap();
        let cache = LstmBatchCache { h_states, c_states };
        (hs_stack, cs_stack, cache)
    }

    /// 批处理序列反向传播
    /// d_h_list: [batch, seq_len, hidden_size]，cache: LstmBatchCache
    /// 返回参数累加梯度
    pub fn backward_batch(&self, d_h_list: &Array3<f64>, cache: &LstmBatchCache) -> LstmCellGradient {
        let (batch_size, seq_len, hidden_size) = d_h_list.dim();
        let input_size = self.cell.input_size;
        let mut grad = LstmCellGradient::zeros(input_size, hidden_size);
        let mut dh_next = Array2::zeros((batch_size, hidden_size));
        let mut dc_next = Array2::zeros((batch_size, hidden_size));
        for t in (0..seq_len).rev() {
            let dht = d_h_list.slice(s![.., t, ..]).to_owned() + &dh_next;
            let c_prev = &cache.c_states[t];
            let mut dw_f = Array2::zeros((hidden_size, input_size + hidden_size));
            let mut dw_i = Array2::zeros((hidden_size, input_size + hidden_size));
            let mut dw_c = Array2::zeros((hidden_size, input_size + hidden_size));
            let mut dw_o = Array2::zeros((hidden_size, input_size + hidden_size));
            let mut db_f = Array1::zeros(hidden_size);
            let mut db_i = Array1::zeros(hidden_size);
            let mut db_c = Array1::zeros(hidden_size);
            let mut db_o = Array1::zeros(hidden_size);
            let mut dh_prev = Array2::zeros((batch_size, hidden_size));
            let mut dc_prev = Array2::zeros((batch_size, hidden_size));
            for b in 0..batch_size {
                // 取每个样本的 cache
                let h_cache = &cache.h_states[t + 1].row(b).to_owned();
                let c_cache = &cache.c_states[t + 1].row(b).to_owned();
                let c_prev_b = &cache.c_states[t].row(b).to_owned();
                // 构造 LstmCache
                let cell_cache = self.cell.forward(
                    &Array1::zeros(input_size), // x_t 不参与反向传播
                    h_cache,
                    c_prev_b,
                ).2;
                let (g, _dx, dhp, dcp) = self.cell.backward(
                    &dht.row(b).to_owned(),
                    &dc_next.row(b).to_owned(),
                    &cell_cache,
                );
                dw_f += &g.dw_f;
                dw_i += &g.dw_i;
                dw_c += &g.dw_c;
                dw_o += &g.dw_o;
                db_f += &g.db_f;
                db_i += &g.db_i;
                db_c += &g.db_c;
                db_o += &g.db_o;
                dh_prev.row_mut(b).assign(&dhp);
                dc_prev.row_mut(b).assign(&dcp);
            }
            grad.dw_f = &grad.dw_f + &dw_f.mapv(|v| v / batch_size as f64);
            grad.dw_i = &grad.dw_i + &dw_i.mapv(|v| v / batch_size as f64);
            grad.dw_c = &grad.dw_c + &dw_c.mapv(|v| v / batch_size as f64);
            grad.dw_o = &grad.dw_o + &dw_o.mapv(|v| v / batch_size as f64);
            grad.db_f = &grad.db_f + &db_f.mapv(|v| v / batch_size as f64);
            grad.db_i = &grad.db_i + &db_i.mapv(|v| v / batch_size as f64);
            grad.db_c = &grad.db_c + &db_c.mapv(|v| v / batch_size as f64);
            grad.db_o = &grad.db_o + &db_o.mapv(|v| v / batch_size as f64);
            dh_next = dh_prev;
            dc_next = dc_prev;
        }
        grad
    }

    /// Many-to-one 前向传播：输入 shape [batch_size, seq_len, input_dim]
    /// 返回每个 batch 的最后一个隐藏状态 [batch_size, hidden_size] 及缓存
    pub fn forward_many_to_one(&self, xs: &Array3<f64>) -> (Array2<f64>, LstmBatchCache) {
        let (hs_stack, _cs_stack, cache) = self.forward_batch(xs);
        let seq_len = hs_stack.shape()[1];
        let last_h = hs_stack.index_axis(Axis(1), seq_len - 1).to_owned();
        (last_h, cache)
    }

    /// Many-to-many 前向传播：输入 shape [batch_size, seq_len, input_dim]
    /// 返回所有时间步的隐藏状态 [batch_size, seq_len, hidden_size] 及缓存
    pub fn forward_many_to_many(&self, xs: &Array3<f64>) -> (Array3<f64>, LstmBatchCache) {
        let (hs_stack, _cs_stack, cache) = self.forward_batch(xs);
        (hs_stack, cache)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, Array3};

    #[test]
    fn test_lstm_layer_forward_batch() {
        let input_size = 2;
        let hidden_size = 3;
        let seq_len = 4;
        let batch_size = 2;
        let layer = LstmLayer::new(input_size, hidden_size);
        let mut xs = Array3::<f64>::zeros((batch_size, seq_len, input_size));
        xs.slice_mut(s![0, .., ..]).assign(&arr2(&[[1.0, 2.0], [2.0, 1.0], [0.5, 1.5], [1.5, 0.5]]));
        xs.slice_mut(s![1, .., ..]).assign(&arr2(&[[0.5, 1.0], [1.5, 0.5], [1.0, 1.0], [0.0, 2.0]]));
        let (hs, cs, _cache) = layer.forward_batch(&xs);
        assert_eq!(hs.shape(), &[batch_size, seq_len, hidden_size]);
        assert_eq!(cs.shape(), &[batch_size, seq_len, hidden_size]);
    }

    #[test]
    fn test_lstm_layer_training_step_many_to_one() {
        let input_size = 2;
        let hidden_size = 3;
        let seq_len = 6;
        let batch_size = 2;
        let learning_rate = 0.05;

        let mut layer = LstmLayer::new(input_size, hidden_size);
        let optimizer = Sgd::new(learning_rate).with_gradient_clipping(1.0);

        // 构造长依赖记忆任务输入 [batch, seq_len, input_size]
        let mut xs = Array3::<f64>::zeros((batch_size, seq_len, input_size));
        // batch 0: 前3步为信号，后3步为干扰
        xs.slice_mut(s![0, 0, ..]).assign(&arr1(&[1.0, 0.0]));
        xs.slice_mut(s![0, 1, ..]).assign(&arr1(&[0.0, 1.0]));
        xs.slice_mut(s![0, 2, ..]).assign(&arr1(&[1.0, 0.0]));
        xs.slice_mut(s![0, 3.., ..]).assign(&arr2(&[[0.5, 0.5]; 3]));
        // batch 1: 前3步为信号，后3步为干扰
        xs.slice_mut(s![1, 0, ..]).assign(&arr1(&[0.0, 1.0]));
        xs.slice_mut(s![1, 1, ..]).assign(&arr1(&[1.0, 0.0]));
        xs.slice_mut(s![1, 2, ..]).assign(&arr1(&[0.0, 1.0]));
        xs.slice_mut(s![1, 3.., ..]).assign(&arr2(&[[0.5, 0.5]; 3]));

        // 目标输出为第2步的信号（模拟记忆任务）
        let ys = arr2(&[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]);

        let mut last_loss = f64::MAX;
        for i in 0..40 {
            let (ps, cache) = layer.forward_many_to_one(&xs);
            let loss = losses::mean_squared_error_batch(&ps, &ys);
            let d_ps = losses::mean_squared_error_derivative_batch(&ps, &ys);
            // 只对最后一个时间步反向传播
            let mut d_h_list = Array3::<f64>::zeros((batch_size, seq_len, hidden_size));
            d_h_list.slice_mut(s![.., seq_len-1, ..]).assign(&d_ps);
            let mut grads = layer.backward_batch(&d_h_list, &cache);
            layer.cell.update(&grads, learning_rate);
            if i > 0 {
                assert!(loss < last_loss + 1e-6, "LSTM Many-to-one loss did not decrease at iter {i}");
            }
            last_loss = loss;
        }
        assert!(last_loss < 0.5);
    }
} 