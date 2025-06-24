use crate::layers::gru_cell::{GruCell, GruCellGradient, GruCache};
use ndarray::{Array1, Array2, Array3, Axis, s};
use crate::layers::bi_rnn_layer::RnnLikeLayer;

/// GRU层的批处理缓存，存储每个batch的中间状态
#[derive(Clone)]
pub struct GruBatchCache {
    pub caches: Vec<Vec<GruCache>>, // [seq_len][batch] 的cache
    pub h_states: Vec<Array2<f64>>, // [seq_len+1] of [batch, hidden_size]
}

/// GRU层，支持批处理序列前向传播
#[derive(Clone)]
pub struct GruLayer {
    pub cell: GruCell,
    pub hidden_size: usize,
}

impl GruLayer {
    /// 创建一个新的GruLayer
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            cell: GruCell::new(input_size, hidden_size),
            hidden_size,
        }
    }

    /// 批处理前向传播，输入shape [batch, seq_len, input_dim]
    /// 返回所有时间步的隐藏状态 [batch, seq_len, hidden_size] 及缓存
    pub fn forward_batch(&self, xs: &Array3<f64>) -> (Array3<f64>, GruBatchCache) {
        let (batch_size, seq_len, _input_dim) = xs.dim();
        let mut h_states = Vec::with_capacity(seq_len + 1);
        let mut caches = Vec::with_capacity(seq_len);
        let mut h_prev = Array2::zeros((batch_size, self.hidden_size));
        h_states.push(h_prev.clone());
        for t in 0..seq_len {
            let x_t = xs.slice(s![.., t, ..]).to_owned();
            let (h_t, cache_t) = self.cell.forward_batch(&x_t, &h_prev);
            h_states.push(h_t.clone());
            caches.push(cache_t);
            h_prev = h_t;
        }
        let hs_stack = ndarray::stack(Axis(1), &h_states[1..].iter().map(|h| h.view()).collect::<Vec<_>>()).unwrap();
        let cache = GruBatchCache { caches, h_states };
        (hs_stack, cache)
    }

    /// Many-to-one前向传播，返回每个batch的最后一个隐藏状态 [batch, hidden_size] 及缓存
    pub fn forward_many_to_one(&self, xs: &Array3<f64>) -> (Array2<f64>, GruBatchCache) {
        let (hs_stack, cache) = self.forward_batch(xs);
        let seq_len = hs_stack.shape()[1];
        let last_h = hs_stack.index_axis(Axis(1), seq_len - 1).to_owned();
        (last_h, cache)
    }

    /// Many-to-many前向传播，返回所有时间步的隐藏状态 [batch, seq_len, hidden_size] 及缓存
    pub fn forward_many_to_many(&self, xs: &Array3<f64>) -> (Array3<f64>, GruBatchCache) {
        self.forward_batch(xs)
    }

    /// 批处理反向传播，输入d_h_list [batch, seq_len, hidden_size]，返回累加梯度
    pub fn backward_batch(&self, d_h_list: &Array3<f64>, cache: &GruBatchCache) -> GruCellGradient {
        let (batch_size, seq_len, hidden_size) = d_h_list.dim();
        let input_size = self.cell.input_size;
        let mut grads = GruCellGradient::zeros(input_size, hidden_size);
        let mut dh_next = Array2::zeros((batch_size, hidden_size));
        for t in (0..seq_len).rev() {
            let dht = d_h_list.slice(s![.., t, ..]).to_owned() + &dh_next;
            let caches_t = &cache.caches[t];
            let mut dw_r = Array2::zeros((hidden_size, input_size + hidden_size));
            let mut dw_z = Array2::zeros((hidden_size, input_size + hidden_size));
            let mut dw_h = Array2::zeros((hidden_size, input_size + hidden_size));
            let mut db_r = Array1::zeros(hidden_size);
            let mut db_z = Array1::zeros(hidden_size);
            let mut db_h = Array1::zeros(hidden_size);
            let mut dh_prev = Array2::zeros((batch_size, hidden_size));
            for b in 0..batch_size {
                let (g, _dx, dhp) = self.cell.backward(&dht.row(b).to_owned(), &caches_t[b]);
                dw_r += &g.dw_r;
                dw_z += &g.dw_z;
                dw_h += &g.dw_h;
                db_r += &g.db_r;
                db_z += &g.db_z;
                db_h += &g.db_h;
                dh_prev.row_mut(b).assign(&dhp);
            }
            grads.dw_r = &grads.dw_r + &dw_r.mapv(|v| v / batch_size as f64);
            grads.dw_z = &grads.dw_z + &dw_z.mapv(|v| v / batch_size as f64);
            grads.dw_h = &grads.dw_h + &dw_h.mapv(|v| v / batch_size as f64);
            grads.db_r = &grads.db_r + &db_r.mapv(|v| v / batch_size as f64);
            grads.db_z = &grads.db_z + &db_z.mapv(|v| v / batch_size as f64);
            grads.db_h = &grads.db_h + &db_h.mapv(|v| v / batch_size as f64);
            dh_next = dh_prev;
        }
        grads
    }
}

impl RnnLikeLayer for GruLayer {
    type Cache = GruBatchCache;
    fn forward_batch(&self, xs: &ndarray::Array3<f64>) -> (ndarray::Array3<f64>, Self::Cache) {
        GruLayer::forward_batch(self, xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::{losses, sgd::Sgd};
    use ndarray::{arr1, arr2, Array3};

    #[test]
    fn test_gru_layer_long_term_memory_task() {
        // 任务：输入序列前几步为信号，后续为干扰，目标是让GRU记住最早的信号，实现长距离依赖记忆。
        // 输入 shape: [batch, seq_len, input_dim]
        let input_size = 2;
        let hidden_size = 3;
        let seq_len = 8;
        let batch_size = 2;
        let learning_rate = 0.05;

        let mut layer = GruLayer::new(input_size, hidden_size);
        let optimizer = Sgd::new(learning_rate).with_gradient_clipping(1.0);

        // 构造输入：batch 0 前2步为信号[1,0]，后6步为干扰[0.5,0.5]；batch 1 前2步为信号[0,1]，后6步为干扰[0.5,0.5]
        let mut xs = Array3::<f64>::zeros((batch_size, seq_len, input_size));
        xs.slice_mut(s![0, 0, ..]).assign(&arr1(&[1.0, 0.0]));
        xs.slice_mut(s![0, 1, ..]).assign(&arr1(&[1.0, 0.0]));
        xs.slice_mut(s![0, 2.., ..]).assign(&arr2(&[[0.5, 0.5]; 6]));
        xs.slice_mut(s![1, 0, ..]).assign(&arr1(&[0.0, 1.0]));
        xs.slice_mut(s![1, 1, ..]).assign(&arr1(&[0.0, 1.0]));
        xs.slice_mut(s![1, 2.., ..]).assign(&arr2(&[[0.5, 0.5]; 6]));

        // 目标输出：希望模型记住第1步的信号
        let ys = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);

        let mut last_loss = f64::MAX;
        for i in 0..200 {
            // 只取最后一个时间步的隐藏状态用于输出
            let (ps, cache) = layer.forward_many_to_one(&xs);
            let loss = losses::mean_squared_error_batch(&ps, &ys);
            let d_ps = losses::mean_squared_error_derivative_batch(&ps, &ys);
            // 只对最后一个时间步反向传播
            let mut d_h_list = Array3::<f64>::zeros((batch_size, seq_len, hidden_size));
            d_h_list.slice_mut(s![.., seq_len-1, ..]).assign(&d_ps);
            let grads = layer.backward_batch(&d_h_list, &cache);
            layer.cell.update(&grads, learning_rate);
            if i > 0 {
                assert!(loss < last_loss + 1e-6, "GRU 长距离记忆任务 loss 未递减，第{i}轮");
            }
            last_loss = loss;
        }
        // 最终loss应足够小，说明模型学会了记忆
        assert!(last_loss < 0.2, "最终loss未收敛，GRU未学会长距离记忆");
    }
} 