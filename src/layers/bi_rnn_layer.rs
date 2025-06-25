use ndarray::{Array3, Array2, ArrayView2, Axis};
use ndarray::s;

/// 通用RNN层Trait，所有RNN/LSTM/GRU层都应实现
pub trait RnnLikeLayer: Clone {
    type Cache: Clone;
    type Grad;
    fn forward_batch(&self, xs: &Array3<f64>) -> (Array3<f64>, Self::Cache);
    fn backward_batch(&self, d_h_list: &Array3<f64>, cache: &Self::Cache) -> Self::Grad;
}

/// 合并策略Trait，支持自定义前向/反向合并
pub trait MergeStrategy: Clone {
    /// 合并前向和反向输出
    fn merge(&self, forward: &ArrayView2<f64>, backward: &ArrayView2<f64>) -> Array2<f64>;
    /// 将合并后的梯度拆分为前向和反向部分
    fn split_grad(
        &self,
        d_merged: &ArrayView2<f64>,
        forward_dim: usize,
        backward_dim: usize,
    ) -> (Array2<f64>, Array2<f64>);
}

/// 通用双向RNN层，支持自定义合并策略
pub struct BiRnnLayer<Layer: RnnLikeLayer, Merge: MergeStrategy> {
    pub forward_layer: Layer,
    pub backward_layer: Layer,
    pub merge_strategy: Merge,
    pub hidden_size: usize,
}

impl<Layer: RnnLikeLayer, Merge: MergeStrategy> BiRnnLayer<Layer, Merge> {
    pub fn new(forward_layer: Layer, backward_layer: Layer, merge_strategy: Merge, hidden_size: usize) -> Self {
        Self {
            forward_layer,
            backward_layer,
            merge_strategy,
            hidden_size,
        }
    }

    /// 批处理前向传播，输入shape [batch, seq_len, input_dim]
    /// 输出shape [batch, seq_len, merge_dim]，merge_dim由合并策略决定
    pub fn forward_batch(&self, xs: &Array3<f64>) -> (Array3<f64>, (Layer::Cache, Layer::Cache)) {
        let (_batch_size, seq_len, _input_dim) = xs.dim();
        // 前向
        let (forward_hs, forward_cache) = self.forward_layer.forward_batch(xs);
        // 反向
        let mut xs_rev = xs.clone();
        xs_rev.invert_axis(Axis(1));
        let (backward_hs_rev, backward_cache) = self.backward_layer.forward_batch(&xs_rev);
        let mut backward_hs = backward_hs_rev.clone();
        backward_hs.invert_axis(Axis(1));
        // 合并
        let mut outputs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let f = forward_hs.index_axis(Axis(1), t);
            let b = backward_hs.index_axis(Axis(1), t);
            outputs.push(self.merge_strategy.merge(&f, &b));
        }
        let merged = ndarray::stack(
            Axis(1),
            &outputs.iter().map(|x| x.view()).collect::<Vec<_>>(),
        ).unwrap();
        (merged, (forward_cache, backward_cache))
    }

    /// 批处理反向传播，输入合并后梯度和cache，返回前向和反向层的梯度
    pub fn backward_batch(
        &self,
        d_merged: &Array3<f64>,
        caches: &(Layer::Cache, Layer::Cache),
    ) -> (Layer::Grad, Layer::Grad) {
        let (batch_size, seq_len, merged_dim) = d_merged.dim();
        let forward_dim = self.hidden_size;
        let backward_dim = merged_dim - forward_dim;
        let mut d_forward = Array3::<f64>::zeros((batch_size, seq_len, forward_dim));
        let mut d_backward = Array3::<f64>::zeros((batch_size, seq_len, backward_dim));
        for t in 0..seq_len {
            let d_merged_t = d_merged.index_axis(Axis(1), t);
            let (df, db) = self.merge_strategy.split_grad(&d_merged_t, forward_dim, backward_dim);
            d_forward.slice_mut(s![.., t, ..]).assign(&df);
            d_backward.slice_mut(s![.., t, ..]).assign(&db);
        }
        let grad_f = self.forward_layer.backward_batch(&d_forward, &caches.0);
        let grad_b = self.backward_layer.backward_batch(&d_backward, &caches.1);
        (grad_f, grad_b)
    }
}

#[derive(Clone)]
pub struct ConcatMerge;

impl MergeStrategy for ConcatMerge {
    fn merge(&self, forward: &ArrayView2<f64>, backward: &ArrayView2<f64>) -> Array2<f64> {
        ndarray::concatenate(Axis(1), &[forward.view(), backward.view()]).unwrap()
    }
    fn split_grad(
        &self,
        d_merged: &ArrayView2<f64>,
        forward_dim: usize,
        backward_dim: usize,
    ) -> (Array2<f64>, Array2<f64>) {
        let df = d_merged.slice(s![.., 0..forward_dim]).to_owned();
        let db = d_merged.slice(s![.., forward_dim..(forward_dim+backward_dim)]).to_owned();
        (df, db)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::gru_layer::GruLayer;
    use crate::optimizers::losses;
    use ndarray::{arr1, arr2, Array3};

    #[test]
    fn test_birnnlayer_bigru_concatmerge_forward() {
        // 任务：输入序列前几步为信号，后续为干扰，目标是让BiGRU记住最早的信号，实现长距离依赖记忆。
        let input_size = 2;
        let hidden_size = 3;
        let seq_len = 8;
        let batch_size = 2;

        let forward_gru = GruLayer::new(input_size, hidden_size);
        let backward_gru = GruLayer::new(input_size, hidden_size);
        let birnn = BiRnnLayer::new(forward_gru, backward_gru, ConcatMerge, hidden_size);

        // 构造输入
        let mut xs = Array3::<f64>::zeros((batch_size, seq_len, input_size));
        xs.slice_mut(ndarray::s![0, 0, ..]).assign(&arr1(&[1.0, 0.0]));
        xs.slice_mut(ndarray::s![0, 1, ..]).assign(&arr1(&[1.0, 0.0]));
        xs.slice_mut(ndarray::s![0, 2.., ..]).assign(&arr2(&[[0.5, 0.5]; 6]));
        xs.slice_mut(ndarray::s![1, 0, ..]).assign(&arr1(&[0.0, 1.0]));
        xs.slice_mut(ndarray::s![1, 1, ..]).assign(&arr1(&[0.0, 1.0]));
        xs.slice_mut(ndarray::s![1, 2.., ..]).assign(&arr2(&[[0.5, 0.5]; 6]));

        // 目标输出：希望模型记住第1步的信号，输出维度为hidden_size*2
        let _ys = arr2(&[[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]]);

        // 前向传播
        let (outputs, (_f_cache, _b_cache)) = birnn.forward_batch(&xs);
        // 检查输出shape应为[batch, seq_len, hidden_size*2]
        assert_eq!(outputs.shape(), &[batch_size, seq_len, hidden_size * 2]);
    }

    #[test]
    fn test_birnnlayer_bigru_concatmerge_training() {
        // 任务：输入序列前几步为信号，后续为干扰，目标是让BiGRU记住最早的信号，实现长距离依赖记忆。
        let input_size = 2;
        let hidden_size = 3;
        let seq_len = 8;
        let batch_size = 2;
        let learning_rate = 0.05;

        let mut forward_gru = GruLayer::new(input_size, hidden_size);
        let mut backward_gru = GruLayer::new(input_size, hidden_size);

        // 构造输入
        let mut xs = Array3::<f64>::zeros((batch_size, seq_len, input_size));
        xs.slice_mut(ndarray::s![0, 0, ..]).assign(&arr1(&[1.0, 0.0]));
        xs.slice_mut(ndarray::s![0, 1, ..]).assign(&arr1(&[1.0, 0.0]));
        xs.slice_mut(ndarray::s![0, 2.., ..]).assign(&arr2(&[[0.5, 0.5]; 6]));
        xs.slice_mut(ndarray::s![1, 0, ..]).assign(&arr1(&[0.0, 1.0]));
        xs.slice_mut(ndarray::s![1, 1, ..]).assign(&arr1(&[0.0, 1.0]));
        xs.slice_mut(ndarray::s![1, 2.., ..]).assign(&arr2(&[[0.5, 0.5]; 6]));

        // 目标输出：希望模型记住第1步的信号，输出维度为hidden_size*2
        let ys = arr2(&[[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]]);

        let mut last_loss = f64::MAX;
        for i in 0..200 {
            let birnn = BiRnnLayer::new(forward_gru.clone(), backward_gru.clone(), ConcatMerge, hidden_size);
            let (outputs, caches) = birnn.forward_batch(&xs);
            let last_ps = outputs.index_axis(ndarray::Axis(1), seq_len - 1).to_owned();
            let loss = losses::mean_squared_error_batch(&last_ps, &ys);
            let d_merged = losses::mean_squared_error_derivative_batch(&last_ps, &ys)
                .into_shape_with_order((batch_size, 1, hidden_size * 2)).unwrap();
            // 只对最后一个时间步反向传播
            let mut d_merged_full = Array3::<f64>::zeros((batch_size, seq_len, hidden_size * 2));
            d_merged_full.slice_mut(ndarray::s![.., seq_len-1, ..]).assign(&d_merged.index_axis(ndarray::Axis(1), 0));
            let (grad_f, grad_b) = birnn.backward_batch(&d_merged_full, &caches);
            // 简单SGD更新
            forward_gru.cell.update(&grad_f, learning_rate);
            backward_gru.cell.update(&grad_b, learning_rate);
            if i > 0 {
                assert!(loss < last_loss + 1e-6, "BiGRU训练loss未递减，第{i}轮");
            }
            last_loss = loss;
        }
        assert!(last_loss < 0.2, "最终loss未收敛，BiGRU未学会长距离记忆");
    }
} 