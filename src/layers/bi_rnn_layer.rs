use ndarray::{Array3, Array2, ArrayView2, Axis};
use ndarray::s;

/// 通用RNN层Trait，所有RNN/LSTM/GRU层都应实现
pub trait RnnLikeLayer: Clone {
    type Cache: Clone;
    fn forward_batch(&self, xs: &Array3<f64>) -> (Array3<f64>, Self::Cache);
    // 反向传播接口可后续扩展
}

/// 合并策略Trait，支持自定义前向/反向合并
pub trait MergeStrategy: Clone {
    /// 合并前向和反向输出
    fn merge(&self, forward: &ArrayView2<f64>, backward: &ArrayView2<f64>) -> Array2<f64>;
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
        let (batch_size, seq_len, _input_dim) = xs.dim();
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
}

#[derive(Clone)]
pub struct ConcatMerge;

impl MergeStrategy for ConcatMerge {
    fn merge(&self, forward: &ArrayView2<f64>, backward: &ArrayView2<f64>) -> Array2<f64> {
        ndarray::concatenate(Axis(1), &[forward.view(), backward.view()]).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::gru_layer::GruLayer;
    use crate::optimizers::{losses, sgd::Sgd};
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

        // 构造输入：batch 0 前2步为信号[1,0]，后6步为干扰[0.5,0.5]；batch 1 前2步为信号[0,1]，后6步为干扰[0.5,0.5]
        let mut xs = Array3::<f64>::zeros((batch_size, seq_len, input_size));
        xs.slice_mut(ndarray::s![0, 0, ..]).assign(&arr1(&[1.0, 0.0]));
        xs.slice_mut(ndarray::s![0, 1, ..]).assign(&arr1(&[1.0, 0.0]));
        xs.slice_mut(ndarray::s![0, 2.., ..]).assign(&arr2(&[[0.5, 0.5]; 6]));
        xs.slice_mut(ndarray::s![1, 0, ..]).assign(&arr1(&[0.0, 1.0]));
        xs.slice_mut(ndarray::s![1, 1, ..]).assign(&arr1(&[0.0, 1.0]));
        xs.slice_mut(ndarray::s![1, 2.., ..]).assign(&arr2(&[[0.5, 0.5]; 6]));

        // 前向传播
        let (outputs, (_f_cache, _b_cache)) = birnn.forward_batch(&xs);
        // 检查输出shape应为[batch, seq_len, hidden_size*2]
        assert_eq!(outputs.shape(), &[batch_size, seq_len, hidden_size * 2]);
    }
} 