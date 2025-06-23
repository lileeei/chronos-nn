use crate::layers::rnn_cell::{RnnCell, RnnCellGradient};

/// 实现随机梯度下降 (SGD) 优化器。
pub struct Sgd {
    learning_rate: f64,
}

impl Sgd {
    /// 创建一个新的 SGD 优化器实例。
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }

    /// 使用计算出的梯度来更新模型的参数。
    ///
    /// # Arguments
    ///
    /// * `params` - 需要更新的模型参数 (可变引用)。
    /// * `grads` - 计算出的梯度。
    pub fn update(&self, params: &mut RnnCell, grads: &RnnCellGradient) {
        params.w_hh.scaled_add(-self.learning_rate, &grads.w_hh);
        params.w_ih.scaled_add(-self.learning_rate, &grads.w_ih);
        params.b_h.scaled_add(-self.learning_rate, &grads.b_h);
    }
} 