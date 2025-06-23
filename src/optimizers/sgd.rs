use crate::layers::rnn_cell::{RnnCell, RnnCellGradient};

/// 实现随机梯度下降 (SGD) 优化器。
pub struct Sgd {
    learning_rate: f64,
    max_norm: Option<f64>,
}

impl Sgd {
    /// 创建一个新的 SGD 优化器实例。
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            max_norm: None,
        }
    }

    /// 设置梯度裁剪的阈值。
    pub fn with_gradient_clipping(mut self, max_norm: f64) -> Self {
        self.max_norm = Some(max_norm);
        self
    }

    /// 使用计算出的梯度来更新模型的参数。
    ///
    /// # Arguments
    ///
    /// * `params` - 需要更新的模型参数 (可变引用)。
    /// * `grads` - 计算出的梯度。
    pub fn update(&self, params: &mut RnnCell, grads: &mut RnnCellGradient) {
        if let Some(max_norm) = self.max_norm {
            let mut total_norm = 0.0;
            for p in [&grads.w_hh, &grads.w_ih] {
                total_norm += p.iter().map(|&g| g.powi(2)).sum::<f64>();
            }
            total_norm += grads.b_h.iter().map(|&g| g.powi(2)).sum::<f64>();
            total_norm = total_norm.sqrt();

            if total_norm > max_norm {
                let scale = max_norm / total_norm;
                grads.w_hh.mapv_inplace(|g| g * scale);
                grads.w_ih.mapv_inplace(|g| g * scale);
                grads.b_h.mapv_inplace(|g| g * scale);
            }
        }

        params.w_hh.scaled_add(-self.learning_rate, &grads.w_hh);
        params.w_ih.scaled_add(-self.learning_rate, &grads.w_ih);
        params.b_h.scaled_add(-self.learning_rate, &grads.b_h);
    }
} 