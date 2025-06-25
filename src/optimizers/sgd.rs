use crate::layers::rnn_cell::{RnnCell, RnnCellGradient};
use crate::layers::lstm_cell::{LstmCell, LstmCellGradient};
use crate::layers::gru_cell::{GruCell, GruCellGradient};

/// 定义梯度裁剪的通用接口。
pub trait GradientClipping {
    /// 计算梯度的总范数。
    fn gradient_norm(&self) -> f64;

    /// 按比例缩放梯度。
    fn scale_gradients(&mut self, scale: f64);

    /// 应用梯度裁剪。
    fn clip_gradients(&mut self, max_norm: f64) {
        let total_norm = self.gradient_norm();
        if total_norm > max_norm {
            let scale = max_norm / total_norm;
            self.scale_gradients(scale);
        }
    }
}

/// 为 RnnCellGradient 实现梯度裁剪。
impl GradientClipping for RnnCellGradient {
    fn gradient_norm(&self) -> f64 {
        let mut total_norm = 0.0;
        total_norm += self.w_hh.iter().map(|&g| g.powi(2)).sum::<f64>();
        total_norm += self.w_ih.iter().map(|&g| g.powi(2)).sum::<f64>();
        total_norm += self.b_h.iter().map(|&g| g.powi(2)).sum::<f64>();
        total_norm.sqrt()
    }

    fn scale_gradients(&mut self, scale: f64) {
        self.w_hh.mapv_inplace(|g| g * scale);
        self.w_ih.mapv_inplace(|g| g * scale);
        self.b_h.mapv_inplace(|g| g * scale);
    }
}

/// 为 LstmCellGradient 实现梯度裁剪。
impl GradientClipping for LstmCellGradient {
    fn gradient_norm(&self) -> f64 {
        let mut total_norm = 0.0;
        total_norm += self.dw_f.iter().map(|&g| g.powi(2)).sum::<f64>();
        total_norm += self.dw_i.iter().map(|&g| g.powi(2)).sum::<f64>();
        total_norm += self.dw_c.iter().map(|&g| g.powi(2)).sum::<f64>();
        total_norm += self.dw_o.iter().map(|&g| g.powi(2)).sum::<f64>();
        total_norm += self.db_f.iter().map(|&g| g.powi(2)).sum::<f64>();
        total_norm += self.db_i.iter().map(|&g| g.powi(2)).sum::<f64>();
        total_norm += self.db_c.iter().map(|&g| g.powi(2)).sum::<f64>();
        total_norm += self.db_o.iter().map(|&g| g.powi(2)).sum::<f64>();
        total_norm.sqrt()
    }

    fn scale_gradients(&mut self, scale: f64) {
        self.dw_f.mapv_inplace(|g| g * scale);
        self.dw_i.mapv_inplace(|g| g * scale);
        self.dw_c.mapv_inplace(|g| g * scale);
        self.dw_o.mapv_inplace(|g| g * scale);
        self.db_f.mapv_inplace(|g| g * scale);
        self.db_i.mapv_inplace(|g| g * scale);
        self.db_c.mapv_inplace(|g| g * scale);
        self.db_o.mapv_inplace(|g| g * scale);
    }
}

/// 为 GruCellGradient 实现梯度裁剪。
impl GradientClipping for GruCellGradient {
    fn gradient_norm(&self) -> f64 {
        let mut total_norm = 0.0;
        total_norm += self.dw_r.iter().map(|&g| g.powi(2)).sum::<f64>();
        total_norm += self.dw_z.iter().map(|&g| g.powi(2)).sum::<f64>();
        total_norm += self.dw_h.iter().map(|&g| g.powi(2)).sum::<f64>();
        total_norm += self.db_r.iter().map(|&g| g.powi(2)).sum::<f64>();
        total_norm += self.db_z.iter().map(|&g| g.powi(2)).sum::<f64>();
        total_norm += self.db_h.iter().map(|&g| g.powi(2)).sum::<f64>();
        total_norm.sqrt()
    }

    fn scale_gradients(&mut self, scale: f64) {
        self.dw_r.mapv_inplace(|g| g * scale);
        self.dw_z.mapv_inplace(|g| g * scale);
        self.dw_h.mapv_inplace(|g| g * scale);
        self.db_r.mapv_inplace(|g| g * scale);
        self.db_z.mapv_inplace(|g| g * scale);
        self.db_h.mapv_inplace(|g| g * scale);
    }
}

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

    /// 使用计算出的梯度来更新RNN模型的参数。
    ///
    /// # Arguments
    ///
    /// * `params` - 需要更新的模型参数 (可变引用)。
    /// * `grads` - 计算出的梯度。
    pub fn update(&self, params: &mut RnnCell, grads: &mut RnnCellGradient) {
        if let Some(max_norm) = self.max_norm {
            grads.clip_gradients(max_norm);
        }

        params.w_hh.scaled_add(-self.learning_rate, &grads.w_hh);
        params.w_ih.scaled_add(-self.learning_rate, &grads.w_ih);
        params.b_h.scaled_add(-self.learning_rate, &grads.b_h);
    }

    /// 使用计算出的梯度来更新LSTM模型的参数。
    ///
    /// # Arguments
    ///
    /// * `params` - 需要更新的模型参数 (可变引用)。
    /// * `grads` - 计算出的梯度。
    pub fn update_lstm(&self, params: &mut LstmCell, grads: &mut LstmCellGradient) {
        if let Some(max_norm) = self.max_norm {
            grads.clip_gradients(max_norm);
        }

        params.w_f.scaled_add(-self.learning_rate, &grads.dw_f);
        params.w_i.scaled_add(-self.learning_rate, &grads.dw_i);
        params.w_c.scaled_add(-self.learning_rate, &grads.dw_c);
        params.w_o.scaled_add(-self.learning_rate, &grads.dw_o);
        params.b_f.scaled_add(-self.learning_rate, &grads.db_f);
        params.b_i.scaled_add(-self.learning_rate, &grads.db_i);
        params.b_c.scaled_add(-self.learning_rate, &grads.db_c);
        params.b_o.scaled_add(-self.learning_rate, &grads.db_o);
    }

    /// 使用计算出的梯度来更新GRU模型的参数。
    ///
    /// # Arguments
    ///
    /// * `params` - 需要更新的模型参数 (可变引用)。
    /// * `grads` - 计算出的梯度。
    pub fn update_gru(&self, params: &mut GruCell, grads: &mut GruCellGradient) {
        if let Some(max_norm) = self.max_norm {
            grads.clip_gradients(max_norm);
        }

        params.w_r.scaled_add(-self.learning_rate, &grads.dw_r);
        params.w_z.scaled_add(-self.learning_rate, &grads.dw_z);
        params.w_h.scaled_add(-self.learning_rate, &grads.dw_h);
        params.b_r.scaled_add(-self.learning_rate, &grads.db_r);
        params.b_z.scaled_add(-self.learning_rate, &grads.db_z);
        params.b_h.scaled_add(-self.learning_rate, &grads.db_h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_gradient_clipping_rnn() {
        let mut grads = RnnCellGradient {
            w_hh: Array2::from_elem((2, 2), 10.0), // 大梯度
            w_ih: Array2::from_elem((2, 3), 10.0),
            b_h: Array1::from_elem(2, 10.0),
        };

        let initial_norm = grads.gradient_norm();
        assert!(initial_norm > 5.0); // 确保初始梯度很大

        grads.clip_gradients(1.0); // 裁剪到最大范数1.0

        let clipped_norm = grads.gradient_norm();
        assert!((clipped_norm - 1.0).abs() < 1e-6); // 裁剪后应该接近1.0
    }

    #[test]
    fn test_gradient_clipping_lstm() {
        let mut grads = LstmCellGradient {
            dw_f: Array2::from_elem((2, 5), 5.0),
            dw_i: Array2::from_elem((2, 5), 5.0),
            dw_c: Array2::from_elem((2, 5), 5.0),
            dw_o: Array2::from_elem((2, 5), 5.0),
            db_f: Array1::from_elem(2, 5.0),
            db_i: Array1::from_elem(2, 5.0),
            db_c: Array1::from_elem(2, 5.0),
            db_o: Array1::from_elem(2, 5.0),
        };

        let initial_norm = grads.gradient_norm();
        assert!(initial_norm > 2.0);

        grads.clip_gradients(1.0);

        let clipped_norm = grads.gradient_norm();
        assert!((clipped_norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_clipping_gru() {
        let mut grads = GruCellGradient {
            dw_r: Array2::from_elem((2, 5), 3.0),
            dw_z: Array2::from_elem((2, 5), 3.0),
            dw_h: Array2::from_elem((2, 5), 3.0),
            db_r: Array1::from_elem(2, 3.0),
            db_z: Array1::from_elem(2, 3.0),
            db_h: Array1::from_elem(2, 3.0),
        };

        let initial_norm = grads.gradient_norm();
        assert!(initial_norm > 1.5);

        grads.clip_gradients(1.0);

        let clipped_norm = grads.gradient_norm();
        assert!((clipped_norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_with_gradient_clipping() {
        use crate::activations::functions::Tanh;

        let mut cell = RnnCell::new(3, 2, Box::new(Tanh));
        let mut grads = RnnCellGradient {
            w_hh: Array2::from_elem((2, 2), 10.0), // 大梯度
            w_ih: Array2::from_elem((2, 3), 10.0),
            b_h: Array1::from_elem(2, 10.0),
        };

        let optimizer = Sgd::new(0.01).with_gradient_clipping(1.0);

        // 记录更新前的参数
        let w_hh_before = cell.w_hh.clone();

        optimizer.update(&mut cell, &mut grads);

        // 验证参数确实被更新了
        assert!((cell.w_hh[[0, 0]] - w_hh_before[[0, 0]]).abs() > 1e-6);

        // 验证梯度被裁剪了
        let final_norm = grads.gradient_norm();
        assert!((final_norm - 1.0).abs() < 1e-6);
    }
}