use ndarray::{Array1, Array2, Axis, concatenate};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use crate::activations::functions::{Sigmoid, Tanh};
use crate::activations::traits::Activation;
use ndarray::s;

/// LSTM 单元的中间缓存，用于反向传播
pub struct LstmCache {
    pub xh: Array1<f64>, // 拼接的 [x_t, h_prev]
    pub f_t: Array1<f64>,
    pub i_t: Array1<f64>,
    pub o_t: Array1<f64>,
    pub c_tilde: Array1<f64>,
    pub c_prev: Array1<f64>,
    pub c_t: Array1<f64>,
}

/// LSTM 单元参数的梯度
pub struct LstmCellGradient {
    pub dw_f: Array2<f64>,
    pub dw_i: Array2<f64>,
    pub dw_c: Array2<f64>,
    pub dw_o: Array2<f64>,
    pub db_f: Array1<f64>,
    pub db_i: Array1<f64>,
    pub db_c: Array1<f64>,
    pub db_o: Array1<f64>,
}

/// LSTM 单元（支持单步和批处理前向传播）
pub struct LstmCell {
    pub w_f: Array2<f64>, // [hidden_size, input_size + hidden_size]
    pub w_i: Array2<f64>,
    pub w_c: Array2<f64>,
    pub w_o: Array2<f64>,
    pub b_f: Array1<f64>, // [hidden_size]
    pub b_i: Array1<f64>,
    pub b_c: Array1<f64>,
    pub b_o: Array1<f64>,
    pub input_size: usize,
    pub hidden_size: usize,
    pub sigmoid: Sigmoid,
    pub tanh: Tanh,
}

impl LstmCell {
    /// 创建并初始化 LstmCell
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let k = input_size + hidden_size;
        let dist = Uniform::new(-0.1, 0.1);
        Self {
            w_f: Array2::random((hidden_size, k), dist),
            w_i: Array2::random((hidden_size, k), dist),
            w_c: Array2::random((hidden_size, k), dist),
            w_o: Array2::random((hidden_size, k), dist),
            b_f: Array1::zeros(hidden_size),
            b_i: Array1::zeros(hidden_size),
            b_c: Array1::zeros(hidden_size),
            b_o: Array1::zeros(hidden_size),
            input_size,
            hidden_size,
            sigmoid: Sigmoid,
            tanh: Tanh,
        }
    }

    /// 单步前向传播
    pub fn forward(&self, x_t: &Array1<f64>, h_prev: &Array1<f64>, c_prev: &Array1<f64>) -> (Array1<f64>, Array1<f64>, LstmCache) {
        let xh = concatenate![Axis(0), x_t.view(), h_prev.view()];
        let f_t = self.sigmoid.forward(&(self.w_f.dot(&xh) + &self.b_f));
        let i_t = self.sigmoid.forward(&(self.w_i.dot(&xh) + &self.b_i));
        let c_tilde = self.tanh.forward(&(self.w_c.dot(&xh) + &self.b_c));
        let c_t = &f_t * c_prev + &i_t * &c_tilde;
        let o_t = self.sigmoid.forward(&(self.w_o.dot(&xh) + &self.b_o));
        let h_t = &o_t * self.tanh.forward(&c_t);
        let cache = LstmCache {
            xh,
            f_t,
            i_t,
            o_t,
            c_tilde,
            c_prev: c_prev.clone(),
            c_t: c_t.clone(),
        };
        (h_t, c_t, cache)
    }

    /// 批处理前向传播
    pub fn forward_batch(&self, x_t: &Array2<f64>, h_prev: &Array2<f64>, c_prev: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let _batch_size = x_t.shape()[0];
        let xh = ndarray::concatenate(Axis(1), &[x_t.view(), h_prev.view()]).unwrap(); // [batch, input+hidden]
        let f_t = self.sigmoid.forward_batch(&(xh.dot(&self.w_f.t()) + &self.b_f));
        let i_t = self.sigmoid.forward_batch(&(xh.dot(&self.w_i.t()) + &self.b_i));
        let c_tilde = self.tanh.forward_batch(&(xh.dot(&self.w_c.t()) + &self.b_c));
        let c_t = &f_t * c_prev + &i_t * &c_tilde;
        let o_t = self.sigmoid.forward_batch(&(xh.dot(&self.w_o.t()) + &self.b_o));
        let h_t = &o_t * self.tanh.forward_batch(&c_t);
        (h_t, c_t)
    }

    /// 单步反向传播
    /// dh_next, dc_next: 来自后续时间步的梯度
    /// cache: 前向传播时的中间变量
    /// 返回：(参数梯度, dx, dh_prev, dc_prev)
    pub fn backward(
        &self,
        dh_next: &Array1<f64>,
        dc_next: &Array1<f64>,
        cache: &LstmCache,
    ) -> (LstmCellGradient, Array1<f64>, Array1<f64>, Array1<f64>) {
        let tanh_c_t = self.tanh.forward(&cache.c_t);
        let do_t = dh_next * &tanh_c_t;
        let dc_t = dh_next * &cache.o_t * (1.0 - tanh_c_t.mapv(|v| v.powi(2))) + dc_next;
        let df_t = &dc_t * &cache.c_prev;
        let di_t = &dc_t * &cache.c_tilde;
        let dc_tilde = &dc_t * &cache.i_t;

        // 门的激活函数导数
        let d_o = do_t * &cache.o_t * (1.0 - &cache.o_t);
        let d_f = df_t * &cache.f_t * (1.0 - &cache.f_t);
        let d_i = di_t * &cache.i_t * (1.0 - &cache.i_t);
        let d_c_tilde = dc_tilde * (1.0 - &cache.c_tilde.mapv(|v| v.powi(2)));

        // 拼接 [x_t, h_prev] 作为输入
        let xh = cache.xh.view(); // [input+hidden]

        // 参数梯度（外积）
        let mut grad = LstmCellGradient::zeros(self.input_size, self.hidden_size);
        grad.dw_o = d_o.view().to_owned().insert_axis(Axis(1)).dot(&xh.insert_axis(Axis(0)));
        grad.dw_f = d_f.view().to_owned().insert_axis(Axis(1)).dot(&xh.insert_axis(Axis(0)));
        grad.dw_i = d_i.view().to_owned().insert_axis(Axis(1)).dot(&xh.insert_axis(Axis(0)));
        grad.dw_c = d_c_tilde.view().to_owned().insert_axis(Axis(1)).dot(&xh.insert_axis(Axis(0)));
        grad.db_o = d_o.clone();
        grad.db_f = d_f.clone();
        grad.db_i = d_i.clone();
        grad.db_c = d_c_tilde.clone();

        // 反传到 xh
        let dxh = self.w_o.t().dot(&d_o)
            + self.w_f.t().dot(&d_f)
            + self.w_i.t().dot(&d_i)
            + self.w_c.t().dot(&d_c_tilde);
        let dx = dxh.slice(s![..self.input_size]).to_owned();
        let dh_prev = dxh.slice(s![self.input_size..]).to_owned();
        let dc_prev = &dc_t * &cache.f_t;

        (grad, dx, dh_prev, dc_prev)
    }

    /// 用梯度和学习率更新参数
    pub fn update(&mut self, grads: &LstmCellGradient, lr: f64) {
        self.w_f = &self.w_f - &(grads.dw_f.mapv(|v| lr * v));
        self.w_i = &self.w_i - &(grads.dw_i.mapv(|v| lr * v));
        self.w_c = &self.w_c - &(grads.dw_c.mapv(|v| lr * v));
        self.w_o = &self.w_o - &(grads.dw_o.mapv(|v| lr * v));
        self.b_f = &self.b_f - &(grads.db_f.mapv(|v| lr * v));
        self.b_i = &self.b_i - &(grads.db_i.mapv(|v| lr * v));
        self.b_c = &self.b_c - &(grads.db_c.mapv(|v| lr * v));
        self.b_o = &self.b_o - &(grads.db_o.mapv(|v| lr * v));
    }
}

impl LstmCellGradient {
    pub fn zeros(input_size: usize, hidden_size: usize) -> Self {
        let k = input_size + hidden_size;
        Self {
            dw_f: Array2::zeros((hidden_size, k)),
            dw_i: Array2::zeros((hidden_size, k)),
            dw_c: Array2::zeros((hidden_size, k)),
            dw_o: Array2::zeros((hidden_size, k)),
            db_f: Array1::zeros(hidden_size),
            db_i: Array1::zeros(hidden_size),
            db_c: Array1::zeros(hidden_size),
            db_o: Array1::zeros(hidden_size),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_lstm_cell_forward() {
        let input_size = 4;
        let hidden_size = 3;
        let cell = LstmCell::new(input_size, hidden_size);
        let x_t = arr1(&[0.1, 0.2, 0.3, 0.4]);
        let h_prev = arr1(&[0.0, 0.0, 0.0]);
        let c_prev = arr1(&[0.0, 0.0, 0.0]);
        let (h_t, c_t, _cache) = cell.forward(&x_t, &h_prev, &c_prev);
        assert_eq!(h_t.len(), hidden_size);
        assert_eq!(c_t.len(), hidden_size);
        for &v in h_t.iter() {
            assert!(v > -1.0 && v < 1.0);
        }
    }
} 