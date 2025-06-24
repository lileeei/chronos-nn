use ndarray::{Array1, Array2, Axis, concatenate};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use crate::activations::functions::{Sigmoid, Tanh};
use ndarray::s;
use crate::activations::traits::Activation;

/// GRU单元的中间缓存，用于反向传播
pub struct GruCache {
    pub xh: Array1<f64>, // 拼接的[x_t, h_prev]
    pub r_t: Array1<f64>,
    pub z_t: Array1<f64>,
    pub h_tilde: Array1<f64>,
    pub h_prev: Array1<f64>,
}

/// GRU单元参数的梯度
pub struct GruCellGradient {
    pub dw_r: Array2<f64>,
    pub dw_z: Array2<f64>,
    pub dw_h: Array2<f64>,
    pub db_r: Array1<f64>,
    pub db_z: Array1<f64>,
    pub db_h: Array1<f64>,
}

/// GRU单元（支持单步和批处理前向传播）
pub struct GruCell {
    pub w_r: Array2<f64>, // [hidden_size, input_size + hidden_size]
    pub w_z: Array2<f64>,
    pub w_h: Array2<f64>,
    pub b_r: Array1<f64>, // [hidden_size]
    pub b_z: Array1<f64>,
    pub b_h: Array1<f64>,
    pub input_size: usize,
    pub hidden_size: usize,
    pub sigmoid: Sigmoid,
    pub tanh: Tanh,
}

impl GruCell {
    /// 创建并初始化GRUCell
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let k = input_size + hidden_size;
        let dist = Uniform::new(-0.1, 0.1);
        Self {
            w_r: Array2::random((hidden_size, k), dist),
            w_z: Array2::random((hidden_size, k), dist),
            w_h: Array2::random((hidden_size, k), dist),
            b_r: Array1::zeros(hidden_size),
            b_z: Array1::zeros(hidden_size),
            b_h: Array1::zeros(hidden_size),
            input_size,
            hidden_size,
            sigmoid: Sigmoid,
            tanh: Tanh,
        }
    }

    /// 单步前向传播
    pub fn forward(&self, x_t: &Array1<f64>, h_prev: &Array1<f64>) -> (Array1<f64>, GruCache) {
        let xh = concatenate![Axis(0), x_t.view(), h_prev.view()];
        let r_t = self.sigmoid.forward(&(self.w_r.dot(&xh) + &self.b_r));
        let z_t = self.sigmoid.forward(&(self.w_z.dot(&xh) + &self.b_z));
        let xh_r = concatenate![Axis(0), x_t.view(), (&r_t * h_prev).view()];
        let h_tilde = self.tanh.forward(&(self.w_h.dot(&xh_r) + &self.b_h));
        let h_t = (1.0 - &z_t) * h_prev + &z_t * &h_tilde;
        let cache = GruCache { xh, r_t, z_t, h_tilde, h_prev: h_prev.clone() };
        (h_t, cache)
    }

    /// 批处理前向传播
    pub fn forward_batch(&self, x_t: &Array2<f64>, h_prev: &Array2<f64>) -> (Array2<f64>, Vec<GruCache>) {
        let batch_size = x_t.shape()[0];
        let mut h_t_out = Array2::<f64>::zeros((batch_size, self.hidden_size));
        let mut caches = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let x = x_t.row(b).to_owned();
            let h = h_prev.row(b).to_owned();
            let (h_t, cache) = self.forward(&x, &h);
            h_t_out.row_mut(b).assign(&h_t);
            caches.push(cache);
        }
        (h_t_out, caches)
    }

    /// 单步反向传播
    /// dh_next: 来自后续时间步的隐藏状态梯度
    /// cache: 前向传播时的中间变量
    /// 返回：(参数梯度, dx, dh_prev)
    pub fn backward(
        &self,
        dh_next: &Array1<f64>,
        cache: &GruCache,
    ) -> (GruCellGradient, Array1<f64>, Array1<f64>) {
        let h_prev = &cache.h_prev;
        let r_t = &cache.r_t;
        let z_t = &cache.z_t;
        let h_tilde = &cache.h_tilde;
        let xh = &cache.xh;
        // h_t = (1-z_t)*h_prev + z_t*h_tilde
        let dh_tilde = dh_next * z_t;
        let dz_t = dh_next * (h_tilde - h_prev);
        let dh_prev = dh_next * (1.0 - z_t);
        // h_tilde = tanh(W_h [x_t, r_t*h_prev] + b_h)
        let dtanh = (1.0 - h_tilde.mapv(|v| v.powi(2))) * dh_tilde;
        // xh_r = [x_t, r_t*h_prev]
        let x_t = xh.slice(s![..self.input_size]).to_owned();
        let h_prev_r = r_t * h_prev;
        let xh_r = concatenate![Axis(0), x_t.view(), h_prev_r.view()];
        // W_h: [hidden, input+hidden]
        let dw_h = dtanh.view().to_owned().insert_axis(Axis(1)).dot(&xh_r.insert_axis(Axis(0)));
        let db_h = dtanh.clone();
        // r_t = sigmoid(W_r [x_t, h_prev] + b_r)
        let dr_t = (self.w_h.slice(s![.., self.input_size..]).t().dot(&dtanh)) * h_prev;
        let dr_t_for_dsigmoid = dr_t.clone();
        let dr_t_for_dhprev = dr_t;
        let dsigmoid_r = dr_t_for_dsigmoid * r_t * (1.0 - r_t);
        let dw_r = dsigmoid_r.view().to_owned().insert_axis(Axis(1)).dot(&xh.clone().insert_axis(Axis(0)));
        let db_r = dsigmoid_r.clone();
        // z_t = sigmoid(W_z [x_t, h_prev] + b_z)
        let dsigmoid_z = dz_t * z_t * (1.0 - z_t);
        let dw_z = dsigmoid_z.view().to_owned().insert_axis(Axis(1)).dot(&xh.clone().insert_axis(Axis(0)));
        let db_z = dsigmoid_z.clone();
        // 反传到xh
        let dxh_z = self.w_z.t().dot(&dsigmoid_z);
        let dxh_r = self.w_r.t().dot(&dsigmoid_r);
        let dxh_h = self.w_h.t().dot(&dtanh);
        let dxh = dxh_z + dxh_r + dxh_h;
        let dx = dxh.slice(s![..self.input_size]).to_owned();
        let dh_prev_total = dxh.slice(s![self.input_size..]).to_owned() + dh_prev + (dr_t_for_dhprev * r_t);
        let mut grad = GruCellGradient::zeros(self.input_size, self.hidden_size);
        grad.dw_h = dw_h;
        grad.db_h = db_h;
        grad.dw_r = dw_r;
        grad.db_r = db_r;
        grad.dw_z = dw_z;
        grad.db_z = db_z;
        (grad, dx, dh_prev_total)
    }

    /// 用梯度和学习率更新参数
    pub fn update(&mut self, grads: &GruCellGradient, lr: f64) {
        self.w_r = &self.w_r - &(grads.dw_r.mapv(|v| lr * v));
        self.w_z = &self.w_z - &(grads.dw_z.mapv(|v| lr * v));
        self.w_h = &self.w_h - &(grads.dw_h.mapv(|v| lr * v));
        self.b_r = &self.b_r - &(grads.db_r.mapv(|v| lr * v));
        self.b_z = &self.b_z - &(grads.db_z.mapv(|v| lr * v));
        self.b_h = &self.b_h - &(grads.db_h.mapv(|v| lr * v));
    }
}

impl GruCellGradient {
    pub fn zeros(input_size: usize, hidden_size: usize) -> Self {
        let k = input_size + hidden_size;
        Self {
            dw_r: Array2::zeros((hidden_size, k)),
            dw_z: Array2::zeros((hidden_size, k)),
            dw_h: Array2::zeros((hidden_size, k)),
            db_r: Array1::zeros(hidden_size),
            db_z: Array1::zeros(hidden_size),
            db_h: Array1::zeros(hidden_size),
        }
    }
} 