use ndarray::Array1;

/// 定义激活函数的通用接口。
pub trait Activation {
    /// 对输入执行前向传播。
    fn forward(&self, x: &Array1<f64>) -> Array1<f64>;

    /// 计算激活函数关于其输入的导数。
    fn derivative(&self, x: &Array1<f64>) -> Array1<f64>;
} 