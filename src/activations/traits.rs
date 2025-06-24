use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;

/// 定义激活函数的通用接口。
pub trait Activation {
    /// 对输入执行前向传播。
    fn forward(&self, x: &Array1<f64>) -> Array1<f64>;

    /// 计算激活函数关于其输入的导数。
    fn derivative(&self, x: &Array1<f64>) -> Array1<f64>;

    /// 批处理前向传播，默认对每一行调用 forward。
    fn forward_batch(&self, x: &Array2<f64>) -> Array2<f64> {
        let rows: Vec<Array1<f64>> = x.rows().into_iter().map(|row| self.forward(&row.to_owned())).collect();
        ndarray::stack(Axis(0), &rows.iter().map(|r| r.view()).collect::<Vec<_>>()).unwrap()
    }

    /// 批处理导数，默认对每一行调用 derivative。
    fn derivative_batch(&self, x: &Array2<f64>) -> Array2<f64> {
        let rows: Vec<Array1<f64>> = x.rows().into_iter().map(|row| self.derivative(&row.to_owned())).collect();
        ndarray::stack(Axis(0), &rows.iter().map(|r| r.view()).collect::<Vec<_>>()).unwrap()
    }
} 