use super::traits::Activation;
use ndarray::Array1;

/// 对数组中的每个元素应用 sigmoid 函数。
///
/// Sigmoid 定义为 `1 / (1 + exp(-x))`。
pub fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

/// 计算 sigmoid 函数的导数。
///
/// 导数定义为 `sigmoid(x) * (1 - sigmoid(x))`。
pub fn sigmoid_derivative(x: &Array1<f64>) -> Array1<f64> {
    let s = sigmoid(x);
    s.mapv(|v| v * (1.0 - v))
}

/// 对数组中的每个元素应用双曲正切 (tanh) 函数。
pub fn tanh(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.tanh())
}

/// 计算 tanh 函数的导数。
///
/// 导数定义为 `1 - tanh(x)^2`。
pub fn tanh_derivative(x: &Array1<f64>) -> Array1<f64> {
    let t = tanh(x);
    t.mapv(|v| 1.0 - v.powi(2))
}

/// Tanh 激活函数的结构体实现。
#[derive(Clone)]
pub struct Tanh;

impl Activation for Tanh {
    fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        tanh(x)
    }

    fn derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        tanh_derivative(x)
    }
}

/// Sigmoid 激活函数的结构体实现。
#[derive(Clone)]
pub struct Sigmoid;

impl Activation for Sigmoid {
    fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        sigmoid(x)
    }

    fn derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        sigmoid_derivative(x)
    }
}

/// 对数组中的每个元素应用 ReLU 函数。
///
/// ReLU 定义为 `max(0, x)`。
pub fn relu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.max(0.0))
}

/// 计算 ReLU 函数的导数。
///
/// 导数定义为 `1 if x > 0, else 0`。
pub fn relu_derivative(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
}

/// ReLU 激活函数的结构体实现。
#[derive(Clone)]
pub struct ReLU;

impl Activation for ReLU {
    fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        relu(x)
    }

    fn derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        relu_derivative(x)
    }
}

/// 对数组中的每个元素应用 Leaky ReLU 函数。
///
/// Leaky ReLU 定义为 `max(alpha * x, x)`，其中 alpha 是一个小的正数。
pub fn leaky_relu(x: &Array1<f64>, alpha: f64) -> Array1<f64> {
    x.mapv(|v| if v > 0.0 { v } else { alpha * v })
}

/// 计算 Leaky ReLU 函数的导数。
///
/// 导数定义为 `1 if x > 0, else alpha`。
pub fn leaky_relu_derivative(x: &Array1<f64>, alpha: f64) -> Array1<f64> {
    x.mapv(|v| if v > 0.0 { 1.0 } else { alpha })
}

/// Leaky ReLU 激活函数的结构体实现。
#[derive(Clone)]
pub struct LeakyReLU {
    pub alpha: f64,
}

impl LeakyReLU {
    /// 创建一个新的 LeakyReLU 激活函数。
    ///
    /// # Arguments
    ///
    /// * `alpha` - 负值的斜率，通常是一个小的正数（如 0.01）。
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Activation for LeakyReLU {
    fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        leaky_relu(x, self.alpha)
    }

    fn derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        leaky_relu_derivative(x, self.alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_sigmoid() {
        let x = arr1(&[0.0, 1.0, -1.0]);
        let expected = arr1(&[0.5, 0.73105858, 0.26894142]);
        let result = sigmoid(&x);
        result
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 1e-6));
    }

    #[test]
    fn test_sigmoid_derivative() {
        let x = arr1(&[0.0, 1.0, -1.0]);
        // sigmoid(0.0) * (1 - sigmoid(0.0)) = 0.5 * 0.5 = 0.25
        // sigmoid(1.0) * (1 - sigmoid(1.0)) ≈ 0.731 * 0.269 = 0.1966
        // sigmoid(-1.0) * (1 - sigmoid(-1.0)) ≈ 0.269 * 0.731 = 0.1966
        let expected = arr1(&[0.25, 0.19661193, 0.19661193]);
        let result = sigmoid_derivative(&x);
        result
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 1e-6));
    }

    #[test]
    fn test_tanh() {
        let x = arr1(&[0.0, 1.0, -1.0]);
        let expected = arr1(&[0.0, 0.76159416, -0.76159416]);
        let result = tanh(&x);
        result
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 1e-6));
    }

    #[test]
    fn test_tanh_derivative() {
        let x = arr1(&[0.0, 1.0, -1.0]);
        // 1 - tanh(0)^2 = 1
        // 1 - tanh(1)^2 ≈ 1 - 0.76159^2 = 0.41997
        // 1 - tanh(-1)^2 ≈ 1 - (-0.76159)^2 = 0.41997
        let expected = arr1(&[1.0, 0.41997434, 0.41997434]);
        let result = tanh_derivative(&x);
        result
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 1e-6));
    }

    #[test]
    fn test_relu() {
        let x = arr1(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let expected = arr1(&[0.0, 0.0, 0.0, 1.0, 2.0]);
        let result = relu(&x);
        result
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 1e-6));
    }

    #[test]
    fn test_relu_derivative() {
        let x = arr1(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let expected = arr1(&[0.0, 0.0, 0.0, 1.0, 1.0]);
        let result = relu_derivative(&x);
        result
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 1e-6));
    }

    #[test]
    fn test_leaky_relu() {
        let x = arr1(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let alpha = 0.01;
        let expected = arr1(&[-0.02, -0.01, 0.0, 1.0, 2.0]);
        let result = leaky_relu(&x, alpha);
        result
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 1e-6));
    }

    #[test]
    fn test_leaky_relu_derivative() {
        let x = arr1(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let alpha = 0.01;
        let expected = arr1(&[0.01, 0.01, 0.01, 1.0, 1.0]);
        let result = leaky_relu_derivative(&x, alpha);
        result
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 1e-6));
    }

    #[test]
    fn test_relu_struct() {
        let relu_fn = ReLU;
        let x = arr1(&[-1.0, 0.0, 1.0]);
        let expected = arr1(&[0.0, 0.0, 1.0]);
        let result = relu_fn.forward(&x);
        result
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 1e-6));
    }

    #[test]
    fn test_leaky_relu_struct() {
        let leaky_relu_fn = LeakyReLU::new(0.1);
        let x = arr1(&[-1.0, 0.0, 1.0]);
        let expected = arr1(&[-0.1, 0.0, 1.0]);
        let result = leaky_relu_fn.forward(&x);
        result
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 1e-6));
    }
}