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
} 