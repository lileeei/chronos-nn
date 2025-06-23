use ndarray::Array1;

/// 计算均方误差 (MSE) 损失。
///
/// MSE 定义为 `(1/n) * Σ(predictions - targets)²`。
pub fn mean_squared_error(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    let diff = predictions - targets;
    let squared_diff = diff.mapv(|v| v.powi(2));
    squared_diff.mean().unwrap_or(0.0)
}

/// 计算均方误差损失的导数。
///
/// 导数定义为 `(2/n) * (predictions - targets)`。
pub fn mean_squared_error_derivative(
    predictions: &Array1<f64>,
    targets: &Array1<f64>,
) -> Array1<f64> {
    let n = predictions.len() as f64;
    (predictions - targets) * (2.0 / n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_mean_squared_error() {
        let predictions = arr1(&[1.0, 2.0, 3.0]);
        let targets = arr1(&[2.0, 3.0, 4.0]);
        let loss = mean_squared_error(&predictions, &targets);
        assert!((loss - 1.0).abs() < 1e-6);

        let predictions_same = arr1(&[1.0, 2.0, 3.0]);
        let targets_same = arr1(&[1.0, 2.0, 3.0]);
        let loss_zero = mean_squared_error(&predictions_same, &targets_same);
        assert!(loss_zero.abs() < 1e-6);
    }

    #[test]
    fn test_mean_squared_error_derivative() {
        let predictions = arr1(&[1.0, 2.0, 3.0]);
        let targets = arr1(&[2.0, 3.0, 4.0]);
        let derivative = mean_squared_error_derivative(&predictions, &targets);
        let expected = arr1(&[-2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0]);
        derivative
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 1e-6));
    }
} 