use ndarray::{Array1, Array2};

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

/// 批处理均方误差 (MSE) 损失。
/// 输入 predictions/targets: [batch_size, dim]
/// 返回 batch loss 均值
pub fn mean_squared_error_batch(predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
    let diff = predictions - targets;
    let squared_diff = diff.mapv(|v| v.powi(2));
    squared_diff.mean().unwrap_or(0.0)
}

/// 批处理均方误差损失的导数。
/// 返回每个样本的导数 [batch_size, dim]
pub fn mean_squared_error_derivative_batch(predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
    let n = predictions.shape()[1] as f64;
    (predictions - targets) * (2.0 / n)
}

/// 计算二元交叉熵损失。
///
/// 二元交叉熵定义为 `-[y*log(p) + (1-y)*log(1-p)]`，其中 p 是预测概率，y 是真实标签。
/// 为了数值稳定性，我们对概率进行裁剪。
pub fn binary_cross_entropy(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    let eps = 1e-15; // 防止log(0)
    let clipped_preds = predictions.mapv(|p| p.max(eps).min(1.0 - eps));
    let loss = targets * clipped_preds.mapv(|p| p.ln()) +
               (1.0 - targets) * clipped_preds.mapv(|p| (1.0 - p).ln());
    -loss.mean().unwrap_or(0.0)
}

/// 计算二元交叉熵损失的导数。
///
/// 导数定义为 `(predictions - targets) / (predictions * (1 - predictions))`。
pub fn binary_cross_entropy_derivative(predictions: &Array1<f64>, targets: &Array1<f64>) -> Array1<f64> {
    let eps = 1e-15;
    let clipped_preds = predictions.mapv(|p| p.max(eps).min(1.0 - eps));
    let n = predictions.len() as f64;
    ((&clipped_preds - targets) / (&clipped_preds * (1.0 - &clipped_preds))) / n
}

/// 批处理二元交叉熵损失。
pub fn binary_cross_entropy_batch(predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
    let eps = 1e-15;
    let clipped_preds = predictions.mapv(|p| p.max(eps).min(1.0 - eps));
    let loss = targets * clipped_preds.mapv(|p| p.ln()) +
               (1.0 - targets) * clipped_preds.mapv(|p| (1.0 - p).ln());
    -loss.mean().unwrap_or(0.0)
}

/// 批处理二元交叉熵损失的导数。
pub fn binary_cross_entropy_derivative_batch(predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
    let eps = 1e-15;
    let clipped_preds = predictions.mapv(|p| p.max(eps).min(1.0 - eps));
    let n = predictions.shape()[0] as f64 * predictions.shape()[1] as f64;
    ((&clipped_preds - targets) / (&clipped_preds * (1.0 - &clipped_preds))) / n
}

/// 计算 Huber 损失（也称为平滑平均绝对误差）。
///
/// Huber 损失对异常值比 MSE 更鲁棒。当误差小于 delta 时使用平方损失，
/// 当误差大于 delta 时使用线性损失。
pub fn huber_loss(predictions: &Array1<f64>, targets: &Array1<f64>, delta: f64) -> f64 {
    let diff = predictions - targets;
    let abs_diff = diff.mapv(|d| d.abs());
    let loss = abs_diff.mapv(|ad| {
        if ad <= delta {
            0.5 * ad.powi(2)
        } else {
            delta * (ad - 0.5 * delta)
        }
    });
    loss.mean().unwrap_or(0.0)
}

/// 计算 Huber 损失的导数。
pub fn huber_loss_derivative(predictions: &Array1<f64>, targets: &Array1<f64>, delta: f64) -> Array1<f64> {
    let diff = predictions - targets;
    let n = predictions.len() as f64;
    diff.mapv(|d| {
        if d.abs() <= delta {
            d / n
        } else {
            delta * d.signum() / n
        }
    })
}

/// 批处理 Huber 损失。
pub fn huber_loss_batch(predictions: &Array2<f64>, targets: &Array2<f64>, delta: f64) -> f64 {
    let diff = predictions - targets;
    let abs_diff = diff.mapv(|d| d.abs());
    let loss = abs_diff.mapv(|ad| {
        if ad <= delta {
            0.5 * ad.powi(2)
        } else {
            delta * (ad - 0.5 * delta)
        }
    });
    loss.mean().unwrap_or(0.0)
}

/// 批处理 Huber 损失的导数。
pub fn huber_loss_derivative_batch(predictions: &Array2<f64>, targets: &Array2<f64>, delta: f64) -> Array2<f64> {
    let diff = predictions - targets;
    let n = predictions.shape()[0] as f64 * predictions.shape()[1] as f64;
    diff.mapv(|d| {
        if d.abs() <= delta {
            d / n
        } else {
            delta * d.signum() / n
        }
    })
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

    #[test]
    fn test_binary_cross_entropy() {
        let predictions = arr1(&[0.9, 0.1, 0.8]);
        let targets = arr1(&[1.0, 0.0, 1.0]);
        let loss = binary_cross_entropy(&predictions, &targets);

        // 手动计算期望值
        let expected = -(1.0 * 0.9_f64.ln() + 1.0 * 0.9_f64.ln() + 1.0 * 0.8_f64.ln()) / 3.0;
        assert!((loss - expected).abs() < 1e-6);
    }

    #[test]
    fn test_binary_cross_entropy_derivative() {
        let predictions = arr1(&[0.9, 0.1, 0.8]);
        let targets = arr1(&[1.0, 0.0, 1.0]);
        let derivative = binary_cross_entropy_derivative(&predictions, &targets);

        // 验证导数的形状和符号
        assert_eq!(derivative.len(), 3);
        // 当预测接近目标时，导数应该接近0
        assert!(derivative[0].abs() < 0.5); // 0.9 vs 1.0
        assert!(derivative[2].abs() < 0.5); // 0.8 vs 1.0
    }

    #[test]
    fn test_huber_loss() {
        let predictions = arr1(&[1.0, 2.0, 5.0]);
        let targets = arr1(&[1.5, 2.0, 1.0]);
        let delta = 1.0;
        let loss = huber_loss(&predictions, &targets, delta);

        // 手动计算：
        // |1.0 - 1.5| = 0.5 <= 1.0, 所以使用平方损失: 0.5 * 0.5^2 = 0.125
        // |2.0 - 2.0| = 0.0 <= 1.0, 所以使用平方损失: 0.5 * 0.0^2 = 0.0
        // |5.0 - 1.0| = 4.0 > 1.0, 所以使用线性损失: 1.0 * (4.0 - 0.5 * 1.0) = 3.5
        let expected = (0.125 + 0.0 + 3.5) / 3.0;
        assert!((loss - expected).abs() < 1e-6);
    }

    #[test]
    fn test_huber_loss_derivative() {
        let predictions = arr1(&[1.0, 2.0, 5.0]);
        let targets = arr1(&[1.5, 2.0, 1.0]);
        let delta = 1.0;
        let derivative = huber_loss_derivative(&predictions, &targets, delta);

        // 验证导数的形状
        assert_eq!(derivative.len(), 3);

        // 验证导数的符号和大小
        assert!(derivative[0] < 0.0); // 预测 < 目标
        assert!(derivative[1].abs() < 1e-6); // 预测 = 目标
        assert!(derivative[2] > 0.0); // 预测 > 目标
    }
}