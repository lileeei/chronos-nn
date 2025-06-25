use ndarray::{Array1, Array2, Array3, Axis};
use crate::activations::functions::tanh;

/// 注意力机制的通用trait
pub trait AttentionMechanism {
    /// 计算注意力权重和上下文向量
    ///
    /// # Arguments
    ///
    /// * `query` - 查询向量 [query_dim]
    /// * `keys` - 键矩阵 [seq_len, key_dim]
    /// * `values` - 值矩阵 [seq_len, value_dim]
    ///
    /// # Returns
    ///
    /// * `(context, attention_weights)` - 上下文向量和注意力权重
    fn forward(&self, query: &Array1<f64>, keys: &Array2<f64>, values: &Array2<f64>) -> (Array1<f64>, Array1<f64>);

    /// 批处理版本的注意力计算
    ///
    /// # Arguments
    ///
    /// * `queries` - 查询矩阵 [batch_size, query_dim]
    /// * `keys` - 键矩阵 [batch_size, seq_len, key_dim]
    /// * `values` - 值矩阵 [batch_size, seq_len, value_dim]
    ///
    /// # Returns
    ///
    /// * `(contexts, attention_weights)` - 上下文矩阵和注意力权重
    fn forward_batch(&self, queries: &Array2<f64>, keys: &Array3<f64>, values: &Array3<f64>) -> (Array2<f64>, Array2<f64>);
}

/// 加性注意力（Additive Attention / Bahdanau Attention）
///
/// 计算公式：
/// e_i = v^T * tanh(W_q * query + W_k * key_i + b)
/// α_i = softmax(e_i)
/// context = Σ(α_i * value_i)
pub struct AdditiveAttention {
    pub w_query: Array2<f64>,  // [hidden_dim, query_dim]
    pub w_key: Array2<f64>,    // [hidden_dim, key_dim]
    pub v: Array1<f64>,        // [hidden_dim]
    pub bias: Array1<f64>,     // [hidden_dim]
}

impl AdditiveAttention {
    /// 创建新的加性注意力层
    ///
    /// # Arguments
    ///
    /// * `query_dim` - 查询向量维度
    /// * `key_dim` - 键向量维度
    /// * `hidden_dim` - 隐藏层维度
    pub fn new(query_dim: usize, key_dim: usize, hidden_dim: usize) -> Self {
        use ndarray_rand::rand_distr::Uniform;
        use ndarray_rand::RandomExt;

        let dist = Uniform::new(-0.1, 0.1);
        
        Self {
            w_query: Array2::random((hidden_dim, query_dim), dist),
            w_key: Array2::random((hidden_dim, key_dim), dist),
            v: Array1::random(hidden_dim, dist),
            bias: Array1::zeros(hidden_dim),
        }
    }

    /// 计算注意力分数
    fn compute_scores(&self, query: &Array1<f64>, keys: &Array2<f64>) -> Array1<f64> {
        let seq_len = keys.shape()[0];
        let mut scores = Array1::zeros(seq_len);

        // 计算 W_q * query
        let query_proj = self.w_query.dot(query);

        for i in 0..seq_len {
            let key_i = keys.row(i);
            // 计算 W_k * key_i
            let key_proj = self.w_key.dot(&key_i);
            // 计算 tanh(W_q * query + W_k * key_i + b)
            let hidden = tanh(&(&query_proj + &key_proj + &self.bias));
            // 计算 v^T * hidden
            scores[i] = self.v.dot(&hidden);
        }

        scores
    }

    /// Softmax函数
    fn softmax(&self, x: &Array1<f64>) -> Array1<f64> {
        let max_val = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_x = x.mapv(|v| (v - max_val).exp());
        let sum_exp = exp_x.sum();
        exp_x / sum_exp
    }
}

impl AttentionMechanism for AdditiveAttention {
    fn forward(&self, query: &Array1<f64>, keys: &Array2<f64>, values: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
        // 计算注意力分数
        let scores = self.compute_scores(query, keys);
        
        // 应用softmax得到注意力权重
        let attention_weights = self.softmax(&scores);
        
        // 计算加权上下文向量
        let seq_len = values.shape()[0];
        let value_dim = values.shape()[1];
        let mut context = Array1::zeros(value_dim);
        
        for i in 0..seq_len {
            let value_i = values.row(i);
            context = context + attention_weights[i] * &value_i;
        }
        
        (context, attention_weights)
    }

    fn forward_batch(&self, queries: &Array2<f64>, keys: &Array3<f64>, values: &Array3<f64>) -> (Array2<f64>, Array2<f64>) {
        let batch_size = queries.shape()[0];
        let seq_len = keys.shape()[1];
        let value_dim = values.shape()[2];
        
        let mut contexts = Array2::zeros((batch_size, value_dim));
        let mut all_weights = Array2::zeros((batch_size, seq_len));
        
        for b in 0..batch_size {
            let query = queries.row(b).to_owned();
            let keys_b = keys.index_axis(Axis(0), b).to_owned();
            let values_b = values.index_axis(Axis(0), b).to_owned();
            
            let (context, weights) = self.forward(&query, &keys_b, &values_b);
            contexts.row_mut(b).assign(&context);
            all_weights.row_mut(b).assign(&weights);
        }
        
        (contexts, all_weights)
    }
}

/// 点积注意力（Dot-Product Attention）
///
/// 计算公式：
/// e_i = query^T * key_i
/// α_i = softmax(e_i)
/// context = Σ(α_i * value_i)
pub struct DotProductAttention;

impl AttentionMechanism for DotProductAttention {
    fn forward(&self, query: &Array1<f64>, keys: &Array2<f64>, values: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
        // 计算点积分数 - keys是[seq_len, key_dim]，query是[key_dim]
        let seq_len = keys.shape()[0];
        let mut scores = Array1::zeros(seq_len);

        for i in 0..seq_len {
            let key_i = keys.row(i);
            scores[i] = key_i.dot(query);
        }
        
        // 应用softmax
        let max_val = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_scores = scores.mapv(|v| (v - max_val).exp());
        let sum_exp = exp_scores.sum();
        let attention_weights = exp_scores / sum_exp;
        
        // 计算加权上下文向量
        let seq_len = values.shape()[0];
        let value_dim = values.shape()[1];
        let mut context = Array1::zeros(value_dim);
        
        for i in 0..seq_len {
            let value_i = values.row(i);
            context = context + attention_weights[i] * &value_i;
        }
        
        (context, attention_weights)
    }

    fn forward_batch(&self, queries: &Array2<f64>, keys: &Array3<f64>, values: &Array3<f64>) -> (Array2<f64>, Array2<f64>) {
        let batch_size = queries.shape()[0];
        let seq_len = keys.shape()[1];
        let value_dim = values.shape()[2];
        
        let mut contexts = Array2::zeros((batch_size, value_dim));
        let mut all_weights = Array2::zeros((batch_size, seq_len));
        
        for b in 0..batch_size {
            let query = queries.row(b).to_owned();
            let keys_b = keys.index_axis(Axis(0), b).to_owned();
            let values_b = values.index_axis(Axis(0), b).to_owned();
            
            let (context, weights) = self.forward(&query, &keys_b, &values_b);
            contexts.row_mut(b).assign(&context);
            all_weights.row_mut(b).assign(&weights);
        }
        
        (contexts, all_weights)
    }
}

/// 缩放点积注意力（Scaled Dot-Product Attention）
///
/// 计算公式：
/// e_i = (query^T * key_i) / sqrt(d_k)
/// α_i = softmax(e_i)
/// context = Σ(α_i * value_i)
pub struct ScaledDotProductAttention {
    pub scale_factor: f64,
}

impl ScaledDotProductAttention {
    /// 创建新的缩放点积注意力
    ///
    /// # Arguments
    ///
    /// * `key_dim` - 键向量的维度，用于计算缩放因子
    pub fn new(key_dim: usize) -> Self {
        Self {
            scale_factor: 1.0 / (key_dim as f64).sqrt(),
        }
    }
}

impl AttentionMechanism for ScaledDotProductAttention {
    fn forward(&self, query: &Array1<f64>, keys: &Array2<f64>, values: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
        // 计算缩放点积分数
        let seq_len = keys.shape()[0];
        let mut scores = Array1::zeros(seq_len);

        for i in 0..seq_len {
            let key_i = keys.row(i);
            scores[i] = key_i.dot(query) * self.scale_factor;
        }
        
        // 应用softmax
        let max_val = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_scores = scores.mapv(|v| (v - max_val).exp());
        let sum_exp = exp_scores.sum();
        let attention_weights = exp_scores / sum_exp;
        
        // 计算加权上下文向量
        let seq_len = values.shape()[0];
        let value_dim = values.shape()[1];
        let mut context = Array1::zeros(value_dim);
        
        for i in 0..seq_len {
            let value_i = values.row(i);
            context = context + attention_weights[i] * &value_i;
        }
        
        (context, attention_weights)
    }

    fn forward_batch(&self, queries: &Array2<f64>, keys: &Array3<f64>, values: &Array3<f64>) -> (Array2<f64>, Array2<f64>) {
        let batch_size = queries.shape()[0];
        let seq_len = keys.shape()[1];
        let value_dim = values.shape()[2];
        
        let mut contexts = Array2::zeros((batch_size, value_dim));
        let mut all_weights = Array2::zeros((batch_size, seq_len));
        
        for b in 0..batch_size {
            let query = queries.row(b).to_owned();
            let keys_b = keys.index_axis(Axis(0), b).to_owned();
            let values_b = values.index_axis(Axis(0), b).to_owned();
            
            let (context, weights) = self.forward(&query, &keys_b, &values_b);
            contexts.row_mut(b).assign(&context);
            all_weights.row_mut(b).assign(&weights);
        }
        
        (contexts, all_weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, Array3};

    #[test]
    fn test_additive_attention() {
        let attention = AdditiveAttention::new(4, 3, 5);

        let query = arr1(&[1.0, 0.5, -0.2, 0.8]);
        let keys = arr2(&[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]);
        let values = arr2(&[
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ]);

        let (context, weights) = attention.forward(&query, &keys, &values);

        // 验证输出维度
        assert_eq!(context.len(), 2);
        assert_eq!(weights.len(), 3);

        // 验证注意力权重和为1
        let weight_sum: f64 = weights.sum();
        assert!((weight_sum - 1.0).abs() < 1e-6);

        // 验证权重都是正数
        for &w in weights.iter() {
            assert!(w >= 0.0);
        }
    }

    #[test]
    fn test_dot_product_attention() {
        let attention = DotProductAttention;

        let query = arr1(&[1.0, 0.5, -0.2]);
        let keys = arr2(&[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]);
        let values = arr2(&[
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ]);

        let (context, weights) = attention.forward(&query, &keys, &values);

        // 验证输出维度
        assert_eq!(context.len(), 2);
        assert_eq!(weights.len(), 3);

        // 验证注意力权重和为1
        let weight_sum: f64 = weights.sum();
        assert!((weight_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_dot_product_attention() {
        let attention = ScaledDotProductAttention::new(3);

        let query = arr1(&[1.0, 0.5, -0.2]);
        let keys = arr2(&[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]);
        let values = arr2(&[
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ]);

        let (context, weights) = attention.forward(&query, &keys, &values);

        // 验证输出维度
        assert_eq!(context.len(), 2);
        assert_eq!(weights.len(), 3);

        // 验证注意力权重和为1
        let weight_sum: f64 = weights.sum();
        assert!((weight_sum - 1.0).abs() < 1e-6);

        // 验证缩放因子
        assert!((attention.scale_factor - 1.0 / (3.0_f64).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_attention_batch_processing() {
        let attention = DotProductAttention;

        let queries = arr2(&[
            [1.0, 0.5, -0.2],
            [0.8, -0.3, 0.6]
        ]);

        let keys = Array3::from_shape_vec(
            (2, 3, 3),
            vec![
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  // batch 0
                0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0   // batch 1
            ]
        ).unwrap();

        let values = Array3::from_shape_vec(
            (2, 3, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,  // batch 0
                2.0, 3.0, 4.0, 5.0, 6.0, 7.0   // batch 1
            ]
        ).unwrap();

        let (contexts, weights) = attention.forward_batch(&queries, &keys, &values);

        // 验证输出维度
        assert_eq!(contexts.shape(), &[2, 2]);  // [batch_size, value_dim]
        assert_eq!(weights.shape(), &[2, 3]);   // [batch_size, seq_len]

        // 验证每个batch的注意力权重和为1
        for b in 0..2 {
            let weight_sum: f64 = weights.row(b).sum();
            assert!((weight_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_attention_focus() {
        let attention = DotProductAttention;

        // 创建一个查询，它与第三个键最相似（更大的点积）
        let query = arr1(&[1.0, 1.0, 1.0]);
        let keys = arr2(&[
            [0.1, 0.2, 0.3],  // 点积 = 0.6
            [0.4, 0.5, 0.6],  // 点积 = 1.5
            [0.7, 0.8, 0.9]   // 点积 = 2.4 (最大)
        ]);
        let values = arr2(&[
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0]
        ]);

        let (context, weights) = attention.forward(&query, &keys, &values);

        // 第三个位置应该有最高的注意力权重
        let max_weight_idx = weights.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        assert_eq!(max_weight_idx, 2);

        // 上下文向量应该更接近第三个值向量
        assert!(context[0] > 2.5);  // 更接近3.0
        assert!(context[1] > 2.5);
    }
}
