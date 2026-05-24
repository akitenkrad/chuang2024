//! 評価指標 (論文 4.3 §6)．
//!
//! 意見分布 `F_o^t` の特徴量を計算する．中心は **Bias B** (意見平均) と
//! **Diversity D** (意見分布の標準偏差)，および分散・クラスタ数・分極指標．
//! `framing_asymmetry` は true/false フレーミング間の比較なので Phase 3 (reproduce)
//! に委ね，ここでは単一実行ごとの指標を扱う．

use serde::Serialize;

use crate::world::{OPINION_MAX, OPINION_MIN};

/// 意見平均 `B = mean(F_o^t)` (Bias)．
pub fn bias(opinions: &[i8]) -> f64 {
    if opinions.is_empty() {
        return 0.0;
    }
    opinions.iter().map(|&o| o as f64).sum::<f64>() / opinions.len() as f64
}

/// 意見の分散 (母分散)．
pub fn variance(opinions: &[i8]) -> f64 {
    if opinions.is_empty() {
        return 0.0;
    }
    let m = bias(opinions);
    opinions
        .iter()
        .map(|&o| (o as f64 - m).powi(2))
        .sum::<f64>()
        / opinions.len() as f64
}

/// 意見分布の標準偏差 `D = std(F_o^t)` (Diversity)．
pub fn diversity(opinions: &[i8]) -> f64 {
    variance(opinions).sqrt()
}

/// クラスタ数 (相異なる意見値の個数)．
///
/// 意見は離散 5 段階なので，占有された意見ビンの個数をそのまま数える．
pub fn n_clusters(opinions: &[i8]) -> usize {
    let mut seen = [false; 5]; // -2..=2 を 0..=4 にマップ
    for &o in opinions {
        let idx = (o.clamp(OPINION_MIN, OPINION_MAX) - OPINION_MIN) as usize;
        seen[idx] = true;
    }
    seen.iter().filter(|&&b| b).count()
}

/// 分極指標 (polarization index)．
///
/// 意見の絶対値平均を意見空間の半径 (=2) で正規化したもの ∈ `[0,1]`．両極
/// (-2 / +2) に質量が集まるほど 1 に近づき，中央 (0) に集まるほど 0 に近づく．
/// 論文の「最頻 2 ピーク間の質量分離度」の簡便な代理指標として用いる．
pub fn polarization(opinions: &[i8]) -> f64 {
    if opinions.is_empty() {
        return 0.0;
    }
    let radius = OPINION_MAX as f64; // = 2.0
    let mean_abs = opinions.iter().map(|&o| (o as f64).abs()).sum::<f64>() / opinions.len() as f64;
    mean_abs / radius
}

/// 1 ステップ分のメトリクス (metrics.csv の 1 行)．
#[derive(Debug, Clone, Serialize)]
pub struct Metrics {
    /// ステップ番号 t．
    pub t: usize,
    /// 意見の分散．
    pub variance: f64,
    /// Bias B (意見平均)．
    pub bias: f64,
    /// Diversity D (意見分布の標準偏差)．
    pub diversity: f64,
    /// クラスタ数 (相異なる意見値の個数)．
    pub n_clusters: usize,
    /// 分極指標 ∈ [0,1]．
    pub polarization: f64,
}

impl Metrics {
    /// 意見ベクトルからメトリクスを計算する．
    pub fn compute(opinions: &[i8], t: usize) -> Self {
        Metrics {
            t,
            variance: variance(opinions),
            bias: bias(opinions),
            diversity: diversity(opinions),
            n_clusters: n_clusters(opinions),
            polarization: polarization(opinions),
        }
    }
}

/// 収束時刻の推定: 意見分散が `tol` 以下になった最初のステップ．
///
/// `variances` は各ステップの分散列 (t 昇順を仮定)．しきい値以下に初めて達した
/// インデックス (= 収束時刻 t) を返す．達しなければ `None`．
pub fn convergence_time(variances: &[f64], tol: f64) -> Option<usize> {
    variances.iter().position(|&v| v < tol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bias_and_diversity_of_consensus() {
        let v = vec![2, 2, 2, 2];
        assert!((bias(&v) - 2.0).abs() < 1e-12);
        assert!(diversity(&v) < 1e-12);
        assert_eq!(n_clusters(&v), 1);
    }

    #[test]
    fn diversity_increases_with_spread() {
        let tight = vec![0, 0, 1, -1];
        let wide = vec![-2, -2, 2, 2];
        assert!(diversity(&wide) > diversity(&tight));
        assert_eq!(n_clusters(&wide), 2);
    }

    #[test]
    fn polarization_extremes() {
        assert!((polarization(&[0, 0, 0]) - 0.0).abs() < 1e-12);
        assert!((polarization(&[-2, 2]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn convergence_time_finds_first_below_tol() {
        let vs = vec![1.0, 0.5, 1e-9, 0.0];
        assert_eq!(convergence_time(&vs, 1e-6), Some(2));
        assert_eq!(convergence_time(&[1.0, 0.5], 1e-6), None);
    }
}
