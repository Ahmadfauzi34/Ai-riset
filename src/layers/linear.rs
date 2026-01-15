use candle_core::{Tensor, Result};
use candle_nn::{Module, VarBuilder};

// 1. Linear / Dense Layer (Stateless, cuma butuh x)
pub struct StrictLinear {
    inner: candle_nn::Linear,
}

impl StrictLinear {
    pub fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let inner = candle_nn::linear(in_dim, out_dim, vb)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

// 2. Dropout (Stateful/Mode-aware, butuh x dan train bool)
pub struct StrictDropout {
    inner: candle_nn::Dropout,
}

impl StrictDropout {
    pub fn new(prob: f32) -> Self {
        let inner = candle_nn::Dropout::new(prob);
        Self { inner }
    }

    // PERBAIKAN DI SINI:
    // Tambahkan parameter 'train: bool'
    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        self.inner.forward(x, train)
    }
}