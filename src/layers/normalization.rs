use candle_core::{Tensor, Result};
// PERBAIKAN: Tambahkan 'ModuleT' di sini
use candle_nn::{Module, ModuleT, VarBuilder}; 

// 1. RMS Norm
pub struct StrictRmsNorm {
    inner: candle_nn::RmsNorm,
}

impl StrictRmsNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let inner = candle_nn::rms_norm(size, eps, vb)?;
        Ok(Self { inner })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

// 2. Layer Norm
pub struct StrictLayerNorm {
    inner: candle_nn::LayerNorm,
}

impl StrictLayerNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let inner = candle_nn::layer_norm(size, eps, vb)?;
        Ok(Self { inner })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

// 3. Batch Norm
pub struct StrictBatchNorm {
    inner: candle_nn::BatchNorm,
}

impl StrictBatchNorm {
    pub fn new(num_features: usize, vb: VarBuilder) -> Result<Self> {
        let inner = candle_nn::batch_norm(num_features, 1e-5, vb)?;
        Ok(Self { inner })
    }

    // Sekarang .forward_t() akan dikenali karena ModuleT sudah di-import
    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        self.inner.forward_t(x, train)
    }
}