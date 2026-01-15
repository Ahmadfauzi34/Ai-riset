use candle_core::{Tensor, Result};
use candle_nn::{Module, VarBuilder, Conv1dConfig, Conv2dConfig};

// 1. Conv1d (Biasanya untuk Audio atau Teks)
pub struct StrictConv1d {
    inner: candle_nn::Conv1d,
}

impl StrictConv1d {
    pub fn new(in_c: usize, out_c: usize, kernel: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv1dConfig::default();
        let inner = candle_nn::conv1d(in_c, out_c, kernel, cfg, vb)?;
        Ok(Self { inner })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

// 2. Conv2d (Biasanya untuk Gambar)
pub struct StrictConv2d {
    inner: candle_nn::Conv2d,
}

impl StrictConv2d {
    pub fn new(in_c: usize, out_c: usize, kernel: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig::default();
        let inner = candle_nn::conv2d(in_c, out_c, kernel, cfg, vb)?;
        Ok(Self { inner })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}