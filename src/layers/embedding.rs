use candle_core::{Tensor, Result};
use candle_nn::{Module, VarBuilder};

pub struct StrictEmbedding {
    inner: candle_nn::Embedding,
}

impl StrictEmbedding {
    pub fn new(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let inner = candle_nn::embedding(vocab_size, hidden_size, vb)?;
        Ok(Self { inner })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}