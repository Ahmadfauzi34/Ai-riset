use candle_core::{Tensor, Result};
use candle_nn::{VarBuilder, LSTMConfig, GRUConfig, RNN}; 

// 1. LSTM
pub struct StrictLSTM {
    inner: candle_nn::LSTM,
}

impl StrictLSTM {
    pub fn new(in_dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = LSTMConfig::default();
        let inner = candle_nn::lstm(in_dim, hidden_dim, cfg, vb)?;
        Ok(Self { inner })
    }
    
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // 1. Dapatkan sequence of states
        let states = self.inner.seq(xs)?;
        
        // 2. Ekstrak hidden state (.h) dari setiap langkah waktu
        // Kita map setiap state menjadi referensi ke tensor h-nya
        let output_tensors: Vec<&Tensor> = states.iter().map(|s| &s.h).collect();
        
        // 3. Tumpuk menjadi satu tensor utuh
        let output = Tensor::stack(&output_tensors, 0)?;
        
        Ok(output)
    }
}

// 2. GRU
pub struct StrictGRU {
    inner: candle_nn::GRU,
}

impl StrictGRU {
    pub fn new(in_dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = GRUConfig::default();
        let inner = candle_nn::gru(in_dim, hidden_dim, cfg, vb)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // 1. Dapatkan sequence of states
        let states = self.inner.seq(xs)?;
        
        // 2. Ekstrak hidden state (.h)
        // GRUState juga punya field .h yang berisi output tensor
        let output_tensors: Vec<&Tensor> = states.iter().map(|s| &s.h).collect();
        
        // 3. Tumpuk
        let output = Tensor::stack(&output_tensors, 0)?;
        
        Ok(output)
    }
}