use wasm_bindgen::prelude::*;
// PERBAIKAN 1: Hapus 'Result' dari sini biar tidak bingung
use candle_core::Tensor; 
use candle_nn::{Module, VarBuilder};
use crate::WasmTensor; 

#[wasm_bindgen]
pub struct StrictLinear {
    inner: candle_nn::Linear,
}

#[wasm_bindgen]
impl StrictLinear {
    #[wasm_bindgen(constructor)]
    // PERBAIKAN 2: Gunakan 'std::result::Result' secara lengkap
    // Ini memberi tahu Rust: "Pakai Result standar, Sukses=StrictLinear, Gagal=JsError"
    pub fn new_random(in_dim: usize, out_dim: usize) -> std::result::Result<StrictLinear, JsError> {
        let device = candle_core::Device::Cpu;
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        
        // Di sini kita map error Candle ke JsError
        let inner = candle_nn::linear(in_dim, out_dim, vb)
            .map_err(|e| JsError::new(&e.to_string()))?;
        
        Ok(StrictLinear { inner })
    }
    
    // PERBAIKAN 3: Gunakan 'std::result::Result' di sini juga
    pub fn forward(&self, input: &WasmTensor) -> std::result::Result<WasmTensor, JsError> {
        let out = self.inner.forward(&input.inner)
            .map_err(|e| JsError::new(&e.to_string()))?;
            
        Ok(WasmTensor { inner: out })
    }
}
