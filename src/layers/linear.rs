use wasm_bindgen::prelude::*;
// HAPUS baris 'use candle_core::Tensor;' agar warning hilang
use candle_nn::{Module, VarBuilder};
use crate::WasmTensor; 

#[wasm_bindgen]
pub struct StrictLinear {
    inner: candle_nn::Linear,
}

#[wasm_bindgen]
impl StrictLinear {
    #[wasm_bindgen(constructor)]
    pub fn new_random(in_dim: usize, out_dim: usize) -> std::result::Result<StrictLinear, JsError> {
        let device = candle_core::Device::Cpu;
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        
        let inner = candle_nn::linear(in_dim, out_dim, vb)
            .map_err(|e| JsError::new(&e.to_string()))?;
        
        Ok(StrictLinear { inner })
    }
    
    pub fn forward(&self, input: &WasmTensor) -> std::result::Result<WasmTensor, JsError> {
        let out = self.inner.forward(&input.inner)
            .map_err(|e| JsError::new(&e.to_string()))?;
            
        Ok(WasmTensor { inner: out })
    }
}
