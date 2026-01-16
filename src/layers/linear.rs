use wasm_bindgen::prelude::*;
use candle_core::{Tensor, Result};
use candle_nn::{Module, VarBuilder};
use crate::WasmTensor; // Ambil koper dari lib.rs

#[wasm_bindgen]
pub struct StrictLinear {
    inner: candle_nn::Linear,
}

#[wasm_bindgen]
impl StrictLinear {
    // Constructor Dummy (Random) untuk tes WASM
    #[wasm_bindgen(constructor)]
    pub fn new_random(in_dim: usize, out_dim: usize) -> Result<StrictLinear, JsError> {
        let device = candle_core::Device::Cpu;
        // Kita pakai VarBuilder zeros biar simpel
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        let inner = candle_nn::linear(in_dim, out_dim, vb)
            .map_err(|e| JsError::new(&e.to_string()))?;
        
        Ok(StrictLinear { inner })
    }
    
    // Forward: Terima Koper -> Proses -> Balikin Koper
    pub fn forward(&self, input: &WasmTensor) -> Result<WasmTensor, JsError> {
        let out = self.inner.forward(&input.inner)
            .map_err(|e| JsError::new(&e.to_string()))?;
            
        Ok(WasmTensor { inner: out })
    }
}
