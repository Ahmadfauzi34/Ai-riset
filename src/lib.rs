use wasm_bindgen::prelude::*;
use candle_core::{Tensor, Device};

pub mod layers;

// --- KOPER TENSOR (WasmTensor) ---
// Ini objek yang nanti akan Anda pakai di JavaScript
#[wasm_bindgen]
pub struct WasmTensor {
    inner: Tensor,
}

#[wasm_bindgen]
impl WasmTensor {
    // 1. Bikin Tensor baru dari Array JS
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[f32], shape: &[usize]) -> Result<WasmTensor, JsError> {
        let device = Device::Cpu;
        let tensor = Tensor::from_slice(data, shape, &device)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmTensor { inner: tensor })
    }

    // 2. Cek Shape (Dimensi)
    pub fn shape(&self) -> Vec<usize> {
        self.inner.dims().to_vec()
    }
    
    // 3. Balikin data ke JS (Float32Array)
    pub fn to_array(&self) -> Result<Vec<f32>, JsError> {
        self.inner.flatten_all()
            .map_err(|e| JsError::new(&e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| JsError::new(&e.to_string()))
    }
}

// Export layer agar bisa dipanggil
pub use layers::linear::StrictLinear;

#[wasm_bindgen]
pub fn init_hooks() {
    console_error_panic_hook::set_once();
}
