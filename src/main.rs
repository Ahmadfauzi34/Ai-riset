use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use candle_nn_catalog::layers::recurrent::StrictLSTM;

fn main() -> anyhow::Result<()> {
    println!("🧪 === LABORATORIUM RISET: LSTM === 🧪");

    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);

    // 1. Buat Layer
    let in_dim = 10;
    let hidden_dim = 20;
    let lstm_layer = StrictLSTM::new(in_dim, hidden_dim, vb.pp("lstm"))?;

    // 2. Input Sequence [Seq_Len=3, Batch=1, Dim=10]
    let input_seq = Tensor::randn(0f32, 1.0, (3, 1, in_dim), &device)?;

    // 3. Forward (Otomatis handle state internal)
    let output = lstm_layer.forward(&input_seq)?;
    
    println!("Output Shape: {:?}", output.shape());
    println!("✅ Eksperimen LSTM Berhasil!");

    Ok(())
}