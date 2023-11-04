use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, SessionBuilder};

pub(crate) fn onnx_session(onnx_model_file: &str) -> Result<ort::Session, anyhow::Error> {
    let environment = Environment::default().into_arc();
    let session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level1)? // Configure model optimization level
        .with_intra_threads(1)? // Configure the number of threads used for inference
        .with_execution_providers([
            // Configure execution providers (e.g., CUDA, CoreML, CPU)
            ExecutionProvider::CUDA(Default::default()),
            ExecutionProvider::CoreML(Default::default()),
            ExecutionProvider::CPU(Default::default()),
        ])?
        .with_model_from_file(onnx_model_file)?;
    Ok(session)
}
