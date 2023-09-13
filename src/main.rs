use image::imageops;
use ndarray::{Array, CowArray};
use ort::{
    Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, SessionBuilder, Value,
};
use std::path::Path;

fn main() -> OrtResult<()> {
    // Input image file path
    let input_img_file = Path::new("images/1.jpg");

    // Output image file path is from the input image file path with "_nbg" suffix
    let output_img_file = input_img_file
        .with_file_name(format!(
            "{}_nbg",
            input_img_file.file_stem().unwrap().to_str().unwrap()
        ))
        .with_extension("png");

    // ONNX model file path
    let onnx_model_file = "assets/medium.onnx";

    // Create an Ort environment, which manages resources for ONNX Runtime
    let environment = Environment::default().into_arc();

    // Create an Ort session for inference using the provided ONNX model
    let session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level1)? // Configure model optimization level
        .with_intra_threads(1)? // Configure the number of threads used for inference
        .with_execution_providers([
            // Configure execution providers (e.g., CUDA, CoreML, CPU)
            ExecutionProvider::CUDA(Default::default()),
            ExecutionProvider::CoreML(Default::default()),
            ExecutionProvider::CPU(Default::default()),
        ])?
        .with_model_from_file(onnx_model_file)?; // Load the ONNX model from a file

    // Get the required input shape: [batch_size, channel, height, width]
    let input_shape = session.inputs[0]
        .dimensions()
        .map(|dim| dim.unwrap())
        .collect::<Vec<usize>>();

    // Read and convert the input image to RGBA8 format
    let input_img = image::open(input_img_file).unwrap().into_rgba8();

    // Calculate the scaling factor for resizing
    let scaling_factor = f32::min(
        1., // Avoid upscaling
        f32::min(
            input_shape[3] as f32 / input_img.width() as f32, // Width ratio
            input_shape[2] as f32 / input_img.height() as f32, // Height ratio
        ),
    );

    // Resize the input image to match the model input shape
    let mut resized_img = imageops::resize(
        &input_img,
        input_shape[3] as u32,
        input_shape[2] as u32,
        imageops::FilterType::Triangle,
    );

    // Create an input tensor from the normalized input image
    let input_tensor = CowArray::from(
        Array::from_shape_fn(input_shape, |indices| {
            let mean = 128.;
            let std = 256.;

            (resized_img[(indices[3] as u32, indices[2] as u32)][indices[1]] as f32 - mean) / std
        })
        .into_dyn(),
    );

    // Prepare an input batch for inference with a single instance
    let inputs = vec![Value::from_array(session.allocator(), &input_tensor)?];

    // Perform the inference
    let outputs = session.run(inputs)?;

    // Extract the output tensor from the output batch
    let output_tensor = outputs[0].try_extract::<f32>()?;

    // Update the alpha channel of the resized image with the predicted values
    for (indices, alpha) in output_tensor.view().indexed_iter() {
        resized_img[(indices[3] as u32, indices[2] as u32)][3] = (alpha * 255.) as u8;
    }

    // Resize the final output image back to the original size if possible using the scaling factor
    let output_img = imageops::resize(
        &resized_img,
        (input_img.width() as f32 * scaling_factor) as u32,
        (input_img.height() as f32 * scaling_factor) as u32,
        imageops::FilterType::Triangle,
    );

    // Save the final output image to the output image file path
    output_img.save(output_img_file).unwrap();

    Ok(())
}
