use anyhow::Result;
use image::{imageops, GenericImage, ImageBuffer, RgbaImage};
use ndarray::{Array, CowArray};
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, SessionBuilder, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() -> Result<()> {
    // Input image file folder path
    let input_images_folder = Path::new("images");
    let output_images_folder = Path::new("output_images");
    fs::create_dir_all(&output_images_folder)?;

    let image_files: Vec<PathBuf> = fs::read_dir(&input_images_folder)?
        .filter_map(|entry| {
            if let Ok(entry) = entry {
                if let Some(extension) = entry.path().extension() {
                    if extension == "jpg" || extension == "png" {
                        return Some(entry.path());
                    }
                }
            }
            None
        })
        .collect();

    for input_img_file in &image_files {
        let output_img_file = output_images_folder.join(
            input_img_file
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap()
                .to_owned()
                + "_nbg.png",
        );

        // Start timing
        let start_time = Instant::now();
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

                (resized_img[(indices[3] as u32, indices[2] as u32)][indices[1]] as f32 - mean)
                    / std
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
        let mut output_img = imageops::resize(
            &resized_img,
            (input_img.width() as f32 * scaling_factor) as u32,
            (input_img.height() as f32 * scaling_factor) as u32,
            imageops::FilterType::Triangle,
        );

        output_img.save(&output_img_file)?;

        let elapsed_time = start_time.elapsed();
        println!(
            "Processed {} in {}.{:03} seconds",
            input_img_file.display(),
            elapsed_time.as_secs(),
            elapsed_time.subsec_millis()
        );
        let alpha_bounds = find_alpha_bounds(&output_img);

        if let Some((min_x, min_y, max_x, max_y)) = alpha_bounds {
            let cropped_img =
                imageops::crop(&mut output_img, min_x, min_y, max_x - min_x, max_y - min_y)
                    .to_image();
            // Convert the cropped image to a full image
            let mut full_cropped_img = ImageBuffer::new(max_x - min_x, max_y - min_y);
            full_cropped_img.copy_from(&cropped_img, 0, 0).ok();

            // Modify the output file path to include "_cropped" before the extension
            let mut output_img_file_cropped = output_img_file.clone();
            if let Some(_extension) = output_img_file_cropped.extension() {
                let file_stem = output_img_file_cropped.file_stem().unwrap();
                let new_file_stem = format!("{}_cropped", file_stem.to_str().unwrap());
                output_img_file_cropped.set_file_name(new_file_stem);
            }

            // Append the original extension to the modified output file path
            if let Some(extension) = output_img_file.extension() {
                output_img_file_cropped.set_extension(extension);
            }

            // Save the cropped image
            full_cropped_img.save(output_img_file_cropped)?;
        }
    }

    Ok(())
}

// Function to find the bounding box containing non-transparent pixels
fn find_alpha_bounds(image: &RgbaImage) -> Option<(u32, u32, u32, u32)> {
    let mut min_x = u32::MAX;
    let mut max_x = 0;
    let mut min_y = u32::MAX;
    let mut max_y = 0;

    for (x, y, pixel) in image.enumerate_pixels() {
        if pixel[3] != 0 {
            // Non-transparent pixel
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
    }

    if min_x <= max_x && min_y <= max_y {
        Some((min_x, min_y, max_x, max_y))
    } else {
        None
    }
}
