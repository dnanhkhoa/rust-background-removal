use anyhow::{Context, Result};
use clap::Parser;

use image::io::Reader as ImageReader;
use image::{imageops, GenericImage, ImageBuffer, RgbaImage};
use image::{DynamicImage, ImageFormat};
use infer::image::is_jpeg;

use ndarray::{Array, CowArray};
use once_cell::sync::OnceCell;
use ort::Value;

use std::fs;
use std::io::{self, Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
// use tokio::io::AsyncWriteExt;

mod http;
mod onnx;

static SESSION: OnceCell<ort::Session> = OnceCell::new();
static THRESHOLD_BG: OnceCell<u8> = OnceCell::new();

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct App {
    #[clap(short, long, value_name = "INPUT_FILE", default_value = "")]
    input_file: String,
    #[clap(
        short = 'I',
        long,
        value_name = "INPUT_FOLDER",
        default_value = "images"
    )]
    input_images_folder: String,
    #[clap(short, long, value_name = "OUTPUT_FILE", default_value = "")]
    output_file: String,
    #[clap(short, long)]
    verbose: bool,
    #[clap(short, long, default_value = "false")]
    crop: bool,
    #[clap(short = 'S', long, conflicts_with("input_file"))]
    stdin: bool, // Flag to read image from stdin
    #[clap(short = 's', long, conflicts_with("output_file"))]
    stdout: bool, // Flag to write cropped image to stdout
    #[clap(short = 'H', long)]
    http: bool,
    #[clap(short, long, default_value = "0.0.0.0")]
    address: String,
    #[clap(short, long, default_value = "9876")]
    port: u16,
    #[clap(short, long, default_value = "10")]
    threshold_bg: u8,
    #[clap(short, long, default_value = "assets/medium.onnx")]
    model: String,
}

//#[tokio::main(flavor = "current_thread")]
#[tokio::main(worker_threads = 10)]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = App::parse();
    THRESHOLD_BG.set(args.threshold_bg).ok();

    if args.http {
        http::start_http_server(&args).await?;
    }
    let session = onnx::onnx_session(&args.model)?;

    // Determine the source of the input image
    let img: Option<DynamicImage> = if args.stdin {
        // Read image from stdin
        let mut buffer = Vec::new();
        io::stdin().read_to_end(&mut buffer)?;
        if is_jpeg(&buffer) {
            Some(
                ImageReader::with_format(io::Cursor::new(buffer), ImageFormat::Jpeg)
                    .decode()
                    .context("Failed to decode image from stdin")?,
            )
        } else {
            Some(
                ImageReader::with_format(io::Cursor::new(buffer), ImageFormat::Png)
                    .decode()
                    .context("Failed to decode image from stdin")?,
            )
        }
    } else {
        None
    };

    if img.is_some() {
        let processed_dynamic_img = process_dynamic_image(&session, img.unwrap())?;
        if args.crop {
            let mut output_img = processed_dynamic_img.to_rgba8();
            let alpha_bounds = find_alpha_bounds(&output_img);
            if let Some((min_x, min_y, max_x, max_y)) = alpha_bounds {
                let cropped_img =
                    imageops::crop(&mut output_img, min_x, min_y, max_x - min_x, max_y - min_y)
                        .to_image();
                // Convert the cropped image to a full image
                let mut full_cropped_img = ImageBuffer::new(max_x - min_x, max_y - min_y);
                full_cropped_img.copy_from(&cropped_img, 0, 0).ok();

                let mut buffer = Cursor::new(Vec::new());
                full_cropped_img.write_to(&mut buffer, ImageFormat::Png)?;
                let buffer_content = buffer.into_inner();
                io::stdout().write_all(&buffer_content)?;
            }
        } else {
            let mut buffer = Cursor::new(Vec::new());
            processed_dynamic_img.write_to(&mut buffer, ImageFormat::Png)?;
            let buffer_content = buffer.into_inner();
            io::stdout().write_all(&buffer_content)?;
        }
        std::process::exit(0);
    }

    // Input image file folder path
    let input_images_folder = Path::new(&args.input_images_folder);
    let output_images_folder = Path::new("output_images");
    fs::create_dir_all(&output_images_folder)?;

    let image_files: Vec<PathBuf>;
    if !args.input_file.is_empty() {
        image_files = vec![PathBuf::from(&args.input_file)];
    } else {
        image_files = fs::read_dir(&input_images_folder)?
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
    }

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
        let mut output_img = process_image(&session, input_img_file, &output_img_file)?;

        let elapsed_time = start_time.elapsed();
        println!(
            "Processed {} in {}.{:03} seconds",
            input_img_file.display(),
            elapsed_time.as_secs(),
            elapsed_time.subsec_millis()
        );
        if args.crop {
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
    }
    Ok(())
}

fn process_image(
    session: &ort::Session,
    input_img_file: &PathBuf,
    output_img_file: &PathBuf,
) -> Result<ImageBuffer<image::Rgba<u8>, Vec<u8>>, anyhow::Error> {
    let input_shape = session.inputs[0]
        .dimensions()
        .map(|dim| dim.unwrap())
        .collect::<Vec<usize>>();
    let input_img = image::open(input_img_file).unwrap().into_rgba8();
    let scaling_factor = f32::min(
        1., // Avoid upscaling
        f32::min(
            input_shape[3] as f32 / input_img.width() as f32, // Width ratio
            input_shape[2] as f32 / input_img.height() as f32, // Height ratio
        ),
    );
    let mut resized_img = imageops::resize(
        &input_img,
        input_shape[3] as u32,
        input_shape[2] as u32,
        imageops::FilterType::Triangle,
    );
    let input_tensor = CowArray::from(
        Array::from_shape_fn(input_shape, |indices| {
            let mean = 128.;
            let std = 256.;

            (resized_img[(indices[3] as u32, indices[2] as u32)][indices[1]] as f32 - mean) / std
        })
        .into_dyn(),
    );
    let inputs = vec![Value::from_array(session.allocator(), &input_tensor)?];
    let outputs = session.run(inputs)?;
    let output_tensor = outputs[0].try_extract::<f32>()?;
    for (indices, alpha) in output_tensor.view().indexed_iter() {
        resized_img[(indices[3] as u32, indices[2] as u32)][3] = (alpha * 255.) as u8;
    }
    let output_img = imageops::resize(
        &resized_img,
        (input_img.width() as f32 * scaling_factor) as u32,
        (input_img.height() as f32 * scaling_factor) as u32,
        imageops::FilterType::Triangle,
    );
    output_img.save(output_img_file)?;
    Ok(output_img)
}

// Function to find the bounding box containing non-transparent pixels
fn find_alpha_bounds(image: &RgbaImage) -> Option<(u32, u32, u32, u32)> {
    let mut min_x = u32::MAX;
    let mut max_x = 0;
    let mut min_y = u32::MAX;
    let mut max_y = 0;
    let thres_b = THRESHOLD_BG.get().unwrap();

    for (x, y, pixel) in image.enumerate_pixels() {
        if pixel[3] > *thres_b {
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
        println!("found NONE {:?}", (min_x, min_y, max_x, max_y));
        None
    }
}

fn process_dynamic_image(
    session: &ort::Session,
    dynamic_img: DynamicImage,
) -> Result<DynamicImage, anyhow::Error> {
    let input_shape = session.inputs[0]
        .dimensions()
        .map(|dim| dim.unwrap())
        .collect::<Vec<usize>>();
    let input_img = dynamic_img.into_rgba8();
    let scaling_factor = f32::min(
        1., // Avoid upscaling
        f32::min(
            input_shape[3] as f32 / input_img.width() as f32, // Width ratio
            input_shape[2] as f32 / input_img.height() as f32, // Height ratio
        ),
    );
    let mut resized_img = imageops::resize(
        &input_img,
        input_shape[3] as u32,
        input_shape[2] as u32,
        imageops::FilterType::Triangle,
    );
    let input_tensor = CowArray::from(
        Array::from_shape_fn(input_shape, |indices| {
            let mean = 128.;
            let std = 256.;

            (resized_img[(indices[3] as u32, indices[2] as u32)][indices[1]] as f32 - mean) / std
        })
        .into_dyn(),
    );
    let inputs = vec![Value::from_array(session.allocator(), &input_tensor)?];
    let outputs = session.run(inputs)?;
    let output_tensor = outputs[0].try_extract::<f32>()?;
    for (indices, alpha) in output_tensor.view().indexed_iter() {
        resized_img[(indices[3] as u32, indices[2] as u32)][3] = (alpha * 255.) as u8;
    }
    let output_img = imageops::resize(
        &resized_img,
        (input_img.width() as f32 * scaling_factor) as u32,
        (input_img.height() as f32 * scaling_factor) as u32,
        imageops::FilterType::Triangle,
    );
    Ok(DynamicImage::ImageRgba8(output_img))
}
