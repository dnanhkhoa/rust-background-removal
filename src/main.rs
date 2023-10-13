use anyhow::{Context, Result};
use clap::Parser;
use image::io::Reader as ImageReader;
use image::{imageops, GenericImage, ImageBuffer, RgbaImage, Rgba};
use image::{DynamicImage, ImageFormat};
use infer::image::is_jpeg;
use ndarray::{Array, CowArray};
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, SessionBuilder, Value};
use std::fs;
use std::io::{self, Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::convert::Infallible;
use hyper::{Body, Request, Response, Server};
use hyper::service::{make_service_fn, service_fn};
use once_cell::sync::OnceCell;
use multer::Multipart;
use tokio::io::AsyncWriteExt;

static SESSION : OnceCell<ort::Session> = OnceCell::new();

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct App {
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
}


//#[tokio::main(flavor = "current_thread")]
#[tokio::main(worker_threads = 10)]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = App::parse();

    if args.http {
        start_http_server().await?;
    }
    let session = onnx_session()?;

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

fn onnx_session() -> Result<ort::Session, anyhow::Error> {
    let onnx_model_file = "assets/medium.onnx";
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

// Function to find the bounding box containing non-transparent pixels
fn find_alpha_bounds(image: &RgbaImage) -> Option<(u32, u32, u32, u32)> {
    let mut min_x = u32::MAX;
    let mut max_x = 0;
    let mut min_y = u32::MAX;
    let mut max_y = 0;

    for (x, y, pixel) in image.enumerate_pixels() {
        if pixel[3] > 10 {
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
        println!("found NONE {:?}",(min_x, min_y, max_x, max_y));
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

async fn start_http_server() -> Result<(), anyhow::Error> {
     let session = onnx_session().unwrap();
     SESSION.set(session).ok();
    // Define a closure to handle incoming HTTP requests
    let make_svc = make_service_fn(|_conn| {
     async {
        Ok::<_, Infallible>(service_fn(move |req| {
            handle_post(req)
        }))
    }
    });

    let addr = ([0, 0, 0, 0], 9876).into();
    let server = Server::bind(&addr).serve(make_svc);
    println!("Listening on http://127.0.0.1:9876/");
    server.await?;
    Ok(())
}

async fn handle_post(req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    let path = req.uri().clone().path().to_owned();
    let session=SESSION.get().unwrap();
    // Check if the request is a multipart POST
    if let Some(content_type) = req.headers().get(hyper::header::CONTENT_TYPE) {
        if let Ok(content_type_str) = content_type.to_str() {
            if content_type_str.starts_with("multipart/form-data") {
    // Parse the boundary from the content_type, exit early if we don't have it.
    let boundary = match content_type_str.split(';').find(|s| s.trim().starts_with("boundary=")) {
        Some(boundary) => boundary.trim().trim_start_matches("boundary=").to_string(),
        None => {
            eprintln!("Error: Boundary not found in CONTENT_TYPE.");
            "".to_string()
        }
    };

    let mut multipart = Multipart::new(req.into_body(), &boundary);
    while let Some(mut field) = multipart.next_field().await.unwrap() {
        let has_filename = field.file_name().map(|s| s.to_string());
        if let Some(file_name) = has_filename {
            let buffer = field.bytes().await.unwrap();
    let img: Option<DynamicImage> = 
        if is_jpeg(&buffer) {
            Some(
                ImageReader::with_format(io::Cursor::new(buffer), ImageFormat::Jpeg)
                    .decode()
                    .context("Failed to decode image from stdin").unwrap()
            )
        } else {
            Some(
                ImageReader::with_format(io::Cursor::new(buffer), ImageFormat::Png)
                    .decode()
                    .context("Failed to decode image from stdin").unwrap()
            )
        };

    if img.is_some() {
        let w=&img.clone().unwrap().clone().width();
        println!("f: {}", file_name);
        let processed_dynamic_img = process_dynamic_image(session, img.unwrap()).unwrap();

            let mut crop_box: String = String::new();
            let mut buffer = Cursor::new(Vec::new());
            if path == "/crop" {
                let mut output_img = processed_dynamic_img.to_rgba8();
                let alpha_bounds = find_alpha_bounds(&output_img);
                if let Some((min_x, min_y, max_x, max_y)) = alpha_bounds {
                    let w=max_x-min_x;
                    let h = max_y-min_y;
                    if w>0 && h>0 {
                        crop_box=format!("{},{},{},{}", min_x, min_y, max_x - min_x, max_y - min_y);
                    let cropped_img =
                        imageops::crop(&mut output_img, min_x, min_y, max_x - min_x, max_y - min_y)
                            .to_image();
                        cropped_img.write_to(&mut buffer, ImageFormat::Png).unwrap();
                    } else {
                        processed_dynamic_img.write_to(&mut buffer, ImageFormat::Png).unwrap();
                    }
                } else {
                    // Set the pixel at (0, 0) to be fully transparent.
                    let mut img: RgbaImage = ImageBuffer::new(1, 1);
                    img.put_pixel(0, 0, Rgba([0, 0, 0, 0]));
                    img.write_to(&mut buffer, ImageFormat::Png).unwrap();
                }
            } else {
                processed_dynamic_img.write_to(&mut buffer, ImageFormat::Png).unwrap();
            }
            let buffer_content = buffer.into_inner();
            let mut response = Response::new(Body::from(buffer_content));
            if !crop_box.is_empty() {
                response.headers_mut().insert(
                "Access-Control-Expose-Headers",
                hyper::header::HeaderValue::from_static("*"),
                               );
               response.headers_mut().insert(
                "X-cropbox",
                hyper::header::HeaderValue::from_str(&crop_box).unwrap(),
            );
            }
            response.headers_mut().insert(
                hyper::header::ACCESS_CONTROL_ALLOW_ORIGIN,
                hyper::header::HeaderValue::from_static("*"),
            );
            return Ok(response)
    } else {
        println!("no image");
    }
        }
    }
                let response = Response::new(Body::from("File uploaded successfully"));
                return Ok(response);
            }
        }
    }

    // If the request is not a multipart POST, return a 400 Bad Request response
    let response = Response::builder()
        .status(400)
        .body(Body::from("Bad Request: Invalid content type"))
        .unwrap();

    Ok(response)
}