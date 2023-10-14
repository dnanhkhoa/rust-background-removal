use crate::App;
use anyhow::{Context, Result};
use clap::Parser;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server};
use image::io::Reader as ImageReader;
use image::{imageops, GenericImage, ImageBuffer, Rgba, RgbaImage};
use image::{DynamicImage, ImageFormat};
use infer::image::is_jpeg;
use multer::Multipart;

use std::convert::Infallible;

use std::io::{self, Cursor, Read};

// use tokio::io::AsyncWriteExt;
use std::net::Ipv4Addr;

pub async fn start_http_server(args: &App) -> Result<(), anyhow::Error> {
    let session = crate::onnx::onnx_session(&args.model).unwrap();
    crate::SESSION.set(session).ok();
    // Define a closure to handle incoming HTTP requests
    let make_svc = make_service_fn(|_conn| async {
        Ok::<_, Infallible>(service_fn(move |req| handle_post(req)))
    });

    let ip_address = args.address.parse::<Ipv4Addr>();
    match ip_address {
        Ok(ip) => {
            let octets = ip.octets();
            let addr: [u8; 4] = octets.into();
            let addr = (addr, args.port).into();
            let server = Server::bind(&addr).serve(make_svc);
            println!("Listening on http://{}:{}/", args.address, args.port);
            server.await?;
        }
        Err(_) => {
            println!("Invalid IP address");
        }
    }
    // Access the parsed values
    Ok(())
}

pub async fn handle_post(req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    let path = req.uri().clone().path().to_owned();
    let session = crate::SESSION.get().unwrap();
    // Check if the request is a multipart POST
    if let Some(content_type) = req.headers().get(hyper::header::CONTENT_TYPE) {
        if let Ok(content_type_str) = content_type.to_str() {
            if content_type_str.starts_with("multipart/form-data") {
                // Parse the boundary from the content_type, exit early if we don't have it.
                let boundary = match content_type_str
                    .split(';')
                    .find(|s| s.trim().starts_with("boundary="))
                {
                    Some(boundary) => boundary.trim().trim_start_matches("boundary=").to_string(),
                    None => {
                        eprintln!("Error: Boundary not found in CONTENT_TYPE.");
                        "".to_string()
                    }
                };

                let mut multipart = Multipart::new(req.into_body(), &boundary);
                while let Some(field) = multipart.next_field().await.unwrap() {
                    let has_filename = field.file_name().map(|s| s.to_string());
                    if let Some(file_name) = has_filename {
                        let buffer = field.bytes().await.unwrap();
                        let img: Option<DynamicImage> = if is_jpeg(&buffer) {
                            Some(
                                ImageReader::with_format(
                                    io::Cursor::new(buffer),
                                    ImageFormat::Jpeg,
                                )
                                .decode()
                                .context("Failed to decode image from stdin")
                                .unwrap(),
                            )
                        } else {
                            Some(
                                ImageReader::with_format(io::Cursor::new(buffer), ImageFormat::Png)
                                    .decode()
                                    .context("Failed to decode image from stdin")
                                    .unwrap(),
                            )
                        };

                        if img.is_some() {
                            println!("f: {}", file_name);
                            let processed_dynamic_img =
                                crate::process_dynamic_image(session, img.unwrap()).unwrap();

                            let mut crop_box: String = String::new();
                            let mut buffer = Cursor::new(Vec::new());
                            if path == "/crop" {
                                let mut output_img = processed_dynamic_img.to_rgba8();
                                let alpha_bounds = crate::find_alpha_bounds(&output_img);
                                if let Some((min_x, min_y, max_x, max_y)) = alpha_bounds {
                                    let w = max_x - min_x;
                                    let h = max_y - min_y;
                                    if w > 0 && h > 0 {
                                        crop_box = format!(
                                            "{},{},{},{}",
                                            min_x,
                                            min_y,
                                            max_x - min_x,
                                            max_y - min_y
                                        );
                                        let cropped_img = imageops::crop(
                                            &mut output_img,
                                            min_x,
                                            min_y,
                                            max_x - min_x,
                                            max_y - min_y,
                                        )
                                        .to_image();
                                        cropped_img
                                            .write_to(&mut buffer, ImageFormat::Png)
                                            .unwrap();
                                    } else {
                                        processed_dynamic_img
                                            .write_to(&mut buffer, ImageFormat::Png)
                                            .unwrap();
                                    }
                                } else {
                                    // Set the pixel at (0, 0) to be fully transparent.
                                    let mut img: RgbaImage = ImageBuffer::new(1, 1);
                                    img.put_pixel(0, 0, Rgba([0, 0, 0, 0]));
                                    img.write_to(&mut buffer, ImageFormat::Png).unwrap();
                                }
                            } else {
                                processed_dynamic_img
                                    .write_to(&mut buffer, ImageFormat::Png)
                                    .unwrap();
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
                            return Ok(response);
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
