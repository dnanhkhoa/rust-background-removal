# 🦀 Background Removal

![Example](images/example.png)

This Rust implementation is a minimal port of the original JavaScript library [@imgly/background-removal](https://github.com/imgly/background-removal-js). It aims to explore **the use of ONNX models in Rust** for background removal tasks. If you're interested in the full capabilities and details of the background removal process, I highly recommend checking out the original JavaScript library's [README](https://github.com/imgly/background-removal-js) here.

**Update:** For the technical details of the model, please check out the paper **Highly Accurate Dichotomous Image Segmentation** mentioned in [this repository](https://github.com/xuebinqin/DIS).

## Usage

```sh
# Download all pretrained models
./download_models.sh

# Update variable input_img_file in src/main.rs

# Run the program
cargo run --release
```

## Limitations

The model resolution is limited to a maximum of 1024x1024 pixels.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Related sites

[pykeio/ort: A Rust wrapper for ONNX Runtime](https://github.com/pykeio/ort)

## Credits

Khoa Duong, David Horner