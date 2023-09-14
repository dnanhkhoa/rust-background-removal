# 🦀 Background Removal

![Example](images/example.png)

This Rust implementation is a minimal port of the original JavaScript library `@imgly/background-removal`. It aims to explore **the use of ONNX models in Rust** for background removal tasks. If you're interested in the full capabilities and details of the background removal process, I highly recommend checking out the original JavaScript library's [README](https://github.com/imgly/background-removal-js) here.

## Usage

```sh
# Download all pretrained models
./download_models.sh

# Update variable input_img_file in src/main.rs

# Run the program
cargo run --release
```

## License

[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
