mkdir -p "assets"
curl -L "https://github.com/imgly/background-removal-js/raw/main/bundle/models/small" -o "assets/small.onnx"
curl -L "https://github.com/imgly/background-removal-js/raw/main/bundle/models/medium" -o "assets/medium.onnx"
curl -L "https://github.com/imgly/background-removal-js/raw/main/bundle/models/large" -o "assets/large.onnx"
