mkdir -p "assets"
curl -L "https://github.com/imgly/background-removal-js/raw/4306d99530d3ae9ec11a892a23802be28f367518/bundle/models/small" -o "assets/small.onnx"
curl -L "https://github.com/imgly/background-removal-js/raw/4306d99530d3ae9ec11a892a23802be28f367518/bundle/models/medium" -o "assets/medium.onnx"
curl -L "https://github.com/imgly/background-removal-js/raw/4306d99530d3ae9ec11a892a23802be28f367518/bundle/models/large" -o "assets/large.onnx"
