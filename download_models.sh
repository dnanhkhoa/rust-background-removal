#!/bin/bash

# Create assets folder
mkdir -p 'assets'

# Download small model
curl -L 'https://github.com/imgly/background-removal-js/raw/main/bundle/models/small' -o 'assets/small.onnx'

# Download medium model
curl -L 'https://github.com/imgly/background-removal-js/raw/main/bundle/models/medium' -o 'assets/medium.onnx'

# Download large model
curl -L 'https://github.com/imgly/background-removal-js/raw/main/bundle/models/large' -o 'assets/large.onnx'
