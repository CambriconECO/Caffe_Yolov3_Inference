#!/bin/bash
pushd convert
    if [ ! -f "yolov3.weights" ]; then
      wget -O yolov3.weights http://video.cambricon.com/models/CambriconECO/Caffe_Yolov3_Inference/yolov3.weights
    else
      echo "yolov3.weights exists."
    fi
    if [ ! -f "yolov3.caffemodel" ]; then
      wget -O yolov3.caffemodel http://video.cambricon.com/models/CambriconECO/Caffe_Yolov3_Inference/yolov3.caffemodel
    else
      echo "yolov3.caffemodel exists."
    fi
popd
