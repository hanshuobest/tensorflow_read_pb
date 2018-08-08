#!bin/bash
python convert.py yolov3-tiny.cfg yolov3-tiny_260000.weights model_data/yolo.h5
python keras_to_tensorflow.py -input_model_file model_data/yolo.h5 
mv yolo.h5.pb yolo.pb


