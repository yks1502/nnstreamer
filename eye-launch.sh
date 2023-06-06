#!/usr/bin/env bash

# require : iris_landmark.tflite 
# process eye_detecting decoder with cam

gst-launch-1.0 -m \
  v4l2src name=cam_src ! videoconvert ! videoscale ! \
    video/x-raw,width=1000,height=1000,format=RGB,pixel-aspect-ratio=1/1,framerate=30/1 ! \
    tee name=t \
    t. ! queue ! videoconvert ! ximagesink sync=false \
    t. ! queue leaky=2 max-size-buffers=2 ! videoscale ! \
    video/x-raw,width=64,height=64,format=RGB ! tensor_converter ! \
    queue ! tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! \
    tensor_filter framework=tensorflow2-lite model=iris_landmark.tflite custom=Delegate:XNNPACK,NumThreads:4 ! \
    other/tensors,num_tensors=2,types=\"float32\,float32\",dimensions=\"213:1:1:1\,15:1:1:1\",format=static ! \
    tensor_decoder mode=eye_detecting option1=64:64 ! \
    other/tensors,num_tensors=1,types=uint32,dimensions=154:1:1:1,format=static ! \
    tensor_sink