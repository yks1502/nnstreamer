#!/usr/bin/env bash

# require : blazeFace tflite ( face_detection_front.tflite )
# display text output of landmark_detecting decoder with cam

gst-launch-1.0 -m \
  tensor_videocrop name=crop_left \
  tensor_videocrop name=crop_right \
  v4l2src name=cam_src ! videoconvert ! videoscale ! \
  video/x-raw,width=1000,height=1000,format=RGB,pixel-aspect-ratio=1/1,framerate=30/1 ! tee name=t \
  t. ! queue ! mix.sink_0 \
  t. ! queue ! crop_left.raw \
  t. ! queue ! crop_right.raw \
  t. ! queue leaky=2 max-size-buffers=2 ! videoscale ! video/x-raw,width=128,height=128,format=RGB ! tensor_converter ! \
       queue ! tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! \
       tensor_filter framework=tensorflow2-lite model=face_detection_front.tflite \
                      output=16:896:1:1,1:896:1:1 \
                      outputname=landmarks,scores \
                      outputtype=float32,float32 ! \
       tensor_decoder mode=landmark_detecting ! tensor_split name=split tensorseg=4:1:1,4:1:1 \
       split.src0 ! queue ! crop_left.info \
       split.src1 ! queue ! crop_right.info \
  crop_left.src ! queue ! videoscale ! video/x-raw,width=64,height=64,format=RGB ! tensor_converter ! \
                  queue ! tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! \
                  tensor_filter framework=tensorflow2-lite model=iris_landmark.tflite custom=Delegate:XNNPACK,NumThreads:4 ! \
                  other/tensors,num_tensors=2,types=float32.float32,dimensions=213:1:1:1.15:1:1:1 ! \
                  tensor_decoder mode=eye_detecting ! mix.sink_1
  crop_right.src ! queue ! videoscale ! video/x-raw,width=64,height=64,format=RGB ! tensor_converter ! \
                   queue ! tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! \
                   tensor_filter framework=tensorflow2-lite model=iris_landmark.tflite custom=Delegate:XNNPACK,NumThreads:4 ! \
                   other/tensors,num_tensors=2,types=float32.float32,dimensions=213:1:1:1.15:1:1:1 ! \
                   tensor_decoder mode=eye_detecting ! mix.sink_2
  compositor name=mix sink_0::zorder=1 sink_1::zorder=2 sink_2::zorder=3 ! videoconvert ! ximagesink sync=false