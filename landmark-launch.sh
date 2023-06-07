#!/usr/bin/env bash

# require : blazeFace tflite ( face_detection_front.tflite )
# display text output of landmark_detecting decoder with cam

gst-launch-1.0 -m \
  tensor_videocrop name=crop_left \
  tensor_videocrop name=crop_right \
  v4l2src name=cam_src ! videoconvert ! videoscale ! \
  video/x-raw,width=1000,height=1000,format=RGB,pixel-aspect-ratio=1/1,framerate=30/1 ! tee name=t \
  t. ! queue ! crop_left.sink \
  t. ! queue ! crop_right.sink \
  t. ! queue ! videoscale ! video/x-raw,width=128,height=128,format=RGB ! tensor_converter ! \
       queue ! tensor_transform mode=arithmetic option=typecast:float32,div:255.0,add:-0.5,div:0.5 ! \
       tensor_filter framework=tensorflow2-lite model=face_detection_front.tflite \
                      output=16:896:1:1,1:896:1:1 \
                      outputname=landmarks,scores \
                      outputtype=float32,float32 ! tee name=tf\
       tf. ! tensor_decoder mode=landmark_detecting option1=left ! crop_left.info \
       tf. ! tensor_decoder mode=landmark_detecting option1=right ! crop_right.info \
  t. ! videoconvert ! videoscale ! ximagesink sync=false \
  crop_left.src ! videoconvert ! videoscale ! ximagesink sync=false \
  crop_right.src ! videoconvert ! videoscale ! ximagesink sync=false

# gst-launch-1.0 -m \
#   v4l2src name=cam_src ! videoconvert ! videoscale ! \
#   video/x-raw,width=100,height=100,format=RGB,framerate=30/1 ! tensor_converter ! tensor_split name=split tensorseg=2:100:100,1:100:100 split.src_0 ! queue ! filesink location=src0.log \
#   split.src_1 ! queue ! filesink location=src1.log

# gst-launch-1.0 -m \
#   tensor_videocrop name=crop_left \
#   tensor_videocrop name=crop_right \
#   v4l2src name=cam_src ! videoconvert ! videoscale ! \
#   video/x-raw,width=1000,height=1000,format=RGB,pixel-aspect-ratio=1/1,framerate=30/1 ! tee name=t \
#   t. ! queue ! crop_left.sink \
#   t. ! queue ! crop_right.sink \
#   t. ! queue ! videoscale ! video/x-raw,width=128,height=128,format=RGB ! tensor_converter ! \
#        queue ! tensor_transform mode=arithmetic option=typecast:float32,div:255.0,add:-0.5,div:0.5 ! \
#        tensor_filter framework=tensorflow2-lite model=face_detection_front.tflite \
#                       output=16:896:1:1,1:896:1:1 \
#                       outputname=landmarks,scores \
#                       outputtype=float32,float32 ! tee name=tf\
#        tf. ! tensor_decoder mode=landmark_detecting option1=left ! crop_left.info \
#        tf. ! tensor_decoder mode=landmark_detecting option1=right ! crop_right.info \
#   crop_left.src ! videoconvert ! ximagesink sync=false \
#   crop_right.src ! videoconvert ! ximagesink sync=false
#     crop_left.src ! queue ! videoscale ! video/x-raw,width=64,height=64,format=RGB ! tensor_converter ! \
#                   queue ! tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! \
#                   tensor_filter framework=tensorflow2-lite model=iris_landmark.tflite custom=Delegate:XNNPACK,NumThreads:4 ! \
#                   other/tensors,num_tensors=2,types=float32.float32,dimensions=213:1:1:1.15:1:1:1 ! \
#                   tensor_decoder mode=eye_detecting ! mix.sink_1
#   crop_right.src ! queue ! videoscale ! video/x-raw,width=64,height=64,format=RGB ! tensor_converter ! \
#                    queue ! tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! \
#                    tensor_filter framework=tensorflow2-lite model=iris_landmark.tflite custom=Delegate:XNNPACK,NumThreads:4 ! \
#                    other/tensors,num_tensors=2,types=float32.float32,dimensions=213:1:1:1.15:1:1:1 ! \
#                    tensor_decoder mode=eye_detecting ! mix.sink_2