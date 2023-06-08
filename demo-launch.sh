#!/usr/bin/env bash

# require : iris_landmark.tflite 
# process eye_detecting decoder with cam

# Cropped eyes with face-landmark
# you can use webcam by using :
# v4l2src name=cam_src ! videoconvert ! videoscale ! \
# you can use still image by using :
# filesrc location=jin.jpg ! decodebin ! videoconvert ! imagefreeze ! videoscale ! \
gst-launch-1.0 -m \
  tensor_videocrop name=crop_left \
  tensor_videocrop name=crop_right \
  v4l2src name=cam_src ! videoconvert ! videoscale ! \
  video/x-raw,width=800,height=800,format=RGB,pixel-aspect-ratio=1/1,framerate=30/1 ! tee name=t \
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
  crop_left.src ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! videoconvert ! tee name=eye_left \
  crop_right.src ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! videoconvert ! tee name=eye_right \
  eye_left. ! mix_left.sink_1 \
  eye_left. ! videoscale ! video/x-raw,width=64,height=64,format=RGB ! tensor_converter ! \
              tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! \
              queue ! tensor_filter framework=tensorflow2-lite model=iris_landmark.tflite custom=Delegate:XNNPACK,NumThreads:4 ! \
              other/tensors,num_tensors=2,types=\"float32\,float32\",dimensions=\"213:1:1:1\,15:1:1:1\",format=static ! \
              tensor_decoder mode=eye_detecting option1=64:64 ! \
              videoconvert ! videoscale ! videoscale ! video/x-raw,width=300,height=300,format=RGBA ! mix_left.sink_0 \
  eye_right. ! mix_right.sink_1 \
  eye_right. ! videoscale ! video/x-raw,width=64,height=64,format=RGB ! tensor_converter ! \
              tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! \
              queue ! tensor_filter framework=tensorflow2-lite model=iris_landmark.tflite custom=Delegate:XNNPACK,NumThreads:4 ! \
              other/tensors,num_tensors=2,types=\"float32\,float32\",dimensions=\"213:1:1:1\,15:1:1:1\",format=static ! \
              tensor_decoder mode=eye_detecting option1=64:64 ! \
              videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGBA ! mix_right.sink_0 \
  compositor name=mix_left sink_0::zorder=2 sink_1::zorder=1 ! videoconvert ! \
             textoverlay text="left_iris" valignment=bottom halignment=center font-desc="Sans, 36" ! ximagesink sync=false \
  compositor name=mix_right sink_0::zorder=2 sink_1::zorder=1 ! videoconvert ! \
             textoverlay text="right_iris" valignment=bottom halignment=center font-desc="Sans, 36" ! ximagesink sync=false \
  eye_right. ! videoconvert ! videoscale ! \
             textoverlay text="right_eye" valignment=bottom halignment=center font-desc="Sans, 36" ! ximagesink sync=false \
  eye_left. ! videoconvert ! videoscale ! \
             textoverlay text="left_eye" valignment=bottom halignment=center font-desc="Sans, 36" ! ximagesink sync=false \
  t. ! videoconvert ! videoscale ! \
             textoverlay text="NNStreamer Script Demo" valignment=bottom halignment=center font-desc="Sans, 20" ! ximagesink sync=false \