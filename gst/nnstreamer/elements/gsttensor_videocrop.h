/**
 * NNStreamer video frame cropping
 * 
 * this element implemented as modification of gstreamer/videocrop element,
 * to crop video frame using tensor instead of property
 * 
 * @see_also: https://github.com/GStreamer/gst-plugins-good/blob/master/gst/videocrop/gstvideocrop.c
 *
 * Copyright (C) 2006 Tim-Philipp Müller <tim centricular net>
 * Copyright (C) 2023 Kiwoong Kim <helloing0119@naver.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Library General Public License for more details.
 * 
 */
/**
 * @file	gsttensor_videocrop.h
 * @date	3 Jul 2023
 * @brief	GStreamer plugin to crop video using tensor
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Kiwoong Kim <helloing0119@naver.com>
 * @bug		need to test.
 *
 */
#ifndef __GST_VIDEO_CROP_H__
#define __GST_VIDEO_CROP_H__

#include <gst/gst.h>
#include <gst/video/gstvideofilter.h>
#include <tensor_common.h>

#define VIDEO_CROP_FORMATS_PACKED_SIMPLE "RGB, BGR, RGB16, RGB15, " \
  "RGBx, xRGB, BGRx, xBGR, RGBA, ARGB, BGRA, ABGR, " \
  "GRAY8, GRAY16_LE, GRAY16_BE, AYUV"
#define VIDEO_CROP_FORMATS_PACKED_COMPLEX "YVYU, YUY2, UYVY"
#define VIDEO_CROP_FORMATS_PLANAR "I420, A420, YV12, Y444, Y42B, Y41B, " \
  "I420_10BE, A420_10BE, Y444_10BE, A444_10BE, I422_10BE, A422_10BE, " \
  "I420_10LE, A420_10LE, Y444_10LE, A444_10LE, I422_10LE, A422_10LE, " \
  "I420_12BE, Y444_12BE, I422_12BE, " \
  "I420_12LE, Y444_12LE, I422_12LE, " \
  "GBR, GBR_10BE, GBR_10LE, GBR_12BE, GBR_12LE, " \
  "GBRA, GBRA_10BE, GBRA_10LE, GBRA_12BE, GBRA_12LE"
#define VIDEO_CROP_FORMATS_SEMI_PLANAR "NV12, NV21"

/* aspectratiocrop uses videocrop. sync caps changes between both */
#define VIDEO_CROP_CAPS                                \
  GST_VIDEO_CAPS_MAKE ("{" \
	VIDEO_CROP_FORMATS_PACKED_SIMPLE "," \
	VIDEO_CROP_FORMATS_PACKED_COMPLEX "," \
	VIDEO_CROP_FORMATS_PLANAR "," \
	VIDEO_CROP_FORMATS_SEMI_PLANAR "}") "; " \
  "video/x-raw(ANY), " \
         "width = " GST_VIDEO_SIZE_RANGE ", " \
         "height = " GST_VIDEO_SIZE_RANGE ", " \
         "framerate = " GST_VIDEO_FPS_RANGE

#define CROP_INFO_CAPS GST_TENSOR_CAP_DEFAULT ";" GST_TENSORS_CAP_WITH_NUM ("1")

G_BEGIN_DECLS
#define GST_TYPE_TENSOR_VIDEO_CROP \
  (gst_tensor_video_crop_get_type())
#define GST_TENSOR_VIDEO_CROP(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_VIDEO_CROP,GstTensorVideoCrop))
#define GST_TENSOR_VIDEO_CROP_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_VIDEO_CROP,GstTensorVideoCropClass))
#define GST_IS_TENSOR_VIDEO_CROP(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_VIDEO_CROP))
#define GST_IS_TENSOR_VIDEO_CROP_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_VIDEO_CROP))

GST_ELEMENT_REGISTER_DECLARE (videocrop);

typedef enum
{
  /* RGB (+ variants), ARGB (+ variants), AYUV, GRAY */
  VIDEO_CROP_PIXEL_FORMAT_PACKED_SIMPLE = 0,
  /* YVYU, YUY2, UYVY */
  VIDEO_CROP_PIXEL_FORMAT_PACKED_COMPLEX,
  /* I420, A420, YV12, Y444, Y42B, Y41B,
   * I420_10BE, A420_10BE, Y444_10BE, A444_10BE, I422_10BE, A422_10BE,
   * I420_10LE, A420_10LE, Y444_10LE, A444_10LE, I422_10LE, A422_10LE,
   * I420_12BE, Y444_12BE, I422_12BE,
   * I420_12LE, Y444_12LE, I422_12LE,
   * GBR, GBR_10BE, GBR_10LE, GBR_12BE, GBR_12LE,
   * GBRA, GBRA_10BE, GBRA_10LE, GBRA_12BE, GBRA_12LE */
  VIDEO_CROP_PIXEL_FORMAT_PLANAR,
  /* NV12, NV21 */
  VIDEO_CROP_PIXEL_FORMAT_SEMI_PLANAR
} VideoCropPixelFormat;

typedef struct _GstVideoCropImageDetails GstVideoCropImageDetails;

typedef struct _GstTensorVideoCrop GstTensorVideoCrop;
typedef struct _GstTensorVideoCropClass GstTensorVideoCropClass;

struct _GstTensorVideoCrop
{
  GstVideoFilter parent;

  /*< private > */
  gfloat prop_left;
  gfloat prop_top;
  gfloat prop_width;
  gfloat prop_height;
  gboolean need_update;

  GstPad *sinkpad_info;
  GstTensorsConfig tensors_config; /**< input tensors info */


  GstVideoInfo in_info;
  GstVideoInfo out_info;

  gint crop_left;
  gint crop_right;
  gint crop_top;
  gint crop_bottom;

  VideoCropPixelFormat packing;
  gint macro_y_off;

  gboolean raw_caps;
};

struct _GstTensorVideoCropClass
{
  GstVideoFilterClass parent_class;
};

GType gst_tensor_video_crop_get_type (void);

G_END_DECLS
#endif /* __GST_VIDEO_CROP_H__ */