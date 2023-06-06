/**
 * NNStreamer video frame cropping
 * 
 * this element implemented as modification of gstreamer/videocrop element,
 * to crop video frame using tensor instead of property
 * 
 * [sinkpads]
 * - tensor_videocrop.raw
 *   : video input
 * - tensor_videocrop.info
 *   : tensor 4:1:1:1 with [letf, top, width, height].
 *     each value should be in range of 0.0 ~ 1.0 (percentage) 
 * 
 * [srcpads]
 * - tensor_videocrop.src
 *   : video output
 * 
 * @see_also: https://github.com/GStreamer/gst-plugins-good/blob/master/gst/videocrop/gstvideocrop.c
 */

/**
 * @file	gsttensor_videocrop.c
 * @date	3 Jul 2023
 * @brief	GStreamer plugin to crop video using tensor
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Kiwoong Kim <helloing0119@naver.com>
 * @bug		need to test.
 *
 */
/**
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <nnstreamer_util.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_log.h>

#include "gsttensor_videocrop.h"
#include <gst/video/video.h>
#include <string.h>

GST_DEBUG_CATEGORY_STATIC (tensor_video_crop_debug);

#define GST_CAT_DEFAULT tensor_video_crop_debug

typedef struct
{
  guint num;
  gfloat left;
  gfloat top;
  gfloat width;
  gfloat height;
} tensor_video_crop_info_s;

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (VIDEO_CROP_CAPS)
    );

static GstStaticPadTemplate raw_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (VIDEO_CROP_CAPS)
    );

static GstStaticPadTemplate info_template = GST_STATIC_PAD_TEMPLATE ("info",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CROP_INFO_CAPS));

#define gst_tensor_video_crop_parent_class parent_class

G_DEFINE_TYPE (GstTensorVideoCrop, gst_tensor_video_crop, GST_TYPE_VIDEO_FILTER);
GST_ELEMENT_REGISTER_DEFINE (gst_tensor_video_crop, "tensor_videocrop", GST_RANK_NONE,
    GST_TYPE_TENSOR_VIDEO_CROP);

static void gst_tensor_video_crop_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_video_crop_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static void gst_tensor_video_crop_before_transform (GstBaseTransform * trans,
    GstBuffer * in);
static GstCaps *gst_tensor_video_crop_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter_caps);
static gboolean gst_tensor_video_crop_src_event (GstBaseTransform * trans,
    GstEvent * event);
static gboolean gst_tensor_video_crop_sink_event (GstPad * pad, GstObject * parent,
    GstEvent * event);

static gboolean gst_tensor_video_crop_sink_chain (GstPad * pad, GstObject * parent,
    GstBuffer * buf);

static gboolean gst_tensor_video_crop_set_info (GstVideoFilter * vfilter, GstCaps * in,
    GstVideoInfo * in_info, GstCaps * out, GstVideoInfo * out_info);
static GstFlowReturn gst_tensor_video_crop_transform_frame (GstVideoFilter * vfilter,
    GstVideoFrame * in_frame, GstVideoFrame * out_frame);

static gboolean gst_tensor_video_crop_decide_allocation (GstBaseTransform * trans,
    GstQuery * query);
static gboolean gst_tensor_video_crop_propose_allocation (GstBaseTransform * trans,
    GstQuery * decide_query, GstQuery * query);
static GstFlowReturn gst_tensor_video_crop_transform_ip (GstBaseTransform * trans,
    GstBuffer * buf);

static void
gst_tensor_video_crop_class_init (GstTensorVideoCropClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *element_class;
  GstBaseTransformClass *basetransform_class;
  GstVideoFilterClass *vfilter_class;

  gobject_class = (GObjectClass *) klass;
  element_class = (GstElementClass *) klass;
  basetransform_class = (GstBaseTransformClass *) klass;
  vfilter_class = (GstVideoFilterClass *) klass;

  gobject_class->set_property = gst_tensor_video_crop_set_property;
  gobject_class->get_property = gst_tensor_video_crop_get_property;

  gst_element_class_add_static_pad_template (element_class, &info_template);
  gst_element_class_add_static_pad_template (element_class, &raw_template);
  gst_element_class_add_static_pad_template (element_class, &src_template);
  gst_element_class_set_static_metadata (element_class,
      "TensorVideoCrop",
      "Crop/Video/Tensor",
      "Crops video into a tensor-defined region",
      "Kiwoong Kim");

  basetransform_class->before_transform =
      GST_DEBUG_FUNCPTR (gst_tensor_video_crop_before_transform);
  basetransform_class->transform_ip_on_passthrough = FALSE;
  basetransform_class->transform_caps =
      GST_DEBUG_FUNCPTR (gst_tensor_video_crop_transform_caps);
  basetransform_class->src_event = GST_DEBUG_FUNCPTR (gst_tensor_video_crop_src_event);
  basetransform_class->decide_allocation =
      GST_DEBUG_FUNCPTR (gst_tensor_video_crop_decide_allocation);
  basetransform_class->propose_allocation =
      GST_DEBUG_FUNCPTR (gst_tensor_video_crop_propose_allocation);
  basetransform_class->transform_ip =
      GST_DEBUG_FUNCPTR (gst_tensor_video_crop_transform_ip);

  vfilter_class->set_info = GST_DEBUG_FUNCPTR (gst_tensor_video_crop_set_info);
  vfilter_class->transform_frame =
      GST_DEBUG_FUNCPTR (gst_tensor_video_crop_transform_frame);
}

static void
gst_tensor_video_crop_init (GstTensorVideoCrop * tvcrop)
{
  GST_DEBUG_CATEGORY_INIT (tensor_video_crop_debug, "tensor_videocrop", 0, "videocrop");

  tvcrop->prop_left = -1;
  tvcrop->prop_top = -1;
  tvcrop->prop_width = -1;
  tvcrop->prop_height = -1;

  tvcrop->crop_left = 0;
  tvcrop->crop_right = 0;
  tvcrop->crop_top = 0;
  tvcrop->crop_bottom = 0;
}

#define ROUND_DOWN_2(n)  ((n)&(~1))

static void
gst_tensor_video_crop_transform_packed_complex (GstTensorVideoCrop * tvcrop,
    GstVideoFrame * in_frame, GstVideoFrame * out_frame, gint x, gint y)
{
  guint8 *in_data, *out_data;
  guint  dx;
  gint i, width, height;
  gint in_stride;
  gint out_stride;

  UNUSED (x);
  UNUSED (y);

  width = GST_VIDEO_FRAME_WIDTH (out_frame);
  height = GST_VIDEO_FRAME_HEIGHT (out_frame);

  in_data = GST_VIDEO_FRAME_PLANE_DATA (in_frame, 0);
  out_data = GST_VIDEO_FRAME_PLANE_DATA (out_frame, 0);

  in_stride = GST_VIDEO_FRAME_PLANE_STRIDE (in_frame, 0);
  out_stride = GST_VIDEO_FRAME_PLANE_STRIDE (out_frame, 0);

  in_data += tvcrop->crop_top * in_stride;

  /** init tensor infos */
  tvcrop->sinkpad_info =
      gst_pad_new_from_static_template (&info_template, "info");
  gst_element_add_pad (GST_ELEMENT (tvcrop), tvcrop->sinkpad_info);
  gst_pad_set_event_function (tvcrop->sinkpad_info,
    GST_DEBUG_FUNCPTR (gst_tensor_video_crop_sink_event));
  gst_pad_set_chain_function (tvcrop->sinkpad_info,
    GST_DEBUG_FUNCPTR (gst_tensor_video_crop_sink_chain));

  gst_tensors_config_init (&tvcrop->tensors_config);

  /* rounding down here so we end up at the start of a macro-pixel and not
   * in the middle of one */
  in_data += ROUND_DOWN_2 (tvcrop->crop_left) *
      GST_VIDEO_FRAME_COMP_PSTRIDE (in_frame, 0);

  dx = width * GST_VIDEO_FRAME_COMP_PSTRIDE (out_frame, 0);

  /* UYVY = 4:2:2 - [U0 Y0 V0 Y1] [U2 Y2 V2 Y3] [U4 Y4 V4 Y5]
   * YUYV = 4:2:2 - [Y0 U0 Y1 V0] [Y2 U2 Y3 V2] [Y4 U4 Y5 V4] = YUY2 */
  if ((tvcrop->crop_left % 2) != 0) {
    for (i = 0; i < height; ++i) {
      gint j;

      memcpy (out_data, in_data, dx);

      /* move just the Y samples one pixel to the left, don't worry about
       * chroma shift */
      for (j = tvcrop->macro_y_off; j < out_stride - 2; j += 2)
        out_data[j] = in_data[j + 2];

      in_data += in_stride;
      out_data += out_stride;
    }
  } else {
    for (i = 0; i < height; ++i) {
      memcpy (out_data, in_data, dx);
      in_data += in_stride;
      out_data += out_stride;
    }
  }
}

static void
gst_tensor_video_crop_transform_packed_simple (GstTensorVideoCrop * tvcrop,
    GstVideoFrame * in_frame, GstVideoFrame * out_frame, gint x, gint y)
{
  guint8 *in_data, *out_data;
  gint width, height;
  guint dx;
  gint i, in_stride, out_stride;

  width = GST_VIDEO_FRAME_WIDTH (out_frame);
  height = GST_VIDEO_FRAME_HEIGHT (out_frame);

  in_data = GST_VIDEO_FRAME_PLANE_DATA (in_frame, 0);
  out_data = GST_VIDEO_FRAME_PLANE_DATA (out_frame, 0);

  in_stride = GST_VIDEO_FRAME_PLANE_STRIDE (in_frame, 0);
  out_stride = GST_VIDEO_FRAME_PLANE_STRIDE (out_frame, 0);

  in_data += (tvcrop->crop_top + y) * in_stride;
  in_data +=
      (tvcrop->crop_left + x) * GST_VIDEO_FRAME_COMP_PSTRIDE (in_frame, 0);

  dx = width * GST_VIDEO_FRAME_COMP_PSTRIDE (out_frame, 0);

  for (i = 0; i < height; ++i) {
    memcpy (out_data, in_data, dx);
    in_data += in_stride;
    out_data += out_stride;
  }
}

static void
gst_tensor_video_crop_transform_planar (GstTensorVideoCrop * tvcrop,
    GstVideoFrame * in_frame, GstVideoFrame * out_frame, gint x, gint y)
{
  const GstVideoFormatInfo *format_info;
  gint crop_top, crop_left;
  guint p;

  format_info = in_frame->info.finfo;
  crop_left = tvcrop->crop_left + x;
  crop_top = tvcrop->crop_top + y;

  for (p = 0; p < GST_VIDEO_FRAME_N_PLANES (in_frame); ++p) {
    guint8 *plane_in, *plane_out;
    guint sub_w_factor, sub_h_factor;
    guint subsampled_crop_left, subsampled_crop_top;
    guint copy_width;
    gint i;
    gsize bytes_per_pixel;

    /* plane */
    plane_in = GST_VIDEO_FRAME_PLANE_DATA (in_frame, p);
    plane_out = GST_VIDEO_FRAME_PLANE_DATA (out_frame, p);

    /* To support > 8bit, we need to add a byte-multiplier that specifies
     * how many bytes are used per pixel value */
    bytes_per_pixel = GST_VIDEO_FRAME_COMP_PSTRIDE (in_frame, p);

    /* apply crop top/left
     * crop_top and crop_left have to be rounded down to the corresponding
     * subsampling factor, since, e.g.: the first line in a subsampled plane
     * describes 2 lines in the actual image. A crop_top of 1 thus should
     * not shift the pointer of the input plane. */
    sub_w_factor = 1 << GST_VIDEO_FORMAT_INFO_W_SUB (format_info, p);
    sub_h_factor = 1 << GST_VIDEO_FORMAT_INFO_H_SUB (format_info, p);
    subsampled_crop_left = GST_ROUND_DOWN_N ((guint) crop_left, sub_w_factor);
    subsampled_crop_top = GST_ROUND_DOWN_N ((guint) crop_top, sub_h_factor);

    plane_in +=
        GST_VIDEO_FORMAT_INFO_SCALE_HEIGHT (format_info, p,
        subsampled_crop_top) * GST_VIDEO_FRAME_PLANE_STRIDE (in_frame, p);
    plane_in +=
        GST_VIDEO_FORMAT_INFO_SCALE_WIDTH (format_info, p,
        subsampled_crop_left) * bytes_per_pixel;
    copy_width = GST_VIDEO_FRAME_COMP_WIDTH (out_frame, p) * bytes_per_pixel;


    for (i = 0; i < GST_VIDEO_FRAME_COMP_HEIGHT (out_frame, p); ++i) {
      memcpy (plane_out, plane_in, copy_width);
      plane_in += GST_VIDEO_FRAME_PLANE_STRIDE (in_frame, p);
      plane_out += GST_VIDEO_FRAME_PLANE_STRIDE (out_frame, p);
    }
  }
}

static void
gst_tensor_video_crop_transform_semi_planar (GstTensorVideoCrop * tvcrop,
    GstVideoFrame * in_frame, GstVideoFrame * out_frame, gint x, gint y)
{
  gint width, height;
  gint i, crop_top, crop_left;
  guint8 *y_out, *uv_out;
  guint8 *y_in, *uv_in;
  guint dx;

  width = GST_VIDEO_FRAME_WIDTH (out_frame);
  height = GST_VIDEO_FRAME_HEIGHT (out_frame);
  crop_left = tvcrop->crop_left + x;
  crop_top = tvcrop->crop_top + y;

  /* Y plane */
  y_in = GST_VIDEO_FRAME_PLANE_DATA (in_frame, 0);
  y_out = GST_VIDEO_FRAME_PLANE_DATA (out_frame, 0);

  /* UV plane */
  uv_in = GST_VIDEO_FRAME_PLANE_DATA (in_frame, 1);
  uv_out = GST_VIDEO_FRAME_PLANE_DATA (out_frame, 1);

  y_in += crop_top * GST_VIDEO_FRAME_PLANE_STRIDE (in_frame, 0) + crop_left;
  dx = width;

  for (i = 0; i < height; ++i) {
    memcpy (y_out, y_in, dx);
    y_in += GST_VIDEO_FRAME_PLANE_STRIDE (in_frame, 0);
    y_out += GST_VIDEO_FRAME_PLANE_STRIDE (out_frame, 0);
  }

  uv_in += (crop_top / 2) * GST_VIDEO_FRAME_PLANE_STRIDE (in_frame, 1);
  uv_in += GST_ROUND_DOWN_2 (crop_left);
  dx = GST_ROUND_UP_2 (width);

  for (i = 0; i < GST_ROUND_UP_2 (height) / 2; i++) {
    memcpy (uv_out, uv_in, dx);
    uv_in += GST_VIDEO_FRAME_PLANE_STRIDE (in_frame, 1);
    uv_out += GST_VIDEO_FRAME_PLANE_STRIDE (out_frame, 1);
  }
}

static GstFlowReturn
gst_tensor_video_crop_transform_frame (GstVideoFilter * vfilter,
    GstVideoFrame * in_frame, GstVideoFrame * out_frame)
{
  GstTensorVideoCrop *tvcrop = GST_TENSOR_VIDEO_CROP (vfilter);
  GstVideoCropMeta *meta = gst_buffer_get_video_crop_meta (in_frame->buffer);
  gint x = 0, y = 0;

  if (G_UNLIKELY (tvcrop->need_update)) {
    if (!gst_tensor_video_crop_set_info (vfilter, NULL, &tvcrop->in_info, NULL,
            &tvcrop->out_info)) {
      return GST_FLOW_ERROR;
    }
  }

  if (meta) {
    x = meta->x;
    y = meta->y;
  }

  switch (tvcrop->packing) {
    case VIDEO_CROP_PIXEL_FORMAT_PACKED_SIMPLE:
      gst_tensor_video_crop_transform_packed_simple (tvcrop, in_frame, out_frame, x, y);
      break;
    case VIDEO_CROP_PIXEL_FORMAT_PACKED_COMPLEX:
      gst_tensor_video_crop_transform_packed_complex (tvcrop, in_frame, out_frame, x,
          y);
      break;
    case VIDEO_CROP_PIXEL_FORMAT_PLANAR:
      gst_tensor_video_crop_transform_planar (tvcrop, in_frame, out_frame, x, y);
      break;
    case VIDEO_CROP_PIXEL_FORMAT_SEMI_PLANAR:
      gst_tensor_video_crop_transform_semi_planar (tvcrop, in_frame, out_frame, x, y);
      break;
    default:
      g_assert_not_reached ();
  }

  return GST_FLOW_OK;
}

static gboolean
gst_tensor_video_crop_decide_allocation (GstBaseTransform * trans, GstQuery * query)
{
  GstTensorVideoCrop *crop = GST_TENSOR_VIDEO_CROP (trans);
  gboolean use_crop_meta;

  use_crop_meta = (gst_query_find_allocation_meta (query,
          GST_VIDEO_CROP_META_API_TYPE, NULL) &&
      gst_query_find_allocation_meta (query, GST_VIDEO_META_API_TYPE, NULL));

  if ((crop->crop_left | crop->crop_right | crop->crop_top | crop->
          crop_bottom) == 0) {
    GST_INFO_OBJECT (crop, "we are using passthrough");
    gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (crop), TRUE);
    gst_base_transform_set_in_place (GST_BASE_TRANSFORM (crop), FALSE);
  } else if (use_crop_meta) {
    GST_INFO_OBJECT (crop, "we are doing in-place transform using crop meta");
    gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (crop), FALSE);
    gst_base_transform_set_in_place (GST_BASE_TRANSFORM (crop), TRUE);
  } else if (crop->raw_caps) {
    GST_INFO_OBJECT (crop, "we are not using passthrough");
    gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (crop), FALSE);
    gst_base_transform_set_in_place (GST_BASE_TRANSFORM (crop), FALSE);
  } else {
    GST_ELEMENT_ERROR (crop, STREAM, WRONG_TYPE,
        ("Dowstream doesn't support crop for non-raw caps"), (NULL));
    return FALSE;
  }

  return GST_BASE_TRANSFORM_CLASS (parent_class)->decide_allocation (trans,
      query);
}

static gboolean
gst_tensor_video_crop_propose_allocation (GstBaseTransform * trans,
    GstQuery * decide_query, GstQuery * query)
{
  /* if we are not passthrough, we can handle video meta and crop meta */
  if (decide_query) {
    GST_DEBUG_OBJECT (trans, "Advertising video meta and crop meta support");
    gst_query_add_allocation_meta (query, GST_VIDEO_META_API_TYPE, NULL);
    gst_query_add_allocation_meta (query, GST_VIDEO_CROP_META_API_TYPE, NULL);
  }

  return GST_BASE_TRANSFORM_CLASS (parent_class)->propose_allocation (trans,
      decide_query, query);
}

static void
gst_tensor_video_crop_before_transform (GstBaseTransform * trans, GstBuffer * in)
{
  GstTensorVideoCrop *self = GST_TENSOR_VIDEO_CROP (trans);
  GstClockTime timestamp, stream_time;

  timestamp = GST_BUFFER_TIMESTAMP (in);
  stream_time =
      gst_segment_to_stream_time (&trans->segment, GST_FORMAT_TIME, timestamp);

  GST_DEBUG_OBJECT (self, "sync to %" GST_TIME_FORMAT,
      GST_TIME_ARGS (timestamp));

  if (GST_CLOCK_TIME_IS_VALID (stream_time))
    gst_object_sync_values (GST_OBJECT (self), stream_time);
}

static GstFlowReturn
gst_tensor_video_crop_transform_ip (GstBaseTransform * trans, GstBuffer * buf)
{
  GstTensorVideoCrop *self = GST_TENSOR_VIDEO_CROP (trans);
  GstVideoFilter *vfilter = GST_VIDEO_FILTER (trans);
  GstVideoMeta *video_meta;
  GstVideoCropMeta *crop_meta;

  GST_LOG_OBJECT (trans, "Transforming in-place");

  if (G_UNLIKELY (self->need_update)) {
    if (!gst_tensor_video_crop_set_info (vfilter, NULL, &self->in_info, NULL,
            &self->out_info)) {
      return GST_FLOW_ERROR;
    }
  }

  /* The video meta is required since we are going to make the caps
   * width/height smaller, which would not result in a usable GstVideoInfo for
   * mapping the buffer. */
  video_meta = gst_buffer_get_video_meta (buf);
  if (!video_meta) {
    video_meta = gst_buffer_add_video_meta (buf, GST_VIDEO_FRAME_FLAG_NONE,
        GST_VIDEO_INFO_FORMAT (&self->in_info), self->in_info.width,
        self->in_info.height);
  }

  crop_meta = gst_buffer_get_video_crop_meta (buf);
  if (!crop_meta)
    crop_meta = gst_buffer_add_video_crop_meta (buf);

  crop_meta->x += self->crop_left;
  crop_meta->y += self->crop_top;
  crop_meta->width = GST_VIDEO_INFO_WIDTH (&self->out_info);
  crop_meta->height = GST_VIDEO_INFO_HEIGHT (&self->out_info);

  return GST_FLOW_OK;
}

static gint
gst_tensor_video_crop_transform_dimension (gint val, gint delta)
{
  gint64 new_val = (gint64) val + (gint64) delta;

  new_val = CLAMP (new_val, 1, G_MAXINT);

  return (gint) new_val;
}

static gboolean
gst_tensor_video_crop_transform_dimension_value (const GValue * src_val,
    gint delta, GValue * dest_val, GstPadDirection direction, gboolean dynamic)
{
  gboolean ret = TRUE;

  if (G_VALUE_HOLDS_INT (src_val)) {
    gint ival = g_value_get_int (src_val);
    ival = gst_tensor_video_crop_transform_dimension (ival, delta);

    if (dynamic) {
      if (direction == GST_PAD_SRC) {
        if (ival == G_MAXINT) {
          g_value_init (dest_val, G_TYPE_INT);
          g_value_set_int (dest_val, ival);
        } else {
          g_value_init (dest_val, GST_TYPE_INT_RANGE);
          gst_value_set_int_range (dest_val, ival, G_MAXINT);
        }
      } else {
        if (ival == 1) {
          g_value_init (dest_val, G_TYPE_INT);
          g_value_set_int (dest_val, ival);
        } else {
          g_value_init (dest_val, GST_TYPE_INT_RANGE);
          gst_value_set_int_range (dest_val, 1, ival);
        }
      }
    } else {
      g_value_init (dest_val, G_TYPE_INT);
      g_value_set_int (dest_val, ival);
    }
  } else if (GST_VALUE_HOLDS_INT_RANGE (src_val)) {
    gint min = gst_value_get_int_range_min (src_val);
    gint max = gst_value_get_int_range_max (src_val);

    min = gst_tensor_video_crop_transform_dimension (min, delta);
    max = gst_tensor_video_crop_transform_dimension (max, delta);

    if (dynamic) {
      if (direction == GST_PAD_SRC)
        max = G_MAXINT;
      else
        min = 1;
    }

    if (min == max) {
      g_value_init (dest_val, G_TYPE_INT);
      g_value_set_int (dest_val, min);
    } else {
      g_value_init (dest_val, GST_TYPE_INT_RANGE);
      gst_value_set_int_range (dest_val, min, max);
    }
  } else if (GST_VALUE_HOLDS_LIST (src_val)) {
    guint i;

    g_value_init (dest_val, GST_TYPE_LIST);

    for (i = 0; i < gst_value_list_get_size (src_val); ++i) {
      const GValue *list_val;
      GValue newval = G_VALUE_INIT;

      list_val = gst_value_list_get_value (src_val, i);
      if (gst_tensor_video_crop_transform_dimension_value (list_val, delta, &newval,
              direction, dynamic))
        gst_value_list_append_value (dest_val, &newval);
      g_value_unset (&newval);
    }

    if (gst_value_list_get_size (dest_val) == 0) {
      g_value_unset (dest_val);
      ret = FALSE;
    }
  } else {
    ret = FALSE;
  }

  return ret;
}

/** TODO */
static GstCaps *
gst_tensor_video_crop_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter_caps)
{
  GstTensorVideoCrop *self;
  GstCaps *other_caps;
  gint dy, dx, i, left, right, bottom, top, width, height;
  gboolean w_dynamic, h_dynamic;

  self = GST_TENSOR_VIDEO_CROP (trans);

  GST_OBJECT_LOCK (self);

  GST_LOG_OBJECT (self, "l=%f,t=%f,w=%f,h=%f",
      self->prop_left, self->prop_top, self->prop_width, self->prop_height);

  width = self->in_info.width;
  height = self->in_info.height;

  w_dynamic = (self->prop_left < 0 || self->prop_width < 0);
  h_dynamic = (self->prop_top < 0 || self->prop_height < 0);

  left = (self->prop_left < 0) ? 0 : (int) (self->prop_left * width);
  right = (self->prop_width < 0) ? 0 : left + (int)(self->prop_width * width);
  top = (self->prop_top < 0) ? 0 : (int) (self->prop_top * height);
  bottom = (self->prop_height < 0) ? 0 : top + (int)(self->prop_height * height);

  GST_OBJECT_UNLOCK (self);

  if (direction == GST_PAD_SRC) {
    dx = left + right;
    dy = top + bottom;
  } else {
    dx = 0 - (left + right);
    dy = 0 - (top + bottom);
  }

  GST_LOG_OBJECT (self, "transforming caps %" GST_PTR_FORMAT, caps);

  other_caps = gst_caps_new_empty ();

  for (i = 0; i < (gint) gst_caps_get_size (caps); ++i) {
    const GValue *v;
    GstStructure *structure, *new_structure;
    GValue w_val = G_VALUE_INIT, h_val = G_VALUE_INIT;
    GstCapsFeatures *features;

    structure = gst_caps_get_structure (caps, i);
    features = gst_caps_get_features (caps, i);

    v = gst_structure_get_value (structure, "width");
    if (!gst_tensor_video_crop_transform_dimension_value (v, dx, &w_val, direction,
            w_dynamic)) {
      GST_WARNING_OBJECT (self, "could not transform width value with dx=%d"
          ", caps structure=%" GST_PTR_FORMAT, dx, structure);
      continue;
    }

    v = gst_structure_get_value (structure, "height");
    if (!gst_tensor_video_crop_transform_dimension_value (v, dy, &h_val, direction,
            h_dynamic)) {
      g_value_unset (&w_val);
      GST_WARNING_OBJECT (self, "could not transform height value with dy=%d"
          ", caps structure=%" GST_PTR_FORMAT, dy, structure);
      continue;
    }

    new_structure = gst_structure_copy (structure);
    gst_structure_set_value (new_structure, "width", &w_val);
    gst_structure_set_value (new_structure, "height", &h_val);
    g_value_unset (&w_val);
    g_value_unset (&h_val);

    GST_LOG_OBJECT (self, "transformed structure %2d: %" GST_PTR_FORMAT
        " => %" GST_PTR_FORMAT "features %" GST_PTR_FORMAT, i, structure,
        new_structure, features);
    gst_caps_append_structure (other_caps, new_structure);

    gst_caps_set_features (other_caps, i, gst_caps_features_copy (features));
  }

  if (!gst_caps_is_empty (other_caps) && filter_caps) {
    GstCaps *tmp = gst_caps_intersect_full (filter_caps, other_caps,
        GST_CAPS_INTERSECT_FIRST);
    gst_caps_replace (&other_caps, tmp);
    gst_caps_unref (tmp);
  }

  return other_caps;
}

/** TODO */
static gboolean
gst_tensor_video_crop_set_info (GstVideoFilter * vfilter, GstCaps * in,
    GstVideoInfo * in_info, GstCaps * out, GstVideoInfo * out_info)
{
  GstTensorVideoCrop *self = GST_TENSOR_VIDEO_CROP (vfilter);
  GstCapsFeatures *features;
  int dx, dy, width, height, left, top;
  width = GST_VIDEO_INFO_WIDTH (in_info);
  height = GST_VIDEO_INFO_HEIGHT (in_info);

  GST_OBJECT_LOCK (self);
  self->need_update = FALSE;
  self->crop_left = left = (self->prop_left < 0) ? 0 : (int) (self->prop_left * width);
  self->crop_right = (self->prop_width < 0) ? 0 : left + (int)(self->prop_width * width);
  self->crop_top = top = (self->prop_top < 0) ? 0 : (int) (self->prop_top * height);
  self->crop_bottom = (self->prop_height < 0) ? 0 : top + (int)(self->prop_height * height);
  GST_OBJECT_UNLOCK (self);

  dx = GST_VIDEO_INFO_WIDTH (in_info) - GST_VIDEO_INFO_WIDTH (out_info);
  dy = GST_VIDEO_INFO_HEIGHT (in_info) - GST_VIDEO_INFO_HEIGHT (out_info);

  if (self->crop_left <= 0 && self->crop_right <= 0) {
    self->crop_left = dx / 2;
    self->crop_right = dx / 2 + (dx & 1);
  } else if (self->crop_left <= 0) {
    if (G_UNLIKELY (self->crop_right > dx))
      goto cropping_too_much;
    self->crop_left = dx - self->crop_right;
  } else if (self->crop_right <= 0) {
    if (G_UNLIKELY (self->crop_left > dx))
      goto cropping_too_much;
    self->crop_right = dx - self->crop_left;
  }

  if (self->crop_top <= 0 && self->crop_bottom <= 0) {
    self->crop_top = dy / 2;
    self->crop_bottom = dy / 2 + (dy & 1);
  } else if (self->crop_top <=0) {
    if (G_UNLIKELY (self->crop_bottom > dy))
      goto cropping_too_much;
    self->crop_top = dy - self->crop_bottom;
  } else if (self->crop_bottom <= 0) {
    if (G_UNLIKELY (self->crop_top > dy))
      goto cropping_too_much;
    self->crop_bottom = dy - self->crop_top;
  }

  if (G_UNLIKELY ((self->crop_left + self->crop_right) >=
          GST_VIDEO_INFO_WIDTH (in_info)
          || (self->crop_top + self->crop_bottom) >=
          GST_VIDEO_INFO_HEIGHT (in_info)))
    goto cropping_too_much;

  if (in && out)
    GST_LOG_OBJECT (self, "incaps = %" GST_PTR_FORMAT ", outcaps = %"
        GST_PTR_FORMAT, in, out);

  if (in) {
    features = gst_caps_get_features (in, 0);
    self->raw_caps = gst_caps_features_is_equal (features,
        GST_CAPS_FEATURES_MEMORY_SYSTEM_MEMORY);
  }

  if (!self->raw_caps)
    goto beach;

  switch (GST_VIDEO_INFO_FORMAT (in_info)) {
    case GST_VIDEO_FORMAT_RGB:
    case GST_VIDEO_FORMAT_BGR:
    case GST_VIDEO_FORMAT_RGB16:
    case GST_VIDEO_FORMAT_RGB15:
    case GST_VIDEO_FORMAT_RGBx:
    case GST_VIDEO_FORMAT_xRGB:
    case GST_VIDEO_FORMAT_BGRx:
    case GST_VIDEO_FORMAT_xBGR:
    case GST_VIDEO_FORMAT_RGBA:
    case GST_VIDEO_FORMAT_ARGB:
    case GST_VIDEO_FORMAT_BGRA:
    case GST_VIDEO_FORMAT_ABGR:
    case GST_VIDEO_FORMAT_GRAY8:
    case GST_VIDEO_FORMAT_GRAY16_LE:
    case GST_VIDEO_FORMAT_GRAY16_BE:
    case GST_VIDEO_FORMAT_AYUV:
      self->packing = VIDEO_CROP_PIXEL_FORMAT_PACKED_SIMPLE;
      break;
    case GST_VIDEO_FORMAT_YVYU:
    case GST_VIDEO_FORMAT_YUY2:
    case GST_VIDEO_FORMAT_UYVY:
      self->packing = VIDEO_CROP_PIXEL_FORMAT_PACKED_COMPLEX;
      if (GST_VIDEO_INFO_FORMAT (in_info) == GST_VIDEO_FORMAT_UYVY) {
        /* UYVY = 4:2:2 - [U0 Y0 V0 Y1] [U2 Y2 V2 Y3] [U4 Y4 V4 Y5] */
        self->macro_y_off = 1;
      } else {
        /* YUYV = 4:2:2 - [Y0 U0 Y1 V0] [Y2 U2 Y3 V2] [Y4 U4 Y5 V4] = YUY2 */
        self->macro_y_off = 0;
      }
      break;
    case GST_VIDEO_FORMAT_I420:
    case GST_VIDEO_FORMAT_I420_10BE:
    case GST_VIDEO_FORMAT_I420_10LE:
    case GST_VIDEO_FORMAT_I420_12BE:
    case GST_VIDEO_FORMAT_I420_12LE:
    case GST_VIDEO_FORMAT_A420:
    case GST_VIDEO_FORMAT_A420_10BE:
    case GST_VIDEO_FORMAT_A420_10LE:
    case GST_VIDEO_FORMAT_YV12:
    case GST_VIDEO_FORMAT_Y444:
    case GST_VIDEO_FORMAT_Y444_10BE:
    case GST_VIDEO_FORMAT_Y444_10LE:
    case GST_VIDEO_FORMAT_Y444_12BE:
    case GST_VIDEO_FORMAT_Y444_12LE:
    case GST_VIDEO_FORMAT_A444_10BE:
    case GST_VIDEO_FORMAT_A444_10LE:
    case GST_VIDEO_FORMAT_Y42B:
    case GST_VIDEO_FORMAT_I422_10BE:
    case GST_VIDEO_FORMAT_I422_10LE:
    case GST_VIDEO_FORMAT_A422_10BE:
    case GST_VIDEO_FORMAT_A422_10LE:
    case GST_VIDEO_FORMAT_I422_12BE:
    case GST_VIDEO_FORMAT_I422_12LE:
    case GST_VIDEO_FORMAT_GBR:
    case GST_VIDEO_FORMAT_GBR_10BE:
    case GST_VIDEO_FORMAT_GBR_10LE:
    case GST_VIDEO_FORMAT_GBR_12BE:
    case GST_VIDEO_FORMAT_GBR_12LE:
    case GST_VIDEO_FORMAT_GBRA:
    case GST_VIDEO_FORMAT_GBRA_10BE:
    case GST_VIDEO_FORMAT_GBRA_10LE:
    case GST_VIDEO_FORMAT_GBRA_12BE:
    case GST_VIDEO_FORMAT_GBRA_12LE:
    case GST_VIDEO_FORMAT_Y41B:
      self->packing = VIDEO_CROP_PIXEL_FORMAT_PLANAR;
      break;
    case GST_VIDEO_FORMAT_NV12:
    case GST_VIDEO_FORMAT_NV21:
      self->packing = VIDEO_CROP_PIXEL_FORMAT_SEMI_PLANAR;
      break;
    default:
      goto unknown_format;
  }

beach:
  self->in_info = *in_info;
  self->out_info = *out_info;

  /* Ensure our decide_allocation will be called again when needed */
  if (gst_base_transform_is_passthrough (GST_BASE_TRANSFORM (self))) {
    gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (self), FALSE);
    gst_base_transform_set_in_place (GST_BASE_TRANSFORM (self), FALSE);
  }

  return TRUE;

  /* ERROR */
cropping_too_much:
  {
    GST_WARNING_OBJECT (self, "we are cropping too much");
    return FALSE;
  }
unknown_format:
  {
    GST_WARNING_OBJECT (self, "Unsupported format");
    return FALSE;
  }
}

/* called with object lock */
static inline void
gst_tensor_video_crop_set_crop (GstTensorVideoCrop * vcrop, gfloat new_value, gfloat * prop)
{
  if (*prop != new_value) {
    *prop = new_value;
    vcrop->need_update = TRUE;
  }
}

static void
gst_tensor_video_crop_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorVideoCrop *self;
  UNUSED (value);
  self = GST_TENSOR_VIDEO_CROP (object);

  GST_OBJECT_LOCK (self);
  switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }

  GST_OBJECT_UNLOCK (self);

  gst_base_transform_reconfigure_src (GST_BASE_TRANSFORM (self));
}

static void
gst_tensor_video_crop_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstTensorVideoCrop *self;
  UNUSED (value);

  self = GST_TENSOR_VIDEO_CROP (object);

  GST_OBJECT_LOCK (self);
  switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
  GST_OBJECT_UNLOCK (self);
}

static gboolean
gst_tensor_video_crop_src_event (GstBaseTransform * trans, GstEvent * event)
{
  return GST_BASE_TRANSFORM_CLASS (parent_class)->src_event (trans, event);
}

static gboolean
gst_tensor_video_crop_parse_caps (GstTensorVideoCrop * tvcrop, GstCaps * caps)
{
  GstStructure *structure;
  GstTensorsConfig *config;

  config = &tvcrop->tensors_config;

  structure = gst_caps_get_structure (caps, 0);
  gst_tensors_config_from_structure (config, structure);

  return gst_tensors_config_validate (config);
}

static gboolean
gst_tensor_video_crop_sink_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  GstTensorVideoCrop *self;

  self = GST_TENSOR_VIDEO_CROP (parent);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps *caps;
      gst_event_parse_caps (event, &caps);
      gst_tensor_video_crop_parse_caps (self, caps);
      break;
    }
    default:
      break;
  }

  return gst_pad_event_default (pad, parent, event);
}


static gboolean
gst_tensor_video_crop_get_info (GstTensorVideoCrop * self, GstBuffer * info,
    tensor_video_crop_info_s * cinfo)
{
  GstMemory *mem;
  GstMapInfo map;
  GstTensorMetaInfo meta;
  gsize hsize, dsize, esize;
  guint i;
  gboolean ret = FALSE;

  i = gst_buffer_n_memory (info);
  g_assert (i > 0);
  if (i > 1) {
    GST_WARNING_OBJECT (self,
        "Info buffer has %u memories, parse first one.", i);
  }

  mem = gst_buffer_peek_memory (info, 0);
  if (!gst_memory_map (mem, &map, GST_MAP_READ)) {
    GST_ERROR_OBJECT (self, "Failed to map the info buffer.");
    return FALSE;
  }

  /* parse crop-info from flex tensor */
  if (!gst_tensor_meta_info_parse_header (&meta, map.data)) {
    GST_ERROR_OBJECT (self, "Failed to get the meta from info buffer.");
    goto done;
  }

  hsize = gst_tensor_meta_info_get_header_size (&meta);
  dsize = gst_tensor_meta_info_get_data_size (&meta);
  esize = gst_tensor_get_element_size (meta.type);

  if (hsize + dsize != map.size) {
    GST_ERROR_OBJECT (self,
        "Invalid meta info, info buffer size is incorrect (received %zd, expected %zd).",
        map.size, hsize + dsize);
    goto done;
  }


  g_assert ((dsize % (esize * 4)) == 0);
  memset (cinfo, 0, sizeof (tensor_video_crop_info_s));

  cinfo->num = dsize / (esize * sizeof (gfloat));
  cinfo->num = MIN (cinfo->num, NNS_TENSOR_SIZE_LIMIT);


  /** hard cording to match cinfo->members*/
  gst_tensor_data_raw_typecast (
    map.data + hsize + (esize * 0),
    meta.type, (guint8 *) (&cinfo->left), _NNS_FLOAT32);
  gst_tensor_data_raw_typecast (
    map.data + hsize + (esize * 1),
    meta.type, (guint8 *) (&cinfo->top), _NNS_FLOAT32);
  gst_tensor_data_raw_typecast (
    map.data + hsize + (esize * 2),
    meta.type, (guint8 *) (&cinfo->width), _NNS_FLOAT32);
  gst_tensor_data_raw_typecast (
    map.data + hsize + (esize * 3),
    meta.type, (guint8 *) (&cinfo->height), _NNS_FLOAT32);
  ret = TRUE;

done:
  gst_memory_unmap (mem, &map);
  return ret;
}

static gboolean
gst_tensor_video_crop_sink_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  GstFlowReturn ret = GST_FLOW_OK;
  GstTensorVideoCrop *self;
  tensor_video_crop_info_s vcinfo;
  guint num_tensors;
  UNUSED (pad);
  self = GST_TENSOR_VIDEO_CROP (parent);

  buf = gst_tensor_buffer_from_config (buf, &self->tensors_config);

  if (gst_tensors_config_is_flexible (&self->tensors_config)) {
    /* cannot get exact number of tensors from config */
    num_tensors = gst_buffer_n_memory (buf);
  } else {
    num_tensors = self->tensors_config.info.num_tensors;

    /* supposed n memory blocks in buffer */
    g_assert (gst_buffer_n_memory (buf) == num_tensors);
  }
  GST_DEBUG_OBJECT (self, " Number of Tensors: %d", num_tensors);

  if (!gst_tensor_video_crop_get_info (self, buf, &vcinfo)) {
    ret = GST_FLOW_ERROR; 
  } else {
    
    GST_OBJECT_LOCK (self);
    self->prop_left = vcinfo.left;
    self->prop_top = vcinfo.top;
    self->prop_width = vcinfo.width;
    self->prop_height = vcinfo.height;
    GST_OBJECT_UNLOCK (self);
  }
  if (buf)
    gst_buffer_unref (buf);

  return ret;


}