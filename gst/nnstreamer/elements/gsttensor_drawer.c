/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer/NNStreamer tensor_drawer
 * Copyright (C) 2023 Hyeonwoo Koo <gusdn1397@gamil.com>
 */
/**
 * @file	gsttensor_drawer.c
 * @date	7 June 2023
 * @brief	GStreamer plugin to help drawing dots with tensor streams.
 *
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Hyeonwoo Koo <gusdn1397@gamil.com>
 */

/**
 * SECTION:element-tensor_drawer
 *
 * A filter that generates a frame with dots from tensor streams.
 * point of the given pipeline. An application writer using an nnstreamer
 * pipeline can use tensor_drawer to draw dots on the specific width and height.
 *
 * Note that this does not support other/tensor, but only supports other/tensors.
 *
 */

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG TRUE
#endif

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include <nnstreamer_log.h>
#include <nnstreamer_util.h>
#include "gsttensor_drawer.h"

GST_DEBUG_CATEGORY_STATIC (gst_tensor_drawer_debug);
#define GST_CAT_DEFAULT gst_tensor_drawer_debug

/**
 * This is a new element created after the obsoletion of other/tensor.
 * Use other/tensors if you want to use tensor_drawer
 */
#define CAPS_STRING "other/tensors, num_tensors=1, types=uint32, dimensions=154:1:1:1, format=static"
#define CAPS_SINK_STRING GST_TENSOR_CAP_DEFAULT ";" GST_TENSORS_CAP_WITH_NUM ("1")
#define VIDEO_CROP_CAPS                                \
  "video/x-raw, format = RGBA" 
/**
 * @brief The capabilities of the inputs
 */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CAPS_SINK_STRING));

/**
 * @brief The capabilities of the outputs
 */
static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (VIDEO_CROP_CAPS));

/**
 * @brief tensor_drawer properties
 */
enum
{
  PROP_0,
  PROP_SIZE,
};

#define C_FLAGS(v) ((guint) v)

#define gst_tensor_drawer_parent_class parent_class
G_DEFINE_TYPE (GstTensorDrawer, gst_tensor_drawer, GST_TYPE_BASE_TRANSFORM);

/* gobject vmethods */
static void gst_tensor_drawer_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_tensor_drawer_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);
static void gst_tensor_drawer_finalize (GObject * object);

/* gstbasetransform vmethods */
static GstFlowReturn gst_tensor_drawer_transform_ip (GstBaseTransform * trans,
    GstBuffer * buffer);
static GstCaps *gst_tensor_drawer_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps);
static gboolean gst_tensor_drawer_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps);

/**
 * @brief Initialize the tensor_drawer's class.
 */
static void
gst_tensor_drawer_class_init (GstTensorDrawerClass * klass)
{
  GObjectClass *object_class;
  GstElementClass *element_class;
  GstBaseTransformClass *trans_class;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_drawer_debug, "tensor_drawer", 0,
      "Element to draw a frame with dots indicated by tensor.");

  trans_class = (GstBaseTransformClass *) klass;
  object_class = (GObjectClass *) klass;
  element_class = (GstElementClass *) klass;

  /* GObjectClass vmethods */
  object_class->set_property = gst_tensor_drawer_set_property;
  object_class->get_property = gst_tensor_drawer_get_property;
  object_class->finalize = gst_tensor_drawer_finalize;

  /**
   * GstTensorDrawer::size:
   *
   * The string to set the size of frame.
   */
  g_object_class_install_property (object_class, PROP_SIZE,
      g_param_spec_string("size", "Size of the output frame", "Size of the output frame. 'width:height' e.g. 640:480",
          "", G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /* set pad template */
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&sink_factory));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&src_factory));

  gst_element_class_set_static_metadata (element_class,
      "TensorDrawer",
      "Filter/Tensor",
      "Help draw a frame with dots indicated by tensors.",
      "Hyeonwoo Koo <gusdn1397@gmail.com>");

  /* GstBaseTransform vmethods */
  trans_class->transform_ip = GST_DEBUG_FUNCPTR (gst_tensor_drawer_transform_ip);

  trans_class->fixate_caps = GST_DEBUG_FUNCPTR (gst_tensor_drawer_fixate_caps);
  trans_class->set_caps = GST_DEBUG_FUNCPTR (gst_tensor_drawer_set_caps);

  /* GstBaseTransform Property */
  trans_class->passthrough_on_same_caps = TRUE;
      /** This won't modify the contents! */
  trans_class->transform_ip_on_passthrough = TRUE;
      /** call transform_ip although it's passthrough */

  /**
   * Note.
   * Without transform_caps and with passthrough_on_same_caps = TRUE,
   * This element is not allowed to touch the contents, but can inspect
   * the contents with transform_ip by setting transform_ip_on_passthrough.
   */
}

/**
 * @brief Initialize tensor_drawer element.
 */
static void
gst_tensor_drawer_init (GstTensorDrawer * self)
{
  /** init properties */
  self->width = 0;
  self->height = 0;
}

/**
 * @brief Function to finalize instance.
 */
static void
gst_tensor_drawer_finalize (GObject * object)
{
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Setter for tensor_drawer properties.
 */
static void
gst_tensor_drawer_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorDrawer *self = GST_TENSOR_DRAWER (object);

  switch (prop_id) {
    case PROP_SIZE:
      gchar *param = g_value_dup_string (value);
      tensor_dim dim;
      int rank = gst_tensor_parse_dimension (param, dim);
      if (param == NULL || *param == '\0') {
        goto size_error;
      }

      if (rank < 2) {
        GST_ERROR
          ("mode-option-1 of eye detection is video output dimension (WIDTH:HEIGHT). The given parameter, \"%s\", is not acceptable.",
          param);
        goto size_error;
      }
      if (rank > 2) {
        GST_WARNING
          ("mode-option-1 of pose estimation is video output dimension (WIDTH:HEIGHT). The third and later elements of the given parameter, \"%s\", are ignored.",
          param);
      }
      self->width = dim[0];
      self->height = dim[1];
      silent_debug (self, "Set width = %d, height = %d", self->width, self->height);
size_error:
      g_free(param);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Getter for tensor_drawer properties.
 */
static void
gst_tensor_drawer_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorDrawer *self = GST_TENSOR_DRAWER (object);

  switch (prop_id) {
    case PROP_SIZE:
      if (self->width > 0 && self->height > 0) {
        gchar *param = g_strdup_printf ("%d:%d", self->width, self->height);
        g_value_set_string (value, param);
        g_free(param);
      } else {
        g_value_set_string (value, "");
      } 
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief The core function that provides the output frame based
 *        on the contents.
 */
static void
_gst_tensor_drawer_output (GstTensorDrawer * self, GstBuffer * outbuf)
{
  GstMemory *out_mem;
  GstMapInfo out_info;
  guint i;
  uint32_t *frame;
  const size_t size = (size_t) self->width * self->height * 4;   /* RGBA */

  /** @todo NYI: do the debug task */
  g_assert (outbuf); /** GST Internal Bug */
  /* Ensure we have outbuf properly allocated */
  if (gst_buffer_get_size (outbuf) == 0) {
    out_mem = gst_allocator_alloc (NULL, size, NULL);
  } else {
    if (gst_buffer_get_size (outbuf) < size) {
      gst_buffer_set_size (outbuf, size);
    }
    out_mem = gst_buffer_get_all_memory (outbuf);
  }
  if (!gst_memory_map (out_mem, &out_info, GST_MAP_WRITE)) {
    gst_memory_unref (out_mem);
    ml_loge ("Cannot map output memory / tensordec-pose.\n");
    return;
  }
  /** reset the buffer with alpha 0 / black */
  memset (out_info.data, 0, size);

  frame = (uint32_t *) out_info.data;

  for (i=0; i < self->width; i++) {
    frame[((self->height)/2)*(self->width) + i] = 0xFFFFFFFF;
  }

  gst_memory_unmap (out_mem, &out_info);
  if (gst_buffer_get_size (outbuf) == 0)
    gst_buffer_append_memory (outbuf, out_mem);
  else
    gst_memory_unref (out_mem);
}

/**
 * @brief in-place transform
 */
static GstFlowReturn
gst_tensor_drawer_transform_ip (GstBaseTransform * trans, GstBuffer * buffer)
{
  GstTensorDrawer *self = GST_TENSOR_DRAWER (trans);

  _gst_tensor_drawer_output (self, buffer);

  return GST_FLOW_OK;
}

/**
 * @brief fixate caps. required vmethod of GstBaseTransform.
 */
static GstCaps *
gst_tensor_drawer_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps)
{
  UNUSED (trans);
  UNUSED (direction);
  UNUSED (caps);
  UNUSED (othercaps);

  return gst_caps_fixate (othercaps);
}

/**
 * @brief set caps. required vmethod of GstBaseTransform.
 */
static gboolean
gst_tensor_drawer_set_caps (GstBaseTransform * trans,
    GstCaps * in_caps, GstCaps * out_caps)
{
  UNUSED (trans);
  UNUSED (in_caps);
  UNUSED (out_caps);

  return TRUE;// gst_caps_can_intersect (in_caps, out_caps);
}
