/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer/NNStreamer tensor_drawer
 * Copyright (C) 2023 Hyeonwoo Koo <gusdn1397@gamil.com>
 */
/**
 * @file	gsttensor_drawer.h
 * @date	7 June 2023
 * @brief	GStreamer plugin to help debug tensor streams.
 *
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Hyeonwoo Koo <gusdn1397@gamil.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_TENSOR_DRAWER_H__
#define __GST_TENSOR_DRAWER_H__

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <tensor_common.h>

G_BEGIN_DECLS

#define GST_TYPE_TENSOR_DRAWER \
  (gst_tensor_drawer_get_type())
#define GST_TENSOR_DRAWER(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_DRAWER,GstTensorDrawer))
#define GST_TENSOR_DRAWER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_DRAWER,GstTensorDrawerClass))
#define GST_IS_TENSOR_DRAWER(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_DRAWER))
#define GST_IS_TENSOR_DRAWER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_DRAWER))
#define GST_TENSOR_DRAWER_CAST(obj)  ((GstTensorDrawer *)(obj))

typedef struct _GstTensorDrawer GstTensorDrawer;
typedef struct _GstTensorDrawerClass GstTensorDrawerClass;

/**
 * @brief Internal data structure for tensor_drawer instances.
 */
struct _GstTensorDrawer
{
  GstBaseTransform element;	/**< This is the parent object */

  GstPad *sinkpad; /**< sink pad */
  GstPad *srcpad; /**< src pad */

  guint width; /**< width of the output frame */
  guint height; /**< height of the output frame */
};

/**
 * @brief GstTensorDrawerClass data structure.
 */
struct _GstTensorDrawerClass
{
  GstBaseTransformClass parent_class; /**< parent class = transform */
};

/**
 * @brief Get Type function required for gst elements
 */
GType gst_tensor_drawer_get_type (void);


G_END_DECLS

#endif /** __GST_TENSOR_DRAWER */
