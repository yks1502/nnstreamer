/*
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Alternatively, the contents of this file may be used under the
 * GNU Lesser General Public License Version 2.1 (the "LGPL"), in
 * which case the following provisions apply instead of the ones
 * mentioned above:
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * @file	tensor_transform.c
 * @date	10 Jul 2018
 * @brief	GStreamer plugin to transform other/tensor dimensions
 *
 * @see		http://github.com/nnsuite/nnstreamer
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs.
 *
 */

#ifndef __GST_TENSOR_TRANSFORM_H__
#define __GST_TENSOR_TRANSFORM_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <tensor_common.h>

G_BEGIN_DECLS

/* #defines don't like whitespacey bits */
#define GST_TYPE_TENSOR_TRANSFORM \
  (gst_tensor_transform_get_type())
#define GST_TENSOR_TRANSFORM(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_TRANSFORM,GstTensor_Transform))
#define GST_TENSOR_TRANSFORM_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_TRANSFORM,GstTensor_TransformClass))
#define GST_IS_TENSOR_TRANSFORM(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_TRANSFORM))
#define GST_IS_TENSOR_TRANSFORM_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_TRANSFORM))
#define GST_TENSOR_TRANSFORM_CAST(obj)  ((GstTensor_Transform *)(obj))

typedef struct _GstTensor_Transform GstTensor_Transform;

typedef struct _GstTensor_TransformClass GstTensor_TransformClass;

typedef enum {
  GTT_DIMCHG = 0, /* Dimension Change. "dimchg" */


  GTT_END,
} tensor_transform_mode;

typedef struct _tensor_transform_dimchg {
  int from;
  int to;
} tensor_transform_dimchg;

/**
 * @brief Internal data structure for tensor_transform instances.
 */
struct _GstTensor_Transform
{
  GstBaseTransform element;	/**< This is the parent object */

  gboolean silent;	/**< True if logging is minimized */
  tensor_transform_mode mode; /**< Transform mode. GTT_END if invalid */
  gchar *option; /**< Stored option value */
  union {
    tensor_transform_dimchg data_dimchg; /**< Parsed option value for "dimchg" mode */
  };
  gboolean loaded; /**< TRUE if mode & option are loaded */

  tensor_dim fromDim; /**< Input dimension */
  tensor_dim toDim; /**< Output dimension */
  tensor_type type; /**< tensor_type of both input and output. They share the same type. */
};

/*
 * @brief GstTensor_TransformClass inherits GstBaseTransformClass.
 *
 * Referring another child (sibiling), GstVideoFilter (abstract class) and
 * its child (concrete class) GstVideoTransform.
 * Note that GstTensor_TransformClass is a concrete class; thus we need to look at both.
 */
struct _GstTensor_TransformClass
{
  GstBaseTransformClass parent_class;	/**< Inherits GstBaseTransformClass */
};

/*
 * @brief Get Type function required for gst elements
 */
GType gst_tensor_transform_get_type (void);

G_END_DECLS

#endif /* __GST_TENSOR_TRANSFORM_H__ */
