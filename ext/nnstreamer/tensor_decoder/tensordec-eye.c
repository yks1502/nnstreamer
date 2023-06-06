/**
 * GStreamer / NNStreamer tensor_decoder subplugin, "eye detecting"
 * Copyright (C) 2018 Jinhyuck Park <jinhyuck83.park@samsung.com>
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file        tensordec-eye.c
 * @date        25 April 2023
 * @brief       NNStreamer tensor-decoder subplugin, "eye detecting",
 *              which converts eye detecting tensors to text stream.
 *
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Hyeonwoo Koo <gusdn1397@gamil.com>
 * @bug         No known bugs except for NYI items
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glib.h>
#include <gst/gstinfo.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_log.h>
#include <nnstreamer_util.h>
#include "tensordecutil.h"

void init_eye (void) __attribute__ ((constructor));
void fini_eye (void) __attribute__ ((destructor));

#define DECODER_EYE_TEXT_CAPS_STR \
    "text/x-raw, format = (string) utf8"

#define DECODER_EYE_TENSOR_CAPS_STR \
    "other/tensors, num_tensors = (int) 1, types = (string) uint32, dimensions = (string) 154:1:1:1, format = (string) static"

/** @brief Internal data structure for output video width and height */
typedef struct
{
  guint width;
  guint height;
} SizeData;

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
eye_init (void **pdata)
{
  SizeData *data;
  data = *pdata = g_new0 (SizeData, 1);
  if (*pdata == NULL) {
    GST_ERROR ("Failed to allocate memory for decoder subplugin.");
    return FALSE;
  }

  data->width = 64;
  data->height = 64;

  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static void
eye_exit (void **pdata)
{
  g_free (*pdata);
  *pdata = NULL;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
eye_setOption (void **pdata, int opNum, const char *param)
{
  SizeData *data = *pdata;

  if (opNum == 0) {
    /* option1 = output video size (width:height) */
    tensor_dim dim;
    int rank = gst_tensor_parse_dimension (param, dim);

    data->width = 64;
    data->height = 64;
    if (param == NULL || *param == '\0')
      return TRUE;

    if (rank < 2) {
      GST_ERROR
          ("mode-option-1 of eye detection is video output dimension (WIDTH:HEIGHT). The given parameter, \"%s\", is not acceptable.",
          param);
      return TRUE;              /* Ignore this param */
    }
    if (rank > 2) {
      GST_WARNING
          ("mode-option-1 of pose estimation is video output dimension (WIDTH:HEIGHT). The third and later elements of the given parameter, \"%s\", are ignored.",
          param);
    }
    data->width = dim[0];
    data->height = dim[1];
    return TRUE;
  }

  GST_INFO ("Property mode-option-%d is ignored", opNum + 1);
  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstCaps *
eye_getOutCaps (void **pdata, const GstTensorsConfig * config)
{
  SizeData *data = *pdata;
  GstCaps *caps;
  int num_tensors, d, i;


  g_return_val_if_fail (config != NULL, NULL);

  /* check the size data */
  g_return_val_if_fail (data->width > 0, NULL);
  g_return_val_if_fail (data->height > 0, NULL);

  /* check whether the input tensors are consisted of 1D-arrays of 213 and 15 tensors.*/
  num_tensors = config->info.num_tensors;
  g_return_val_if_fail (num_tensors == 2, NULL);
  g_return_val_if_fail (config->info.info[0].dimension[0] == 213, NULL);
  g_return_val_if_fail (config->info.info[1].dimension[0] == 15, NULL);
  for (d = 0; d < num_tensors; d++) {
    for (i = 1; i < NNS_TENSOR_RANK_LIMIT; i++)
      g_return_val_if_fail (config->info.info[d].dimension[i] == 1, NULL);
  }

  /* set out capacities */
  caps = gst_caps_from_string (DECODER_EYE_TENSOR_CAPS_STR);
  setFramerateFromConfig (caps, config);

  return caps;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static size_t
eye_getTransformSize (void **pdata, const GstTensorsConfig * config,
    GstCaps * caps, size_t size, GstCaps * othercaps, GstPadDirection direction)
{
  UNUSED (pdata);
  UNUSED (config);
  UNUSED (caps);
  UNUSED (size);
  UNUSED (othercaps);
  UNUSED (direction);

  return 0;
  /** @todo Use max_word_length if that's appropriate */
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstFlowReturn
eye_decode (void **pdata, const GstTensorsConfig * config,
    const GstTensorMemory * input, GstBuffer * outbuf)
{
  GstMapInfo out_info;
  GstMemory *out_mem;
  size_t size;
  const GstTensorMemory  *data;
  SizeData *size_data = *pdata;
  int num_eye_tensor, num_pupil_tensor;
  int i;
  uint32_t *out_data;

  num_pupil_tensor = (config->info.info[0].dimension[0]);
  num_eye_tensor = (config->info.info[1].dimension[0]);

  /* [width, height, x_1, y_1, ..., x_76, y_76] */
  size = (size_t) sizeof(uint32_t) * 2 * (1 + num_eye_tensor + num_pupil_tensor);

  if (gst_buffer_get_size (outbuf) == 0) {
    out_mem = gst_allocator_alloc (NULL, size, NULL);
  } else {
    if (gst_buffer_get_size (outbuf) < size) {
      gst_buffer_set_size (outbuf, size);
    }
    out_mem = gst_buffer_get_all_memory (outbuf);
  }

  if (!gst_memory_map (out_mem, &out_info, GST_MAP_WRITE)) {
    ml_loge ("Cannot map output memory / tensordec-eye.\n");
    gst_memory_unref (out_mem);
    return GST_FLOW_ERROR;
  }

  out_data = (uint32_t *) out_info.data;

  out_data[0] = size_data->width;
  out_data[1] = size_data->height;

  /* @todo apply offset to each dot */
  data = &input[0];
  for (i = 0; i < num_eye_tensor; i++) {
    out_data[2 + 2*i] = ((float*) data->data)[3*i];
    out_data[2 + 2*i + 1] = ((float*) data->data)[3*i + 1];
  }

  data = &input[1];
  for (i = 0; i < num_pupil_tensor; i++) {
    out_data[(2 + 2*num_eye_tensor) + (2*i)] = ((float*) data->data)[3*i];
    out_data[(2 + 2*num_eye_tensor) + (2*i + 1)] = ((float*) data->data)[3*i + 1];
  }

  gst_memory_unmap (out_mem, &out_info);
  gst_buffer_append_memory (outbuf, out_mem);

  return GST_FLOW_OK;
}

static gchar decoder_subplugin_eye_detecting[] = "eye_detecting";

/** @brief Image Labeling tensordec-plugin GstTensorDecoderDef instance */
static GstTensorDecoderDef eyeDetecting = {
  .modename = decoder_subplugin_eye_detecting,
  .init = eye_init,
  .exit = eye_exit,
  .setOption = eye_setOption,
  .getOutCaps = eye_getOutCaps,
  .getTransformSize = eye_getTransformSize,
  .decode = eye_decode
};

/** @brief Initialize this object for tensordec-plugin */
void
init_eye (void)
{
  nnstreamer_decoder_probe (&eyeDetecting);
}

/** @brief Destruct this object for tensordec-plugin */
void
fini_eye (void)
{
  nnstreamer_decoder_exit (eyeDetecting.modename);
}
