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

/* font.c */
extern uint8_t rasters[][13];

#define NUM_EYE_TENSOR                           (71)
#define NUM_PUPIL_TENSOR                         (5)
#define DEFAULT_WIDTH                            (64)
#define DEFAULT_HEIGHT                           (64)
#define EYE_PIXEL_VALUE                          (0xFF0000FF)    /* RED 100% in RGBA */
#define PUPIL_PIXEL_VALUE                        (0x0000FFFF)    /* BLUE 100% in RGBA */

#define DECODER_EYE_TEXT_CAPS_STR \
    "text/x-raw, format = (string) utf8"

#define DECODER_EYE_TENSOR_CAPS_STR \
    "other/tensors, num_tensors = (int) 1, types = (string) uint32, dimensions = (string) 154:1:1:1, format = (string) static"

/**
 * @todo Fill in the value at build time or hardcode this. It's const value
 * @brief The bitmap of characters
 * [Character (ASCII)][Height][Width]
 */
static singleLineSprite_t singleLineSprite;

/** @brief Internal data structure for output video width and height */
typedef struct
{
  guint width;
  guint height;

  GArray *eye_tensor_x;
  GArray *eye_tensor_y;

  GArray *pupil_tensor_x;
  GArray *pupil_tensor_y;
} eye_data;

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
eye_init (void **pdata)
{
  eye_data *data;
  data = *pdata = g_new0 (eye_data, 1);
  if (*pdata == NULL) {
    GST_ERROR ("Failed to allocate memory for decoder subplugin.");
    return FALSE;
  }

  data->width = DEFAULT_WIDTH;
  data->height = DEFAULT_HEIGHT;

  data->eye_tensor_x =
    g_array_sized_new (TRUE, TRUE, sizeof (guint), NUM_EYE_TENSOR);
  data->eye_tensor_y =
    g_array_sized_new (TRUE, TRUE, sizeof (guint), NUM_EYE_TENSOR);

  data->pupil_tensor_x =
    g_array_sized_new (TRUE, TRUE, sizeof (guint), NUM_PUPIL_TENSOR);
  data->pupil_tensor_y =
    g_array_sized_new (TRUE, TRUE, sizeof (guint), NUM_PUPIL_TENSOR);

  initSingleLineSprite (singleLineSprite, rasters, EYE_PIXEL_VALUE);

  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static void
eye_exit (void **pdata)
{
  eye_data *data = *pdata;

  g_array_free (data->eye_tensor_x, TRUE);
  g_array_free (data->eye_tensor_y, TRUE);
  g_array_free (data->pupil_tensor_x, TRUE);
  g_array_free (data->pupil_tensor_y, TRUE);

  g_free (*pdata);
  *pdata = NULL;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
eye_setOption (void **pdata, int opNum, const char *param)
{
  eye_data *data = *pdata;

  if (opNum == 0) {
    /* option1 = output video size (width:height) */
    tensor_dim dim;
    int rank = gst_tensor_parse_dimension (param, dim);

    data->width = DEFAULT_WIDTH;
    data->height = DEFAULT_HEIGHT;
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
  eye_data *data = *pdata;
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

/**
 * @brief Draw with the given results (eye) to the output buffer
 * @param[out] out_info The output buffer (RGBA plain)
 * @param[in] data The eye internal data.
 */
static void
draw (GstMapInfo * out_info, eye_data * data)
{
  uint32_t *frame = (uint32_t *) out_info->data;
  guint i;

  for (i = 0; i < data->eye_tensor_x->len; i++) {
    guint *pos_x = &g_array_index (data->eye_tensor_x, guint, i);
    guint *pos_y = &g_array_index (data->eye_tensor_y, guint, i);

    uint32_t *pos = &frame[(*pos_y) * data->width + (*pos_x)];
    *pos = EYE_PIXEL_VALUE;
  }

  for (i = 0; i < data->pupil_tensor_x->len; i++) {
    guint *pos_x = &g_array_index (data->pupil_tensor_x, guint, i);
    guint *pos_y = &g_array_index (data->pupil_tensor_y, guint, i);

    uint32_t *pos = &frame[(*pos_y) * data->width + (*pos_x)];
    *pos = PUPIL_PIXEL_VALUE;
  }
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstFlowReturn
eye_decode (void **pdata, const GstTensorsConfig * config,
    const GstTensorMemory * input, GstBuffer * outbuf)
{
  eye_data *data = *pdata;
  const size_t size = (size_t) data->width * data->height * 4;
  GstMapInfo out_info;
  GstMemory *out_mem;
  const GstTensorMemory *in_data;
  guint i, num_pupil_tensor, num_eye_tensor;

  num_pupil_tensor = (config->info.info[0].dimension[0]);
  num_eye_tensor = (config->info.info[1].dimension[0]);

  g_array_set_size (data->eye_tensor_x, num_eye_tensor);
  g_array_set_size (data->eye_tensor_y, num_eye_tensor);
  g_array_set_size (data->pupil_tensor_x, num_pupil_tensor);
  g_array_set_size (data->pupil_tensor_y, num_pupil_tensor);

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
    ml_loge ("Cannot map output memory / tensordec-eye.\n");
    gst_memory_unref (out_mem);
    return GST_FLOW_ERROR;
  }
  /** reset the buffer with alpha 0 / black */
  memset (out_info.data, 0, size);

  /* @todo apply offset to each dot */
  in_data = &input[0];
  for (i = 0; i < num_eye_tensor; i++) {
    guint *pos_x = &g_array_index (data->eye_tensor_x, guint, i);
    guint *pos_y = &g_array_index (data->eye_tensor_y, guint, i);
    *pos_x = (guint)((float*) in_data->data)[3 * i];
    *pos_y = (guint)((float*) in_data->data)[3 * i + 1];
  }

  in_data = &input[1];
  for (i = 0; i < num_pupil_tensor; i++) {
    guint *pos_x = &g_array_index (data->pupil_tensor_x, guint, i);
    guint *pos_y = &g_array_index (data->pupil_tensor_y, guint, i);
    *pos_x = (guint)((float*) in_data->data)[3 * i];
    *pos_y = (guint)((float*) in_data->data)[3 * i + 1];
  }

  draw (&out_info, data);

  gst_memory_unmap (out_mem, &out_info);
  if (gst_buffer_get_size (outbuf) == 0)
    gst_buffer_append_memory (outbuf, out_mem);
  else
    gst_memory_unref (out_mem);

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
