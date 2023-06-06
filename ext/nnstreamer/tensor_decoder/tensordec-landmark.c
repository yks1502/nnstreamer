/**
 * GStreamer / NNStreamer tensor_decoder subplugin, "landmark detecting"
 * Copyright (C) 2023 Kiwoong Kim <helloing0119@naver.com>
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
 * @file        tensordec-landmark.c
 * @date        3 Jun 2023
 * @brief       NNStreamer tensor-decoder subplugin, "landmark detecting",
 *              which converts landmark detecting tensors to text stream.
 *
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Kiwoong Kim <helloing0119@naver.com>
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

void init_landmark (void) __attribute__ ((constructor));
void fini_landmark (void) __attribute__ ((destructor));

#define DECODER_LANDMARK_TEXT_CAPS_STR \
    "text/x-raw, format = (string) utf8"

#define LANDMARK_IDX_LOCATIONS_DEFAULT 0
#define LANDMARK_IDX_SCORES_DEFAULT 1
#define LANDMARK_THRESHOLD_DEFAULT G_MINFLOAT

typedef struct {
  float left;
  float top;
  float width;
  float height;
} detectedEye;

typedef struct
{
  float ymin;
  float xmin;
  float ymax;
  float xmax;
  float right_eye_x;
  float right_eye_y;
  float left_eye_x;
  float left_eye_y;
  float nose_x;
  float nose_y;
  float mouth_x;
  float mouth_y;
  float right_ear_x;
  float right_ear_y;
  float left_ear_x;
  float left_ear_y;
  float score;
} detectedFace;

#define _get_faces(_type, typename, locationinput, scoreinput, config, results) \
  case typename: \
  { \
    int d; \
    size_t boxbpi; \
    _type * locations_ = (_type *) locationinput; \
    _type * scores_ = (_type *) scoreinput; \
    int locations_idx =  LANDMARK_IDX_LOCATIONS_DEFAULT; \
    results = g_array_sized_new (FALSE, TRUE, sizeof (detectedFace), 896); \
    boxbpi = config->info.info[locations_idx].dimension[0]; \
    for (d = 0; d < 896; d++) { \
      detectedFace face; \
      if (scores_[d] < 80) \
        continue; \
      face.ymin = locations_[d * boxbpi + 1]; \
      face.xmin = locations_[d * boxbpi + 1]; \
      face.ymax = locations_[d * boxbpi + 1]; \
      face.xmax = locations_[d * boxbpi + 1]; \
      face.right_eye_x = locations_[d * boxbpi + 1]; \
      face.right_eye_y = locations_[d * boxbpi + 1]; \
      face.left_eye_x = locations_[d * boxbpi + 1]; \
      face.left_eye_y = locations_[d * boxbpi + 1]; \
      face.nose_x = locations_[d * boxbpi + 1]; \
      face.nose_y = locations_[d * boxbpi + 1]; \
      face.mouth_x = locations_[d * boxbpi + 1]; \
      face.mouth_y = locations_[d * boxbpi + 1]; \
      face.right_ear_x = locations_[d * boxbpi + 1]; \
      face.right_ear_y = locations_[d * boxbpi + 1]; \
      face.left_ear_x = locations_[d * boxbpi + 1]; \
      face.left_ear_y = locations_[d * boxbpi + 1]; \
      face.score = scores_[d]; \
      g_array_append_val (results, face); \
    } \
  } \
  break

#define _get_faces_(type, typename) \
  _get_faces (type, typename, (mem_locations->data), (mem_scores->data), config, results)


/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
landmark_init (void **pdata)
{
  UNUSED (pdata);
  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static void
landmark_exit (void **pdata)
{
  UNUSED (pdata);
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
landmark_setOption (void **pdata, int opNum, const char *param)
{
  UNUSED (pdata);
  UNUSED (opNum);
  UNUSED (param);
  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstCaps *
landmark_getOutCaps (void **pdata, const GstTensorsConfig * config)
{
//  const uint32_t *dim;
  GstCaps *caps;
//  int i;
//  UNUSED (pdata);
//
//  g_return_val_if_fail (config != NULL, NULL);
//  g_return_val_if_fail (config->info.num_tensors == 2, NULL);
//
//  /* Even if it's multi-tensor, we use the first tensor only in image labeling */
//  dim = config->info.info[0].dimension;
//  /* This allows N:1 only! */
//  g_return_val_if_fail (dim[0] > 0 && dim[1] == 1, NULL);
//  for (i = 2; i < NNS_TENSOR_RANK_LIMIT; i++)
//    g_return_val_if_fail (dim[i] == 1, NULL);

  caps = gst_caps_from_string (DECODER_LANDMARK_TEXT_CAPS_STR);
//  setFramerateFromConfig (caps, config);
  UNUSED (pdata);
  UNUSED (config);
  return caps;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static size_t
landmark_getTransformSize (void **pdata, const GstTensorsConfig * config,
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

/** @brief Search for max. Macro for tensor_element union */
#define search_max(type, i, max_index, max_val, bpe, data, num_data) \
do {\
  unsigned int i;\
  type *cursor = (type *) (data);\
  max_val = cursor[0];\
  max_index = 0;\
  for (i = 1; i < (num_data); i++) {\
    if (cursor[i] > (max_val)) {\
      max_val = cursor[i];\
      max_index = i;\
    }\
  }\
} while (0);

/** @brief Shorter case statement for search_max */
#define search_max_case(type, typename) \
case typename:\
  search_max(type, i, max_index, max_val._##type, bpe, input_data, num_data);\
  break;


static void
tensorize_face (GstMapInfo * out_info, GArray * results)
{
   /* Let's draw per pixel (4bytes) */
  float *out_tensor = (float *) out_info->data;
  unsigned int i, best_idx;
  float best_score = 0;
  detectedFace *best_face;
  
  for (i = 0; i < results->len; i++) {
    detectedFace *cur = &g_array_index (results, detectedFace, i);
    if (cur->score > best_score) {
      best_score = cur->score;
      best_idx = i;
    }
  }
  best_face = &g_array_index (results, detectedFace, best_idx);
  /** @todo: convert best_face to eye */
  out_tensor[0] = best_face->left_eye_x;
  out_tensor[1] = best_face->left_eye_y;
  out_tensor[2] = 0.2;
  out_tensor[3] = 0.2;
  out_tensor[4] = best_face->right_eye_x - 0.2;
  out_tensor[5] = best_face->right_eye_y - 0.2;
  out_tensor[2] = 0.2;
  out_tensor[3] = 0.2;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstFlowReturn
landmark_decode (void **pdata, const GstTensorsConfig * config,
    const GstTensorMemory * input, GstBuffer * outbuf)
{
  GstMapInfo out_info;
  GstMemory *out_mem;

  const GstTensorMemory *mem_locations, *mem_scores;
  gboolean need_output_alloc;

  GArray *results = NULL;
  const size_t size = (size_t) sizeof(float) * 8; /* 2 * [l, t, w, h] */

  int locations_idx, scores_idx;

  g_assert (outbuf);
  need_output_alloc = gst_buffer_get_size (outbuf) == 0;

  /* Ensure we have outbuf properly allocated */
  if (need_output_alloc) {
    out_mem = gst_allocator_alloc (NULL, size, NULL);
  } else {
    if (gst_buffer_get_size (outbuf) < size) {
      gst_buffer_set_size (outbuf, size);
    }
    out_mem = gst_buffer_get_all_memory (outbuf);
  }
  if (!gst_memory_map (out_mem, &out_info, GST_MAP_WRITE)) {
    ml_loge ("Cannot map output memory / tensordec-landmark.\n");
    goto error_free;
  }

  /** reset the buffer with 0.0 */
  memset (out_info.data, 0, size);
  locations_idx = LANDMARK_IDX_LOCATIONS_DEFAULT;
  scores_idx = LANDMARK_IDX_SCORES_DEFAULT;

  mem_locations = &input[locations_idx];
  mem_scores = &input[scores_idx];

  switch (config->info.info[locations_idx].type) {
      _get_faces_ (uint8_t, _NNS_UINT8);
      _get_faces_ (int8_t, _NNS_INT8);
      _get_faces_ (uint16_t, _NNS_UINT16);
      _get_faces_ (int16_t, _NNS_INT16);
      _get_faces_ (uint32_t, _NNS_UINT32);
      _get_faces_ (int32_t, _NNS_INT32);
      _get_faces_ (uint64_t, _NNS_UINT64);
      _get_faces_ (int64_t, _NNS_INT64);
      _get_faces_ (float, _NNS_FLOAT32);
      _get_faces_ (double, _NNS_FLOAT64);
    default:
      g_assert (0);
  }

  UNUSED (pdata);
  UNUSED (config);

  tensorize_face (&out_info, results);
  g_array_free (results, TRUE);

  gst_memory_unmap (out_mem, &out_info);

  if (need_output_alloc)
    gst_buffer_append_memory (outbuf, out_mem);
  else
    gst_memory_unref (out_mem);

  return GST_FLOW_OK;

error_free:
  gst_memory_unref (out_mem);

  return GST_FLOW_ERROR;
}

static gchar decoder_subplugin_landmark_detecting[] = "landmark_detecting";

/** @brief Image Labeling tensordec-plugin GstTensorDecoderDef instance */
static GstTensorDecoderDef landmarkDetecting = {
  .modename = decoder_subplugin_landmark_detecting,
  .init = landmark_init,
  .exit = landmark_exit,
  .setOption = landmark_setOption,
  .getOutCaps = landmark_getOutCaps,
  .getTransformSize = landmark_getTransformSize,
  .decode = landmark_decode
};

/** @brief Initialize this object for tensordec-plugin */
void
init_landmark (void)
{
  nnstreamer_decoder_probe (&landmarkDetecting);
}

/** @brief Destruct this object for tensordec-plugin */
void
fini_landmark (void)
{
  nnstreamer_decoder_exit (landmarkDetecting.modename);
}
