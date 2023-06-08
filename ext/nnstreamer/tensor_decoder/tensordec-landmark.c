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
#include <math.h>
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
    "other/tensor, " \
    "type = (string)float32"

// define DECODER_LANDMARK_TEXT_CAPS_STR 
//     "text/x-raw, format = (string) utf8, framerate=30/1"

#define LANDMARK_IDX_LOCATIONS_DEFAULT 0
#define LANDMARK_IDX_SCORES_DEFAULT 1
#define LANDMARK_THRESHOLD_DEFAULT G_MINFLOAT

#define SCORE_THRESHOLD 0.7
#define DIMENSION_TOTAL 896
#define INPUT_SIZE_DEFAULT 128
#define OFFSET_DEFAULT 0.5
#define NUM_LAYERS_DEFAULT 4
#define NUM_LAYERS_MAXIMUM 4
#define MIN_SCALE_DEFAULT 0.15625
#define MAX_SCALE_DEFAULT 0.75

#define SELECT_LEFT_EYE 0
#define SELECT_RIGHT_EYE 1

typedef struct {
  GstTensorsConfig config;
  GArray *anchors;
  gint selection;
  gint input_size;
  gint num_layers;
  gint strides[4];
  gfloat offset_x;
  gfloat offset_y;
  gfloat min_scale;
  gfloat max_scale;
  gint generated;
} landmarkPluginData;

typedef struct {
  float x_center;
  float y_center;
  float w;
  float h;
} anchor;

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
  int index;
} detectedFace;

#define _get_faces(_type, typename, locationinput, scoreinput, config, results) \
  case typename: \
  { \
    int d; \
    size_t boxbpi; \
    _type * locations_ = (_type *) locationinput; \
    _type * scores_ = (_type *) scoreinput; \
    int locations_idx =  LANDMARK_IDX_LOCATIONS_DEFAULT; \
    float threshold = log (SCORE_THRESHOLD / (1 - SCORE_THRESHOLD)); \
    results = g_array_sized_new (FALSE, TRUE, sizeof (detectedFace), DIMENSION_TOTAL); \
    boxbpi = config->info.info[locations_idx].dimension[0]; \
    for (d = 0; d < DIMENSION_TOTAL; d++) { \
      detectedFace face; \
      if (scores_[d] < threshold) \
        continue; \
      num_detection++; \
      face.ymin = locations_[d * boxbpi + 0]; \
      face.xmin = locations_[d * boxbpi + 1]; \
      face.ymax = locations_[d * boxbpi + 2]; \
      face.xmax = locations_[d * boxbpi + 3]; \
      face.right_eye_x = locations_[d * boxbpi + 4]; \
      face.right_eye_y = locations_[d * boxbpi + 5]; \
      face.left_eye_x = locations_[d * boxbpi + 6]; \
      face.left_eye_y = locations_[d * boxbpi + 7]; \
      face.nose_x = locations_[d * boxbpi + 8]; \
      face.nose_y = locations_[d * boxbpi + 9]; \
      face.mouth_x = locations_[d * boxbpi + 10]; \
      face.mouth_y = locations_[d * boxbpi + 11]; \
      face.right_ear_x = locations_[d * boxbpi + 12]; \
      face.right_ear_y = locations_[d * boxbpi + 13]; \
      face.left_ear_x = locations_[d * boxbpi + 14]; \
      face.left_ear_y = locations_[d * boxbpi + 15]; \
      face.score = 1.0 / (1.0 + exp (scores_[d])); \
      face.index = d; \
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
  landmarkPluginData *ldata; 
  ldata = *pdata = g_new0 (landmarkPluginData, 1);
  gst_tensors_config_init (&(ldata->config));
  ldata->anchors = g_array_new(FALSE, TRUE, sizeof(anchor));
  ldata->selection = SELECT_LEFT_EYE;
  ldata->input_size = INPUT_SIZE_DEFAULT;
  ldata->num_layers = NUM_LAYERS_DEFAULT;
  ldata->offset_x = OFFSET_DEFAULT;
  ldata->offset_y = OFFSET_DEFAULT;
  ldata->min_scale = MIN_SCALE_DEFAULT;
  ldata->max_scale = MAX_SCALE_DEFAULT;
  ldata->strides[0] = 8;
  ldata->strides[1] = 16;
  ldata->strides[2] = 16;
  ldata->strides[3] = 16;
  ldata->generated = 0;
  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static void
landmark_exit (void **pdata)
{
  landmarkPluginData *ldata = *pdata;
  gst_tensors_config_free(&(ldata->config));
  g_array_free(ldata->anchors, TRUE);
  g_free (*pdata);
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static int
landmark_setOption (void **pdata, int opNum, const char *param)
{
  landmarkPluginData *ldata = *pdata;
  const char *eye_selection[] = {
    [SELECT_LEFT_EYE] = "left",
    [SELECT_RIGHT_EYE] = "right",
  };
  if (opNum == 0) {
    /* option1 = eye selection */
    if (NULL == param || *param == '\0') {
      GST_ERROR ("Please set the valid mode at option1");
      return FALSE;
    }

    ldata->selection = find_key_strv (eye_selection, param);
    return TRUE;
  }
  return TRUE;
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstCaps *
landmark_getOutCaps (void **pdata, const GstTensorsConfig * config)
{
  GstCaps *caps;
  int i;
  landmarkPluginData * ldata = *pdata;
  GstTensorsConfig *p_config = &(ldata->config);
  if(p_config->info.info[0].dimension[0] != 4) {
    p_config->info.info[0].dimension[0] = 4;
    for (i = 1; i < NNS_TENSOR_RANK_LIMIT; i++) {
      p_config->info.info[0].dimension[i] = 1;
    }
    p_config->info.info[0].type = _NNS_FLOAT32;
    p_config->info.num_tensors = 1;
    p_config->info.format = 0;
  }
  caps = gst_tensor_caps_from_config (p_config);
  setFramerateFromConfig (caps, config);
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

/**
 * @brief Calculate anchor scale
 */
static gfloat
_calculate_scale (float min_scale, float max_scale, int stride_index, int num_strides)
{
  if (num_strides == 1) {
    return (min_scale + max_scale) * 0.5f;
  } else {
    return min_scale + (max_scale - min_scale) * 1.0 * stride_index / (num_strides - 1.0f);
  }
}

static void
_generate_anchors (landmarkPluginData *ldata)
{
  int layer_id = 0;
  int strides[NUM_LAYERS_MAXIMUM];
  int idx;
  guint i;

  gint num_layers = ldata->num_layers;
  gfloat offset_x = ldata->offset_x;
  gfloat offset_y = ldata->offset_y;

  for (idx = 0; idx < num_layers; idx++) {
    strides[idx] = ldata->strides[idx];
  }

  while (layer_id < num_layers) {
    GArray *aspect_ratios = g_array_new(FALSE, TRUE, sizeof(gfloat));
    GArray *scales = g_array_new(FALSE, TRUE, sizeof(gfloat));
    GArray *anchor_height = g_array_new(FALSE, TRUE, sizeof(gfloat));
    GArray *anchor_width = g_array_new(FALSE, TRUE, sizeof(gfloat));

    int last_same_stride_layer = layer_id;

    while (last_same_stride_layer < num_layers
           && strides[last_same_stride_layer] == strides[layer_id]) {
      gfloat scale;
      gfloat ratio = 1.0f;
      g_array_append_val(aspect_ratios, ratio);
      g_array_append_val(aspect_ratios, ratio);
      scale = _calculate_scale(ldata->min_scale, ldata->max_scale,
                                     last_same_stride_layer, num_layers);
      g_array_append_val(scales, scale);
      scale = _calculate_scale(ldata->min_scale, ldata->max_scale,
                                     last_same_stride_layer + 1, num_layers);
      g_array_append_val(scales, scale);
      last_same_stride_layer++;
    }

    for (i = 0; i < aspect_ratios->len; ++i) {
      const float ratio_sqrts = sqrt(g_array_index (aspect_ratios, gfloat, i));
      const gfloat sc = g_array_index (scales, gfloat, i);
      gfloat anchor_height_ = sc / ratio_sqrts;
      gfloat anchor_width_ = sc * ratio_sqrts;
      g_array_append_val(anchor_height, anchor_height_);
      g_array_append_val(anchor_width, anchor_width_);
    }

    {
      int feature_map_height = 0;
      int feature_map_width = 0;
      int x, y;
      int anchor_id;

      const int stride = strides[layer_id];
      feature_map_height = ceil(1.0f * INPUT_SIZE_DEFAULT / stride);
      feature_map_width = ceil(1.0f * INPUT_SIZE_DEFAULT / stride);

      for (y = 0; y < feature_map_height; ++y) {
        for (x = 0; x < feature_map_width; ++x) {
          for (anchor_id = 0; anchor_id < (int)aspect_ratios->len; ++anchor_id) {
            const float x_center = (x + offset_x) * 1.0f / feature_map_width;
            const float y_center = (y + offset_y) * 1.0f / feature_map_height;

            const anchor a = {.x_center = x_center, .y_center = y_center,
              .w = g_array_index (anchor_width, gfloat, anchor_id), .h = g_array_index (anchor_height, gfloat, anchor_id)};
            g_array_append_val(ldata->anchors, a);
          }
        }
      }
      layer_id = last_same_stride_layer;
    }

    g_array_free(aspect_ratios, FALSE);
  }
}

static void
tensorize_face (GstMapInfo * out_info, GArray * results, GArray * anchors, size_t hsize, int selection)
{
   /* Let's draw per pixel (4bytes) */
  float *out_tensor = (float *) out_info->data;
  float nose_y, nose_x, draw_x, draw_y, eye_x, eye_y;
  float x_center, y_center,  modifier, temp;
  unsigned int i, best_idx;
  float best_score = -1;
  detectedFace *best_face;
  anchor *current_anchor;
  
  for (i = 0; i < results->len; i++) {
    detectedFace *cur = &g_array_index (results, detectedFace, i);
    if (cur->score > best_score) {
      best_score = cur->score;
      best_idx = i;
    }
  }
  if(best_score > 0) {
    best_face = &g_array_index (results, detectedFace, best_idx);
    current_anchor = &g_array_index (anchors, anchor, best_face->index);

    if (selection == SELECT_LEFT_EYE) {
      eye_x = best_face->left_eye_x;
      eye_y = best_face->left_eye_y;
    } else {
      eye_x = best_face->right_eye_x;
      eye_y = best_face->right_eye_y;
    }
    nose_x = best_face->nose_x;
    nose_y = best_face->nose_y;

    x_center = current_anchor->x_center;
    y_center = current_anchor->y_center;

    eye_x = (eye_x + x_center * INPUT_SIZE_DEFAULT) / INPUT_SIZE_DEFAULT;
    eye_y = (eye_y + y_center * INPUT_SIZE_DEFAULT) / INPUT_SIZE_DEFAULT;
    nose_x = (nose_x + x_center * INPUT_SIZE_DEFAULT) / INPUT_SIZE_DEFAULT;
    nose_y = (nose_y + y_center * INPUT_SIZE_DEFAULT) / INPUT_SIZE_DEFAULT;

    modifier = (nose_x - eye_x) / 1.2;
    modifier = modifier < 0 ? -modifier : modifier;
    temp = (nose_y - eye_y) / 1.2;
    temp = temp < 0 ? -temp : temp;
    modifier = modifier > temp ? modifier : temp;

    draw_x = eye_x - modifier;
    draw_y = eye_y - modifier;
  
    out_tensor[hsize/4 + 0] = draw_x < 0 ? 0 : \
                              draw_x + (modifier * 2) >= 1 ? \
                              1 - (2 * modifier) : draw_x;
    out_tensor[hsize/4 + 1] = draw_y < 0 ? 0 : \
                              draw_y + (modifier * 2) >= 1 ? \
                              1 - (2 * modifier) : draw_y;
    out_tensor[hsize/4 + 2] = 2 * modifier;
    out_tensor[hsize/4 + 3] = 2 * modifier;
  } else {
    out_tensor[hsize/4 + 0] = 0.5;
    out_tensor[hsize/4 + 1] = 0.5;
    out_tensor[hsize/4 + 2] = 0.2;
    out_tensor[hsize/4 + 3] = 0.2;
  }

  GST_WARNING("\n\n%deye : %f %f\n\nout : %f %f %f %f\n\n", selection, eye_x, eye_y, out_tensor[hsize/4+0], out_tensor[hsize/4+1], out_tensor[hsize/4+2], out_tensor[hsize/4+3]);
}

/** @brief tensordec-plugin's GstTensorDecoderDef callback */
static GstFlowReturn
landmark_decode (void **pdata, const GstTensorsConfig * config,
    const GstTensorMemory * input, GstBuffer * outbuf)
{
  GstMapInfo out_info;
  GstMemory *out_mem;
  GstBuffer *temp_buf;
  gint num_detection = 0;
  GstTensorMetaInfo meta;
  landmarkPluginData * ldata = *pdata;
  GstTensorsConfig *p_config = &(ldata->config);
  int selection = ldata->selection;

  const GstTensorMemory *mem_locations, *mem_scores;
  gboolean need_output_alloc;
  GArray *results = NULL;
  size_t hsize, dsize, size;
  int locations_idx, scores_idx;

  if(!ldata->generated) {
    _generate_anchors(ldata);
  }

  p_config->rate_d = config->rate_d;
  p_config->rate_n = config->rate_n;

  gst_tensor_info_convert_to_meta(&p_config->info.info[0], &meta);
  hsize = gst_tensor_meta_info_get_header_size(&meta);
  dsize = gst_tensor_info_get_size(&(p_config->info.info[0]));
  size  = hsize + dsize;

  g_assert (outbuf);
  need_output_alloc = gst_buffer_get_size (outbuf) == 0;

  /* Ensure we have outbuf properly allocated */
  if (!need_output_alloc && gst_buffer_get_size (outbuf) < size) {
    gst_buffer_set_size (outbuf, size);
  }

  /** memory allocate */
  out_mem = gst_allocator_alloc (NULL, size, NULL);

  if (!gst_memory_map (out_mem, &out_info, GST_MAP_WRITE)) {
    gst_memory_unref (out_mem);
    ml_loge ("Cannot map output memory / tensordec-landmark.\n");
    return GST_FLOW_ERROR;
  }

  memset (out_info.data, 0, size);

  /** face detection */
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

  /** memory write */
  tensorize_face (&out_info, results, ldata->anchors, hsize, selection);
  memcpy(out_info.data, &meta, sizeof(GstTensorMetaInfo));
  g_array_free (results, TRUE);

  gst_memory_unmap (out_mem, &out_info);


  if (need_output_alloc) {
    gst_buffer_append_memory (outbuf, out_mem);
  } else {
    temp_buf = gst_buffer_new ();
    gst_buffer_append_memory(temp_buf, out_mem);
    gst_buffer_copy_into(outbuf, temp_buf, GST_BUFFER_COPY_DEEP,0,-1);
    // gst_buffer_unref (temp_buf);
    // gst_memory_unref (out_mem);
    // gst_memory_unref (meta_mem);
  }

  UNUSED (config);

  GST_WARNING("parse_eye : done");
  return GST_FLOW_OK;
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
