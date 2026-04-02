/**
 * @file tennis.cpp
 * @brief 网球检测核心实现
 */

#include "tennis.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "cviruntime.h"

#define LOGI(fmt, ...) printf("[INFO] " fmt "\n", ##__VA_ARGS__)
#define LOGE(fmt, ...) printf("[ERROR] " fmt "\n", ##__VA_ARGS__)

// ============================================================================
// 内部数据结构
// ============================================================================

typedef struct {
    float x, y, w, h;
} Box;

typedef struct {
    Box bbox;
    float score;
} Detection;

// ============================================================================
// 全局变量 (模型上下文)
// ============================================================================

static CVI_MODEL_HANDLE g_model = NULL;
static CVI_TENSOR* g_input_tensors = NULL;
static CVI_TENSOR* g_output_tensors = NULL;
static int32_t g_input_num = 0;
static int32_t g_output_num = 0;
static int g_model_w = 0;
static int g_model_h = 0;
static CVI_SHAPE g_output_shape;
static bool g_initialized = false;

static const float CONF_THRESHOLD = 0.5f;
static const float IOU_THRESHOLD = 0.5f;

// ============================================================================
// 内部辅助函数
// ============================================================================

static float calculateIoU(const Box& a, const Box& b) {
    float area_a = a.w * a.h;
    float area_b = b.w * b.h;
    
    float inter_w = std::min(a.x + a.w / 2, b.x + b.w / 2) - 
                    std::max(a.x - a.w / 2, b.x - b.w / 2);
    float inter_h = std::min(a.y + a.h / 2, b.y + b.h / 2) - 
                    std::max(a.y - a.h / 2, b.y - b.h / 2);
    
    float inter_area = std::max(inter_w, 0.0f) * std::max(inter_h, 0.0f);
    return inter_area / (area_a + area_b - inter_area);
}

static void applyNMS(std::vector<Detection>& dets, float thresh) {
    if (dets.empty()) return;
    
    std::sort(dets.begin(), dets.end(), 
              [](const Detection& a, const Detection& b) { 
                  return a.score > b.score; 
              });
    
    std::vector<bool> removed(dets.size(), false);
    
    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (removed[j]) continue;
            if (calculateIoU(dets[i].bbox, dets[j].bbox) > thresh) {
                removed[j] = true;
            }
        }
    }
    
    std::vector<Detection> result;
    for (size_t i = 0; i < dets.size(); ++i) {
        if (!removed[i]) {
            result.push_back(dets[i]);
        }
    }
    dets = std::move(result);
}

static void mapBoxesToOriginalImage(std::vector<Detection>& dets, 
                                     int img_h, int img_w, 
                                     int input_h, int input_w) {
    float scale = std::min((float)input_w / img_w, (float)input_h / img_h);
    int pad_left = (input_w - (int)(img_w * scale)) / 2;
    int pad_top = (input_h - (int)(img_h * scale)) / 2;

    for (auto& det : dets) {
        float x1 = det.bbox.x - det.bbox.w / 2;
        float y1 = det.bbox.y - det.bbox.h / 2;
        float x2 = det.bbox.x + det.bbox.w / 2;
        float y2 = det.bbox.y + det.bbox.h / 2;

        x1 = std::max(0.0f, (x1 - pad_left) / scale);
        y1 = std::max(0.0f, (y1 - pad_top) / scale);
        x2 = std::min((float)img_w, (x2 - pad_left) / scale);
        y2 = std::min((float)img_h, (y2 - pad_top) / scale);

        det.bbox.x = (x1 + x2) / 2;
        det.bbox.y = (y1 + y2) / 2;
        det.bbox.w = x2 - x1;
        det.bbox.h = y2 - y1;
    }
}

static cv::Mat preprocessImage(const cv::Mat& src, int target_w, int target_h) {
    cv::Mat dst = cv::Mat::zeros(target_h, target_w, CV_8UC3);
    
    float scale = std::min((float)target_w / src.cols, (float)target_h / src.rows);
    int new_w = (int)(src.cols * scale);
    int new_h = (int)(src.rows * scale);
    int pad_left = (target_w - new_w) / 2;
    int pad_top = (target_h - new_h) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));
    resized.copyTo(dst(cv::Rect(pad_left, pad_top, new_w, new_h)));

    return dst;
}

static int parseDetections(CVI_TENSOR* output, 
                           int input_h, int input_w,
                           CVI_SHAPE output_shape, 
                           float conf_thresh,
                           std::vector<Detection>& dets) {
    if (!output) return 0;

    float* data = (float*)CVI_NN_TensorPtr(&output[0]);
    if (!data) return 0;

    int batch = output_shape.dim[0];
    int total_anchors = output_shape.dim[2];
    
    const float strides[3] = {8.0f, 16.0f, 32.0f};
    int anchor_counts[3];
    for (int i = 0; i < 3; i++) {
        anchor_counts[i] = (input_h / strides[i]) * (input_w / strides[i]);
    }

    int count = 0;
    
    for (int b = 0; b < batch; b++) {
        int anchor_offset = 0;
        for (int s = 0; s < 3; s++) {
            for (int a = 0; a < anchor_counts[s]; a++) {
                int idx = anchor_offset + a;
                float conf = data[4 * total_anchors + idx];
                
                if (conf <= conf_thresh) continue;
                
                Detection det;
                det.bbox.x = data[0 * total_anchors + idx];
                det.bbox.y = data[1 * total_anchors + idx];
                det.bbox.w = data[2 * total_anchors + idx];
                det.bbox.h = data[3 * total_anchors + idx];
                det.score = conf;
                
                dets.push_back(det);
                count++;
            }
            anchor_offset += anchor_counts[s];
        }
    }
    
    return count;
}

// ============================================================================
// 公共 API 函数
// ============================================================================

int initTennisDetector(const char* model_path) {
    if (g_initialized) {
        LOGI("检测器已初始化");
        return 0;
    }

    int ret = CVI_NN_RegisterModel(model_path, &g_model);
    if (ret != CVI_RC_SUCCESS) {
        LOGE("模型加载失败: %d", ret);
        return -1;
    }

    CVI_NN_GetInputOutputTensors(g_model, &g_input_tensors, &g_input_num, 
                                  &g_output_tensors, &g_output_num);

    CVI_TENSOR* input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, 
                                                g_input_tensors, g_input_num);
    CVI_SHAPE input_shape = CVI_NN_TensorShape(input);
    g_model_h = input_shape.dim[2];
    g_model_w = input_shape.dim[3];
    
    g_output_shape = CVI_NN_TensorShape(&g_output_tensors[0]);
    
    g_initialized = true;
    LOGI("检测器初始化成功，模型输入: %dx%d", g_model_w, g_model_h);
    
    return 0;
}

void deinitTennisDetector(void) {
    if (g_initialized && g_model) {
        CVI_NN_CleanupModel(g_model);
        g_model = NULL;
        g_initialized = false;
        LOGI("检测器已释放");
    }
}

int detectTennis(const char* jpeg_data, int len, TennisResult* result) {
    if (!g_initialized) {
        LOGE("检测器未初始化");
        return -1;
    }
    
    if (!jpeg_data || len <= 0 || !result) {
        LOGE("无效参数");
        return -1;
    }

    memset(result, 0, sizeof(TennisResult));

    // 解码 JPEG
    std::vector<uchar> buf(jpeg_data, jpeg_data + len);
    cv::Mat image = cv::imdecode(buf, cv::IMREAD_COLOR);
    if (image.empty()) {
        LOGE("JPEG 解码失败");
        return -1;
    }

    result->image_width = image.cols;
    result->image_height = image.rows;

    // 预处理
    cv::Mat preprocessed = preprocessImage(image, g_model_w, g_model_h);
    cv::Mat rgb;
    cv::cvtColor(preprocessed, rgb, cv::COLOR_BGR2RGB);

    cv::Mat channels[3];
    cv::split(rgb, channels);

    CVI_TENSOR* input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, 
                                                g_input_tensors, g_input_num);
    int8_t* tensor_ptr = (int8_t*)CVI_NN_TensorPtr(input);
    int channel_size = g_model_h * g_model_w;
    for (int i = 0; i < 3; ++i) {
        memcpy(tensor_ptr + i * channel_size, channels[i].data, channel_size);
    }

    // 推理
    CVI_NN_Forward(g_model, g_input_tensors, g_input_num, 
                   g_output_tensors, g_output_num);

    // 后处理
    std::vector<Detection> detections;
    parseDetections(g_output_tensors, g_model_h, g_model_w, 
                    g_output_shape, CONF_THRESHOLD, detections);
    applyNMS(detections, IOU_THRESHOLD);
    mapBoxesToOriginalImage(detections, image.rows, image.cols, 
                            g_model_h, g_model_w);

    // 填充结果
    result->count = std::min((int)detections.size(), MAX_DETECTIONS);
    for (int i = 0; i < result->count; i++) {
        const Detection& det = detections[i];
        TennisBox& box = result->boxes[i];
        
        box.cx = det.bbox.x;
        box.cy = det.bbox.y;
        box.w = det.bbox.w;
        box.h = det.bbox.h;
        box.x1 = det.bbox.x - det.bbox.w / 2;
        box.y1 = det.bbox.y - det.bbox.h / 2;
        box.x2 = det.bbox.x + det.bbox.w / 2;
        box.y2 = det.bbox.y + det.bbox.h / 2;
        box.score = det.score;
    }

    return 0;
}

void printTennisResult(const TennisResult* result) {
    if (!result) return;
    
    LOGI("图像尺寸: %d x %d", result->image_width, result->image_height);
    LOGI("检测到 %d 个网球:", result->count);
    
    for (int i = 0; i < result->count; i++) {
        const TennisBox& box = result->boxes[i];
        LOGI("  #%d: 置信度=%.1f%%, 位置=(%.0f,%.0f)-(%.0f,%.0f), 中心=(%.0f,%.0f)", 
             i + 1, box.score * 100, box.x1, box.y1, box.x2, box.y2, box.cx, box.cy);
    }
}
