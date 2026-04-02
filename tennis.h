/**
 * @file tennis.h
 * @brief 网球检测 API 头文件
 */

#ifndef TENNIS_H
#define TENNIS_H

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_DETECTIONS 32

/**
 * @brief 单个检测结果
 */
typedef struct {
    float x1, y1;    // 左上角坐标
    float x2, y2;    // 右下角坐标
    float cx, cy;    // 中心点坐标
    float w, h;      // 宽度和高度
    float score;     // 置信度 (0.0 ~ 1.0)
} TennisBox;

/**
 * @brief 检测结果集合
 */
typedef struct {
    int count;                       // 检测到的目标数量
    int image_width;                 // 原图宽度
    int image_height;                // 原图高度
    TennisBox boxes[MAX_DETECTIONS]; // 检测框数组
} TennisResult;

/**
 * @brief 初始化检测器
 * @param model_path 模型文件路径 (.cvimodel)
 * @return 0 成功，-1 失败
 */
int initTennisDetector(const char* model_path);

/**
 * @brief 释放检测器资源
 */
void deinitTennisDetector(void);

/**
 * @brief 检测 JPEG 图像中的网球
 * @param jpeg_data JPEG 图像数据
 * @param len 数据长度
 * @param result 检测结果输出
 * @return 0 成功，-1 失败
 */
int detectTennis(const char* jpeg_data, int len, TennisResult* result);

/**
 * @brief 打印检测结果
 */
void printTennisResult(const TennisResult* result);

#ifdef __cplusplus
}
#endif

#endif // TENNIS_H
