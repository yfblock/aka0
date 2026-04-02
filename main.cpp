/**
 * @file main.cpp
 * @brief 网球检测命令行工具
 * 
 * 用法:
 *   ./tennis detect-img <model_path> <input_img> <output_img>
 *   ./tennis camera <model_path> <output_img>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <opencv2/opencv.hpp>
#include "tennis.h"

#define LOGI(fmt, ...) printf("[INFO] " fmt "\n", ##__VA_ARGS__)
#define LOGE(fmt, ...) printf("[ERROR] " fmt "\n", ##__VA_ARGS__)

// ============================================================================
// 摄像头相关定义
// ============================================================================

#define CVI_CAM_IOCTL_INIT      1u
#define CVI_CAM_IOCTL_GET_INFO  2u
#define CVI_CAM_IOCTL_GET_FRAME 3u

#define FRAME_BUF_SIZE (2u * 1024u * 1024u)  // 2MB

struct camera_info {
    uint16_t width;
    uint16_t height;
    uint8_t format;
    uint8_t connected;
} __attribute__((packed));

// ============================================================================
// 辅助函数
// ============================================================================

static void printUsage(const char* program) {
    printf("网球检测工具\n\n");
    printf("用法:\n");
    printf("  %s detect-img <模型文件> <输入图片> <输出图片>\n", program);
    printf("  %s camera <模型文件> <输出图片>\n\n", program);
    printf("命令:\n");
    printf("  detect-img    检测图片中的网球并保存结果\n");
    printf("  camera        从摄像头捕获，检测到网球时保存并退出\n\n");
    printf("示例:\n");
    printf("  %s detect-img model.cvimodel input.jpg output.jpg\n", program);
    printf("  %s camera model.cvimodel detected.jpg\n", program);
}

/**
 * @brief 读取文件内容
 */
static char* readFile(const char* path, long* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        LOGE("无法打开文件: %s", path);
        return NULL;
    }
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char* data = (char*)malloc(size);
    if (!data) {
        LOGE("内存分配失败");
        fclose(f);
        return NULL;
    }
    
    fread(data, 1, size, f);
    fclose(f);
    
    if (out_size) *out_size = size;
    return data;
}

/**
 * @brief 在图像上绘制检测结果 (彩色样式)
 */
static void drawResults(cv::Mat& image, const TennisResult* result) {
    for (int i = 0; i < result->count; i++) {
        const TennisBox& box = result->boxes[i];
        
        // 绘制边界框 (黄色, 粗线)
        cv::rectangle(image, 
                      cv::Point((int)box.x1, (int)box.y1),
                      cv::Point((int)box.x2, (int)box.y2),
                      cv::Scalar(0, 255, 255), 3);
        
        // 准备标签文字
        char label[64];
        snprintf(label, sizeof(label), "tennis %.0f%%", box.score * 100);
        
        // 计算文字大小
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                              0.7, 2, &baseline);
        
        // 绘制标签背景 (红色)
        int label_y = (int)box.y1 - 5;
        if (label_y - text_size.height < 0) {
            label_y = (int)box.y1 + text_size.height + 5;
        }
        
        cv::rectangle(image,
                      cv::Point((int)box.x1, label_y - text_size.height - 5),
                      cv::Point((int)box.x1 + text_size.width + 10, label_y + 5),
                      cv::Scalar(0, 0, 200), -1);
        
        // 绘制标签文字 (白色)
        cv::putText(image, label, 
                    cv::Point((int)box.x1 + 5, label_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }
}

/**
 * @brief 在图像上绘制检测结果 (黑色字体样式)
 */
static void drawResultsBlack(cv::Mat& image, const TennisResult* result) {
    for (int i = 0; i < result->count; i++) {
        const TennisBox& box = result->boxes[i];
        
        // 绘制边界框 (绿色, 粗线)
        cv::rectangle(image, 
                      cv::Point((int)box.x1, (int)box.y1),
                      cv::Point((int)box.x2, (int)box.y2),
                      cv::Scalar(0, 255, 0), 3);
        
        // 准备标签文字
        char label[64];
        snprintf(label, sizeof(label), "tennis %.0f%%", box.score * 100);
        
        // 计算文字大小
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                              0.7, 2, &baseline);
        
        // 绘制标签背景 (白色)
        int label_y = (int)box.y1 - 5;
        if (label_y - text_size.height < 0) {
            label_y = (int)box.y1 + text_size.height + 5;
        }
        
        cv::rectangle(image,
                      cv::Point((int)box.x1, label_y - text_size.height - 5),
                      cv::Point((int)box.x1 + text_size.width + 10, label_y + 5),
                      cv::Scalar(255, 255, 255), -1);
        
        // 绘制标签文字 (黑色)
        cv::putText(image, label, 
                    cv::Point((int)box.x1 + 5, label_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    }
}

// ============================================================================
// 命令处理函数
// ============================================================================

/**
 * @brief 处理 detect-img 命令
 */
static int cmdDetectImage(int argc, char** argv) {
    if (argc < 5) {
        LOGE("参数不足");
        printf("\n用法: %s detect-img <模型文件> <输入图片> <输出图片>\n", argv[0]);
        return -1;
    }
    
    const char* model_path = argv[2];
    const char* input_path = argv[3];
    const char* output_path = argv[4];
    
    LOGI("========================================");
    LOGI("网球检测 - 图片模式");
    LOGI("========================================");
    LOGI("模型: %s", model_path);
    LOGI("输入: %s", input_path);
    LOGI("输出: %s", output_path);
    LOGI("----------------------------------------");
    
    // 初始化检测器
    if (initTennisDetector(model_path) != 0) {
        return -1;
    }
    
    // 读取图片文件
    long file_size = 0;
    char* jpeg_data = readFile(input_path, &file_size);
    if (!jpeg_data) {
        deinitTennisDetector();
        return -1;
    }
    LOGI("文件大小: %ld 字节", file_size);
    
    // 执行检测
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    
    TennisResult result;
    int ret = detectTennis(jpeg_data, file_size, &result);
    
    gettimeofday(&t2, NULL);
    long elapsed_ms = ((t2.tv_sec - t1.tv_sec) * 1000000 + 
                       (t2.tv_usec - t1.tv_usec)) / 1000;
    
    free(jpeg_data);
    
    if (ret != 0) {
        LOGE("检测失败");
        deinitTennisDetector();
        return -1;
    }
    
    LOGI("检测耗时: %ld ms", elapsed_ms);
    LOGI("----------------------------------------");
    
    // 打印结果
    printTennisResult(&result);
    
    // 绘制并保存结果图片
    LOGI("----------------------------------------");
    cv::Mat image = cv::imread(input_path);
    if (image.empty()) {
        LOGE("无法读取图片用于绘制");
        deinitTennisDetector();
        return -1;
    }
    
    drawResults(image, &result);
    
    if (cv::imwrite(output_path, image)) {
        LOGI("结果已保存: %s", output_path);
    } else {
        LOGE("保存图片失败: %s", output_path);
    }
    
    // 清理
    deinitTennisDetector();
    
    LOGI("========================================");
    LOGI("完成");
    
    return 0;
}

/**
 * @brief 处理 camera 命令 - 从摄像头捕获并检测
 */
static int cmdCamera(int argc, char** argv) {
    if (argc < 4) {
        LOGE("参数不足");
        printf("\n用法: %s camera <模型文件> <输出图片>\n", argv[0]);
        return -1;
    }
    
    const char* model_path = argv[2];
    const char* output_path = argv[3];
    const char* dev_path = "/dev/cvi-camera";
    
    LOGI("========================================");
    LOGI("网球检测 - 摄像头模式");
    LOGI("========================================");
    LOGI("模型: %s", model_path);
    LOGI("输出: %s", output_path);
    LOGI("设备: %s", dev_path);
    LOGI("----------------------------------------");
    
    // 初始化检测器
    if (initTennisDetector(model_path) != 0) {
        return -1;
    }
    
    // 打开摄像头设备
    int fd = open(dev_path, O_RDWR);
    if (fd < 0) {
        LOGE("无法打开摄像头设备: %s", dev_path);
        deinitTennisDetector();
        return -1;
    }
    
    // 初始化摄像头
    if (ioctl(fd, CVI_CAM_IOCTL_INIT, (unsigned long)0) < 0) {
        LOGE("摄像头初始化失败");
        close(fd);
        deinitTennisDetector();
        return -1;
    }
    
    // 获取摄像头信息
    struct camera_info info;
    memset(&info, 0, sizeof(info));
    if (ioctl(fd, CVI_CAM_IOCTL_GET_INFO, (unsigned long)&info) < 0) {
        LOGE("获取摄像头信息失败");
        close(fd);
        deinitTennisDetector();
        return -1;
    }
    
    LOGI("摄像头: %ux%u, format=%u, connected=%u",
         (unsigned)info.width, (unsigned)info.height,
         (unsigned)info.format, (unsigned)info.connected);
    LOGI("----------------------------------------");
    LOGI("等待检测到网球...");
    
    // 分配帧缓冲区
    unsigned char* frame_buf = (unsigned char*)malloc(FRAME_BUF_SIZE);
    if (!frame_buf) {
        LOGE("内存分配失败");
        close(fd);
        deinitTennisDetector();
        return -1;
    }
    
    int frame_count = 0;
    TennisResult result;
    bool detected = false;
    long frame_size = 0;
    
    // 循环捕获直到检测到网球
    while (!detected) {
        frame_count++;
        
        // 获取一帧
        long nbytes = ioctl(fd, CVI_CAM_IOCTL_GET_FRAME, (unsigned long)frame_buf);
        if (nbytes <= 0) {
            LOGE("获取帧失败");
            continue;
        }
        
        if ((size_t)nbytes > FRAME_BUF_SIZE) {
            LOGE("帧数据过大: %ld", nbytes);
            continue;
        }
        
        frame_size = nbytes;
        
        // 检测
        struct timeval t1, t2;
        gettimeofday(&t1, NULL);
        
        int ret = detectTennis((const char*)frame_buf, nbytes, &result);
        
        gettimeofday(&t2, NULL);
        long elapsed_ms = ((t2.tv_sec - t1.tv_sec) * 1000000 + 
                           (t2.tv_usec - t1.tv_usec)) / 1000;
        
        if (ret != 0) {
            LOGE("检测失败");
            continue;
        }
        
        // 输出帧信息
        printf("\r[帧 %d] 大小: %ld 字节, 耗时: %ld ms, 检测到: %d 个目标", 
               frame_count, nbytes, elapsed_ms, result.count);
        fflush(stdout);
        
        // 检查是否检测到网球
        if (result.count > 0) {
            detected = true;
            printf("\n");
            LOGI("----------------------------------------");
            LOGI("检测到网球!");
        }
    }
    
    // 打印检测结果
    printTennisResult(&result);
    
    // 解码图像并绘制结果
    LOGI("----------------------------------------");
    std::vector<uchar> buf(frame_buf, frame_buf + frame_size);
    cv::Mat image = cv::imdecode(buf, cv::IMREAD_COLOR);
    
    if (image.empty()) {
        LOGE("图像解码失败");
        free(frame_buf);
        close(fd);
        deinitTennisDetector();
        return -1;
    }
    
    // 绘制检测框 (黑色字体)
    drawResultsBlack(image, &result);
    
    // 保存结果
    if (cv::imwrite(output_path, image)) {
        LOGI("结果已保存: %s", output_path);
    } else {
        LOGE("保存图片失败: %s", output_path);
    }
    
    // 清理
    free(frame_buf);
    close(fd);
    deinitTennisDetector();
    
    LOGI("========================================");
    LOGI("完成");
    
    return 0;
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return -1;
    }
    
    const char* command = argv[1];
    
    if (strcmp(command, "detect-img") == 0) {
        return cmdDetectImage(argc, argv);
    } else if (strcmp(command, "camera") == 0) {
        return cmdCamera(argc, argv);
    } else if (strcmp(command, "-h") == 0 || strcmp(command, "--help") == 0) {
        printUsage(argv[0]);
        return 0;
    } else {
        LOGE("未知命令: %s", command);
        printUsage(argv[0]);
        return -1;
    }
}
