/**
 * @file main.cpp
 * @brief 网球检测命令行工具
 * 
 * 用法:
 *   ./tennis detect-img <model_path> <input_img> <output_img>
 *   ./tennis camera <model_path> <output_img>
 *   ./tennis run <model_path>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
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
// 机器人控制相关定义
// ============================================================================

#define ROBO_INIT        1
#define ROBO_TURN_LEFT   2
#define ROBO_TURN_RIGHT  3
#define ROBO_FORWARD     4
#define ROBO_BACKWARD    5
#define ROBO_GRAB        6
#define ROBO_RELEASE     7
#define ROBO_STOP        8

// 视觉伺服参数
#define CENTER_BAND   0.30f   // 中心区域宽度比例
#define GRAB_RATIO    0.40f   // 抓取时网球面积占比阈值 (40%)
#define TURN_MS       50      // 转向时间 (ms) - 每次转一点点
#define FORWARD_MS    500     // 前进时间 (ms)

// ============================================================================
// 辅助函数
// ============================================================================

static void printUsage(const char* program) {
    printf("网球检测工具\n\n");
    printf("用法:\n");
    printf("  %s detect-img <模型文件> <输入图片> <输出图片>\n", program);
    printf("  %s camera <模型文件> <输出图片>\n", program);
    printf("  %s run <模型文件>\n", program);
    printf("  %s follow <模型文件>\n\n", program);
    printf("命令:\n");
    printf("  detect-img    检测图片中的网球并保存结果\n");
    printf("  camera        从摄像头捕获，检测到网球时保存并退出\n");
    printf("  run           机器人模式：检测网球并控制机器人抓取\n");
    printf("  follow        跟随模式：跟着网球走，不抓取\n\n");
    printf("示例:\n");
    printf("  %s detect-img model.cvimodel input.jpg output.jpg\n", program);
    printf("  %s camera model.cvimodel detected.jpg\n", program);
    printf("  %s run model.cvimodel\n", program);
    printf("  %s follow model.cvimodel\n", program);
}

static void msleep(unsigned int ms) {
    usleep(ms * 1000u);
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
        
        cv::rectangle(image, 
                      cv::Point((int)box.x1, (int)box.y1),
                      cv::Point((int)box.x2, (int)box.y2),
                      cv::Scalar(0, 255, 255), 3);
        
        char label[64];
        snprintf(label, sizeof(label), "tennis %.0f%%", box.score * 100);
        
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                              0.7, 2, &baseline);
        
        int label_y = (int)box.y1 - 5;
        if (label_y - text_size.height < 0) {
            label_y = (int)box.y1 + text_size.height + 5;
        }
        
        cv::rectangle(image,
                      cv::Point((int)box.x1, label_y - text_size.height - 5),
                      cv::Point((int)box.x1 + text_size.width + 10, label_y + 5),
                      cv::Scalar(0, 0, 200), -1);
        
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
        
        cv::rectangle(image, 
                      cv::Point((int)box.x1, (int)box.y1),
                      cv::Point((int)box.x2, (int)box.y2),
                      cv::Scalar(0, 255, 0), 3);
        
        char label[64];
        snprintf(label, sizeof(label), "tennis %.0f%%", box.score * 100);
        
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                              0.7, 2, &baseline);
        
        int label_y = (int)box.y1 - 5;
        if (label_y - text_size.height < 0) {
            label_y = (int)box.y1 + text_size.height + 5;
        }
        
        cv::rectangle(image,
                      cv::Point((int)box.x1, label_y - text_size.height - 5),
                      cv::Point((int)box.x1 + text_size.width + 10, label_y + 5),
                      cv::Scalar(255, 255, 255), -1);
        
        cv::putText(image, label, 
                    cv::Point((int)box.x1 + 5, label_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    }
}

/**
 * @brief 获取置信度最高的检测结果
 */
static const TennisBox* getBestDetection(const TennisResult* result) {
    if (result->count <= 0) return NULL;
    
    const TennisBox* best = &result->boxes[0];
    for (int i = 1; i < result->count; i++) {
        if (result->boxes[i].score > best->score) {
            best = &result->boxes[i];
        }
    }
    return best;
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
    
    if (initTennisDetector(model_path) != 0) {
        return -1;
    }
    
    long file_size = 0;
    char* jpeg_data = readFile(input_path, &file_size);
    if (!jpeg_data) {
        deinitTennisDetector();
        return -1;
    }
    LOGI("文件大小: %ld 字节", file_size);
    
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
    
    printTennisResult(&result);
    
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
    
    if (initTennisDetector(model_path) != 0) {
        return -1;
    }
    
    int fd = open(dev_path, O_RDWR);
    if (fd < 0) {
        LOGE("无法打开摄像头设备: %s", dev_path);
        deinitTennisDetector();
        return -1;
    }
    
    if (ioctl(fd, CVI_CAM_IOCTL_INIT, (unsigned long)0) < 0) {
        LOGE("摄像头初始化失败");
        close(fd);
        deinitTennisDetector();
        return -1;
    }
    
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
    
    while (!detected) {
        frame_count++;
        
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
        
        printf("\r[帧 %d] 大小: %ld 字节, 耗时: %ld ms, 检测到: %d 个目标", 
               frame_count, nbytes, elapsed_ms, result.count);
        fflush(stdout);
        
        if (result.count > 0) {
            detected = true;
            printf("\n");
            LOGI("----------------------------------------");
            LOGI("检测到网球!");
        }
    }
    
    printTennisResult(&result);
    
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
    
    drawResultsBlack(image, &result);
    
    if (cv::imwrite(output_path, image)) {
        LOGI("结果已保存: %s", output_path);
    } else {
        LOGE("保存图片失败: %s", output_path);
    }
    
    free(frame_buf);
    close(fd);
    deinitTennisDetector();
    
    LOGI("========================================");
    LOGI("完成");
    
    return 0;
}

/**
 * @brief 处理 run 命令 - 机器人检测与控制
 */
static int cmdRun(int argc, char** argv) {
    if (argc < 3) {
        LOGE("参数不足");
        printf("\n用法: %s run <模型文件>\n", argv[0]);
        return -1;
    }
    
    const char* model_path = argv[2];
    const char* cam_dev = "/dev/cvi-camera";
    const char* robo_dev = "/dev/robo-ctl";
    
    LOGI("========================================");
    LOGI("网球检测 - 机器人模式");
    LOGI("========================================");
    LOGI("模型: %s", model_path);
    LOGI("摄像头: %s", cam_dev);
    LOGI("机器人: %s", robo_dev);
    LOGI("----------------------------------------");
    
    // 初始化检测器
    if (initTennisDetector(model_path) != 0) {
        return -1;
    }
    
    // 打开摄像头设备
    int fd_cam = open(cam_dev, O_RDWR);
    if (fd_cam < 0) {
        LOGE("无法打开摄像头: %s (%s)", cam_dev, strerror(errno));
        deinitTennisDetector();
        return -1;
    }
    
    // 打开机器人设备
    int fd_robo = open(robo_dev, O_RDWR);
    if (fd_robo < 0) {
        LOGE("无法打开机器人: %s (%s)", robo_dev, strerror(errno));
        close(fd_cam);
        deinitTennisDetector();
        return -1;
    }
    
    // 初始化摄像头
    LOGI("初始化摄像头...");
    if (ioctl(fd_cam, CVI_CAM_IOCTL_INIT, 0UL) < 0) {
        LOGE("摄像头初始化失败");
        close(fd_robo);
        close(fd_cam);
        deinitTennisDetector();
        return -1;
    }
    
    struct camera_info cam_info;
    memset(&cam_info, 0, sizeof(cam_info));
    if (ioctl(fd_cam, CVI_CAM_IOCTL_GET_INFO, (unsigned long)&cam_info) < 0) {
        LOGE("获取摄像头信息失败");
        close(fd_robo);
        close(fd_cam);
        deinitTennisDetector();
        return -1;
    }
    LOGI("摄像头: %ux%u, format=%u", cam_info.width, cam_info.height, cam_info.format);
    
    // 初始化机器人
    LOGI("初始化机器人...");
    if (ioctl(fd_robo, ROBO_INIT, 0UL) < 0) {
        LOGE("机器人初始化失败");
        close(fd_robo);
        close(fd_cam);
        deinitTennisDetector();
        return -1;
    }
    msleep(500);
    
    LOGI("----------------------------------------");
    LOGI("开始连续检测网球...");
    LOGI("----------------------------------------");
    
    // 分配帧缓冲区
    unsigned char* frame_buf = (unsigned char*)malloc(FRAME_BUF_SIZE);
    if (!frame_buf) {
        LOGE("内存分配失败");
        close(fd_robo);
        close(fd_cam);
        deinitTennisDetector();
        return -1;
    }
    
    int frame_no = 0;
    
    while (1) {
        frame_no++;
        
        // 1. 获取帧
        long nbytes = ioctl(fd_cam, CVI_CAM_IOCTL_GET_FRAME, (unsigned long)frame_buf);
        if (nbytes <= 0) {
            LOGE("第 %d 帧: 获取失败, 重试...", frame_no);
            msleep(300);
            continue;
        }
        
        // 2. TPU 推理
        TennisResult result;
        if (detectTennis((const char*)frame_buf, (int)nbytes, &result) != 0) {
            LOGE("第 %d 帧: 推理失败", frame_no);
            msleep(200);
            continue;
        }
        
        // 3. 获取最佳检测结果
        const TennisBox* ball = getBestDetection(&result);
        if (!ball) {
            printf("[帧 %d] 未发现网球, 原地旋转搜索...\n", frame_no);
            ioctl(fd_robo, ROBO_TURN_LEFT, 0UL);
            msleep(TURN_MS);
            ioctl(fd_robo, ROBO_STOP, 0UL);
            msleep(50);
            continue;
        }
        
        // 计算位置参数
        float img_w = (float)result.image_width;
        float img_h = (float)result.image_height;
        float img_area = img_w * img_h;
        float ball_area = ball->w * ball->h;
        float cx_norm = ball->cx / img_w;        // 归一化中心 x 坐标
        float area_ratio = ball_area / img_area; // 面积占比
        
        LOGI("[帧 %d] 网球: 置信度=%.0f%%, 中心=(%.0f,%.0f), 大小=%.0fx%.0f, 面积占比=%.1f%%",
             frame_no, ball->score * 100,
             ball->cx, ball->cy, ball->w, ball->h,
             area_ratio * 100);
        
        // 4. 视觉伺服决策
        float left_edge = (1.0f - CENTER_BAND) / 2.0f;   // 0.35
        float right_edge = 1.0f - left_edge;              // 0.65
        
        if (cx_norm < left_edge) {
            // 网球偏左，左转
            LOGI("  -> 网球偏左(%.0f%%), 左转", cx_norm * 100);
            ioctl(fd_robo, ROBO_TURN_LEFT, 0UL);
            msleep(TURN_MS);
            ioctl(fd_robo, ROBO_STOP, 0UL);
            
        } else if (cx_norm > right_edge) {
            // 网球偏右，右转
            LOGI("  -> 网球偏右(%.0f%%), 右转", cx_norm * 100);
            ioctl(fd_robo, ROBO_TURN_RIGHT, 0UL);
            msleep(TURN_MS);
            ioctl(fd_robo, ROBO_STOP, 0UL);
            
        } else if (area_ratio < GRAB_RATIO) {
            // 网球居中但较远，前进
            LOGI("  -> 网球居中但较远(面积%.1f%%), 前进", area_ratio * 100);
            ioctl(fd_robo, ROBO_FORWARD, 0UL);
            msleep(FORWARD_MS);
            ioctl(fd_robo, ROBO_STOP, 0UL);
            
        } else {
            // 网球就位，执行抓取
            LOGI("  -> 网球就位, 执行抓取!");
            ioctl(fd_robo, ROBO_STOP, 0UL);
            msleep(200);
            
            // 抓取
            LOGI("     抓取中...");
            ioctl(fd_robo, ROBO_GRAB, 0UL);
            msleep(3000);
            
            // 后退
            LOGI("     后退...");
            ioctl(fd_robo, ROBO_BACKWARD, 0UL);
            msleep(1500);
            ioctl(fd_robo, ROBO_STOP, 0UL);
            
            // 释放
            LOGI("     释放...");
            ioctl(fd_robo, ROBO_RELEASE, 0UL);
            msleep(2000);
            
            LOGI("     抓取完成!");
        }
        
        msleep(50);
    }
    
    // 清理 (实际上不会执行到这里，因为是无限循环)
    ioctl(fd_robo, ROBO_STOP, 0UL);
    free(frame_buf);
    close(fd_robo);
    close(fd_cam);
    deinitTennisDetector();
    
    return 0;
}

/**
 * @brief 处理 follow 命令 - 跟随网球
 */
static int cmdFollow(int argc, char** argv) {
    if (argc < 3) {
        LOGE("参数不足");
        printf("\n用法: %s follow <模型文件>\n", argv[0]);
        return -1;
    }
    
    const char* model_path = argv[2];
    const char* cam_dev = "/dev/cvi-camera";
    const char* robo_dev = "/dev/robo-ctl";
    
    LOGI("========================================");
    LOGI("网球检测 - 跟随模式");
    LOGI("========================================");
    LOGI("模型: %s", model_path);
    LOGI("摄像头: %s", cam_dev);
    LOGI("机器人: %s", robo_dev);
    LOGI("----------------------------------------");
    
    // 初始化检测器
    if (initTennisDetector(model_path) != 0) {
        return -1;
    }
    
    // 打开摄像头设备
    int fd_cam = open(cam_dev, O_RDWR);
    if (fd_cam < 0) {
        LOGE("无法打开摄像头: %s (%s)", cam_dev, strerror(errno));
        deinitTennisDetector();
        return -1;
    }
    
    // 打开机器人设备
    int fd_robo = open(robo_dev, O_RDWR);
    if (fd_robo < 0) {
        LOGE("无法打开机器人: %s (%s)", robo_dev, strerror(errno));
        close(fd_cam);
        deinitTennisDetector();
        return -1;
    }
    
    // 初始化摄像头
    LOGI("初始化摄像头...");
    if (ioctl(fd_cam, CVI_CAM_IOCTL_INIT, 0UL) < 0) {
        LOGE("摄像头初始化失败");
        close(fd_robo);
        close(fd_cam);
        deinitTennisDetector();
        return -1;
    }
    
    struct camera_info cam_info;
    memset(&cam_info, 0, sizeof(cam_info));
    if (ioctl(fd_cam, CVI_CAM_IOCTL_GET_INFO, (unsigned long)&cam_info) < 0) {
        LOGE("获取摄像头信息失败");
        close(fd_robo);
        close(fd_cam);
        deinitTennisDetector();
        return -1;
    }
    LOGI("摄像头: %ux%u, format=%u", cam_info.width, cam_info.height, cam_info.format);
    
    // 初始化机器人
    LOGI("初始化机器人...");
    if (ioctl(fd_robo, ROBO_INIT, 0UL) < 0) {
        LOGE("机器人初始化失败");
        close(fd_robo);
        close(fd_cam);
        deinitTennisDetector();
        return -1;
    }
    msleep(500);
    
    LOGI("----------------------------------------");
    LOGI("开始跟随网球...");
    LOGI("----------------------------------------");
    
    // 分配帧缓冲区
    unsigned char* frame_buf = (unsigned char*)malloc(FRAME_BUF_SIZE);
    if (!frame_buf) {
        LOGE("内存分配失败");
        close(fd_robo);
        close(fd_cam);
        deinitTennisDetector();
        return -1;
    }
    
    int frame_no = 0;
    
    while (1) {
        frame_no++;
        
        // 1. 获取帧
        long nbytes = ioctl(fd_cam, CVI_CAM_IOCTL_GET_FRAME, (unsigned long)frame_buf);
        if (nbytes <= 0) {
            LOGE("第 %d 帧: 获取失败, 重试...", frame_no);
            msleep(300);
            continue;
        }
        
        // 2. TPU 推理
        TennisResult result;
        if (detectTennis((const char*)frame_buf, (int)nbytes, &result) != 0) {
            LOGE("第 %d 帧: 推理失败", frame_no);
            msleep(200);
            continue;
        }
        
        // 3. 获取最佳检测结果
        const TennisBox* ball = getBestDetection(&result);
        if (!ball) {
            // 没有找到网球，停止并等待
            printf("[帧 %d] 未发现网球, 等待...\n", frame_no);
            ioctl(fd_robo, ROBO_STOP, 0UL);
            msleep(100);
            continue;
        }
        
        // 计算位置参数
        float img_w = (float)result.image_width;
        float img_h = (float)result.image_height;
        float img_area = img_w * img_h;
        float ball_area = ball->w * ball->h;
        float cx_norm = ball->cx / img_w;        // 归一化中心 x 坐标
        float area_ratio = ball_area / img_area; // 面积占比
        
        LOGI("[帧 %d] 网球: 置信度=%.0f%%, 中心=(%.0f,%.0f), 大小=%.0fx%.0f, 面积占比=%.1f%%",
             frame_no, ball->score * 100,
             ball->cx, ball->cy, ball->w, ball->h,
             area_ratio * 100);
        
        // 4. 跟随决策
        float left_edge = (1.0f - CENTER_BAND) / 2.0f;   // 0.35
        float right_edge = 1.0f - left_edge;              // 0.65
        
        if (cx_norm < left_edge) {
            // 网球偏左，左转
            LOGI("  -> 网球偏左(%.0f%%), 左转", cx_norm * 100);
            ioctl(fd_robo, ROBO_TURN_LEFT, 0UL);
            msleep(TURN_MS);
            ioctl(fd_robo, ROBO_STOP, 0UL);
            
        } else if (cx_norm > right_edge) {
            // 网球偏右，右转
            LOGI("  -> 网球偏右(%.0f%%), 右转", cx_norm * 100);
            ioctl(fd_robo, ROBO_TURN_RIGHT, 0UL);
            msleep(TURN_MS);
            ioctl(fd_robo, ROBO_STOP, 0UL);
            
        } else if (area_ratio < GRAB_RATIO) {
            // 网球居中但较远，前进
            LOGI("  -> 网球居中但较远(面积%.1f%%), 前进", area_ratio * 100);
            ioctl(fd_robo, ROBO_FORWARD, 0UL);
            msleep(FORWARD_MS);
            ioctl(fd_robo, ROBO_STOP, 0UL);
            
        } else {
            // 网球已经很近，停止
            LOGI("  -> 网球已到达, 停止");
            ioctl(fd_robo, ROBO_STOP, 0UL);
        }
        
        msleep(50);
    }
    
    // 清理
    ioctl(fd_robo, ROBO_STOP, 0UL);
    free(frame_buf);
    close(fd_robo);
    close(fd_cam);
    deinitTennisDetector();
    
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
    } else if (strcmp(command, "run") == 0) {
        return cmdRun(argc, argv);
    } else if (strcmp(command, "follow") == 0) {
        return cmdFollow(argc, argv);
    } else if (strcmp(command, "-h") == 0 || strcmp(command, "--help") == 0) {
        printUsage(argv[0]);
        return 0;
    } else {
        LOGE("未知命令: %s", command);
        printUsage(argv[0]);
        return -1;
    }
}
