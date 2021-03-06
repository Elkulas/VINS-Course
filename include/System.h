#pragma once

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>

#include <fstream>
#include <condition_variable>

// #include <cv.h>
// #include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>

#include "estimator.h"
#include "parameters.h"
#include "feature_tracker.h"


//imu for vio
struct IMU_MSG
{
    double header;
    Eigen::Vector3d linear_acceleration;
    Eigen::Vector3d angular_velocity;
};
// 指向imu数据的指针
typedef std::shared_ptr<IMU_MSG const> ImuConstPtr;

//image for vio    
struct IMG_MSG {
    // 时间戳
    double header;
    vector<Vector3d> points;
    // 相机点的id,第n个相机中的第i个点就是 n*CAM_NUM+i
    vector<int> id_of_point;
    // 特征点的uv位置
    vector<float> u_of_point;
    vector<float> v_of_point;
    // 速度,在readimage中被修改,也就是减去上一帧除掉时间
    vector<float> velocity_x_of_point;
    vector<float> velocity_y_of_point;
};
// 指向图像数据的指针
typedef std::shared_ptr <IMG_MSG const > ImgConstPtr;
    
class System
{
public:
    // 使用config文件进行构造
    System(std::string sConfig_files);

    ~System();

    // 图像数据发布
    void PubImageData(double dStampSec, cv::Mat &img);

    // imu数据发布
    void PubImuData(double dStampSec, const Eigen::Vector3d &vGyr, 
        const Eigen::Vector3d &vAcc);

    // thread: visual-inertial odometry
    // 后端优化
    void ProcessBackEnd();
    void Draw();
    
    pangolin::OpenGlRenderState s_cam;
    pangolin::View d_cam;
    // 特征点类
    FeatureTracker trackerData[NUM_OF_CAM];

#ifdef __APPLE__
    void InitDrawGL(); 
    void DrawGLFrame();
#endif

private:

    //feature tracker
    std::vector<uchar> r_status;
    std::vector<float> r_err;
    // std::queue<ImageConstPtr> img_buf;

    // ros::Publisher pub_img, pub_match;
    // ros::Publisher pub_restart;

    double first_image_time;
    int pub_count = 1;
    bool first_image_flag = true;
    double last_image_time = 0;
    bool init_pub = 0;

    //estimator
    Estimator estimator;

    std::condition_variable con;
    double current_time = -1;
    std::queue<ImuConstPtr> imu_buf;
    std::queue<ImgConstPtr> feature_buf;
    // std::queue<PointCloudConstPtr> relo_buf;
    int sum_of_wait = 0;

    std::mutex m_buf;
    std::mutex m_state;
    std::mutex i_buf;
    std::mutex m_estimator;

    double latest_time;
    Eigen::Vector3d tmp_P;
    Eigen::Quaterniond tmp_Q;
    Eigen::Vector3d tmp_V;
    Eigen::Vector3d tmp_Ba;
    Eigen::Vector3d tmp_Bg;
    Eigen::Vector3d acc_0;
    Eigen::Vector3d gyr_0;
    // 检测是不是第一个帧
    // 第一帧需要进行跳过
    bool init_feature = 0;
    bool init_imu = 1;
    double last_imu_t = 0;
    std::ofstream ofs_pose;
    std::vector<Eigen::Vector3d> vPath_to_draw;
    bool bStart_backend;
    // 原本属于estimator
    // 该函数的主要功能是对imu和图像数据进行对齐并组合，返回的是(IMUs, img_msg)s，
    // 即图像帧所对应的所有IMU数据，并将其放入一个容器vector中。
    std::vector<std::pair<std::vector<ImuConstPtr>, ImgConstPtr>> getMeasurements();
    
};
