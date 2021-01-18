#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "utility/tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

// 判断跟踪的特征点是否在图像边界内
bool inBorder(const cv::Point2f &pt);

// 去除无法跟踪的特征点
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker();

    // 对图像使用光流法进行特征点跟踪
    void readImage(const cv::Mat &_img,double _cur_time);

    // 对跟踪点进行排序并去除密集点
    // 对跟踪到的特征点，按照被追踪到的次数排序并依次选点
    // 使用mask进行类似非极大抑制的方法，半径为30，去掉分部密集的点，使特征点分布均匀
    void setMask();

    // 添将新检测到的特征点n_pts，ID初始化-1，跟踪次数1
    void addPoints();

    // 更新特征点id
    bool updateID(unsigned int i);

    // 读取相机内参
    void readIntrinsicParameter(const string &calib_file);

    // 显示去畸变之后的特征点
    void showUndistortion(const string &name);

    // 通过F矩阵去除outliers
    // 该函数主要是通过基本矩阵（F）去除外点outliers。首先将将图像坐标畸变矫正后转换为像素坐标，
    // 通过cv::findFundamentalMat()计算F矩阵，利用得到的status通过reduceVector()去除outliers 。
    void rejectWithF();

    // 对特征点的图像坐标去畸变矫正，并计算每个角点的速度
    void undistortedPoints();

    // 图像掩码
    cv::Mat mask;
    // 鱼眼相机的mask,用来去除边缘噪点
    cv::Mat fisheye_mask;

    // prev_img 上一帧的图像数据
    // cur_img 光流跟踪前一帧的数据
    // forw_img 光流跟踪后一帧的数据
    cv::Mat prev_img, cur_img, forw_img;

    // 每一帧中新提取的特征点
    vector<cv::Point2f> n_pts;
    // 对应帧中提取的特征点
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    // 归一化相机坐标系下的坐标
    // xx_pts is with image coordinate, 
    // while xx_un_pts is in ideal image plane after undistortion and normalization
    vector<cv::Point2f> prev_un_pts, cur_un_pts;
    // 当前帧相对前一帧特征点沿x,y方向的像素移动速度
    vector<cv::Point2f> pts_velocity;
    // 能够被跟踪到的特征点的id
    vector<int> ids;
    // 当前帧forw_img中每个特征点被追踪的时间次数
    vector<int> track_cnt;


    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;

    // 相机模型
    camodocal::CameraPtr m_camera;
    double cur_time;
    double prev_time;

    // 用来作为特征点id，每检测到一个新的特征点，就将n_id作为该特征点的id，然后n_id加1
    static int n_id;
};
