// https://sm.ms/image/pno4mY3GFQH6eNR

/*
三者之间的串联结构,可以获得基本数据point了
for (auto &it_per_id : f_manager.feature)
{
   ......
   for (auto &it_per_frame : it_per_id.feature_per_frame)
   {
     Vector3d pts_j = it_per_frame.point;// 3D特征点坐标
 }
}
*/
#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
#include <map>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

// #include <ros/console.h>
// #include <ros/assert.h>

#include "parameters.h"

// 指的是每帧基本的数据：特征点[x,y,z,u,v,vx,vy]和td IMU与cam同步时间差
// 某点在某帧下的基本数据
class FeaturePerFrame
{
public:
  FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
  {
    point.x() = _point(0);
    point.y() = _point(1);
    point.z() = _point(2);
    uv.x() = _point(3);
    uv.y() = _point(4);
    velocity.x() = _point(5);
    velocity.y() = _point(6);
    cur_td = td;
  }
  double cur_td;
  Vector3d point;
  Vector2d uv;
  Vector2d velocity;

  // 特征点的深度
  double z;
  // 是否被用了
  bool is_used;
  // 视差
  double parallax;
  // 变换矩阵
  MatrixXd A;
  VectorXd b;
  double dep_gradient;
};

// 指的是某feature_id下的所有FeaturePerFrame。常用feature_id和观测第一帧start_frame、最后一帧endFrame()
class FeaturePerId
{
public:
  // 特征点ID索引
  const int feature_id;
  // 首次被观测到时，该帧的索引
  int start_frame;

  // 能够观测到某个特征点的所有相关帧
  vector<FeaturePerFrame> feature_per_frame;

  // 该特征出现的次数
  int used_num;
  // 是否外点
  bool is_outlier;
  // 是否Marg边缘化
  bool is_margin;
  // 估计的逆深度
  double estimated_depth;
  int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

  Vector3d gt_p;

  FeaturePerId(int _feature_id, int _start_frame)
      : feature_id(_feature_id), start_frame(_start_frame),
        used_num(0), estimated_depth(-1.0), solve_flag(0)
  {
  }

  // 返回最后一次观测到这个特征点的图像帧ID
  int endFrame();
};


// 管理所有特征点，通过list容器存储特征点属性
// https://sm.ms/image/pno4mY3GFQH6eNR
class FeatureManager
{
public:
  FeatureManager(Matrix3d _Rs[]);

  void setRic(Matrix3d _ric[]);

  void clearState();

  // 窗口中被跟踪的特征点数量
  int getFeatureCount();

  // 特征点进入时检查视差,是否为关键帧
  bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
  void debugShow();
  // 前后两帧之间匹配特征点三维坐标
  // 得到frame_count_l与frame_count_r两帧之间的对应特征点3D坐标
  vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

  //void updateDepth(const VectorXd &x);
  // 设置特征点逆深度
  void setDepth(const VectorXd &x);
  void removeFailures();
  void clearDepth(const VectorXd &x);
  VectorXd getDepthVector();
  // 特征点三角化求解深度
  void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
  void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
  // 边缘化最老帧,直接将特征点保存的帧号往前移
  void removeBack();
  // 边缘化次新帧,将特征点在次新帧中的信息进行移除
  void removeFront(int frame_count);
  // 外点移除
  void removeOutlier();

  // 重要！！ 通过FeatureManager可以得到滑动窗口内所有的角点信息
  // 滑动窗口内所有的角点信息
  list<FeaturePerId> feature;
  int last_track_num;

private:
  double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
  const Matrix3d *Rs;
  Matrix3d ric[NUM_OF_CAM];
};

#endif