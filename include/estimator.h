#pragma once

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"

#include "factor/integration_base.h"

#include "backend/problem.h"

#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>

class Estimator
{
  public:
    Estimator();

    // 设置部分参数
    void setParameter();

    // interface
    // 处理imu数据,预积分
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    
    // 处理图像特征数据
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double header);
    void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r);

    // internal
    // 清空或初始化滑动窗口中所有的状态量
    void clearState();

    // 视觉的结构初始化
    bool initialStructure();

    // 视觉惯性联合初始化
    // 流程
    // VisualIMUAlignment（）函数计算陀螺仪偏置bg，尺度s，重力加速度g和速度v
    // f_manager.triangulate（）计算特征点深度estimated_depth
    // repropagate（）陀螺仪的偏置bgs改变，重新计算预积分
    // 将Ps、Vs、depth进行更新
    // 将重力旋转到Z轴，将Ps、Vs、Rs从相机参考坐标系c0旋转到世界坐标系w。
    bool visualInitialAlign();

    // 判断两帧有足够视差30且内点数目大于12则可进行初始化，同时得到R和T
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);

    // 滑动窗口法
    void slideWindow();

    // VIO非线性优化求解里程计
    void solveOdometry();
    void slideWindowNew();
    void slideWindowOld();

    // 基于滑动窗口的紧耦合的非线性优化，残差项的构造和求解
    void optimization();
    void backendOptimization();

    // 和一家博士写得处理方式
    void problemSolve();
    void MargOldFrame();
    void MargNewFrame();

    void vector2double();
    void double2vector();
    bool failureDetection();


    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };
//////////////// OUR SOLVER ///////////////////
    MatXX Hprior_;
    VecX bprior_;
    VecX errprior_;
    MatXX Jprior_inv_;

    Eigen::Matrix2d project_sqrt_info_;
//////////////// OUR SOLVER //////////////////
    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Vector3d g;
    MatrixXd Ap[2], backup_A;
    VectorXd bp[2], backup_b;

    // TODO: RT对于imu和camera
    // cam到body 的变换
    // tbc, rbc
    Matrix3d ric[NUM_OF_CAM];
    Vector3d tic[NUM_OF_CAM];

    // 经过初始化之后变成从imu系到world系的变换
    // Pwb, Vwb, Rwb
    Vector3d Ps[(WINDOW_SIZE + 1)];
    Vector3d Vs[(WINDOW_SIZE + 1)];
    Matrix3d Rs[(WINDOW_SIZE + 1)];
    Vector3d Bas[(WINDOW_SIZE + 1)];
    Vector3d Bgs[(WINDOW_SIZE + 1)];
    double td;

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    // 滑动窗口内的所有帧的时间戳
    double Headers[(WINDOW_SIZE + 1)];

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    // number of frames in the window.
    // 窗口内帧的个数
    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;


    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];

    int loop_window_index;

    // MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;

    // 时间戳,帧对象,所有帧
    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration;

    //relocalization variable
    bool relocalization_info;
    double relo_frame_stamp;
    double relo_frame_index;
    int relo_frame_local_index;
    vector<Vector3d> match_points;
    double relo_Pose[SIZE_POSE];
    Matrix3d drift_correct_r;
    Vector3d drift_correct_t;
    Vector3d prev_relo_t;
    Matrix3d prev_relo_r;
    Vector3d relo_relative_t;
    Quaterniond relo_relative_q;
    double relo_relative_yaw;
};
