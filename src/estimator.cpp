#include "estimator.h"

#include "backend/vertex_inverse_depth.h"
#include "backend/vertex_pose.h"
#include "backend/vertex_speedbias.h"
#include "backend/edge_reprojection.h"
#include "backend/edge_imu.h"

#include <ostream>
#include <fstream>

using namespace myslam;

Estimator::Estimator() : f_manager{Rs}
{
    // ROS_INFO("init begins");

    for (size_t i = 0; i < WINDOW_SIZE + 1; i++)
    {
        pre_integrations[i] = nullptr;
    }
    for(auto &it: all_image_frame)
    {
        it.second.pre_integration = nullptr;
    }
    tmp_pre_integration = nullptr;
    
    clearState();
}

void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        // cout << "1 Estimator::setParameter tic: " << tic[i].transpose()
        //     << " ric: " << ric[i] << endl;
    }
    cout << "1 Estimator::setParameter FOCAL_LENGTH: " << FOCAL_LENGTH << endl;
    f_manager.setRic(ric);
    project_sqrt_info_ = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
}

void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    
    tmp_pre_integration = nullptr;
    
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
    relocalization_info = 0;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}

/**
 * @brief   处理IMU数据
 * @Description IMU预积分，中值积分得到当前PQV作为优化初值
 * @param[in]   dt 时间间隔
 * @param[in]   linear_acceleration 线加速度
 * @param[in]   angular_velocity 角速度
 * @return  void
*/
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    // 1. 判断是不是第一个imu消息，如果是第一个imu消息，则将当前传入的线加速度和角速度作为初始的加速度和角速度
    // first_imu为false表示当前上报的imu数据为第一个imu数据
    if (!first_imu)
    {

        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    // 2. 初始化一个预计分的值
    // 创建预积分对象
    // IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }

    // 当滑动窗口内有图像帧数据时，进行预积分 frame_count表示滑动窗口中图像数据的个数
    if (frame_count != 0)
    {
        // 3. 进行预积分操作
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        // 4. 将dt,线速度,角速度加到buf中
        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;
        // 5. 采用中值积分的方式
        // 绿笔记中值法离散积分处
        // a0=Q(a^-ba)-g 已知上一帧imu速度
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        // w=0.5(w0+w1)-bg
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        // 旋转四元数更新
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        // a1 当前imu的速度
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        // 中值积分下的加速度 0.5(a0+a1)
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        // p_k+1 = p_k + v_k*dt + 0.5 * a * dt * dt
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        // vk+1 = vk + a*dt
        Vs[j] += dt * un_acc;
    }
    // 6. 更新上一帧的加速度和角速度
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

/**
 * @brief   处理图像特征数据
 * @Description addFeatureCheckParallax()添加特征点到feature中，计算点跟踪的次数和视差，判断是否是关键帧               
 *              判断并进行外参标定
 *              进行视觉惯性联合初始化或基于滑动窗口非线性优化的紧耦合VIO
 * @param[in]   image 某帧所有特征点的[camera_id,[x,y,z,u,v,vx,vy]]s构成的map,索引为feature_id
 * @param[in]   header 某帧图像的头信息
 * @return  void
*/
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double header)
{
    //ROS_DEBUG("new image coming ------------------------------------------");
    cout << "Adding feature points: " << image.size()<<endl;

    // 添加之前检测到的特征点到feature容器中，计算每一个点跟踪的次数，以及它的视差
    // 通过检测两帧之间的视差决定次新帧是否作为关键帧
    // 判断是marg掉哪个量
    // f_mamnager 滑动窗口内特征管理器类的对象
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
        marginalization_flag = MARGIN_OLD; // 0
    else
        marginalization_flag = MARGIN_SECOND_NEW; //1

    //ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    //ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    //ROS_DEBUG("Solving %d", frame_count);
    cout << "number of feature: " << f_manager.getFeatureCount()<<endl;
    
    Headers[frame_count] = header;

    // 2. 填充imageframe的容器以及更新临时预积分初始值
    // ImageFrame类包括特征点、时间、位姿Rt、预积分对象pre_integration、是否关键帧
    ImageFrame imageframe(image, header);
    // tmp_pre_integration是之前IMU 预积分计算的
    // 存到imageframe中
    imageframe.pre_integration = tmp_pre_integration;
    // map<double, ImageFrame> all_image_frame;
    // 存一下这个时间戳
    all_image_frame.insert(make_pair(header, imageframe));
    // 更新零时预计分值
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    // 3. 如果没有外参则标定IMU到相机的外参
    if (ESTIMATE_EXTRINSIC == 2)
    {
        cout << "calibrating extrinsic param, rotation movement is needed" << endl;
        if (frame_count != 0)
        {
            // 得到两帧之间归一化的特征点
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            // 标定imu与camera之间的外参数
            Matrix3d calib_ric;
            // 标定外参数
            // 可以参见粉色笔记本中的笔记
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                // ROS_WARN("initial extrinsic rotation calib success");
                // ROS_WARN_STREAM("initial extrinsic rotation: " << endl
                                                            //    << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    // 4. 初始化
    if (solver_flag == INITIAL)
    {
        // frame_count是滑动窗口中图像帧的数量，一开始初始化为0，滑动窗口总帧数WINDOW_SIZE=10
        // 确保有足够的frame参与初始化
        if (frame_count == WINDOW_SIZE)
        {
            bool result = false;
            // 有外参且当前帧时间戳大于初始化时间戳0.1秒，就进行初始化操作
            if (ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
            {
                // cout << "1 initialStructure" << endl;
                // 4.1 重要！！！ 视觉惯性联合初始化
                result = initialStructure();
                // 4.2 更新初始化间隔
                initial_timestamp = header;
            }
            // 初始化成功
            if (result)
            {
                // 先进行一次滑动窗口非线性优化，得到当前帧与第一帧的位姿
                // 也就是对目前初始化结束的部分做一次非线性优化,优化所有帧的位姿
                // 初始化更改为非线性
                solver_flag = NON_LINEAR;
                // 4.3 非线性化求解里程计
                solveOdometry();
                // 4.4 滑动窗口
                slideWindow();
                // 4.5 剔除feature中估计失败的点（solve_flag == 2）0 haven't solve yet; 1 solve succ; 2 solve fail;
                f_manager.removeFailures();
                cout << "Initialization finish!" << endl;
                // 4.6 初始化窗口中PVQ
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
            }
            else
                // 初始化失败直接滑动窗口
                slideWindow();
        }
        else
            frame_count++;
    }
    // 5. 紧耦合的非线性优化
    else
    {
        TicToc t_solve;
        // 5.1 非线性化求解里程计
        solveOdometry();
        //ROS_DEBUG("solver costs: %fms", t_solve.toc());

        // 5.2 故障检测与恢复,一旦检测到故障，系统将切换回初始化阶段
        if (failureDetection())
        {
            // ROS_WARN("failure detection!");
            failure_occur = 1;
            // 清空状态
            clearState();
            // 重新设置参数
            setParameter();
            // ROS_WARN("system reboot!");
            return;
        }

        TicToc t_margin;
        // 5.3 滑动窗口
        slideWindow();
        // 5.4 移除失败
        f_manager.removeFailures();
        //ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}

// 视觉结构初始化过程至关重要，多传感器融合过程中，
// 当单个传感器数据不确定性较高，需要依赖其他传感器降低不确定性。先对纯视觉SFM初始化相机位姿，再和IMU对齐。
// 1、纯视觉SFM估计滑动窗内相机位姿和路标点逆深度
// 2、视觉惯性联合校准，SFM与IMU积分对齐。
// 使用松耦合的方式来进行初始化矫正,对应粉色笔记本上的内容

bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    // 1. 通过加速度标准差判断 IMU是否有充分运动以初始化
    {
        // 信息帧的迭代器
        map<double, ImageFrame>::iterator frame_it;
        // 统计帧内imu数据的加速度的和
        // 1.1 求加速度均值aver_g
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        // 计算平均加速度值
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        // 1.2 计算加速度的标准差,线加速度
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        // 判断IMU的激励,如果不足0.25则表示imu没有充足激励
        if (var < 0.25)
        {
            cerr << "IMU excitation not enouth!"<< endl;
            //return false;
        }
    }
    // global sfm
    // 初始化变量
    // 旋转四元数Q
    Quaterniond Q[frame_count + 1];
    // 位姿变化T
    Vector3d T[frame_count + 1];
    // 存储SFM重建出特征点的坐标
    map<int, Vector3d> sfm_tracked_points;
    // SFMFeature三角化状态、特征点索引、平面观测、位置坐标、深度
    vector<SFMFeature> sfm_f;

    // 2. 遍历滑窗中的所有角点
    // 2. 将f_manager中的所有feature保存到存有SFMFeature对象的sfm_f中
    // list<FeaturePerId> feature滑动窗口内所有角点
    for (auto &it_per_id : f_manager.feature)
    {
        // 第一次观测到特征点的帧数-1
        // 这里就是imu 的id
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;

        // 遍历能观测到该点的所有关键帧
        // 也就是遍历这个特征点的所有观测信息
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            // 遍历到点的信息
            Vector3d pts_j = it_per_frame.point;
            // 保存该点的所有观测信息以及对应的imuid序列
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } // 遍历所有角点结束

    // 3. 返回滑动窗口中第一个满足视差的帧，为l帧，以及RT,可以三角化。
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    // 保证具有足够的视差,由E矩阵恢复R、t
    // 这里的第L帧是从第一帧开始到滑动窗口中第一个满足与当前帧的平均视差足够大的帧，
    // 会作为参考帧到下面的全局sfm使用，得到的Rt为当前帧到第l帧的坐标系变换Rt
    if (!relativePose(relative_R, relative_T, l))
    {
        cout << "Not enough features or parallax; Move device around" << endl;
        return false;
    }

    //4. 对窗口中每个图像帧都求解sfm问题
    // 得到滑动窗口内所有图像帧相对于参考帧l的姿态四元数Q 平移向量T 和特征点坐标sfm_tracked_points 
    // Q、T存储的是每一帧相对于第l帧(参考帧)的相对位姿。
    GlobalSFM sfm;
    if (!sfm.construct(frame_count + 1, Q, T, l,
                       relative_R, relative_T,
                       sfm_f, sfm_tracked_points))
    {
        // 求解失败则边缘化最早一帧并滑动窗口
        cout << "global SFM failed!" << endl;
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    //对于所有的图像帧，包括不在滑动窗口中的，提供初始的RT估计
    // 实际代码中所显示的,就是使用sfm优化得到的这些个三维点来求解剩下帧的位姿
    // 前提是这些帧也有这些点的投影
    // 然后solvePnP进行优化,得到每一帧的姿态
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    // 遍历所有帧
    for (int i = 0; frame_it != all_image_frame.end(); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        // 查看该帧是否是滑动窗口内的关键帧(也就是那十个)
        // 检查方式就是检查header时间戳
        if ((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if ((frame_it->first) > Headers[i])
        {
            i++;
        }
        // 这里说明该帧不是滑动窗口中的关键帧
        // 给定一个初始值,初始值是最近的滑动窗口那一帧的数据
        // 变换成第l帧到该帧的位姿
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        // 罗德里格斯公式将旋转矩阵转换成旋转向量
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        // 遍历该帧的所有点
        for (auto &id_pts : frame_it->second.points)
        {
            // 获取该特征点的特征id
            int feature_id = id_pts.first;
            // 遍历观察到该点的所有cam
            for (auto &i_p : id_pts.second)
            {
                // 找到该id在视觉sfm中的位置
                it = sfm_tracked_points.find(feature_id);
                // 这个点在sfm中出现过(不然等于.end)
                if (it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    // 存储三维点
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    // 存储二维点
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }// 遍历所有cam
        }// 遍历所有点

        // 保证至少这个帧里有五个特征点
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (pts_3_vector.size() < 6)
        {
            cout << "Not enough points for solve pnp pts_3_vector size " << pts_3_vector.size() << endl;
            return false;
        }
        // 可以!解pnp
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            cout << " solve pnp fail!" << endl;
            return false;
        }
        // 这里也同样需要将坐标变换矩阵转变成图像帧位姿，并转换为IMU坐标系的位姿
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }// 遍历所有帧
    
    /// 至此,已经完成视觉上的相对位姿的初始化和解算,可以获得一个关于第l帧的一个相对位姿曲线
    /// 接下来就是将视觉和imu数据进行联合对齐工作

    // 进行视觉imu联合初始化
    if (visualInitialAlign())
        return true;
    else
    {
        cout << "misalign visual structure with IMU" << endl;
        return false;
    }
}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    // 1. 初始化速度,偏置bg,尺度s,重力加速度
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if (!result)
    {
        //ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    // 2、得到所有图像帧的位姿Ps、Rs，并将其置为关键帧
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    // 3、重新计算特征点的深度depth,因为已经有了scale
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep);

    //triangulat on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for (int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);

    // 4、陀螺仪的偏置bgs改变，重新计算预积分
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    // 5 将Ps,Vs,depth进行更新
    // 5.1 目的是将姿态从相机坐标系c0转换到IMU坐标系中。
    for (int i = frame_count; i >= 0; i--)
        // 之前相机第l帧为参考系，转换到IMU bo为基准坐标系
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            // 5.2 Vs为优化得到的速度
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    // 5.3 逆深度depth更新
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    // 6、所有变量从参考坐标系c0到世界坐标系。
    // 通过将重力旋转到z轴上，得到世界坐标系与摄像机坐标系c0之间的旋转矩阵rot_diff
    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;

    // 7/所有变量从参考坐标系c0旋转到世界坐标系w
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    //ROS_DEBUG_STREAM("g0     " << g.transpose());
    //ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

    return true;
}


/**
 * @brief   返回滑动窗口中第一个满足视差的帧，为l帧，以及RT,可以三角化。
 * @Description    判断每帧到窗口最后一帧(也就是当前帧)对应特征点的平均视差是否大于30
                solveRelativeRT()通过基础矩阵计算当前帧与第l帧之间的R和T,并判断内点数目是否足够
                找到滑动窗口中第一个与最后一帧具有足够的共视特征以及视差的帧的索引 l，
                具体方式为在滑动窗内从第一帧开始计算每一帧和最后一帧匹配的特征点
 * @param[out]   relative_R 当前帧到第l帧之间的旋转矩阵R
 * @param[out]   relative_T 当前帧到第l帧之间的平移向量T
 * @param[out]   L 从第一帧开始到滑动窗口中第一个满足视差足够的帧，这里的l帧之后作为参考帧做全局SFM用
 * @return  bool 1:可以进行初始化;0:不满足初始化条件
*/
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    // 遍历滑动窗口中的帧
    // 寻找第i帧到窗口最后一帧(当前帧)的对应特征点
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        // 先得到第i帧和最后一帧的特征匹配
        // 找到corres满足的帧i
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        // 1. 如果corres数目足够
        if (corres.size() > 20)
        {
            // 2. 计算平均视差
            double sum_parallax = 0;
            double average_parallax;

            // 遍历所有匹配的特征点
            for (int j = 0; j < int(corres.size()); j++)
            {
                // 第j个对应点在第i帧和最后一帧的(x,y)
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }// 遍历所有匹配的特征点结束
            average_parallax = 1.0 * sum_parallax / int(corres.size());

            // 判断是否满足初始化条件:视差>30并且内点inlier数目要满足要求
            // 同时返回窗口最后一帧(当前帧)到第l帧(参考帧)的Rt
            if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                //ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::solveOdometry()
{
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        // 将所有的点进行三角化求解estimated depth
        f_manager.triangulate(Ps, tic, ric);
        //cout << "triangulation costs : " << t_tri.toc() << endl;        
        backendOptimization();
    }
}

// 从Ps、Rs、Vs、Bas、Bgs转化为para_Pose（6维，相机位姿）和para_SpeedBias（9维，相机速度、加速度偏置、角速度偏置）
// 从tic和q转化为para_Ex_Pose （6维，Cam到IMU外参）
// 从dep到para_Feature（1维，特征点深度）
// 从td转化为para_Td（1维，标定同步时间）
void Estimator::vector2double()
{
    // 对窗口内的帧进行变换
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        // 平移+旋转
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        // 速度
        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        // 加速度bias
        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();
        
        // 陀螺仪bias
        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }

    // 从tic和q转化为para_Ex_Pose （6维，Cam到IMU外参）
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    // 从dep到para_Feature（1维，特征点深度
    // 这里获得的是逆深度 getvecdepth
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

// 从Ps、Rs、Vs、Bas、Bgs转化为para_Pose（6维，相机位姿）和para_SpeedBias（9维，相机速度、加速度偏置、角速度偏置）
// 从tic和q转化为para_Ex_Pose （6维，Cam到IMU外参）
// 从dep到para_Feature（1维，特征点深度）
// 从td转化为para_Td（1维，标定同步时间）

// 后端优化之后的变量更新
// 这是因为在后端滑动窗口的非线性优化时，我们并没有固定住第一帧的位姿不变，而是将其作为优化变量进行调整。
// 但是，因为相机的偏航角yaw是不可观测的，也就是说对于任意的yaw都满足优化目标函数，因此优化之后我们将偏航角旋转至优化前的初始状态。
void Estimator::double2vector()
{
    // 原来的最初始的Rs,Ps
    // 获得yaw pitch row数据
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    // 如果出现错误情况,那么就沿用上次没优化前的数据
    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    // 优化之后的第一帧旋转初始值
    // 获得yaw pitch row数据
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                     para_Pose[0][3],
                                                     para_Pose[0][4],
                                                     para_Pose[0][5])
                                             .toRotationMatrix());
    // 获得不客观的yaw偏差
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    // yaw偏差计算得到的旋转矩阵
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));

    // 如果yaw角变化比较小,那么直接使用原来的旋转和优化之后旋转的一个差值作为rotation diff
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        //ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5])
                               .toRotationMatrix()
                               .transpose();
    }

    // 对所有帧优化之后的位姿,motion参数放回RS PS vs bas bgs变量中
    // 其中位姿和速度项同时乘以rot_diff来保证整条轨迹没有飘
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) +
                origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }
    // 优化完毕的外参数
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5])
                     .toRotationMatrix();
    }

    // 更新优化之后所有feature 的depth 参数
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    // 回环
    if (relocalization_info)
    {
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) +
                 origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;
    }
}

bool Estimator::failureDetection()
{
    if (f_manager.last_track_num < 2)
    {
        //ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        //ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        //ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        //ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        //ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        //ROS_INFO(" big z translation");
        return true;
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        //ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

// heyijia博士写的
void Estimator::MargOldFrame()
{
    // 定义lossfunc
    backend::LossFunction *lossfunction;
    lossfunction = new backend::CauchyLoss(1.0);

    // step1. 构建 problem,边缘化problem
    backend::Problem problem(backend::Problem::ProblemType::SLAM_PROBLEM);
    vector<shared_ptr<backend::VertexPose>> vertexCams_vec;
    vector<shared_ptr<backend::VertexSpeedBias>> vertexVB_vec;
    int pose_dim = 0;

    // 添加残差项,边缘化残差
    // 先把 外参数 节点加入图优化，这个节点在以后一直会被用到，所以我们把他放在第一个
    shared_ptr<backend::VertexPose> vertexExt(new backend::VertexPose());
    {
        Eigen::VectorXd pose(7);
        pose << para_Ex_Pose[0][0], para_Ex_Pose[0][1], para_Ex_Pose[0][2], para_Ex_Pose[0][3], para_Ex_Pose[0][4], para_Ex_Pose[0][5], para_Ex_Pose[0][6];
        vertexExt->SetParameters(pose);
        problem.AddVertex(vertexExt);
        pose_dim += vertexExt->LocalDimension();
    }

    // 位姿节点
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        shared_ptr<backend::VertexPose> vertexCam(new backend::VertexPose());
        Eigen::VectorXd pose(7);
        pose << para_Pose[i][0], para_Pose[i][1], para_Pose[i][2], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5], para_Pose[i][6];
        vertexCam->SetParameters(pose);
        vertexCams_vec.push_back(vertexCam);
        problem.AddVertex(vertexCam);
        pose_dim += vertexCam->LocalDimension();

        shared_ptr<backend::VertexSpeedBias> vertexVB(new backend::VertexSpeedBias());
        Eigen::VectorXd vb(9);
        vb << para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2],
            para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5],
            para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8];
        vertexVB->SetParameters(vb);
        vertexVB_vec.push_back(vertexVB);
        problem.AddVertex(vertexVB);
        pose_dim += vertexVB->LocalDimension();
    }

    // IMU 残差项
    {
        if (pre_integrations[1]->sum_dt < 10.0)
        {
            std::shared_ptr<backend::EdgeImu> imuEdge(new backend::EdgeImu(pre_integrations[1]));
            std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
            edge_vertex.push_back(vertexCams_vec[0]);
            edge_vertex.push_back(vertexVB_vec[0]);
            edge_vertex.push_back(vertexCams_vec[1]);
            edge_vertex.push_back(vertexVB_vec[1]);
            imuEdge->SetVertex(edge_vertex);
            problem.AddEdge(imuEdge);
        }
    }

    // Visual Factor
    {
        int feature_index = -1;
        // 遍历每一个特征
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;

            ++feature_index;

            int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
            if (imu_i != 0)
                continue;

            Vector3d pts_i = it_per_id.feature_per_frame[0].point;

            shared_ptr<backend::VertexInverseDepth> verterxPoint(new backend::VertexInverseDepth());
            VecX inv_d(1);
            inv_d << para_Feature[feature_index][0];
            verterxPoint->SetParameters(inv_d);
            problem.AddVertex(verterxPoint);

            // 遍历所有的观测
            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                if (imu_i == imu_j)
                    continue;

                Vector3d pts_j = it_per_frame.point;

                std::shared_ptr<backend::EdgeReprojection> edge(new backend::EdgeReprojection(pts_i, pts_j));
                std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
                edge_vertex.push_back(verterxPoint);
                edge_vertex.push_back(vertexCams_vec[imu_i]);
                edge_vertex.push_back(vertexCams_vec[imu_j]);
                edge_vertex.push_back(vertexExt);

                edge->SetVertex(edge_vertex);
                edge->SetInformation(project_sqrt_info_.transpose() * project_sqrt_info_);

                edge->SetLossFunction(lossfunction);
                problem.AddEdge(edge);
            }
        }
    }

    // 先验
    {
        // 已经有 Prior 了
        if (Hprior_.rows() > 0)
        {
            problem.SetHessianPrior(Hprior_); // 告诉这个 problem
            problem.SetbPrior(bprior_);
            problem.SetErrPrior(errprior_);
            problem.SetJtPrior(Jprior_inv_);
            problem.ExtendHessiansPriorSize(15); // 但是这个 prior 还是之前的维度，需要扩展下装新的pose,多添加15
        }
        else
        {
            Hprior_ = MatXX(pose_dim, pose_dim);
            Hprior_.setZero();
            bprior_ = VecX(pose_dim);
            bprior_.setZero();
            problem.SetHessianPrior(Hprior_); // 告诉这个 problem
            problem.SetbPrior(bprior_);
        }
    }

    // 定义要marg掉的顶点
    std::vector<std::shared_ptr<backend::Vertex>> marg_vertex;
    // old 帧
    marg_vertex.push_back(vertexCams_vec[0]);
    marg_vertex.push_back(vertexVB_vec[0]);
    // 在此处进行marg操作
    problem.Marginalize(marg_vertex, pose_dim);
    Hprior_ = problem.GetHessianPrior();
    bprior_ = problem.GetbPrior();
    errprior_ = problem.GetErrPrior();
    Jprior_inv_ = problem.GetJtPrior();
}
void Estimator::MargNewFrame()
{

    // step1. 构建 problem
    backend::Problem problem(backend::Problem::ProblemType::SLAM_PROBLEM);
    vector<shared_ptr<backend::VertexPose>> vertexCams_vec;
    vector<shared_ptr<backend::VertexSpeedBias>> vertexVB_vec;
    //    vector<backend::Point3d> points;
    int pose_dim = 0;

    // 先把 外参数 节点加入图优化，这个节点在以后一直会被用到，所以我们把他放在第一个
    shared_ptr<backend::VertexPose> vertexExt(new backend::VertexPose());
    {
        Eigen::VectorXd pose(7);
        pose << para_Ex_Pose[0][0], para_Ex_Pose[0][1], para_Ex_Pose[0][2], para_Ex_Pose[0][3], para_Ex_Pose[0][4], para_Ex_Pose[0][5], para_Ex_Pose[0][6];
        vertexExt->SetParameters(pose);
        problem.AddVertex(vertexExt);
        pose_dim += vertexExt->LocalDimension();
    }

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        shared_ptr<backend::VertexPose> vertexCam(new backend::VertexPose());
        Eigen::VectorXd pose(7);
        pose << para_Pose[i][0], para_Pose[i][1], para_Pose[i][2], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5], para_Pose[i][6];
        vertexCam->SetParameters(pose);
        vertexCams_vec.push_back(vertexCam);
        problem.AddVertex(vertexCam);
        pose_dim += vertexCam->LocalDimension();

        shared_ptr<backend::VertexSpeedBias> vertexVB(new backend::VertexSpeedBias());
        Eigen::VectorXd vb(9);
        vb << para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2],
            para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5],
            para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8];
        vertexVB->SetParameters(vb);
        vertexVB_vec.push_back(vertexVB);
        problem.AddVertex(vertexVB);
        pose_dim += vertexVB->LocalDimension();
    }

    // 先验
    {
        // 已经有 Prior 了
        if (Hprior_.rows() > 0)
        {
            problem.SetHessianPrior(Hprior_); // 告诉这个 problem
            problem.SetbPrior(bprior_);
            problem.SetErrPrior(errprior_);
            problem.SetJtPrior(Jprior_inv_);

            problem.ExtendHessiansPriorSize(15); // 但是这个 prior 还是之前的维度，需要扩展下装新的pose
        }
        else
        {
            Hprior_ = MatXX(pose_dim, pose_dim);
            Hprior_.setZero();
            bprior_ = VecX(pose_dim);
            bprior_.setZero();
        }
    }

    std::vector<std::shared_ptr<backend::Vertex>> marg_vertex;
    // 把窗口倒数第二个帧 marg 掉
    marg_vertex.push_back(vertexCams_vec[WINDOW_SIZE - 1]);
    marg_vertex.push_back(vertexVB_vec[WINDOW_SIZE - 1]);
    problem.Marginalize(marg_vertex, pose_dim);
    Hprior_ = problem.GetHessianPrior();
    bprior_ = problem.GetbPrior();
    errprior_ = problem.GetErrPrior();
    Jprior_inv_ = problem.GetJtPrior();
}


// 相比于optimization,没有加入relocalization_info的优化与统计
// prior就是对应了marginlizationfactor边缘化残差
void Estimator::problemSolve()
{
    // 这个部分就像是之前的course_5的部分
    backend::LossFunction *lossfunction;
    lossfunction = new backend::CauchyLoss(1.0);
    //    lossfunction = new backend::TukeyLoss(1.0);

    // step1. 构建 problem
    backend::Problem problem(backend::Problem::ProblemType::SLAM_PROBLEM);
    vector<shared_ptr<backend::VertexPose>> vertexCams_vec;
    vector<shared_ptr<backend::VertexSpeedBias>> vertexVB_vec;
    int pose_dim = 0;

    // 先把 外参数 节点加入图优化，这个节点在以后一直会被用到，所以我们把他放在第一个
    // 外参
    shared_ptr<backend::VertexPose> vertexExt(new backend::VertexPose());
    {
        Eigen::VectorXd pose(7);
        pose << para_Ex_Pose[0][0], para_Ex_Pose[0][1], para_Ex_Pose[0][2], para_Ex_Pose[0][3], para_Ex_Pose[0][4], para_Ex_Pose[0][5], para_Ex_Pose[0][6];
        vertexExt->SetParameters(pose);

        if (!ESTIMATE_EXTRINSIC)
        {
            //ROS_DEBUG("fix extinsic param");
            // TODO:: set Hessian prior to zero
            vertexExt->SetFixed();
        }
        else{
            //ROS_DEBUG("estimate extinsic param");
        }
        problem.AddVertex(vertexExt);
        pose_dim += vertexExt->LocalDimension();
    }

    // 遍历所有的pose,添加pose节点
    // 添加所有速度项节点,也就是关于速度 bg ba的节点
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        shared_ptr<backend::VertexPose> vertexCam(new backend::VertexPose());
        Eigen::VectorXd pose(7);
        pose << para_Pose[i][0], para_Pose[i][1], para_Pose[i][2], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5], para_Pose[i][6];
        vertexCam->SetParameters(pose);
        vertexCams_vec.push_back(vertexCam);
        problem.AddVertex(vertexCam);
        pose_dim += vertexCam->LocalDimension();

        shared_ptr<backend::VertexSpeedBias> vertexVB(new backend::VertexSpeedBias());
        Eigen::VectorXd vb(9);
        vb << para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2],
            para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5],
            para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8];
        vertexVB->SetParameters(vb);
        vertexVB_vec.push_back(vertexVB);
        problem.AddVertex(vertexVB);
        pose_dim += vertexVB->LocalDimension();
    }

    // 顶点添加全部结束
    // 添加边开始
    // IMU
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;

        std::shared_ptr<backend::EdgeImu> imuEdge(new backend::EdgeImu(pre_integrations[j]));
        std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
        edge_vertex.push_back(vertexCams_vec[i]);
        edge_vertex.push_back(vertexVB_vec[i]);
        edge_vertex.push_back(vertexCams_vec[j]);
        edge_vertex.push_back(vertexVB_vec[j]);
        imuEdge->SetVertex(edge_vertex);
        problem.AddEdge(imuEdge);
    }

    // Visual Factor
    // 视觉项
    vector<shared_ptr<backend::VertexInverseDepth>> vertexPt_vec;
    {
        int feature_index = -1;
        // 遍历每一个特征
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;

            ++feature_index;

            int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point;

            // 添加逆深度节点
            shared_ptr<backend::VertexInverseDepth> verterxPoint(new backend::VertexInverseDepth());
            VecX inv_d(1);
            inv_d << para_Feature[feature_index][0];
            verterxPoint->SetParameters(inv_d);
            problem.AddVertex(verterxPoint);
            vertexPt_vec.push_back(verterxPoint);

            // 遍历所有的观测
            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                if (imu_i == imu_j)
                    continue;

                Vector3d pts_j = it_per_frame.point;

                std::shared_ptr<backend::EdgeReprojection> edge(new backend::EdgeReprojection(pts_i, pts_j));
                std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
                edge_vertex.push_back(verterxPoint);
                edge_vertex.push_back(vertexCams_vec[imu_i]);
                edge_vertex.push_back(vertexCams_vec[imu_j]);
                // 注意这里多增加了外参数
                edge_vertex.push_back(vertexExt);

                edge->SetVertex(edge_vertex);
                edge->SetInformation(project_sqrt_info_.transpose() * project_sqrt_info_);

                edge->SetLossFunction(lossfunction);
                problem.AddEdge(edge);
            }
        }
    }

    // 先验
    {
        // 已经有 Prior 了
        if (Hprior_.rows() > 0)
        {
            // 外参数先验设置为 0. TODO:: 这个应该放到 solver 里去弄
            //            Hprior_.block(0,0,6,Hprior_.cols()).setZero();
            //            Hprior_.block(0,0,Hprior_.rows(),6).setZero();

            problem.SetHessianPrior(Hprior_); // 告诉这个 problem
            problem.SetbPrior(bprior_);
            problem.SetErrPrior(errprior_);
            problem.SetJtPrior(Jprior_inv_);
            problem.ExtendHessiansPriorSize(15); // 但是这个 prior 还是之前的维度，需要扩展下装新的pose
        }
    }

    // 优化求解迭代十次
    problem.Solve(10);

    // update bprior_,  Hprior_ do not need update
    if (Hprior_.rows() > 0)
    {
        std::cout << "----------- update bprior -------------\n";
        std::cout << "             before: " << bprior_.norm() << std::endl;
        std::cout << "                     " << errprior_.norm() << std::endl;
        bprior_ = problem.GetbPrior();
        errprior_ = problem.GetErrPrior();
        std::cout << "             after: " << bprior_.norm() << std::endl;
        std::cout << "                    " << errprior_.norm() << std::endl;
    }

    // update parameter
    // 更新优化完毕后的参数
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        // 更新pose
        VecX p = vertexCams_vec[i]->Parameters();
        for (int j = 0; j < 7; ++j)
        {
            para_Pose[i][j] = p[j];
        }

        // 更新motion项
        VecX vb = vertexVB_vec[i]->Parameters();
        for (int j = 0; j < 9; ++j)
        {
            para_SpeedBias[i][j] = vb[j];
        }
    }

    // 遍历每一个特征
    // 更新逆深度
    for (int i = 0; i < vertexPt_vec.size(); ++i)
    {
        VecX f = vertexPt_vec[i]->Parameters();
        para_Feature[i][0] = f[0];
    }
}

// 这里就是对应vinsmono中原来的Estimator::optimization函数
void Estimator::backendOptimization()
{
    TicToc t_solver;
    // 借助 vins 框架，维护变量
    vector2double();
    // 构建求解器
    // 这里就是进行修改的地方
    problemSolve();
    // 优化后的变量处理下自由度
    double2vector();
    //ROS_INFO("whole time for solver: %f", t_solver.toc());

    // 维护 marg
    // 优化完毕之后开始marg
    // 两步marg策略
    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        vector2double();
        // 边缘化最老帧
        // 对应源代码中对于marginlization_info的操作
        // 如果次新帧是关键帧，将边缘化最老帧，及其看到的路标点和IMU数据，将其转化为先验。
        MargOldFrame();

        // 调整参数块在下一次窗口中对应的位置（往前移一格），注意这里是指针，后面slideWindow中会赋新值，这里只是提前占座
        // 这里仅仅是相当于将指针进行了一次移动，指针对应的数据还是旧数据，
        // 因此需要结合后面调用的slideWindow()函数才能实现真正的滑窗移动
        // TODO: 这里没理解
        std::unordered_map<long, double *> addr_shift; // prior 中对应的保留下来的参数地址
        // 从1开始,因为0,也就是第一帧不要
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            //第i的位置存放的的是i-1的内容，这就意味着窗口向前移动了一格
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
    }
    else
    {
        // 如果存在先验
        if (Hprior_.rows() > 0)
        {

            vector2double();

            MargNewFrame();

            // 调整参数块在下一次窗口中对应的位置
            // 去掉次新帧
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }
        }
    }
    
}

// 这个部分还是原来的部分
void Estimator::slideWindow()
{
    TicToc t_margin;
    // 最老帧
    if (marginalization_flag == MARGIN_OLD)
    {
        // 保存最老帧信息到back_变量中
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            // 遍历窗口内帧
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                // swap操作,将最老帧也就是0代表的帧一致swap到第Rs[WINDOW_SIZE]处
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }// 遍历窗口内帧

            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;

                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }

                all_image_frame.erase(all_image_frame.begin(), it_0);
                all_image_frame.erase(t_0);
            }
            slideWindowOld();
        }
    }
    // MARGIN_SECOND_NEW 边缘化次新帧，但是不删除IMU约束
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                // 取出最新一帧的信息
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            slideWindowNew();
        }
    }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    // 1、统计一共多少次marg滑窗第一帧情况
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        // R0P0 -> 要marg掉的帧的位姿 Rwc, Rwc
        // R1P1 -> 当前滑动窗口中最老的帧的位姿 Rwc, Rwc
        Matrix3d R0, R1;
        Vector3d P0, P1;
        //back_R0、back_P0为窗口中最老帧的位姿
        //Rs[0]、Ps[0]为滑动窗口后第0帧的位姿，即原来的第1帧
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        // 首次在原来最老帧出现的特征点转移到现在现在最老帧
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
    // TODO: 没有搞懂这个逻辑
        f_manager.removeBack();
}
