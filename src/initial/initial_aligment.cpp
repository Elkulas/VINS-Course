#include "initial/initial_alignment.h"

// 注意得到了新的Bias后保存在Bgs[]中，对应的预积分需要重新计算一遍repropagate。
/**
 * @brief   陀螺仪偏置校正
 * @optional    根据视觉SFM的结果来校正陀螺仪Bias -> Paper V-B-1
 *              主要是将相邻帧之间SFM求解出来的旋转矩阵与IMU预积分的旋转量对齐
 *              注意得到了新的Bias后对应的预积分需要repropagate
 *              https://blog.csdn.net/try_again_later/article/details/104783107
 * @param[in]   all_image_frame所有图像帧构成的map,图像帧保存了位姿、预积分量和关于角点的信息
 * @param[out]  Bgs 陀螺仪偏置
 * @return      void
*/
void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
{
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    // 遍历所有的frame
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        // 获得视觉qbibj
        // R_ij = (R^c0_bk)^-1 * (R^c0_bk+1) 转换为四元数 q_ij = (q^c0_bk)^-1 * (q^c0_bk+1)
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
        // tmp_A = J_j_bw
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        // tmp_b = 2 * (r^bk_bk+1)^-1 * (q^c0_bk)^-1 * (q^c0_bk+1)
        //       = 2 * (r^bk_bk+1)^-1 * q_ij
        // .vec是从四元数变化为vector
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
        // 对所有frame进行叠加
        // 在求解 Ax=b的最小二乘解时，两边同乘以A矩阵的转置得到的AT*A一定是可逆
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;

    }
    // tmp_A * delta_bg = tmp_b

    delta_bg = A.ldlt().solve(b);
    // ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

    // 求解得到新的bg
    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;

    // 对新的bg重新计算预计分
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
    }
}


MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

/**
 * @brief   重力矢量细化
 * @optional    重力细化，在其切线空间上用两个变量重新参数化重力 -> Paper V-B-3 
                g^ = ||g|| * (g^-) + w1b1 + w2b2 
                https://blog.csdn.net/try_again_later/article/details/104783107
 * @param[in]   all_image_frame 所有图像帧构成的map,图像帧保存了位姿，预积分量和关于角点的信息
 * @param[out]  g 重力加速度
 * @param[out]  x 待优化变量，窗口中每帧的速度V[0:n]、二自由度重力参数w[w1,w2]^T、尺度s
 * @return      void
*/
void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    //g0 = (g^-)*||g||
    Vector3d g0 = g.normalized() * G.norm();
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for(int k = 0; k < 4; k++)
    {
        //lxly = b = [b1,b2]
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;

            // tmp_A(6,9) = [-I*dt           0             (R^bk_c0)*dt*dt*b/2   (R^bk_c0)*((p^c0_ck+1)-(p^c0_ck))  ] 
            //              [ -I    (R^bk_c0)*(R^c0_bk+1)      (R^bk_c0)*dt*b                  0                    ]
            // tmp_b(6,1) = [ (a^bk_bk+1)+(R^bk_c0)*(R^c0_bk+1)*p^b_c-p^b_c - (R^bk_c0)*dt*dt*||g||*(g^-)/2 , (b^bk_bk+1)-(R^bk_c0)dt*||g||*(g^-)]^T
            // tmp_A * x = tmp_b 求解最小二乘问题
            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
            A = A * 1000.0;
            b = b * 1000.0;
            x = A.ldlt().solve(b);

            //dg = [w1,w2]^T
            VectorXd dg = x.segment<2>(n_state - 3);
            g0 = (g0 + lxly * dg).normalized() * G.norm();
            //double s = x(n_state - 1);
    } 
    // 返回refine之后的值  
    g = g0;
}

/**
 * @brief   计算尺度，重力加速度和速度
 * @optional    速度、重力向量和尺度初始化Paper -> V-B-2
 *              相邻帧之间的位置和速度与IMU预积分出来的位置和速度对齐，求解最小二乘
 *              重力细化 -> Paper V-B-3
 *              https://blog.csdn.net/try_again_later/article/details/104783107    
 * @param[in]   all_image_frame 所有图像帧构成的map,图像帧保存了位姿，预积分量和关于角点的信息
 * @param[out]  g 重力加速度
 * @param[out]  x 待优化变量，窗口中每帧的速度V[0:n]、重力g、尺度s
 * @return      void
*/
bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    // 优化变量的总维度
    int n_state = all_frame_count * 3 + 3 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    // 遍历所有frame
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        // H
        MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        // b
        VectorXd tmp_b(6);
        tmp_b.setZero();
        // 获得dt
        double dt = frame_j->second.pre_integration->sum_dt;

        // tmp_A(6,10) = H^bk_bk+1 = [-I*dt           0             (R^bk_c0)*dt*dt/2   (R^bk_c0)*((p^c0_ck+1)-(p^c0_ck))  ] 
        //                           [ -I    (R^bk_c0)*(R^c0_bk+1)      (R^bk_c0)*dt                  0                    ]
        // tmp_b(6,1 ) = z^bk_bk+1 = [ (a^bk_bk+1)+(R^bk_c0)*(R^c0_bk+1)*p^b_c-p^b_c , (b^bk_bk+1)]^T
        // tmp_A * x = tmp_b 求解最小二乘问题
        // -I*dt
        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        // (R^bk_c0)*dt*dt/2
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        // (R^bk_c0)*((p^c0_ck+1)-(p^c0_ck))
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;  
        // (a^bk_bk+1)+(R^bk_c0)*(R^c0_bk+1)*p^b_c-p^b_c   
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        
        // -I
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        // (R^bk_c0)*(R^c0_bk+1)
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        // (R^bk_c0)*dt  
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        // (b^bk_bk+1)
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        // 在A中分块填补
        // 速度部分 bk时候的速度以及bk+1时候的速度
        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        // 重力和sacle部分
        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        // 速度和重力scale之间的相关项,也就是在大矩阵的B,C区域
        // 可以想象,在矩阵A中A区域的范围是最大的,BC区域特别狭窄,D区域就一点点
        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }

    // TODO: 为什么要*1000
    A = A * 1000.0;
    b = b * 1000.0;
    // 求解
    x = A.ldlt().solve(b);
    double s = x(n_state - 1) / 100.0;
    // ROS_DEBUG("estimated scale: %f", s);
    g = x.segment<3>(n_state - 4);
    // ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
    if(fabs(g.norm() - G.norm()) > 1.0 || s < 0)
    {
        return false;
    }

    // 对应粉笔记中的第四步,对重力分向量进行refine
    RefineGravity(all_image_frame, g, x);
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    // ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
    if(s < 0.0 )
        return false;   
    else
        return true;
}

// 求解陀螺仪偏置bg、重力加速度g、每帧速度v、尺度s和相机到IMU的外参估计
// 利用旋转约束估计陀螺仪偏置bg
// 利用平移约束估计重力加速度g、每帧速度v、尺度s
// 对重力向量g^{c0}进一步优化
// 对应粉笔记上理论的2,3,4步骤
bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
    // 使用旋转约束求解陀螺仪bias bg,对应粉笔记第二步
    solveGyroscopeBias(all_image_frame, Bgs);

    // 计算尺度\重力\速度,对应粉笔记第三步
    if(LinearAlignment(all_image_frame, g, x))
        return true;
    else 
        return false;
}
