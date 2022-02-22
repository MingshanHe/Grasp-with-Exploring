#include "control_strategy/control_strategy.h"

Control_Strategy::Control_Strategy(
    const ros::NodeHandle &nh_,
    std::vector<double>     workspace_limits_,
    std::vector<double>     home_pose_,
    std::vector<double>     work_start_pose_,
    std::vector<double>     predict_map_size_) :
    nh(nh_), workspace_limits(workspace_limits_.data()),
    home_pose(home_pose_.data()),work_start_pose(work_start_pose_.data()),
    predict_map_size(predict_map_size_.data())
{
    force_x = 0.0;
    force_y = 0.0;
    force_x_pre = 0.0;
    force_y_pre = 0.0;
    // ROS Service
    switch_controller_client = nh.serviceClient<controller_manager_msgs::SwitchController>(SwitchController_Topic);
    // ROS Pub&Sub
    Cartesian_Pose_Pub  = nh.advertise<geometry_msgs::Pose>(CartesianPose_Topic, 1, true);
    Cartesian_Twist_Pub = nh.advertise<geometry_msgs::Twist>(CartesianTwist_Topic, 1, true);
    Predict_IMG_Pub     = nh.advertise<std_msgs::Float64MultiArray>(Predict_IMG_Topic, 1, true);
    Cartesian_State_Sub = nh.subscribe(CartesianState_Topic, 5, &Control_Strategy::Cartesian_State_Cb, this, ros::TransportHints().reliable().tcpNoDelay());
    Wrench_Sub          = nh.subscribe(Wrench_Topic, 5, &Control_Strategy::Wrench_Cb, this, ros::TransportHints().reliable().tcpNoDelay());

    Predict_IMG = cv::Mat(predict_map_size(0),predict_map_size(1),CV_8UC1);
    uchar* data = (uchar*)Predict_IMG.data;
    for ( int i=0; i<predict_map_size(0); i++ )
    {
        for ( int j=0; j<predict_map_size(1); j++ )
        {
            int index = i*predict_map_size(0) + j;
            data[index] = 0;
        }
    }
    data[50] = 0;
    logger.save_image(Predict_IMG, 0);
}

void Control_Strategy::Switch_Controller(const int &cognition)
{
    std::string cartesian_position_controller("cartesian_position_controller");
    std::string cartesian_velocity_controller("cartesian_velocity_controller");
    switch (cognition)
    {
    case 0:
        start_controllers.clear();
        stop_controllers.clear();

        start_controllers.push_back(cartesian_position_controller);
        switch_controller_srv.request.start_controllers = start_controllers;
        switch_controller_srv.request.stop_controllers = stop_controllers;
        switch_controller_srv.request.strictness = 2;
        if(switch_controller_client.call(switch_controller_srv))
        {
            ROS_INFO("Switch 'cartesian_position_controller' Successfully.");
        }
        else
        {
            ROS_ERROR("Switch 'cartesian_position_controller' Failed. Please Check Code");
        }
        break;
    case 1:
        start_controllers.clear();
        stop_controllers.clear();

        start_controllers.push_back(cartesian_velocity_controller);
        switch_controller_srv.request.start_controllers = start_controllers;
        switch_controller_srv.request.stop_controllers = stop_controllers;
        switch_controller_srv.request.strictness = 2;
        if(switch_controller_client.call(switch_controller_srv))
        {
            ROS_INFO("Switch 'cartesian_velocity_controller' Successfully.");
        }
        else
        {
            ROS_ERROR("Switch 'cartesian_velocity_controller' Failed. Please Check Code");
        }
        break;
    case 2:
        start_controllers.clear();
        stop_controllers.clear();

        start_controllers.push_back(cartesian_position_controller);
        stop_controllers.push_back(cartesian_velocity_controller);
        switch_controller_srv.request.start_controllers = start_controllers;
        switch_controller_srv.request.stop_controllers = stop_controllers;
        switch_controller_srv.request.strictness = 2;
        if(switch_controller_client.call(switch_controller_srv))
        {
            ROS_INFO("Switch 'cartesian_position_controller' Successfully.");
        }
        else
        {
            ROS_ERROR("Switch 'cartesian_position_controller' Failed. Please Check Code");
        }
        break;
    case 3:
        start_controllers.clear();
        stop_controllers.clear();

        start_controllers.push_back(cartesian_velocity_controller);
        stop_controllers.push_back(cartesian_position_controller);
        switch_controller_srv.request.start_controllers = start_controllers;
        switch_controller_srv.request.stop_controllers = stop_controllers;
        switch_controller_srv.request.strictness = 2;
        if(switch_controller_client.call(switch_controller_srv))
        {
            ROS_INFO("Switch 'cartesian_velocity_controller' Successfully.");
        }
        else
        {
            ROS_ERROR("Switch 'cartesian_velocity_controller' Failed. Please Check Code");
        }
        break;
    default:
        ROS_ERROR("Switch Controller Cognition Failed. Please Check Code and Choose ( 0 or 1 ).");
        break;
    }
}
void Control_Strategy::Go_Home(void)
{
    ros::Rate loop_rate(10);
    geometry_msgs::Pose msg;
    // msg.position.x = home_pose(0);
    // msg.position.y = home_pose(1);
    // msg.position.z = home_pose(2);
    // msg.orientation.x = home_pose(3);
    // msg.orientation.y = home_pose(4);
    // msg.orientation.z = home_pose(5);
    // msg.orientation.w = home_pose(6);
    msg.position.x = home_pose(0);
    msg.position.y = home_pose(1);
    msg.position.z = home_pose(2);
    msg.orientation.x = home_pose(3);
    msg.orientation.y = home_pose(4);
    msg.orientation.z = home_pose(5);
    msg.orientation.w = home_pose(6);
    size_t i = 3;
    while (i>0)
    {
        Cartesian_Pose_Pub.publish(msg);
        ros::spinOnce();
        loop_rate.sleep();
        i--;
    }

}
void Control_Strategy::Go_Work(void)
{
    ros::Rate loop_rate(10);
    geometry_msgs::Pose msg;
    msg.position.x = work_start_pose(0);
    msg.position.y = work_start_pose(1);
    msg.position.z = work_start_pose(2);
    msg.orientation.x = work_start_pose(3);
    msg.orientation.y = work_start_pose(4);
    msg.orientation.z = work_start_pose(5);
    msg.orientation.w = work_start_pose(6);
    size_t i = 3;
    while (i>0)
    {
        Cartesian_Pose_Pub.publish(msg);
        ros::spinOnce();
        loop_rate.sleep();
        i--;
    }
}
void Control_Strategy::Go(Eigen::Vector3d Position)
{
    ros::Rate loop_rate(10);
    geometry_msgs::Pose msg;

    msg.position.x = Position(0);
    msg.position.y = Position(1);
    msg.position.z = Position(2);
    msg.orientation.x = home_pose(3);
    msg.orientation.y = home_pose(4);
    msg.orientation.z = home_pose(5);
    msg.orientation.w = home_pose(6);
    size_t i = 3;
    while (i>0)
    {
        Cartesian_Pose_Pub.publish(msg);
        ros::spinOnce();
        loop_rate.sleep();
        i--;
    }
}
void Control_Strategy::Explore(void)
{
    ros::Rate loop_rate(10);
    geometry_msgs::Twist msg;
    msg.linear.x = 0.0;
    msg.linear.y = 0.05;
    msg.linear.z = 0.0;
    size_t i = 3;
    while (i>0)
    {
        Cartesian_Twist_Pub.publish(msg);
        ros::spinOnce();
        loop_rate.sleep();
        i--;
    }
}

void Control_Strategy::Image_Process(Eigen::Vector3d start_position, Eigen::Vector3d stop_position, bool predict)
{
    Eigen::Vector2d start_pixel, stop_pixel, line_1, line_2, point, vec_1, vec_2;
    double distance;
    start_pixel <<  static_cast<int>((start_position(0) - workspace_limits(0))/(workspace_limits(1) - workspace_limits(0))*predict_map_size(0)),
                    static_cast<int>((start_position(1) - workspace_limits(2))/(workspace_limits(3) - workspace_limits(2))*predict_map_size(1));
    stop_pixel <<   static_cast<int>((stop_position(0) - workspace_limits(0))/(workspace_limits(1) - workspace_limits(0))*predict_map_size(0)),
                    static_cast<int>((stop_position(1) - workspace_limits(2))/(workspace_limits(3) - workspace_limits(2))*predict_map_size(1));
    line_1 << start_pixel(1),start_pixel(0);
    line_2 << stop_pixel(1), stop_pixel(0);
    uchar* data = (uchar*)Predict_IMG.data;
    for (size_t i = 0; i < predict_map_size(0); i++)
    {
        for (size_t j = 0; j < predict_map_size(1); j++)
        {
            point << i, j;
            vec_1 = line_1 - point;
            vec_2 = line_2 - point;
            distance = fabs(vec_1(0)*vec_2(1) - vec_2(0)*vec_1(1))/(line_1 - line_2).norm();
            if (distance < 15)
            {
                if ((vec_1(0)*vec_2(0) + vec_1(1)*vec_2(1)) < 0)
                {
                    if (distance < 3)
                    {
                        int index = i*predict_map_size(0) + j;
                        data[index] = 255;
                    }
                    else if (distance < 5)
                    {
                        int index = i*predict_map_size(0) + j;
                        data[index] = 150;
                    }
                    else if (distance < 10)
                    {
                        int index = i*predict_map_size(0) + j;
                        data[index] = 50;
                    }
                    else
                    {
                        int index = i*predict_map_size(0) + j;
                        data[index] = 20;
                    }
                }
            }
            //TODO: Add actan2(vec_2(0),vec_2(1)) and force sensor.
        }
    }
    logger.save_image(Predict_IMG, 0);
    if (predict)
    {
        /* code */
    }
    else
    {
        /* code */
    }
}

void Control_Strategy::Image_msg_Create()
{
    for (size_t i = 0; i < predict_map_size(0); i++)
    {
        for (size_t j = 0; j < predict_map_size(1); j++)
        {
            Predict_IMG_msg.data.push_back(255);
        }
    }
    Predict_IMG_Pub.publish(Predict_IMG_msg);
}

void Control_Strategy::Wrench_Cb(const geometry_msgs::WrenchStampedConstPtr &msg)
{
    force_x = msg->wrench.force.x;
    force_y = msg->wrench.force.y;
    force_z = msg->wrench.force.z;
    force_x = (1 - 0.2)*force_x_pre + 0.2*force_x;
    force_x_pre = force_x;
    force_y = (1 - 0.2)*force_y_pre + 0.2*force_y;
    force_y_pre = force_y;
    // Eigen::Vector3d force;
    // force << force_x, force_y, force_z;
    // force = filter.LowPassFilter(force);
    logger.save_wrench( force_x, force_y, force_z,
                        msg->wrench.torque.x, msg->wrench.torque.y, msg->wrench.torque.z);
    // logger.save_wrench( msg->wrench.force.x, msg->wrench.force.y, msg->wrench.force.z,
    //                     msg->wrench.torque.x, msg->wrench.torque.y, msg->wrench.torque.z);
}

void Control_Strategy::Cartesian_State_Cb(const cartesian_state_msgs::PoseTwistConstPtr &msg)
{
    Cartesian_State <<  msg->pose.position.x, msg->pose.position.y, msg->pose.position.z,
                        msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z, msg->pose.orientation.w;
}

void Control_Strategy::Force_Check(void)
{
    // std::cout<<(force_x*force_x + force_y*force_y)<<std::endl;
    if((force_x*force_x + force_y*force_y)>3)
    {
        std::cout<<force_x<<" "<<force_y<<std::endl;
        std::cout<<(force_x*force_x + force_y*force_y)<<std::endl;
        ros::Rate loop_rate(10);
        geometry_msgs::Twist msg;
        msg.linear.x = 0.0;
        msg.linear.y = 0.0;
        msg.linear.z = 0.0;
        size_t i = 3;
        while (i>0)
        {
            Cartesian_Twist_Pub.publish(msg);
            ros::spinOnce();
            loop_rate.sleep();
            i--;
        }
        Eigen::Vector3d start_position, stop_position, Pre_grasp_position;
        start_position << 0.0, 0.25, 0.25;
        stop_position << Cartesian_State(0), Cartesian_State(1), Cartesian_State(2);
        Image_Process(start_position, stop_position, true);

        Switch_Controller(2);
        sleep(3);
        Pre_grasp_position << Cartesian_State(0), Cartesian_State(1), home_pose(2);
        Go(Pre_grasp_position);
        Pre_grasp_position << Cartesian_State(0)+force_x*-0.05, Cartesian_State(1)+force_y*-0.05, home_pose(2);
        Go(Pre_grasp_position);
    }
}