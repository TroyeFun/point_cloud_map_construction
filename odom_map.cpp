#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h> 
#include <pcl/visualization/pcl_visualizer.h>
#include <string>
#include <chrono>
#include <cmath>

using namespace std;
using namespace cv;
using namespace pcl;

typedef pcl::PointXYZRGB PointT;
typedef Eigen::Matrix4d TF;

// instinct parameters
double cx = 325.5;
double cy = 253.5;
double fx = 518.0;
double fy = 519.0;
double depthScale = 1800.0;
double imu_height = 0.1;
double yaw_imu2cam = -2 * M_PI / 180;
TF tf_imu2cam;



string path = "./Data/";
string dir_color = "20181215_164023_460_color/";
string dir_depth = "20181215_164023_460_depth/";
string file_img_time = "20181215_164023_460_color_timestamp.log";
string file_odom = "20181215_164023_611.odom";
string save_name;

/////////////////////////////////////



void exportToPCD(vector <Mat>& colorImgs, vector <Mat>& depthImgs, vector<Eigen::Matrix4d> TFs){
    pcl::PointCloud<PointT>::Ptr pointCloud( new pcl::PointCloud<PointT> );
    for ( int i=0; i<colorImgs.size(); i++ )
    {
        cout<<"转换图像中: "<<i+1<<endl; 
        cv::Mat color = colorImgs[i]; 
        cv::Mat depth = depthImgs[i];
        
        bool flag = false;
        for ( int v=0; v<color.rows; v++ )
            for ( int u=0; u<color.cols; u++ )
            {
                unsigned int d = depth.ptr<unsigned short> (v)[u]; // 深度值
                if ( d==0 ) continue; // 为0表示没有测量到
                Eigen::Vector4d point; 
                /////////////////////////////////////////////////////// 内参needed！
                point[2] = double(d)/depthScale; 
                point[0] = (u-cx)*point[2]/fx;
                point[1] = (v-cy)*point[2]/fy;
                point[3] = 1.0; 
                
                point=TFs[i]*point;
                Eigen::Vector3d pointWorld = Eigen::Vector3d(point[0], point[1], point[2]);
                
                if(!flag){
                    cout<<v<<' '<<u<<endl;
                    cout<<point<<endl;
                    cout<<TFs[i]<<endl;
                    // cout<<pointWorld<<endl;
                    flag=1;
                }
                PointT p ;
                p.x = pointWorld[0];
                p.y = pointWorld[1];
                p.z = pointWorld[2];
                p.b = color.data[ v*color.step+u*color.channels() ];
                p.g = color.data[ v*color.step+u*color.channels()+1 ];
                p.r = color.data[ v*color.step+u*color.channels()+2 ];
                pointCloud->points.push_back( p );
            }
        
    }

    pointCloud->is_dense = false;
    cout<<"点云共有"<<pointCloud->size()<<"个点."<<endl;
    pcl::io::savePCDFileBinary(save_name, *pointCloud );
}



int next_time_stamp(ifstream& fimg_time, int step)
{
    int time_stamp = 0;
    for (int i = 0; i < step; i++)
    {
        fimg_time >> time_stamp;
    }
    return time_stamp;
}


TF get_tf_world2imu(int img_time, int last_odom_time, int next_odom_time, 
    float lastx, float nextx, float lasty , float nexty, float last_yaw, float next_yaw)
{
    // linear interpolation
    float x = 1.0 * (lastx * (next_odom_time - img_time) + nextx * (img_time - last_odom_time))
        / (next_odom_time - last_odom_time);
    float y = 1.0 * (lasty * (next_odom_time - img_time) + nexty * (img_time - last_odom_time))
        / (next_odom_time - last_odom_time);
    float yaw = 1.0 * (last_yaw * (next_odom_time - img_time) + next_yaw * (img_time - last_odom_time))
        / (next_odom_time - last_odom_time);

    TF tf;
    tf << cos(yaw) , -sin(yaw) , 0 , x 
       , sin(yaw) , cos(yaw)  , 0 , y
       , 0        , 0         , 1 , imu_height
       , 0        , 0         , 0 , 1;
    return tf;
}

int main ( int argc, char** argv )
{
    if ( argc != 5 )
    {
        cout<<"usage: odom_map [start_image_id] [number_of_images] [step] [save_name]"<<endl;
        return 1;
    }
    int start_id = stoi(argv[1], NULL, 10);
    int img_number = stoi(argv[2], NULL, 10);
    int step = stoi(argv[3], NULL, 10);
    save_name = argv[4];

    vector<TF> TFs;
    vector <Mat> colorSeq;
    vector <Mat> depthSeq;
    TFs.resize(img_number);
    tf_imu2cam << sin(yaw_imu2cam), 0, cos(yaw_imu2cam), 0.11
                , -cos(yaw_imu2cam), 0, sin(yaw_imu2cam), 0
                , 0, -1, 0, 0
                , 0, 0, 0, 1;


    ifstream fimg_time, fodom; 
    fimg_time.open(path + file_img_time);
    fodom.open(path+file_odom);

    int img_time = next_time_stamp(fimg_time, start_id);
    int odom_time, last_odom_time, last_pulse, pulse, last_backdis, backdis;
    float lastx, x, lasty, y, last_yaw, yaw, last_speed, speed;
    fodom >> last_odom_time >> lastx >> lasty >> last_yaw >> last_speed >> last_pulse >> last_backdis;

    if (last_odom_time > img_time)
    {
        cout <<"No position information at Start Image. Please increase start_image_id." << endl;
        return 1;
    }


    for (int i = 0; i < img_number; i++)
    {

        int img_id = i * step + start_id;
        Mat rgb, depth;
        colorSeq.push_back(imread ( path + dir_color + to_string(img_id) + ".jpg", CV_LOAD_IMAGE_COLOR ));
        depthSeq.push_back(imread ( path + dir_depth + to_string(img_id) + ".png", CV_LOAD_IMAGE_UNCHANGED ));

        while (fodom >> odom_time >> x >> y >> yaw >> speed >> pulse >> backdis)
        {
            if (odom_time >= img_time) {
                cout << "odom_time=" << odom_time << ", img_time=" << img_time << ", last=" << 
                    lastx << ',' << lasty << ',' << last_yaw << ", now=" << x << ','<< y << ',' << yaw << endl;
                break;
            }
            last_odom_time = odom_time;
            lastx = x;
            lasty = y;
            last_yaw = yaw;
            last_speed = speed;
            last_pulse = pulse;
            last_backdis = backdis;
        }

        TF tf_world2imu = get_tf_world2imu(img_time, last_odom_time, odom_time, lastx, x, lasty, y, last_yaw, yaw);
        TFs[i] = tf_world2imu * tf_imu2cam;

        img_time = next_time_stamp(fimg_time, step);

    }

    // 根据Poses重新对raw数据进行重建
    exportToPCD(colorSeq, depthSeq, TFs);
}
