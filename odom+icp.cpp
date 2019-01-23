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
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/convolution_3d.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/geometry.h>

#include <string>
#include <chrono>
#include <cmath>

using namespace std;
using namespace cv;
using namespace pcl;
using namespace Eigen;


/////////////////////////////////////

bool U;
typedef pcl::PointXYZRGB PointT;
typedef Eigen::Matrix4d TF;
typedef Eigen::Isometry3d tf;

// instinct parameters
double cx = 325.5;
double cy = 253.5;
double fx = 518.0;
double fy = 519.0;
double depthScale = 1800.0;
double imu_height = 0.1;
double yaw_imu2cam = -2 * M_PI / 180;
TF tf_imu2cam;
Mat K = ( Mat_<double> ( 3,3 ) << fx, 0, cx, 0, fy, cy, 0, 0, 1 );

int recallNum=5;
int windowsize=5;
int maxIterationN=200;
int IterationN=250;
pcl::PointCloud<PointT>::Ptr final_pc( new pcl::PointCloud<PointT> );

vector <Mat> colorSeq;
vector <Mat> depthSeq;
vector<TF> TFs;

string path = "./Data/";
string dir_color = "20181215_164023_460_color/";
string dir_depth = "20181215_164023_460_depth/";
string file_img_time = "20181215_164023_460_color_timestamp.log";
string file_odom = "20181215_164023_611.odom";
string save_name;

// 提供rgb图，将匹配到的特征点放入keypoint1/2，以及将匹配组合放入matches
void find_feature_matches (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches )
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 iHammng 距离
    vector<DMatch> match;
   // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    //printf ( "-- Max dist : %f \n", max_dist );
   // printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 50.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
}

// 像素坐标转相机归一化坐标
Point2d pixel2cam ( const Point2d& p, const Mat& K ){
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}

// frame2pc得到的是未经过TFs投影的结果
pcl::PointCloud<PointT> frame2pc(int i){
    // 将当前帧转化为点云
    pcl::PointCloud<PointT> p_cloud;

    cv::Mat color = colorSeq[i]; 
    cv::Mat depth = depthSeq[i];
    
    for ( int v=0; v<color.rows; v++ )
        for ( int u=60; u<color.cols; u++ )
        {
            unsigned int d = depth.ptr<unsigned short> (v)[u]; // 深度值
            if ( d==0 ) continue; // 为0表示没有测量到
            Eigen::Vector4d point; 

            point[2] = double(d)/depthScale; 
            
            // 对点云的深度筛选
            if(point[2] > 5)
                continue;
            
            point[0] = (u-cx)*point[2]/fx;
            point[1] = (v-cy)*point[2]/fy;
            point[3] = 1.0; 
            
            //point=TFs[i]*point;
            
            Eigen::Vector3d pointWorld = Eigen::Vector3d(point[0], point[1], point[2]);
            
            PointT p ;
            p.x = pointWorld[0];
            p.y = pointWorld[1];
            p.z = pointWorld[2];
            p.b = color.data[ v*color.step+u*color.channels() ];
            p.g = color.data[ v*color.step+u*color.channels()+1 ];
            p.r = color.data[ v*color.step+u*color.channels()+2 ];
            p_cloud.points.push_back( p );
        }

    return p_cloud;
}

// 给进一组匹配点，利用SVD算法将最优的转换矩阵放入TF 中
// ps. e=p1-TFp2
void pose_estimation_3d3d (
    const vector<Vector4d>& pts1,
    const vector<Vector4d>& pts2,
    TF& TF
){
    Vector4d p1=Vector4d::Zero();
    Vector4d p2=Vector4d::Zero();   // center of mass
    
    int N = pts1.size();

    for ( int i=0; i<N; i++ )
    {
        p1 += pts1[i];
        p2 += pts2[i];
    }

    p1 = Vector4d( (p1) / N);
    p2 = Vector4d( (p2) / N);
    vector<Vector4d>     q1 ( N ), q2 ( N ); // remove the center
    for ( int i=0; i<N; i++ )
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
        q1[i][3]=q2[i][3]=1.0;
    }
    
    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for ( int i=0; i<N; i++ )
    {
        W += Eigen::Vector3d ( q1[i][0], q1[i][1], q1[i][2] ) * Eigen::Vector3d ( q2[i][0], q2[i][1], q2[i][2] ).transpose();
    }
    //cout<<"W="<<W<<endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd ( W, Eigen::ComputeFullU|Eigen::ComputeFullV );
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    
    if (U.determinant() * V.determinant() < 0)
	{
        for (int x = 0; x < 3; ++x)
        {
            U(x, 2) *= -1;
        }
	}

    Eigen::Matrix3d R = U* ( V.transpose() );
    Eigen::Vector3d t = Eigen::Vector3d ( p1[0], p1[1], p1[2] ) - R * Eigen::Vector3d ( p2[0], p2[1], p2[2] );

    for(int i = 0; i < 4; i ++){
        for (int j =0; j < 4; j++){
            if(i<3 && j <3)
                TF(i,j)=R(i,j);
            if (j==3)
                TF(i,j)=t(i);
            if (i==3){
                if (j==3)
                    TF(i,j)=1;
                else TF(i,j)=0;
            }
        }
    }

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

void odom(int s, int numOfImages)
{
    int start_id = s;
    int img_number = numOfImages;
    int step = 50;

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
        return ;
    }

    for (int i = 0; i < img_number; i++)
    {
        int img_id = i * step + start_id;

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
}

void preprocessing(pcl::PointCloud<PointT>::Ptr inputcloud){
        //Create the input and filtered cloud objects
    typedef pcl::PointXYZRGB PointT;
    
    pcl::PointCloud<PointT>::Ptr nearcloud (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr Gaussiancloud (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr Sampledcloud (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr outputcloud (new pcl::PointCloud<PointT>);


    // 距离过滤
    for(int i = 0; i < inputcloud->points.size(); i++){
        if(inputcloud->points[i].z>0.6 && inputcloud->points[i].z<5)
            nearcloud->points.push_back(inputcloud->points[i]);
    }

    // 增采样
    pcl::MovingLeastSquares<PointT, PointT> filter;
    filter.setInputCloud(nearcloud);
    //建立搜索对象 
    pcl::search::KdTree<PointT>::Ptr kdtree2;
    filter.setSearchMethod(kdtree2);
    //设置搜索邻域的半径为3cm 
    filter.setSearchRadius(0.03);
    // Upsampling 采样的方法有 DISTINCT_CLOUD, RANDOM_UNIFORM_DENSITY 
    filter.setUpsamplingMethod(pcl::MovingLeastSquares<PointT, PointT>::SAMPLE_LOCAL_PLANE);
    // 采样的半径是 
    filter.setUpsamplingRadius(0.03);
    // 采样步数的大小 
    filter.setUpsamplingStepSize(0.02);

    filter.process(*Sampledcloud);
   // std::cout << "upSampled" << std::endl;

    // 下采样
    // 创建滤波对象 
    pcl::VoxelGrid<PointT> sampler;
    sampler.setInputCloud(Sampledcloud);
    // 设置体素栅格的大小为 1x1x1cm 
    sampler.setLeafSize(0.015f, 0.015f, 0.015f);
    sampler.filter(*inputcloud);
   // std::cout << "downSampled" << std::endl;
}

int main ( int argc, char** argv )
{
    
    if ( argc != 3 )
    {
        cout<<"usage: main beginFrameNumber endFrameNumber(in sample_1 data) "<<endl;
        return 1;
    }
    // 读入 照片/深度图 （对应的）序列

    int s=strtol(argv[1], NULL, 10);
    int e=strtol(argv[2], NULL, 10);

    for(int i = s; i <= e; i+=50){
        colorSeq.push_back(imread ( "./Data/color/"+to_string(i)+".jpg", CV_LOAD_IMAGE_COLOR ));
        depthSeq.push_back(imread ( "./Data/depth/"+to_string(i)+".png", CV_LOAD_IMAGE_UNCHANGED ));// 深度图为16位无符号数，单通道图像
    }
        
    // 位姿矩阵
    // 航位推算得到TF初值
    odom(s, (e-s)/50+1);
    for (int i =0; i < TFs.size(); i++){
        cout<<TFs[i]<<endl;
    }
    pcl::PointCloud<PointT>::Ptr unItedfinal( new pcl::PointCloud<PointT> );

    cout<<"preprocessing the first frame..."<<endl;

    // 转换第一帧为点云
    pcl::PointCloud<PointT>::Ptr p_cloud( new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr p_cloud_odom( new pcl::PointCloud<PointT>);
    *p_cloud = frame2pc(0);
    transformPointCloud(*p_cloud, *p_cloud_odom, TFs[0]);
    *unItedfinal+=*p_cloud_odom;
    *final_pc+=*p_cloud_odom;

    //preprocessing(final_pc);
    pcl::VoxelGrid<PointT> sampler2;
    sampler2.setInputCloud(final_pc);
    sampler2.setLeafSize(0.01f, 0.01f, 0.01f);
    
    vector<double> residuals;
    // 一帧帧加入
    for(int i=1; i < colorSeq.size(); i++){
        cout<<"\n\n\n////////////////////////////\ninserting frame"<<i*50+s<<endl;
        residuals.push_back(0.0);
        int preMark=i*50+s;
        if(preMark >= 4200 && preMark<=5200){
            cout<<"skip!"<<endl;
            continue;
        }

        // 匹配点集
        int recallNumT;
        if (i < recallNum){
            recallNumT=i;
        }
        // vector<Vector4d> ORB_p_pts, ORB_f_pts; // ORB
        vector<Vector4d> Closest_p_pts, Closest_f_pts; // closest 

        cout<<" building pointcloud for present frame..."<<endl;
        pcl::PointCloud<PointT>::Ptr p_cloud( new pcl::PointCloud<PointT>);
        pcl::PointCloud<PointT>::Ptr p_cloud_odom( new pcl::PointCloud<PointT>);
        *p_cloud = frame2pc(i);
        
        // 最近点对准备
            // 下采样
        pcl::PointCloud<PointT>::Ptr samplePcTemp( new pcl::PointCloud<PointT> );
        pcl::PointCloud<PointT>::Ptr samplePc( new pcl::PointCloud<PointT> );

        pcl::VoxelGrid<PointT> sampler;
        sampler.setInputCloud(p_cloud);
        sampler.setLeafSize(0.08f, 0.08f, 0.5f);
        sampler.filter(*samplePcTemp);

            // 距离过滤
        for(int i = 0; i < samplePcTemp->points.size(); i++){
            if(samplePcTemp->points[i].z<4) // 被TF 转换以后纵深的轴变成x
                samplePc->points.push_back(samplePcTemp->points[i]);
        }
        transformPointCloud(*samplePc, *samplePc, TFs[i]);

        transformPointCloud(*p_cloud, *p_cloud_odom, TFs[i]);
        *unItedfinal+=*p_cloud_odom;

        cout<<" preprocessing..."<<endl;
        // 对当前帧的预处理 preprocessing
        //preprocessing(p_cloud);
        cout<<" preprocessing done. start sampling..."<<endl;
        
        // bonus：点云特征匹配        

        cout<<" sampling done, Get "<<samplePc->points.size()<<" pairs of points, start iteration..."<<endl;
        if (samplePc->points.size()!=0){
            pcl::io::savePCDFileBinary("sample.pcd", *samplePc );
        }
        
        
        // 开始迭代
        cout<<"     iteration round:\n";
        for(int k = 0; ; k++){
            cout<<k<<' ';
            double sum = 0.0;

            pcl::KdTreeFLANN<PointT> kdtree; 
            kdtree.setInputCloud (final_pc);

            for (int p = 0; p < samplePc->points.size(); p++){
                vector<int> pointIdxNKNSearch(1);
                vector<float> pointNKNSquaredDistance(1);
                if ( kdtree.nearestKSearch (samplePc->points[p], 1, pointIdxNKNSearch, pointNKNSquaredDistance) <= 0 ){   
                    cout<<"can;t find one!"<<endl;
                }

                if(pointNKNSquaredDistance[0] > 0.8)
                    continue;

                Closest_f_pts.push_back(Vector4d(final_pc->points[ pointIdxNKNSearch[0] ].x, final_pc->points[ pointIdxNKNSearch[0] ].y, final_pc->points[ pointIdxNKNSearch[0] ].z, 1.0));
                Closest_p_pts.push_back(Vector4d(samplePc->points[p].x, samplePc->points[p].y, samplePc->points[p].z, 1.0));
            }

            vector<Vector4d> p_pts, f_pts;
            //p_pts.insert(p_pts.end(),ORB_p_pts.begin(),ORB_p_pts.end());
            p_pts.insert(p_pts.end(),Closest_p_pts.begin(),Closest_p_pts.end());
            
            //f_pts.insert(f_pts.end(),ORB_f_pts.begin(),ORB_f_pts.end());
            f_pts.insert(f_pts.end(),Closest_f_pts.begin(),Closest_f_pts.end());
            
            sum = 0.0;
            for (int i = 0; i < p_pts.size(); i++){
                float d = (f_pts[i]-p_pts[i]).norm();
                sum += d;
            }
            if(k%20==0)
                cout<<sum<<' '<<endl;

            *residuals.rbegin() = sum;


            TF tempTF;
            pose_estimation_3d3d( f_pts, p_pts, tempTF);


            transformPointCloud (*samplePc, *samplePc, tempTF);
            
            //if ( k > maxIterationN || ((k > IterationN ||(tempTF-TF::Identity()).maxCoeff()<1e-6) && sum < 10.0)){
            if (((tempTF-TF::Identity()).maxCoeff()<1e-6 && sum < 10.0) || k > maxIterationN){
                cout<<"converged after "<<k<<" rounds."<<endl;
                break;
            }
            else{
                TFs[i] =  tempTF*TFs[i];
            }

            Closest_p_pts.clear();
            Closest_f_pts.clear();


        }
        cout<<" iteration done, with result:\n"<<TFs[i]<<endl;

        
        if(residuals[i] > 30.0)
            continue;
        // 放到一起，然后整理
        pcl::PointCloud<PointT>::Ptr transformed_pc( new pcl::PointCloud<PointT> );
        pcl::transformPointCloud (*p_cloud, *transformed_pc, TFs[i]);

        cout<<"点云共有"<<final_pc->size()<<"个点."<<endl;
        *final_pc+=*transformed_pc;
        sampler2.filter(*final_pc);

        cout<<" finish mixing."<<endl;
    }
    for (int i =0; i < TFs.size(); i++){
        cout<<TFs[i]<<endl;
        cout<<residuals[i]<<endl;
    }

    // 对航位推算结果进行采样生成点云
    pcl::VoxelGrid<PointT> sampler3;
    sampler3.setInputCloud(unItedfinal);
    sampler3.setLeafSize(0.01f, 0.01f, 0.01f);
    sampler3.filter(*unItedfinal);
    pcl::io::savePCDFileBinary("unIted_result.pcd", *unItedfinal );

    // 输出
    final_pc->is_dense = false;
    cout<<"点云共有"<<final_pc->size()<<"个点."<<endl;
    pcl::io::savePCDFileBinary(to_string(s)+"_"+to_string(e)+"odom_result.pcd", *final_pc );
}
