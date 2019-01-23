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
#include <string>
#include <chrono>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace pcl;

// instinct parameters
double cx = 325.5;
double cy = 253.5;
double fx = 518.0;
double fy = 519.0;
double depthScale = 1800.0;
/////////////////////////////////////
bool U;
typedef pcl::PointXYZRGB PointT;
typedef Eigen::Isometry3d tf;

// 首张位姿 T0
// Eigen::Quaterniond q( 0.993042, -0.0004327, -0.113131, -0.0326832 );
Eigen::Quaterniond q( 1.0, 0.0, 0.0, 0.0 );
Eigen::Isometry3d T0(q);

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

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 20.0 ) )
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

// 给进一组匹配点，利用icp算法将最优的转换矩阵放入TF 中
// ps. e=p1-TFp2
void pose_estimation_3d3d (
    const vector<Point3f>& pts1,
    const vector<Point3f>& pts2,
    Eigen::Matrix4d& TF
){
    Point3f p1, p2;     // center of mass
    int N = pts1.size();
    cout<<"N:"<<N<<endl;
    for ( int i=0; i<N; i++ )
    {
        cout<<i<<pts1[i]<<' '<<pts2[i]<<endl;
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = Point3f( Vec3f(p1) / N);
    p2 = Point3f( Vec3f(p2) / N);
    vector<Point3f>     q1 ( N ), q2 ( N ); // remove the center
    for ( int i=0; i<N; i++ )
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    cout<<"p1,p2:"<<p1<<p2<<endl;
    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for ( int i=0; i<N; i++ )
    {
        W += Eigen::Vector3d ( q1[i].x, q1[i].y, q1[i].z ) * Eigen::Vector3d ( q2[i].x, q2[i].y, q2[i].z ).transpose();
    }
    cout<<"W="<<W<<endl;

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
    
    cout<<"U="<<U<<endl;
    cout<<"V="<<V<<endl;

    Eigen::Matrix3d R = U* ( V.transpose() );
    Eigen::Vector3d t = Eigen::Vector3d ( p1.x, p1.y, p1.z ) - R * Eigen::Vector3d ( p2.x, p2.y, p2.z );

    cout<<"R="<<R<<endl;
    cout<<"T="<<t<<endl;

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


void exportToPCD(vector <Mat>& colorImgs, vector <Mat>& depthImgs, vector<Eigen::Matrix4d> TFs, string save_name){
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

void bgr2gray_scale(Mat &inputImg, Mat &outputImg)
{
    cvtColor(inputImg, outputImg, CV_BGR2GRAY);
    for (int r = 0; r < outputImg.rows; r++)
    {
        for (int c = 0; c < outputImg.cols; c++)
        {
            uchar pix = outputImg.ptr<uchar>(r)[c];
            outputImg.ptr<uchar>(r)[c] = min(255, max(0, 180+(pix-180)*8));
        }
    }

}


int main ( int argc, char** argv )
{
    if ( argc != 5 )
    {
        cout<<"usage: main start_id number_of_image step save_name"<<endl;
        return 1;
    }
    // 读入 照片/深度图 （对应的）序列
    vector <Mat> colorSeq;
    vector <Mat> depthSeq;
    vector <Mat> graySeq;

    int start_id = atoi(argv[1]);
    int img_number = atoi(argv[2]);
    int step = atoi(argv[3]);
    string save_name = argv[4];
    string path = "./Data/";
    string dir_color = "20181215_164023_460_color/";
    string dir_depth = "20181215_164023_460_depth/";

    for(int i = 0; i < img_number; i++){
        Mat rawColor = imread ( path + dir_color + to_string(start_id + i*step)+".jpg", CV_LOAD_IMAGE_COLOR );
        Mat gray;
        bgr2gray_scale(rawColor, gray);
        colorSeq.push_back(rawColor);
        graySeq.push_back(gray);
        depthSeq.push_back(imread ( path + dir_depth + to_string(start_id + i*step)+".png", CV_LOAD_IMAGE_UNCHANGED ));// 深度图为16位无符号数，单通道图像
    }
        
    // 位姿矩阵
    vector<Eigen::Matrix4d> TFs;
    TFs.resize(colorSeq.size());
    for (int i =0;i < 4; i++){
        for (int j = 0; j < 4; j++){
            if (i==j)
                TFs[0](i,j)=1.0;
            else 
                TFs[0](i,j)=0.0;
        }
    }
    
    T0.pretranslate( Eigen::Vector3d( 0.0, 0.0, 0.06 ));

    // 遍历，两两进行匹配
    for(int i=1; i < colorSeq.size(); i++){
        // 以下都是两图匹配的算法

        // 图像特征点准备
        vector<KeyPoint> keypoints_1, keypoints_2;
        vector<DMatch> matches;
        find_feature_matches ( graySeq[i-1], graySeq[i], keypoints_1, keypoints_2, matches );
        cout<<"找到了"<<matches.size()<<"对匹配点。"<<endl;
                
        // 匹配点对（用）
        vector<Point3f> pts1, pts2;

        // 图像特征点到stereo keypoint的转换
        Mat K = ( Mat_<double> ( 3,3 ) << fx, 0, cx, 0, fy, cy, 0, 0, 1 );
        for ( DMatch m:matches )
        {
            int x1 = int ( keypoints_1[m.queryIdx].pt.x );
            int y1 = int ( keypoints_1[m.queryIdx].pt.y );
            int x2 = int ( keypoints_2[m.trainIdx].pt.x );
            int y2 = int ( keypoints_2[m.trainIdx].pt.y );
            ushort d1 = depthSeq[i-1].ptr<unsigned short> (y1) [x1];
            ushort d2 = depthSeq[i].ptr<unsigned short> (y2) [x2];
            if ( d1==0 || d2==0 || (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2) > 120*120){   // bad depth
                continue;
            }
            Point2d p1 = pixel2cam ( keypoints_1[m.queryIdx].pt, K );
            Point2d p2 = pixel2cam ( keypoints_2[m.trainIdx].pt, K );
            float dd1 = float ( d1 ) /depthScale;
            float dd2 = float ( d2 ) /depthScale;
            pts1.push_back ( Point3f ( p1.x*dd1, p1.y*dd1, dd1 ) );
            pts2.push_back ( Point3f ( p2.x*dd2, p2.y*dd2, dd2 ) );
        }
        cout << "有效匹配点: " << pts1.size() << endl;

        pose_estimation_3d3d( pts1, pts2, TFs[i]);
        // tf:相机位姿的转换矩阵

        if(i>0){
            TFs[i]=TFs[i-1]*TFs[i];
        }
    }

    // 根据Poses重新对raw数据进行重建
    exportToPCD(colorSeq, depthSeq, TFs, save_name);
}
