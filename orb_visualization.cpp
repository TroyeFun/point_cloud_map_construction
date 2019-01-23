#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <string>
#include <chrono>

#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
/////////////////////////////////////
bool U;
typedef Eigen::Isometry3d tf;
string path = "./Data/20181215_164023_460_color/";
string depthpath = "./Data/20181215_164023_460_depth/";


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

// erase pixels whose corresponding depth is 0
void erase(Mat& dstImage, int imgID, vector<DMatch>& matches, vector<KeyPoint>& keypoints_1, vector<KeyPoint>& keypoints_2)
{
    Mat dep1, dep2;
    dep1=imread ( depthpath+to_string(imgID)+".png", CV_LOAD_IMAGE_UNCHANGED );
    dep2=imread ( depthpath+to_string(imgID+50)+".png", CV_LOAD_IMAGE_UNCHANGED );
    for (int r = 0; r < dep1.rows; r++)
    {
        for (int c = 0; c < dep1.cols; c++)
        {
            unsigned int d1 = dep1.ptr<unsigned short> (r)[c];            
            if (d1 == 0)
            {
                dstImage.at<Vec3b>(r, c)[0] = 0;
                dstImage.at<Vec3b>(r, c)[1] = 0;
                dstImage.at<Vec3b>(r, c)[2] = 0;
            }
            unsigned int d2 = dep2.ptr<unsigned short> (r)[c];            
            if (d2 == 0)
            {
                dstImage.at<Vec3b>(r, c + dep1.cols)[0] = 0;
                dstImage.at<Vec3b>(r, c + dep1.cols)[1] = 0;
                dstImage.at<Vec3b>(r, c + dep1.cols)[2] = 0;
            }
        }
    }

    int validMatch = 0;
    for ( DMatch m:matches )
    {
        int x1 = int ( keypoints_1[m.queryIdx].pt.x );
        int y1 = int ( keypoints_1[m.queryIdx].pt.y );
        int x2 = int ( keypoints_2[m.trainIdx].pt.x );
        int y2 = int ( keypoints_2[m.trainIdx].pt.y );
        ushort d1 = dep1.ptr<unsigned short> (y1) [x1];
        ushort d2 = dep2.ptr<unsigned short> (y2) [x2];
        if ( d1==0 || d2==0 || (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2) > 120*120){   // bad depth
            continue;
        }
        validMatch++;
    }
    cout << "有效匹配点" << validMatch << endl;
}

int main ( int argc, char** argv )
{
    Mat im1, im2, proc_im1, proc_im2;
    Mat dstImage;

    namedWindow("image[Orb]", CV_WINDOW_NORMAL);

    int num1 = atoi(argv[1]);
    im1=imread ( path+to_string(num1)+".jpg", CV_LOAD_IMAGE_COLOR );
    im2=imread ( path+to_string(num1+50)+".jpg", CV_LOAD_IMAGE_COLOR );
    
    bgr2gray_scale(im1, proc_im1);
    bgr2gray_scale(im2, proc_im2);

    // imshow("gray", proc_im1);
    // cv::waitKey(0);

    // 以下都是两图匹配的算法

    // 图像特征点准备
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches ( proc_im1, proc_im2, keypoints_1, keypoints_2, matches );

    // 距离筛选：只取0.6～5m之间的点作为rgbd相机的置信距离
    // for(int i = 0; i < matches.size(); i++){
    //     if (matches[i].)
    // }

    cout<<"找到了"<<matches.size()<<"对匹配点。"<<endl;
    cout << keypoints_1.size() << endl;

    //绘制并显示匹配窗口 
    cv::Mat resultImage; 

    // drawMatches(im1,keypoints_1, im2,keypoints_2,matches,dstImage);

    drawMatches(proc_im1,keypoints_1, proc_im2,keypoints_2,matches,dstImage);
    // erase(dstImage, num1, matches, keypoints_1, keypoints_2); 
    
    imshow("image[Orb]",dstImage); 

    cv::waitKey(0); 
    return 0; 

}
