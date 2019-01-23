## point cloud map reconstruction based on RGB-D data

### 项目文件说明:

odom+icp.cpp：
核心文件，同时使用了odom数据和icp算法，生成程序ODOM_ICP。

odom_map.cpp:
使用了odom数据，用于生成航位推算结果。

ORB+proc.cpp:
使用ORB特征点作为关键点进行匹配的建图程序。(没有迭代取点的过程)

orb_visualization.cpp:
用于显示orb匹配效果的程序。

### 数据说明
./Data/Sample_1中存放的是从初始序列中每50帧采样得到的数据，其中300-4200帧是向前直行的过程，5200-8300是反向行驶的过程。

### 示例点云结果说明
示例点云放在 ./pre/special\ resualts/下，可以用pcl_viewer打开。
名字中两个数字表示采用帧号的始末，step都是50。

通过对300_3000_noodom.pcd和300_3000_odom_result.pcd的对比可以看出在直道上odom数据提供的初值的作用。
通过对2500_6000_odom.pcd和2500_6000_odom_only.pcd的对比可以看出ICP算法对航位推算的矫正作用。

5000_8300_left.pcd是单纯用航位推算法修正过利用回程帧建立的地图，同时将构建过程从左手系变换到右手系，纠正了点云左右相反的问题
