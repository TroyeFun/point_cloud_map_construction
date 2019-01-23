cmake_minimum_required( VERSION 2.8 )
project( HW )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# opencv
find_package( OpenCV 3.1 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# eigen
include_directories( "/usr/include/eigen3/" )

# pcl 
find_package( PCL REQUIRED ) #COMPONENT common io 
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )


add_executable( ICP_PROC icp_proc.cpp )
target_link_libraries( ICP_PROC 
   ${OpenCV_LIBS}
   ${PCL_LIBRARIES}
)

add_executable( ODOM_ICP odom+icp.cpp )
target_link_libraries( ODOM_ICP 
   ${OpenCV_LIBS}
   ${PCL_LIBRARIES}
)