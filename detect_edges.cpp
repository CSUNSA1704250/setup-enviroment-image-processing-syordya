#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <math.h> 

int main( int argc, char** argv ) {
  
  cv::Mat image;
  cv::Mat img_gray;
  cv::Mat img_blur;
  cv::Mat img_answer;
  image = cv::imread("tiger.jpeg" ,cv::IMREAD_COLOR);

  if(!image.data ) {
    std::cout <<  "No hay imagen" << std::endl ;
    return -1;
  }

  cv::cvtColor(image,img_gray,cv::COLOR_BGR2GRAY);
  
  cv::GaussianBlur(img_gray, img_blur, cv::Size(3,3), 0);

  cv::cvtColor( img_blur, img_answer, cv::COLOR_BGR2GRAY );

  cv::Size sz = image.size();
  int imageWidth = sz.width;
  int imageHeight = sz.height;


  std::vector<std::vector<int>> kernel_mat(3);
  for (int i = 0; i < 3; ++i)
  {
    kernel_mat[i].resize(3);
  }

  kernel_mat[0][0] = 1;
  kernel_mat[0][1] = 2;
  kernel_mat[0][2] = 1;

  kernel_mat[1][0] = 0;
  kernel_mat[1][1] = 0;
  kernel_mat[1][2] = 0;

  kernel_mat[2][0] = -1;
  kernel_mat[2][1] = -2;
  kernel_mat[2][2] = -1;
  int x, y;

  for(int i = 1; i < imageHeight-1; i++)
  {
    for(int j = 1; j < imageWidth-1; j++)
    {
      x=img_blur.at<int>(i+1,j-1)*kernel_mat[2][0]+img_blur.at<int>(i+1,j)*kernel_mat[2][1]+img_blur.at<int>(i+1,j+1)*kernel_mat[2][2]+(img_blur.at<int>(i-1,j-1)*kernel_mat[0][0]+img_blur.at<int>(i-1,j)*kernel_mat[0][1]+img_blur.at<int>(i-1,j+1)*kernel_mat[0][2]);
      y=(img_blur.at<int>(i-1,j+1)*kernel_mat[0][2]+img_blur.at<int>(i,j+1)*kernel_mat[1][2]+img_blur.at<int>(i+1,j+1)*kernel_mat[2][2])+(img_blur.at<int>(i-1, j-1)*kernel_mat[0][0]+img_blur.at<int>(i,j-1)*kernel_mat[1][0]+img_blur.at<int>(i+1,j-1)*kernel_mat[2][0]);
      img_answer.at<int>(i-1,j-1)=sqrt(pow(x,2)+pow(y,2));
    }
        
  }
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(img_answer,contours, hierarchy, cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
  cv::drawContours(img_answer,contours,-1,cv::Scalar(255,0,0),1);


  cv::imshow( "OpenCV Test Program", img_answer);
  
  cv::waitKey(0);
  return 0;
}
