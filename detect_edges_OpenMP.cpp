#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <math.h> 
#include "hpc_helpers.hpp"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>



int imageWidth;
int imageHeight;
std::vector<std::vector<float>> kernel_mat(3);
cv::Mat image;  
cv::Mat img_blur;
cv::Mat img_answer;
int cant_threads;

void Convert_grayScale_Blur(cv::Mat &imagen_entrada, cv::Mat &imagen_salida)
{
  cv::Mat img_gray;

  // Conversion a escala de grises
  cv::cvtColor(imagen_entrada,img_gray,cv::COLOR_BGR2GRAY);
  
  // Aplicación de Blur a la imagen con un tamaño de filtro de 3x3
  cv::GaussianBlur(img_gray, imagen_salida, cv::Size(3,3), 0);

}



void filtro_gaussiano_inicializar(std::vector<std::vector<float>> &kernel_mat)
{
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
}



int main( int argc, char** argv ) {
  
  if(argc != 2) 
  {
    std::cout << "Establezca los threads" << std::endl;
    return -1;
  }

 

  // Manejo de threads

  if (argv[1] == NULL) return -1;

  double start, finish;
  cant_threads = strtol(argv[1], NULL, 10);
  omp_set_num_threads(cant_threads);
  
  
  // Lectura de imagen
  image = cv::imread("tiger.jpeg" ,cv::IMREAD_COLOR);

  if(!image.data ) {
    std::cout <<  "No hay imagen" << std::endl ;
    return -1;
  }
  
  // Procesamiento de imagen inicial
  Convert_grayScale_Blur(image, img_blur);

  // Imagen respuesta  
  cv::cvtColor( image, img_answer, cv::COLOR_BGR2GRAY );
 

  //Filtro gausiano
  filtro_gaussiano_inicializar(kernel_mat);


  // Convolución de la matriz por el kernel gaussiano
  cv::Size sz = image.size();
  imageWidth = sz.width;
  imageHeight = sz.height;
 
  TIMERSTART(start);
  uint8_t x, y;
  #pragma omp parallel for private(x, y)
    
    for(int i = 1; i < imageHeight-1; i++)
    {
      for(int j = 1; j < imageWidth-1; j++)
      {
        x=img_blur.at<uint8_t>(i+1,j-1)*kernel_mat[2][0]+img_blur.at<uint8_t>(i+1,j)*kernel_mat[2][1]+img_blur.at<uint8_t>(i+1,j+1)*kernel_mat[2][2]+(img_blur.at<uint8_t>(i-1,j-1)*kernel_mat[0][0]+img_blur.at<uint8_t>(i-1,j)*kernel_mat[0][1]+img_blur.at<uint8_t>(i-1,j+1)*kernel_mat[0][2]);
        y=(img_blur.at<uint8_t>(i-1,j+1)*kernel_mat[0][2]+img_blur.at<uint8_t>(i,j+1)*kernel_mat[1][2]+img_blur.at<uint8_t>(i+1,j+1)*kernel_mat[2][2])+(img_blur.at<uint8_t>(i-1, j-1)*kernel_mat[0][0]+img_blur.at<uint8_t>(i,j-1)*kernel_mat[1][0]+img_blur.at<uint8_t>(i+1,j-1)*kernel_mat[2][0]);
        img_answer.at<uint8_t>(i-1,j-1)=sqrt(x*x+y*y);      
      }       
    }
   TIMERSTOP(start);
   std::cout<<"\t------------------------------------\n";





  // Extracción de contornos de la imagen
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(img_answer,contours, hierarchy, cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

  // Dibujo de contornos en base a los puntos hallados previamente
  cv::drawContours(img_answer,contours,-1,cv::Scalar(255,0,0),1, cv::LINE_8, hierarchy, 0 );
  
  /*
  cv::imshow( "OpenCV Test Program", img_answer);  
  cv::waitKey(0);
  */

  cv::imwrite("bordes_tigre.png", img_answer);
  return 0;
}
