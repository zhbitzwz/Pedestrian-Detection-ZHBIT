//-----------------------------------------------------------------------------
// 作    者：ZWZ
// 描    述：
// 版    本：
//-----------------------------------------------------------------------------
// 历史更新纪录
//-----------------------------------------------------------------------------
// 版    本：           修改时间：           修改人：          
// 修改内容：
//-----------------------------------------------------------------------------
// Copyright (C) 2016-ZHBITZWZ
//-----------------------------------------------------------------------------
#include <stdio.h>
#include <opencv2/opencv.hpp>
// #include "opencv2/highgui/highgui.hpp"
// #include "opencv2/imgproc/imgproc.hpp"
//#include < opencv2\gpu\gpu.hpp>

using namespace cv;
using namespace std;


#ifdef _DEBUG
#pragma comment(lib, "opencv_core300d.lib")
#pragma comment(lib, "opencv_imgproc300d.lib")   //MAT processing
#pragma comment(lib, "opencv_objdetect300d.lib") //HOGDescriptor
//#pragma comment(lib, "opencv_gpu300d.lib")
//#pragma comment(lib, "opencv_features2d300d.lib")
#pragma comment(lib, "opencv_highgui300d.lib")
//#pragma comment(lib, "opencv_ml300d.lib")
//#pragma comment(lib, "opencv_stitching300d.lib");
//#pragma comment(lib, "opencv_nonfree300d.lib");

#else
#pragma comment(lib, "opencv_core300.lib")
#pragma comment(lib, "opencv_imgproc300.lib")
#pragma comment(lib, "opencv_objdetect300.lib")
//#pragma comment(lib, "opencv_gpu300.lib")
//#pragma comment(lib, "opencv_features2d300.lib")
#pragma comment(lib, "opencv_highgui300.lib")
//#pragma comment(lib, "opencv_ml300.lib")
//#pragma comment(lib, "opencv_stitching300.lib");
//#pragma comment(lib, "opencv_nonfree300.lib");
#endif


int main()
{
 //variables
 char FullFileName[100];
 string filename = "./Extent_source/lena.ppm";
 char SaveHogDesFileName[100] = "Positive.xml";
 int FileNum=1000000;

 vector< vector < float> > v_descriptorsValues;
 vector< vector < Point> > v_locations;


 for(int i=0; i< FileNum; ++i)
 {
  //sprintf_s(FullFileName, "%s%d.png", FirstFileName, i+1);
  printf("%s\n", filename.c_str());

  //read image file
  Mat img, img_gray;
  img = imread(filename);

  if(!img.data) {                       // Check for invalid input
      cout <<  "Could not open or find the image" << std::endl ;
      return -1;
  }

  //resizing调整大小
  //resize(img, img, Size(64,48) ); //Size(64,48) ); //Size(32*2,16*2)); //Size(80,72) );
  //gray灰色图片
  cvtColor(img, img_gray, CV_RGB2GRAY);

  //extract feature特征提取
  HOGDescriptor d( Size(32,16), Size(8,8), Size(4,4), Size(4,4), 9);
  vector< float> descriptorsValues;
  vector< Point> locations;
  d.compute( img_gray, descriptorsValues, Size(0,0), Size(0,0), locations);

  //printf("descriptor number =%d\n", descriptorsValues.size() );
  v_descriptorsValues.push_back( descriptorsValues );
  v_locations.push_back( locations );
  //show image
  imshow("origin", img);

  waitKey(5);
 }

 //refer to this address -> http://feelmare.blogspot.kr/2014/04/the-example-source-code-of-2d-vector.html
 //save to xml保存在xml
 FileStorage hogXml(SaveHogDesFileName, FileStorage::WRITE); //FileStorage::READ
 //2d vector to Mat 二维矢量
 int row=v_descriptorsValues.size(), col=v_descriptorsValues[0].size();
 printf("col=%d, row=%d\n", row, col);
 Mat M(row,col,CV_32F);
 //save Mat to XML
 for(int i=0; i< row; ++i)
  memcpy( &(M.data[col * i * sizeof(float) ]) ,v_descriptorsValues[i].data(),col*sizeof(float));
 //write xml
 write(hogXml, "Descriptor_of_images",  M);

 //write(hogXml, "Descriptor", v_descriptorsValues );
 //write(hogXml, "locations", v_locations );
 hogXml.release();

}
