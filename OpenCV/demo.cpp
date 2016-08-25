#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;


#define PosSamNO 2400  //original positive num
#define NegSamNO 2400 // original negative num
#define HardExampleNO 3600 // hard negative num
#define AugPosSamNO 2400 //Aug positive num

#define TRAIN false 
#define CENTRAL_CROP true

int main()
{
	//winsize(64,128),blocksize(16,16),blockstep(8,8),cellsize(8,8),bins9
	HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);
	int DescriptorDim;
	Ptr<SVM> svm = SVM::create();
	if(TRAIN)
	{
		string ImgName;
		ifstream finPos("DATA/INRIAPerson96X160PosList.txt");
		ifstream finNeg("DATA/NoPersonFromINRIAList.txt");

		if (!finPos || !finNeg)
		{
			cout << "Pos/Neg imglist reading failed..." << endl;
			return 1;
		}

		Mat sampleFeatureMat;
		Mat sampleLabelMat;

		//loading original positive examples...
		for(int num=0; num < PosSamNO && getline(finPos,ImgName); num++)
		{
			cout <<"Now processing original positive image: " << ImgName << endl;
			ImgName = "DataSet/INRIAPerson/train_64x128_H96/pos/" + ImgName;
			Mat src = imread(ImgName);

			if(CENTRAL_CROP)
				src = src(Rect(16,16,64,128));

			vector<float> descriptors;
			hog.compute(src, descriptors, Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)


			if( 0 == num )
			{
				DescriptorDim = descriptors.size();
				sampleFeatureMat = Mat::zeros(PosSamNO +AugPosSamNO +NegSamNO +HardExampleNO, DescriptorDim, CV_32FC1);
				sampleLabelMat = Mat::zeros(PosSamNO +AugPosSamNO +NegSamNO +HardExampleNO, 1, CV_32SC1);//sampleLabelMat的数据类型必须为有符号整数型
			}

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for(int i=0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num,i) = descriptors[i];
			sampleLabelMat.at<int>(num,0) = 1;
		}
		finPos.close();

		//positive examples augmenting...
		if (AugPosSamNO > 0)
		{
			ifstream finAug("DATA/AugPosImgList.txt");
			if (!finAug)
			{
				cout << "Aug positive imglist reading failed..." << endl;
				return 1;
			}
			for (int num = 0; num < AugPosSamNO && getline(finAug, ImgName); ++num)
			{
				cout << "Now processing Aug positive image: " << ImgName << endl;
				ImgName = "DATA/INRIAPerson/AugPos/" + ImgName;
				Mat src = imread(ImgName);
				vector<float> descriptors;
				hog.compute(src, descriptors, Size(8,8));
				for (int i = 0; i < DescriptorDim; ++i)
					sampleFeatureMat.at<float>(num +PosSamNO, i) = descriptors[i];
				sampleLabelMat.at<int>(num +PosSamNO, 0) = 1;
			}
			finAug.close();
		}

		//loading original negative examples...
		for(int num = 0; num < NegSamNO && getline(finNeg,ImgName); num++)
		{
			cout<<"Now processing original negative image: "<<ImgName<<endl;
			ImgName = "DATA/INRIAPerson/Neg/" + ImgName;
			Mat src = imread(ImgName);

			vector<float> descriptors;
			hog.compute(src,descriptors,Size(8,8));

			for(int i=0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num+PosSamNO,i) = descriptors[i];
			sampleLabelMat.at<int>(num +PosSamNO +AugPosSamNO, 0) = -1;

		}
		finNeg.close();

		//loading hard examples...
		if(HardExampleNO > 0)
		{
			ifstream finHardExample("DATA/INRIAPersonHardNegList.txt");
			if (!finHardExample)
			{
				cout << "HardExample list reading failed..." << endl;
				return 1;
			}

			for(int num=0; num < HardExampleNO && getline(finHardExample, ImgName); num++)
			{
				cout<<"Now processing hard negative image: "<<ImgName<<endl;
				ImgName = "DATA/INRIAPerson/HardNeg/" + ImgName;
				Mat src = imread(ImgName);

				vector<float> descriptors;
				hog.compute(src,descriptors,Size(8,8));

				for(int i=0; i<DescriptorDim; i++)
					sampleFeatureMat.at<float>(num+PosSamNO+NegSamNO,i) = descriptors[i];
				sampleLabelMat.at<int>(num +PosSamNO +AugPosSamNO +NegSamNO, 0) = -1;
			}
			finHardExample.close();
		}

		svm ->setType(SVM::C_SVC);
		svm ->setC(0.01);
		svm ->setKernel(SVM::LINEAR);
		svm ->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 3000, 1e-6));

		cout<<"Starting training..."<<endl;
		svm ->train(sampleFeatureMat, ROW_SAMPLE, sampleLabelMat);
		cout<<"Finishing training..."<<endl;

		svm ->save("SVM_HOG.xml");

	}
	else {
		svm = SVM::load<SVM>("SVM_HOG_1.xml"); //或者svm = Statmodel::load<SVM>("SVM_HOG.xml"); static function
		// svm->load<SVM>("SVM_HOG.xml"); 这样使用不行
	}

	Mat svecsmat = svm ->getSupportVectors();//svecsmat元素的数据类型为float
	int svdim = svm ->getVarCount();//特征向量位数
	int numofsv = svecsmat.rows;

	Mat alphamat = Mat::zeros(numofsv, svdim, CV_32F);//alphamat和svindex必须初始化，否则getDecisionFunction()函数会报错
	Mat svindex = Mat::zeros(1, numofsv,CV_64F);

	Mat Result;
	double rho = svm ->getDecisionFunction(0, alphamat, svindex);
	alphamat.convertTo(alphamat, CV_32F);//将alphamat元素的数据类型重新转成CV_32F
	Result = -1 * alphamat * svecsmat;//float

	vector<float> vec;
	for (int i = 0; i < svdim; ++i)
	{
		vec.push_back(Result.at<float>(0, i));
	}
	vec.push_back(rho);

	//saving HOGDetectorForOpenCV.txt
	/*ofstream fout("HOGDetectorForOpenCV.txt");
	for (int i = 0; i < vec.size(); ++i)
	{
	fout << vec[i] << endl;
	}
	*/
	/*********************************Testing**************************************************/

	HOGDescriptor hog_test;
	hog_test.setSVMDetector(vec);

	VideoCapture capture("4.avi");
	while(1)
	{
		//Mat src = imread("");
		Mat src;
		capture>>src;
		vector<Rect> found, found_filtered;
		hog_test.detectMultiScale(src, found, 0, Size(8,8), Size(32,32), 1.05, 2);

		cout<<"found.size 人数 : "<<found.size()<<endl;

		//找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中
		for(int i=0; i < found.size(); i++)
		{
			Rect r = found[i];
			int j=0;
			for(; j < found.size(); j++)
				if(j != i && (r & found[j]) == r)
					break;
			if( j == found.size())
				found_filtered.push_back(r);
		}


		//画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整
		for(int i=0; i<found_filtered.size(); i++)
		{
			Rect r = found_filtered[i];
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.07);
			r.height = cvRound(r.height*0.8);
			rectangle(src, r.tl(), r.br(), Scalar(0,255,0), 3);
		}

		//imwrite("ImgProcessed.jpg",src);
		namedWindow("src",0);
		imshow("src",src);
		if(waitKey(33) == 27) break;
	}
	return 0;
}
