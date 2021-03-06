#pragma once
#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <ctime>

#include "createCSV.h"

using namespace cv;
using namespace cv::face;
using namespace std;

class Eigenfaces
{
private:
	Mat norm_0_255(InputArray _src);
	void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';');
	void predictSample(string path,int testLabel);
	Ptr<BasicFaceRecognizer> loadEigenYML();
	int widthData = 92;
	int heightData = 112;
	string output_folder = "output/";
	string fn_csv = "at.txt";
	clock_t start;
	double duration;
	createCSV CSV;

public :
	void run();
	Ptr<BasicFaceRecognizer> initializeEigen(double threshold);
	void trainEigen(string csv);
	void predict(Mat predictSample, int testLabel);
};