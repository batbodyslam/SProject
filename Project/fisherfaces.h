#pragma once
#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdio>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;

class Fisherfaces
{
private:
	Mat norm_0_255(InputArray _src);
	void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';');
	void trainFisher(string csv);
	void predictSample(string path, int testLabel);
	Ptr<BasicFaceRecognizer> loadFisherYML();
	string output_folder = "/output";
	int widthData = 92;
	int heightData = 112;
	string fn_csv = "at.txt";
	clock_t start;
	double duration;
public:
	void run();
	Ptr<BasicFaceRecognizer> initializeFisher(double threshold);
	void predict(Mat predictSample, int testLabel);
};