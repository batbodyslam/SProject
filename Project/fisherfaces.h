#pragma once
#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;

class Fisherfaces
{
private:
	Mat norm_0_255(InputArray _src);
	void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';');
	void saveFisherYML(Ptr<BasicFaceRecognizer> model);
	Ptr<BasicFaceRecognizer> loadFisherYML();
	string output_folder = "/output";
	string fn_csv = "at.txt";
public:
	void run();

};