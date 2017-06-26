#include <iostream>
#include <sstream>

#include "app.h"
#include "createCSV.h"
#include "eigenfaces.h"
#include "fisherfaces.h"
int main(int argc, char* argv[])
{
	
	//createCSV csv;
	//csv.run();
	Mat testSample = imread("C:\\Users\\Pete\\Documents\\Visual Studio 2015\\Projects\\Project\\Project\\faceMat.png");
	
	/*
	imshow("TestSample", testSample);
	
	Eigenfaces _eigen;
	std::cout << "eigenfaces running...." << endl;
	_eigen.predict(testSample, 20);
	std::cout << "eigenfaces Finnish...." << endl;
	Fisherfaces _fisher;
	std::cout << "fisherfaces running...." << endl;
	_fisher.predict(testSample, 20);
	std::cout << "fihserfaces Finish...." << endl;
	*/

	 //kinect
	try {
		Kinect kinect;
		kinect.run();
	}
	catch (std::exception& ex) {
		std::cout << ex.what() << std::endl;
	}
	
	waitKey(0);
	return 0;
}