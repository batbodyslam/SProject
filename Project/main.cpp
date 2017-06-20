#include <iostream>
#include <sstream>

#include "app.h"
#include "createCSV.h"
#include "eigenfaces.h"
int main(int argc, char* argv[])
{
	
	createCSV csv;
	csv.run();
	Eigenfaces _eigen;
	_eigen.run();

	/* //kinect
	try {
		Kinect kinect;
		kinect.run();
	}
	catch (std::exception& ex) {
		std::cout << ex.what() << std::endl;
	}
	*/
	waitKey(0);

	return 0;
}