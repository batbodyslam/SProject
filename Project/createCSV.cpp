#include "createCSV.h"

void createCSV::run(){

	ofstream myfile;
	myfile.open("at.txt");
	std::cout << "Creating CSV Files....." << std::endl;
	if (myfile.is_open())
	{
		for(int i=1;i<20;i++)
			for (int j = 1; j < 6; j++) {
				myfile << "C:/Users/Pete/Documents/Visual Studio 2015/Projects/Project/Project/att_faces/s" + to_string(i) + "/" + to_string(j) + ".pgm;" + to_string(i - 1) + "\n";
			}
		for (int i = 1; i<=3; i++)
			for(int j=1;j<=10;j++){
				myfile << "C:/Users/Pete/Documents/Visual Studio 2015/Projects/Project/Project/att_faces/p"+ to_string(i)+"/"+ "testface" + to_string(j) +  ".png;"+to_string(20+i-1) +"\n";
			}
			myfile.close();
	}
	else cout << "Unable to open file";
	cout << "Finish create CSV Files" << endl;
}