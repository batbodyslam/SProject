/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */

#include "fisherfaces.h"

using namespace cv;
using namespace cv::face;
using namespace std;

Mat Fisherfaces::norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

void Fisherfaces::read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator ) {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(Error::StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}



void Fisherfaces::trainFisher(string fn_csv) {

	// These vectors hold the images and corresponding labels.
	vector<Mat> images;
	vector<int> labels;
	// Read in the data. This can fail if no valid
	// input filename is given.
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}

	int height = images[0].rows;
	int width = images[0].cols;
	int channel = images[0].channels();
	cout << "height = " << height << " width =" << width << " channels=" << channel << endl;

	// Quit if there are not enough images for this demo.
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(Error::StsError, error_message);
	}

	Ptr<BasicFaceRecognizer> trainModel = createFisherFaceRecognizer();
	trainModel->train(images, labels);
	trainModel->save("fisherfaces_at.yml");
}


Ptr<BasicFaceRecognizer> Fisherfaces::loadFisherYML() {
	Ptr<BasicFaceRecognizer> load = createFisherFaceRecognizer();
	load->load("fisherfaces_at.yml");
	if (load.empty()) {
		throw std::runtime_error("failed cv::face::FaceRecognizer::load()");
	}
	return load;
}

Ptr<BasicFaceRecognizer> Fisherfaces::initializeFisher(double threshold) {
	Ptr<BasicFaceRecognizer> _fisherRecognize = createFisherFaceRecognizer();
	_fisherRecognize = loadFisherYML();
	_fisherRecognize->setThreshold(threshold);
	return _fisherRecognize;
}

void Fisherfaces::predictSample(string path, int testLabel) {

	//Mat testSample = imread("C:\\Users\\Pete\\Documents\\Visual Studio 2015\\Projects\\Project\\Project\\s1\\1.pgm");
	Mat testSample = imread(path);
	cv::resize(testSample, testSample, cv::Size(widthData, heightData));
	cv::cvtColor(testSample, testSample, CV_BGR2GRAY);
	cout << "testSample Height" << testSample.rows << "Width " << testSample.cols << "CH " << testSample.channels();

	Ptr<BasicFaceRecognizer> model = createFisherFaceRecognizer();
	//model->train(images, labels);
	model = loadFisherYML();
	// The following line predicts the label of a given
	// test image:
	int predictedLabel = model->predict(testSample);
	//
	// To get the confidence of a prediction call the model with:
	//
	//      int predictedLabel = -1;
	//      double confidence = 0.0;
	//      model->predict(testSample, predictedLabel, confidence);
	//
	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	cout << result_message << endl;
	duration = (clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "duration = " << duration << endl;


}

void Fisherfaces::predict(Mat predictSample, int testLabel) {

	//Mat testSample = imread("C:\\Users\\Pete\\Documents\\Visual Studio 2015\\Projects\\Project\\Project\\s1\\1.pgm");
	//Mat testSample = imread(path);
	Mat testSample = predictSample;
	cv::resize(testSample, testSample, cv::Size(widthData, heightData));
	cv::cvtColor(testSample, testSample, CV_BGR2GRAY);
	cout << "testSample Height" << testSample.rows << "Width " << testSample.cols << "CH " << testSample.channels();

	Ptr<BasicFaceRecognizer> model = createFisherFaceRecognizer();
	//model->train(images, labels);
	model = loadFisherYML();
	// The following line predicts the label of a given
	// test image:
	int predictedLabel = model->predict(testSample);
	//
	// To get the confidence of a prediction call the model with:
	//
	//      int predictedLabel = -1;
	//      double confidence = 0.0;
	//      model->predict(testSample, predictedLabel, confidence);
	//
	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	cout << result_message << endl;
	duration = (clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "duration = " << duration << endl;


}

void Fisherfaces::run() {
	start = clock();
	
    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    vector<int> labels;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Quit if there are not enough images for this demo.
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(Error::StsError, error_message);
    }
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size:
    int height = images[0].rows;
    // The following lines simply get the last images from
    // your dataset and remove it from the vector. This is
    // done, so that the training data (which we learn the
    // cv::BasicFaceRecognizer on) and the test data we test
    // the model with, do not overlap.
	Mat testSample = imread("C:\\Users\\Pete\\Documents\\Visual Studio 2015\\Projects\\Project\\Project\\s1\\1.pgm");
    //Mat testSample = images[images.size() - 1];
    int testLabel = 1;
	imshow("testSample", testSample);

    images.pop_back();
    labels.pop_back();
    // The following lines create an Fisherfaces model for
    // face recognition and train it with the images and
    // labels read from the given CSV file.
    // If you just want to keep 10 Fisherfaces, then call
    // the factory method like this:
    //
    //      cv::createFisherFaceRecognizer(10);
    //
    // However it is not useful to discard Fisherfaces! Please
    // always try to use _all_ available Fisherfaces for
    // classification.
    //
    // If you want to create a FaceRecognizer with a
    // confidence threshold (e.g. 123.0) and use _all_
    // Fisherfaces, then call it with:
    //
    //      cv::createFisherFaceRecognizer(0, 123.0);
    //
    Ptr<BasicFaceRecognizer> model = createFisherFaceRecognizer();
    //model->train(images, labels);
	//trainFisher(images, labels);
	model = loadFisherYML();

    // The following line predicts the label of a given
    // test image:
    int predictedLabel = model->predict(testSample);
    //
    // To get the confidence of a prediction call the model with:
    //
    //      int predictedLabel = -1;
    //      double confidence = 0.0;
    //      model->predict(testSample, predictedLabel, confidence);
    //
    string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
    cout << result_message << endl;
	duration = (clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "duration =" << duration << endl;
    // Here is how to get the eigenvalues of this Eigenfaces model:
    Mat eigenvalues = model->getEigenValues();
    // And we can do the same to display the Eigenvectors (read Eigenfaces):
    Mat W = model->getEigenVectors();
    // Get the sample mean from the training data
    Mat mean = model->getMean();
    // Display or save:
   // if(argc == 2) {
    //    imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));
    //} else {
        imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));
    //}
    // Display or save the first, at most 16 Fisherfaces:
    for (int i = 0; i < min(16, W.cols); i++) {
        string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
        cout << msg << endl;
        // get eigenvector #i
        Mat ev = W.col(i).clone();
        // Reshape to original size & normalize to [0...255] for imshow.
        Mat grayscale = norm_0_255(ev.reshape(1, height));
        // Show the image & apply a Bone colormap for better sensing.
        Mat cgrayscale;
        applyColorMap(grayscale, cgrayscale, COLORMAP_BONE);
        // Display or save:
      //  if(argc == 2) {
       //     imshow(format("fisherface_%d", i), cgrayscale);
       // } else {
            imwrite(format("%s/fisherface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));
       // }
    }
    // Display or save the image reconstruction at some predefined steps:
    for(int num_component = 0; num_component < min(16, W.cols); num_component++) {
        // Slice the Fisherface from the model:
        Mat ev = W.col(num_component);
        Mat projection = LDA::subspaceProject(ev, mean, images[0].reshape(1,1));
        Mat reconstruction = LDA::subspaceReconstruct(ev, mean, projection);
        // Normalize the result:
        reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
        // Display or save:
       // if(argc == 2) {
       //     imshow(format("fisherface_reconstruction_%d", num_component), reconstruction);
       // } else {
            imwrite(format("%s/fisherface_reconstruction_%d.png", output_folder.c_str(), num_component), reconstruction);
      //  }
    }
	/*
    // Display if we are not writing to an output folder:
    if(argc == 2) {
        waitKey(0);
    }
    return 0;
*/}
