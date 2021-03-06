#ifndef __APP__
#define __APP__

#include <Windows.h>
#include <Kinect.h>
#include <Kinect.Face.h>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

#include <vector>
#include <array>
#include <string>

#include <wrl/client.h>
#include "eigenfaces.h"
#include "fisherfaces.h"
using namespace Microsoft::WRL;

#include <array>

class Kinect
{
private:
    // Sensor
    ComPtr<IKinectSensor> kinect;

    // Coordinate Mapper
    ComPtr<ICoordinateMapper> coordinateMapper;

    // Reader
    ComPtr<IColorFrameReader> colorFrameReader;
    ComPtr<IBodyFrameReader> bodyFrameReader;
    std::array<ComPtr<IFaceFrameReader>, BODY_COUNT> faceFrameReader;

    // Color Buffer
    std::vector<BYTE> colorBuffer;
    int colorWidth;
    int colorHeight;
    unsigned int colorBytesPerPixel;
    cv::Mat colorMat;

    // Body Buffer
    std::array<IBody*, BODY_COUNT> bodies = { nullptr };

    // Face Buffer
    std::array<ComPtr<IFaceFrameResult>, BODY_COUNT> results;

    // Face Recognition
	cv::Ptr<cv::face::FaceRecognizer> recognizer;
	int numcomponents = 0;
    const std::string model = "eigenfaces.xml"; // Pre-Trained Model File Path ( *.xml or *.yaml )
    const double threshold = 4000.0; // Max Matching Distance = 40
    std::array<int, BODY_COUNT> labels;
    std::array<double, BODY_COUNT> distances;
	Fisherfaces _fisherfaces;
	Eigenfaces _eigenfaces;

public:
    // Constructor
    Kinect();

    // Destructor
    ~Kinect();

    // Processing
    void run();
	// Initialize
	void initialize();

private:
    

    // Initialize Sensor
    inline void initializeSensor();

    // Initialize Color
    inline void initializeColor();

    // Initialize Body
    inline void initializeBody();

    // Initialize Face
    inline void initializeFace();
	
    // Initialize Recognition
    inline void initializeRecognition();
	
    // Finalize
    void finalize();

    // Update Data
    void update();

    // Update Color
    inline void updateColor();

    // Update Body
    inline void updateBody();

    // Update Face
    inline void updateFace();

    // Update Recognition
    inline void updateRecognition();

    // Draw Data
    void draw();

    // Draw Color
    inline void drawColor();

    // Draw Recognition
    inline void drawRecognition();

	// Draw Face Points
	inline void drawFacePoints(cv::Mat& image, const std::array<PointF, FacePointType::FacePointType_Count>& points, const int radius, const cv::Vec3b& color, const int thickness = -1);

	// Draw Face Rotation
	inline void drawFaceRotation(cv::Mat& image, Vector4& quaternion, const RectI& box, const double fontScale, const cv::Vec3b& color, const int thickness = 2);

	// Convert Quaternion to Degree
	inline void quaternion2degree(const Vector4* quaternion, int* pitch, int* yaw, int* roll);

    // Draw Face Bounding Box
    inline void drawFaceBoundingBox( cv::Mat& image, const RectI& box, const cv::Vec3b& color, const int thickness = 1 );

    // Draw Recognition Results
    inline void drawRecognitionResults( cv::Mat& image, const int label, const double distance, const cv::Point& point, const double scale, const cv::Vec3b& color, const int thickness = 2 );

    // Show Data
    void show();

    // Show Recognition
    inline void showRecognition();
};

#endif // __APP__