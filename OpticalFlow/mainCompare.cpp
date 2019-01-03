#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <ctype.h>
#include "PyrLKTracker.h"
#define MY_PYRLK
//#define OPENCV_PYRLK
using namespace cv;
using namespace std;
static void help()
{
	// print a welcome message, and the OpenCV version
	cout << "\nThis is a demo of Lukas-Kanade optical flow lkdemo(),\n"
		"Using OpenCV version " << CV_VERSION << endl;
	cout << "\nIt uses camera by default, but you can provide a path to video as an argument.\n";
	cout << "\nHot keys: \n"
		"\tESC - quit the program\n"
		"\tr - auto-initialize tracking\n"
		"\tc - delete all the points\n"
		"\tn - switch the \"night\" mode on/off\n"
		"To add/remove a feature point click it\n" << endl;
}

string input = "test4.mp4";
int main(int argc, char** argv)
{
	VideoCapture cap;
	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	Size subPixWinSize(10, 10), LKwinSize(31, 31);
	const int MAX_COUNT = 500;
	int prevFeaturePointNum = 0;
	bool needToInit = true;
	help();
	/*
	cv::CommandLineParser parser(argc, argv, "{@input|0|}");
	string input = parser.get<string>("@input");
	if (input.size() == 1 && isdigit(input[0]))
		cap.open(input[0] - '0');
	else
		cap.open(input);*/
	cap.open(input);
	double fps = cap.get(CV_CAP_PROP_FPS);
	int fourcc = cap.get(CV_CAP_PROP_FOURCC);
	int frameWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int frameHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	VideoWriter videoOut("PyrLK3_" + input, fourcc, fps, Size(frameWidth, frameHeight));

	if (!cap.isOpened())
	{
		cout << "Could not initialize capturing...\n";
		system("pause");
		return 0;
	}
	namedWindow("LK Demo", 1);
	Mat gray, prevGray, image, frame;
#ifdef OPENCV_PYRLK
	vector<Point2f> points[2];
#endif // OPENCV_PYRLK
#ifdef MY_PYRLK
	PyrLKTracker tracker(5, true);
	Size winSize;
	vector<Point2f> points;
#endif // MY_PYRLK
	for (;;)
	{
		cap >> frame;
		if (frame.empty())
			break;
		frame.copyTo(image);
		cvtColor(image, gray, COLOR_BGR2GRAY);
#ifdef OPENCV_PYRLK
		if (needToInit)
		{
			goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
			cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
			prevFeaturePointNum = points[1].size();
		}
#endif // OPENCV_PYRLK
#ifdef MY_PYRLK		
		if (needToInit)
		{
			goodFeaturesToTrack(gray, points, MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
			cornerSubPix(gray, points, subPixWinSize, Size(-1, -1), termcrit);
			prevFeaturePointNum = points.size();

			winSize = gray.size();
			tracker.init(winSize.height, winSize.width);
			tracker.trackPoints = points;
			// µÚÒ»Ö¡
			vector<Byte> grayImg;
			grayImg.resize(winSize.height * winSize.width);
			for (size_t i = 0; i < winSize.height; i++) {
				uchar* data = gray.ptr<uchar>(i);
				for (size_t j = 0; j < winSize.width; j++) {
					grayImg[i*winSize.width + j] = data[j];
				}
			}
			tracker.setFirstFrame(grayImg);
			// 
			gray.copyTo(prevGray);
		}
#endif // MY_PYRLK
#ifdef OPENCV_PYRLK
		else if (!points[0].empty())
		{
			vector<uchar> status;
			vector<float> err;
			if (prevGray.empty())
				gray.copyTo(prevGray);
			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, LKwinSize,
				3, termcrit, 0, 0.001);
			size_t i, k;
			for (i = k = 0; i < points[1].size(); i++)
			{
				if (!status[i])
					continue;
				points[1][k++] = points[1][i];
				circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 2);
				line(image, points[0][i], points[1][i], Scalar(0, 255, 0), 1);
			}
			points[1].resize(k);
		}
#endif // OPENCV_PYRLK
#ifdef MY_PYRLK
		else
		{
			vector<Byte> grayImg;
			grayImg.resize(winSize.height * winSize.width);
			for (size_t i = 0; i < winSize.height; i++) {
				uchar* data = gray.ptr<uchar>(i);
				for (size_t j = 0; j < winSize.width; j++) {
					grayImg[i*winSize.width + j] = data[j];
				}
			}
			tracker.setNextFrame(grayImg);
			tracker.runOneFrame();
			for (size_t i = 0; i < tracker.trackPoints.size(); i++)
			{
				circle(image, tracker.trackPoints[i], 3, Scalar(0, 0, 255), -1, 2);
				line(image, tracker.featurePoints[i], tracker.trackPoints[i], Scalar(0, 0, 255), 1);
			}
		}
#endif // MY_PYRLK
#ifdef OPENCV_PYRLK
		if (points[1].size() < 0.8 * prevFeaturePointNum) {
			if (points[1].size() < 0.8 * MAX_COUNT) {
				vector<Point2f> tmppoints;
				goodFeaturesToTrack(gray, tmppoints, MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
				cornerSubPix(gray, tmppoints, subPixWinSize, Size(-1, -1), termcrit);
				if (tmppoints.size() > 1.2 * points[1].size()) {
					swap(points[1], tmppoints);
					prevFeaturePointNum = points[1].size();
				}
				cout << prevFeaturePointNum << endl;
			}
		}
#endif // OPENCV_PYRLK
#ifdef MY_PYRLK
		if (tracker.trackPoints.size() < 0.8 * prevFeaturePointNum) {
			if (tracker.trackPoints.size() < 0.8 * MAX_COUNT) {
				vector<Point2f> tmppoints;
				goodFeaturesToTrack(gray, tmppoints, MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
				cornerSubPix(gray, tmppoints, subPixWinSize, Size(-1, -1), termcrit);
				if (tmppoints.size() > 1.2 * tracker.trackPoints.size()) {
					swap(tracker.trackPoints, tmppoints);
					prevFeaturePointNum = tracker.trackPoints.size();
				}
				cout << prevFeaturePointNum << endl;
			}
		}
#endif // MY_PYRLK
		needToInit = false;
		videoOut.write(image);
		imshow("LK Demo", image);
		char c = (char)waitKey(10);
		if (c == 27)
			break;
		switch (c)
		{
		case 'r':
			needToInit = true;
			break;
		}
		std::swap(points[1], points[0]);
		cv::swap(prevGray, gray);
	}
	cap.release();
	videoOut.release();
	return 0;
}