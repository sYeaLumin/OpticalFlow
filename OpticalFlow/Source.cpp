#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <ctype.h>
#include "PyrLKTracker.h"
using namespace cv;
using namespace std;

Point2f point;
bool addRemovePt = false;
static void onMouse(int event, int x, int y, int /*flags*/, void* /*param*/)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << x << " " << y << endl;
	}
}
int main(int argc, char** argv)
{
	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);
	const int MAX_COUNT = 500;

	namedWindow("My LK", WINDOW_AUTOSIZE);
	namedWindow("LK Demo", WINDOW_AUTOSIZE);
	setMouseCallback("LK Demo", onMouse, 0);
	Mat gray, prevGray, image, frame;
	vector<Point2f> points[2];
	Mat frame0 = imread("Backyard\\frame07.png");
	Mat frame1 = imread("Backyard\\frame08.png");
	// 处理第一帧
	frame0.copyTo(image);
	cvtColor(image, prevGray, COLOR_BGR2GRAY);
	goodFeaturesToTrack(prevGray, points[0], MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
	cornerSubPix(prevGray, points[0], subPixWinSize, Size(-1, -1), termcrit);
	// 处理第二帧
	frame1.copyTo(image);
	cvtColor(image, gray, COLOR_BGR2GRAY);

	// My LK 预处理
	Size size = prevGray.size();
	PyrLKTracker tracker(5, true);
	tracker.init(size.height, size.width);
	// 输入目标点
	tracker.trackPoints = points[0];
	// 第一帧
	vector<Byte> prevGrayImg;
	prevGrayImg.resize(size.height * size.width);
	for (size_t i = 0; i < size.height; i++) {
		uchar* data = prevGray.ptr<uchar>(i);
		for (size_t j = 0; j < size.width; j++) {
			prevGrayImg[i*size.width + j] = data[j];
		}
	}
	tracker.setFirstFrame(prevGrayImg);
	// 第二帧
	vector<Byte> grayImg;
	grayImg.resize(size.height * size.width);
	for (size_t i = 0; i < size.height; i++) {
		uchar* data = gray.ptr<uchar>(i);
		for (size_t j = 0; j < size.width; j++) {
			grayImg[i*size.width + j] = data[j];
		}
	}
	tracker.setNextFrame(grayImg);

	// 对照组处理
	vector<uchar> status;
	vector<float> err;
	calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
		3, termcrit, 0, 0.001);
	size_t i, k;
	for (i = k = 0; i < points[1].size(); i++)
	{
		if (!status[i])
			continue;
		points[1][k++] = points[1][i];
		line(image, points[0][i], points[1][i], Scalar(0, 255, 0), 2);
	}
	points[1].resize(k);
	imshow("LK Demo", image);

	Mat imageCompare;
	image.copyTo(imageCompare);
	// My LK 处理
	tracker.runOneFrame();
	for (size_t i = 0; i < tracker.trackPoints.size(); i++)
	{
		line(imageCompare, points[0][i], tracker.trackPoints[i], Scalar(0, 0, 255), 2);
	}
	imshow("My LK", imageCompare);
	waitKey(0);
	return 0;
}