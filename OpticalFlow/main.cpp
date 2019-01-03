#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <ctype.h>
#include "PyrLKTracker.h"
//#define PYRLK_COMPARE
using namespace cv;
using namespace std;
static void help()
{
	cout << "Using OpenCV version " << CV_VERSION << endl;
	cout << "\nIt uses camera by default, but you can provide a path to video as an argument.\n";
	cout << "\nHot keys: \n"
		"\tESC - quit the program\n"
		"\tSPAcE - pause the video\n"
		"\tr - auto-initialize tracking\n" << endl;
}

string input = "test.mp4";
int main(int argc, char** argv)
{
	VideoCapture cap;
	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	Size subPixWinSize(10, 10), LKwinSize(31, 31);
	const int MAX_COUNT = 500;
	int prevFeaturePointNum = 0;
	bool needToInit = true;
	help();
	cv::CommandLineParser parser(argc, argv, "{@input|0|}");
	string input = parser.get<string>("@input");
	if (input.size() == 1 && isdigit(input[0]))
	cap.open(input[0] - '0');
	else
	cap.open(input);

	double fps = cap.get(CV_CAP_PROP_FPS);
	int fourcc = cap.get(CV_CAP_PROP_FOURCC);
	int frameWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int frameHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	VideoWriter videoOut("PyrLK_" + input, fourcc, fps, Size(frameWidth, frameHeight));

	if (!cap.isOpened())
	{
		cout << "Could not initialize capturing...\n";
		system("pause");
		return 0;
	}

	namedWindow("LK Demo", 1);
	Mat gray, prevGray, image, frame;
	PyrLKTracker tracker(5, true);
	Size winSize;
	vector<Point2f> featurePoints;
#ifdef PYRLK_COMPARE
	vector<Point2f> compareFeaturePoints[2];
#endif // PYRLK_COMPARE

	while(true)
	{
		cap >> frame;
		if (frame.empty())
			break;
		frame.copyTo(image);
		cvtColor(image, gray, COLOR_BGR2GRAY);
		if (needToInit)
		{
			goodFeaturesToTrack(gray, featurePoints, MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
			cornerSubPix(gray, featurePoints, subPixWinSize, Size(-1, -1), termcrit);
			prevFeaturePointNum = featurePoints.size();
#ifdef PYRLK_COMPARE
			compareFeaturePoints[1].assign(featurePoints.begin(), featurePoints.end());
#endif // PYRLK_COMPARE
			winSize = gray.size();
			tracker.init(winSize.height, winSize.width);
			tracker.trackPoints = featurePoints;

			vector<Byte> grayImg;
			grayImg.resize(winSize.height * winSize.width);
			for (size_t i = 0; i < winSize.height; i++) {
				uchar* data = gray.ptr<uchar>(i);
				for (size_t j = 0; j < winSize.width; j++) {
					grayImg[i*winSize.width + j] = data[j];
				}
			}
			tracker.setFirstFrame(grayImg); 
			gray.copyTo(prevGray);
		}
		else
		{
#ifdef PYRLK_COMPARE
			vector<uchar> status;
			vector<float> err;
			calcOpticalFlowPyrLK(prevGray, gray, compareFeaturePoints[0], compareFeaturePoints[1], status, err, LKwinSize,
				3, termcrit, 0, 0.001);
			size_t i, k;
			for (i = k = 0; i < compareFeaturePoints[1].size(); i++)
			{
				if (!status[i])
					continue;
				compareFeaturePoints[1][k++] = compareFeaturePoints[1][i];
				circle(image, compareFeaturePoints[1][i], 3, Scalar(0, 255, 0), -1);
				line(image, compareFeaturePoints[0][i], compareFeaturePoints[1][i], Scalar(0, 255, 0), 1);
			}
			compareFeaturePoints[1].resize(k);
#endif // PYRLK_COMPARE

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
				circle(image, tracker.trackPoints[i], 3, Scalar(0, 0, 255), -1);
				line(image, tracker.featurePoints[i], tracker.trackPoints[i], Scalar(0, 0, 255), 1);
			}
		}

		// ¸ú×Ùµã¸üÐÂ
		if (tracker.trackPoints.size() < 0.8 * prevFeaturePointNum) {
			if (tracker.trackPoints.size() < 0.8 * MAX_COUNT) {
				vector<Point2f> tmppoints;
				goodFeaturesToTrack(gray, tmppoints, MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
				cornerSubPix(gray, tmppoints, subPixWinSize, Size(-1, -1), termcrit);
				if (tmppoints.size() > 1.2 * tracker.trackPoints.size()) {
#ifdef PYRLK_COMPARE
					vector<Point2f>().swap(compareFeaturePoints[1]);
					compareFeaturePoints[1].assign(tmppoints.begin(), tmppoints.end());
#endif // PYRLK_COMPARE
					swap(tracker.trackPoints, tmppoints);
					prevFeaturePointNum = tracker.trackPoints.size();
				}
			}
		}
		needToInit = false;
		videoOut.write(image);
		imshow("LK Demo", image);
		char c = (char)waitKey(10);
		if (c == 27)
			break;
		if (c == 32)
			waitKey(0);
		switch (c)
		{
		case 'r':
			needToInit = true;
			break;
		}
#ifdef PYRLK_COMPARE
		std::swap(compareFeaturePoints[1], compareFeaturePoints[0]);
#endif // PYRLK_COMPARE
		cv::swap(prevGray, gray);
	}
	cap.release();
	videoOut.release();
	return 0;
}