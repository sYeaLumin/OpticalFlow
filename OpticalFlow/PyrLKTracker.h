#pragma once
#include "opencv2/highgui.hpp"
#include <iostream>
#include <algorithm>
#include <assert.h>
#include <vector>
typedef unsigned char Byte;
using cv::Point2f;
using cv::Mat;
using std::vector;
using std::cout;
class PyrLKTracker
{
private:
	vector<int> height;
	vector<int> width;
	vector<vector<Byte>> prePyramid;
	vector<vector<Byte>> nextPyramid;
	unsigned int maxLayer;
	unsigned int windowRadius;
	bool ifUsePyramid;
	double accuracyThreshold = 0.00001;
	int maxIter = 50;
public:
	vector<Point2f> featurePoints;
	vector<Point2f> trackPoints;

private:
	void buildPyramid(vector<vector<Byte>>&original_gray);
	void calc(vector<uchar>&state);
	void getMaxLayer(const int nh, const int nw);
	void pyramidSample(vector<Byte>&src, const int srcH, const int srcW,
		vector<Byte>& dst, int&dstH, int&dstW);
	double interpolator(vector<Byte>&src, int h, int w, const Point2f& point);
	void matrixInverse(double *pMatrix, double * _pMatrix, int dim);
	bool matrixMul(double *src1, int h1, int w1, double *src2, int h2, int w2, double *dst);
public:
	PyrLKTracker(const int windowRadius, bool usePyr, int maxIter = 50, double threshold = 0.00001);
	~PyrLKTracker();
	void init(const int nh, const int nw);
	void setFirstFrame(vector<Byte>&gray);
	void setNextFrame(vector<Byte>&gray);
	void runOneFrame();
	vector<Byte>& getPrePyramid(size_t i) { return prePyramid[i]; }
	vector<Byte>& getNextPyramid(size_t i) { return nextPyramid[i]; }
	int getPyramidH(int th) { return height[th]; }
	int getPyramidW(int th) { return width[th]; }
};