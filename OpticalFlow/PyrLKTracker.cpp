#include "PyrLKTracker.h"

PyrLKTracker::PyrLKTracker(const int windowRadius, bool usePyr, int maxIter, float threshold)
	:windowRadius(windowRadius), ifUsePyramid(usePyr), maxIter(maxIter), accuracyThreshold(threshold)
{

}


PyrLKTracker::~PyrLKTracker()
{
}

void PyrLKTracker::init(const int nh, const int nw)
{
	if (ifUsePyramid)
		getMaxLayer(nh, nw);
	else
		maxLayer = 1;
	prePyramid.resize(maxLayer);
	nextPyramid.resize(maxLayer);
	height.resize(maxLayer);
	width.resize(maxLayer);
	height[0] = nh;
	width[0] = nw;
}

void PyrLKTracker::setFirstFrame(vector<Byte>&gray)
{
	nextPyramid[0] = gray;
	buildPyramid(nextPyramid);
}

void  PyrLKTracker::setNextFrame(vector<Byte>&gray)
{
	swap(prePyramid, nextPyramid);
	nextPyramid[0] = gray;
	buildPyramid(nextPyramid);
	swap(featurePoints, trackPoints);
}

void PyrLKTracker::pyramidSample(vector<Byte>&src, const int srcH, const int srcW, 
	                                                            vector<Byte>& dst, int&dstH, int&dstW)
{
	dstH = srcH / 2;
	dstW = srcW / 2;
	assert(dstW > 3 && dstH > 3);
	dst.resize(dstH*dstW);
	for (int i = 0; i < dstH - 1; i++)
		for (int j = 0; j < dstW - 1; j++)
		{
			int srcY = 2 * i + 1;
			int srcX = 2 * j + 1;
			float re = src[srcY*srcW + srcX] * 0.25;
			re += src[(srcY - 1)*srcW + srcX] * 0.125;
			re += src[(srcY + 1)*srcW + srcX] * 0.125;
			re += src[srcY*srcW + srcX - 1] * 0.125;
			re += src[srcY*srcW + srcX + 1] * 0.125;
			re += src[(srcY - 1)*srcW + srcX + 1] * 0.0625;
			re += src[(srcY - 1)*srcW + srcX - 1] * 0.0625;
			re += src[(srcY + 1)*srcW + srcX - 1] * 0.0625;
			re += src[(srcY + 1)*srcW + srcX + 1] * 0.0625;
			dst[i*dstW + j] = re;
		}
	for (int i = 0; i < dstH; i++)
		dst[i*dstW + dstW - 1] = dst[i*dstW + dstW - 2];
	for (int i = 0; i < dstW; i++)
		dst[(dstH - 1)*dstW + i] = dst[(dstH - 2)*dstW + i];
}

//bilinear interplotation
float PyrLKTracker::interpolator(vector<Byte>&src, int h, int w, const Point2f& point)
{
	int floorX = floor(point.x);
	int floorY = floor(point.y);

	if (floorX < 0 && floorY < 0) {
		return src[0];
	}
	if (floorX >= w - 1 && floorY >= h - 1) {
		return src.back();
	}
	float fractX = point.x - floorX;
	float fractY = point.y - floorY;
	float ceilX, ceilY;
	if (floorX < 0)
		floorX = ceilX = 0;
	else if (floorX >= w - 1)
		floorX = ceilX = w - 1;
	else
		ceilX = floorX + 1;

	if (floorY < 0)
		floorY = ceilY = 0;
	else if (floorY >= h - 1)
		floorY = ceilY = h - 1;
	else
		ceilY = floorY + 1;

	return ((1.0 - fractX) * (1.0 - fractY) * src[floorX + w* floorY])
		+ (fractX * (1.0 - fractY) * src[ceilX + floorY*w])
		+ ((1.0 - fractX) * fractY * src[floorX + ceilY*w])
		+ (fractX * fractY * src[ceilX + ceilY*w]);
}


void PyrLKTracker::getMaxLayer(const int nh, const int nw)
{
	int layer = 0;
	int windowsize = 2 * windowRadius + 1;
	int tmp = nh > nw ? nw : nh;
	if (tmp > ((1 << 5) * windowsize))
	{
		maxLayer = 5;
		return;
	}
	tmp = tmp / 2;
	while (tmp > 2 * windowsize)
	{
		layer++;
		tmp = tmp / 2;
	}
	maxLayer = layer;
}

void PyrLKTracker::buildPyramid(vector<vector<Byte>>&pyramid)
{
	for (int i = 1; i < maxLayer; i++)
		pyramidSample(pyramid[i - 1], height[i - 1], width[i - 1], pyramid[i], height[i], width[i]);
}

void PyrLKTracker::runOneFrame()
{
	vector<uchar> states;
	calc(states);
	vector<Point2f> tureFeaturePoints;
	vector<Point2f> tureTrackPoints;
	for (size_t i = 0; i < states.size(); i++) {
		if (states[i]) {
			tureFeaturePoints.push_back(featurePoints[i]);
			tureTrackPoints.push_back(trackPoints[i]);
		}
	}
	swap(tureFeaturePoints, featurePoints);
	swap(tureTrackPoints, trackPoints);
}

void PyrLKTracker::calc(vector<uchar>&states)
{
	states.resize(featurePoints.size());
	for (auto &state : states) 
		state = true;
	trackPoints.resize(featurePoints.size());
	vector<float> derivativeXs;
	derivativeXs.resize((2 * windowRadius + 1)*(2 * windowRadius + 1));
	vector<float> derivativeYs;
	derivativeYs.resize((2 * windowRadius + 1)*(2 * windowRadius + 1));

	for (int i = 0; i < featurePoints.size(); i++)
	{
		//cout << "featurePoints:" << i << endl;
		float g[2] = { 0 };
		float finalOpticalFlow[2] = { 0 };

		for (int layer = maxLayer - 1; layer >= 0; layer--)
		{
			//cout << "pyramidLayer:" << j << endl;
			Point2f currPoint;
			currPoint.x = featurePoints[i].x / (1 << layer);
			currPoint.y = featurePoints[i].y / (1 << layer);
			float xLeft = currPoint.x - windowRadius;
			float xRight = currPoint.x + windowRadius;
			float yLeft = currPoint.y - windowRadius;
			float yRight = currPoint.y + windowRadius;

			int idx = 0;
			float A11 = 0; 
			float A12 = 0; 
			float A22 = 0;
			for (float xx = xLeft; xx < xRight + 0.01; xx += 1.0)
				for (float yy = yLeft; yy < yRight + 0.01; yy += 1.0)
				{
					assert(xx < width[layer] + (float)windowRadius 
						&& yy < height[layer] + (float)windowRadius
						&& xx >= -(float)windowRadius 
						&& yy >= -(float)windowRadius);
					float derivativeX =
						interpolator(prePyramid[layer], height[layer], width[layer], Point2f(xx + 1.0, yy)) -
						interpolator(prePyramid[layer], height[layer], width[layer], Point2f(xx - 1.0, yy));
					derivativeX /= 2.0;

					float derivativeY = 
						interpolator(prePyramid[layer], height[layer], width[layer], Point2f(xx, yy + 1.0)) -
						interpolator(prePyramid[layer], height[layer], width[layer], Point2f(xx, yy - 1.0));
					derivativeY /= 2.0;

					derivativeXs[idx] = derivativeX;
					derivativeYs[idx] = derivativeY;
					idx++;
					A11 += derivativeX * derivativeX;
					A12 += derivativeX * derivativeY;
					A22 += derivativeY * derivativeY;
				}
			float minEig = (A22 + A11 - std::sqrt((A11 - A22)*(A11 - A22) +
				4.f*A12*A12)) / (2 * derivativeXs.size());
			if (minEig < 0.001) {
				if (i == 0 && states[i]) {
					states[i] = false;
				}
				continue;
			}

			Mat grad(2, 2, CV_32FC1, cv::Scalar::all(0));
			grad.at<float>(0, 0) = A11;
			grad.at<float>(0, 1) = A12;
			grad.at<float>(1, 0) = A12;
			grad.at<float>(1, 1) = A22;
			Mat gradInverse(2, 2, CV_32FC1, cv::Scalar::all(0));
			gradInverse = grad.inv();

			float opticalFlow[2] = { 0 };
			float opticalflowResidual = 1;
			int iteration = 0;
			while (iteration<maxIter && opticalflowResidual>accuracyThreshold)
			{
				iteration++;
				if (xRight + g[0] + opticalFlow[0] >= width[layer] + (float)windowRadius
					|| yRight + g[1] + opticalFlow[1] >= height[layer] + (float)windowRadius
					|| xLeft + g[0] + opticalFlow[0] < -(float)windowRadius
					|| yLeft + g[1] + opticalFlow[1] < -(float)windowRadius) {
					if (layer == 0 && states[i])
						states[i] = false;
					break;
				}
				Mat b_k(2, 1, CV_32FC1, cv::Scalar::all(0));
				idx = 0;
				for (float xx = xLeft; xx < xRight + 0.001; xx += 1.0)
					for (float yy = yLeft; yy < yRight + 0.001; yy += 1.0)
					{
						float nextX = xx + g[0] + opticalFlow[0];
						float nextY = yy + g[1] + opticalFlow[1];
						assert(nextX < width[layer] + (float)windowRadius 
							&& nextY < height[layer] + (float)windowRadius
							&& nextX >= -(float)windowRadius 
							&& nextY >= -(float)windowRadius);
						float pixelDifference =
							interpolator(prePyramid[layer], height[layer], width[layer], Point2f(xx, yy)) -
							interpolator(nextPyramid[layer], height[layer], width[layer], Point2f(nextX, nextY));
						b_k.at<float>(0, 0) += pixelDifference*derivativeXs[idx];
						b_k.at<float>(1, 0) += pixelDifference*derivativeYs[idx];
						idx++;
					}
				Mat eta_k(2, 1, CV_32FC1, cv::Scalar::all(0));
				eta_k = gradInverse*b_k;
				opticalFlow[0] += eta_k.at<float>(0, 0);
				opticalFlow[1] += eta_k.at<float>(1, 0);
				opticalflowResidual = abs(
					eta_k.at<float>(0, 0) + 
					eta_k.at<float>(1, 0)
					);
			}
			if (layer == 0)
			{
				finalOpticalFlow[0] = opticalFlow[0];
				finalOpticalFlow[1] = opticalFlow[1];
			}
			else
			{
				g[0] = 2 * (g[0] + opticalFlow[0]);
				g[1] = 2 * (g[1] + opticalFlow[1]);
			}
		}
		finalOpticalFlow[0] += g[0];
		finalOpticalFlow[1] += g[1];
		trackPoints[i].x = featurePoints[i].x + finalOpticalFlow[0];
		trackPoints[i].y = featurePoints[i].y + finalOpticalFlow[1];
	}
}


void PyrLKTracker::matrixInverse(float *pMatrix, float * _pMatrix, int dim)
{
	float *tMatrix = new float[2 * dim*dim];
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++)
			tMatrix[i*dim * 2 + j] = pMatrix[i*dim + j];
	}
	for (int i = 0; i < dim; i++) {
		for (int j = dim; j < dim * 2; j++)
			tMatrix[i*dim * 2 + j] = 0.0;
		tMatrix[i*dim * 2 + dim + i] = 1.0;
	}
   
	for (int i = 0; i < dim; i++)
	{
		float basic = tMatrix[i*dim * 2 + i];
		assert(fabs(basic) > 1e-300);
		for (int j = 0; j < dim; j++)  
		{
			if (j == i) continue;
			float times = tMatrix[j*dim * 2 + i] / basic;
			for (int k = 0; k < dim * 2; k++)  
			{
				tMatrix[j*dim * 2 + k] = tMatrix[j*dim * 2 + k] - times*tMatrix[i*dim * 2 + k];
			}
		}
		for (int k = 0; k < dim * 2; k++) {
			tMatrix[i*dim * 2 + k] /= basic;
		}
	}
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
			_pMatrix[i*dim + j] = tMatrix[i*dim * 2 + j + dim];
	}
	delete[] tMatrix;
}

bool PyrLKTracker::matrixMul(float *src1, int h1, int w1, float *src2, int h2, int w2, float *dst)
{
	int i, j, k;
	float sum = 0;
	float *first = src1;
	float *second = src2;
	float *dest = dst;
	int Step1 = w1;
	int Step2 = w2;

	if (src1 == nullptr || src2 == nullptr || dest == nullptr || h2 != w1)
		return false;

	for (j = 0; j < h1; j++)
	{
		for (i = 0; i < w2; i++)
		{
			sum = 0;
			second = src2 + i;
			for (k = 0; k < w1; k++)
			{
				sum += first[k] * (*second);
				second += Step2;
			}
			dest[i] = sum;
		}
		first += Step1;
		dest += Step2;
	}
	return true;
}
