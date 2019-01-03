#include "PyrLKTracker.h"

PyrLKTracker::PyrLKTracker(const int windowRadius, bool usePyr, int maxIter, double threshold)
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
			double re = src[srcY*srcW + srcX] * 0.25;
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
double PyrLKTracker::interpolator(vector<Byte>&src, int h, int w, const Point2f& point)
{
	int floorX = floor(point.x);
	int floorY = floor(point.y);

	if (floorX < 0 && floorY < 0) {
		return src[0];
	}
	if (floorX >= w - 1 && floorY >= h - 1) {
		return src.back();
	}
	double fractX = point.x - floorX;
	double fractY = point.y - floorY;
	double ceilX, ceilY;
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
	for (const auto&state : states) {
		if (!state)
			cout << "F";
	}
}

void PyrLKTracker::calc(vector<uchar>&states)
{
	states.resize(featurePoints.size());
	for (auto &state : states) 
		state = true;
	trackPoints.resize(featurePoints.size());
	vector<double> derivativeXs;
	derivativeXs.resize((2 * windowRadius + 1)*(2 * windowRadius + 1));
	vector<double> derivativeYs;
	derivativeYs.resize((2 * windowRadius + 1)*(2 * windowRadius + 1));

	for (int i = 0; i < featurePoints.size(); i++)
	{
		//cout << "featurePoints:" << i << endl;
		double g[2] = { 0 };
		double finalOpticalFlow[2] = { 0 };

		for (int j = maxLayer - 1; j >= 0; j--)
		{
			//cout << "pyramidLayer:" << j << endl;
			Point2f currPoint;
			currPoint.x = featurePoints[i].x / (1 << j);
			currPoint.y = featurePoints[i].y / (1 << j);
			double xLeft = currPoint.x - windowRadius;
			double xRight = currPoint.x + windowRadius;
			double yLeft = currPoint.y - windowRadius;
			double yRight = currPoint.y + windowRadius;

			int idx = 0;
			Mat grad(2, 2, CV_64FC1, cv::Scalar::all(0));
			//double gradient[4] = { 0 };
			for (double xx = xLeft; xx < xRight + 0.01; xx += 1.0)
				for (double yy = yLeft; yy < yRight + 0.01; yy += 1.0)
				{
					assert(xx < 1000 && yy < 1000
						&& xx >= -(double)windowRadius && yy >= -(double)windowRadius);
					double derivativeX =
						interpolator(prePyramid[j], height[j], width[j], Point2f(xx + 1.0, yy)) -
						interpolator(prePyramid[j], height[j], width[j], Point2f(xx - 1.0, yy));
					derivativeX /= 2.0;

					double derivativeY = 
						interpolator(prePyramid[j], height[j], width[j], Point2f(xx, yy + 1.0)) -
						interpolator(prePyramid[j], height[j], width[j], Point2f(xx, yy - 1.0));
					derivativeY /= 2.0;

					derivativeXs[idx] = derivativeX;
					derivativeYs[idx] = derivativeY;
					idx++;
					grad.at<double>(0, 0) += derivativeX * derivativeX;
					grad.at<double>(0, 1) += derivativeX * derivativeY;
					grad.at<double>(1, 0) += derivativeX * derivativeY;
					grad.at<double>(1, 1) += derivativeY * derivativeY;
				}
			Mat gradInverse(2, 2, CV_64FC1, cv::Scalar::all(0));
			gradInverse = grad.inv();
			Mat eValues, eVectors;
			cv::eigen(grad, eValues, eVectors);
			if (eValues.at<double>(1, 0) < 0.001) {
				if (i == 0 && states[i]) {
					states[i] = false;
				}
				continue;
			}

			double opticalFlow[2] = { 0 };
			double opticalflowResidual = 1;
			int iteration = 0;
			while (iteration<maxIter && opticalflowResidual>accuracyThreshold)
			{
				iteration++;
				Mat b_k(2, 1, CV_64FC1, cv::Scalar::all(0));
				idx = 0;
				for (double xx = xLeft; xx < xRight + 0.001; xx += 1.0)
					for (double yy = yLeft; yy < yRight + 0.001; yy += 1.0)
					{
						double nextX = xx + g[0] + opticalFlow[0];
						double nextY = yy + g[1] + opticalFlow[1];
						double pixelDifference =
							interpolator(prePyramid[j], height[j], width[j], Point2f(xx, yy)) -
							interpolator(nextPyramid[j], height[j], width[j], Point2f(nextX, nextY));
						b_k.at<double>(0, 0) += pixelDifference*derivativeXs[idx];
						b_k.at<double>(1, 0) += pixelDifference*derivativeYs[idx];
						idx++;
					}
				Mat eta_k(2, 1, CV_64FC1, cv::Scalar::all(0));
				eta_k = gradInverse*b_k;
				opticalFlow[0] += eta_k.at<double>(0, 0);
				opticalFlow[1] += eta_k.at<double>(1, 0);
				opticalflowResidual = abs(
					eta_k.at<double>(0, 0) + 
					eta_k.at<double>(1, 0)
					);
			}
			if (j == 0)
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


void PyrLKTracker::matrixInverse(double *pMatrix, double * _pMatrix, int dim)
{
	double *tMatrix = new double[2 * dim*dim];
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
		double basic = tMatrix[i*dim * 2 + i];
		assert(fabs(basic) > 1e-300);
		for (int j = 0; j < dim; j++)  
		{
			if (j == i) continue;
			double times = tMatrix[j*dim * 2 + i] / basic;
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

bool PyrLKTracker::matrixMul(double *src1, int h1, int w1, double *src2, int h2, int w2, double *dst)
{
	int i, j, k;
	double sum = 0;
	double *first = src1;
	double *second = src2;
	double *dest = dst;
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
