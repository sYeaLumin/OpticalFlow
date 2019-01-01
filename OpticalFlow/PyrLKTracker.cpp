#include "PyrLKTracker.h"

PyrLKTracker::PyrLKTracker(const int windowRadius, bool usePyr)
	:windowRadius(windowRadius), ifUsePyramid(usePyr)
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

void PyrLKTracker::pyramidSample(vector<Byte>&src_gray_data,
	const int src_h, const int src_w, vector<Byte>& dst, int&dst_h, int&dst_w)
{
	dst_h = src_h / 2;
	dst_w = src_w / 2;
	int ii = height[1];
	int hh = width[1];
	assert(dst_w > 3 && dst_h > 3);
	dst.resize(dst_h*dst_w);
	for (int i = 0; i < dst_h - 1; i++)
		for (int j = 0; j < dst_w - 1; j++)
		{
			int srcY = 2 * i + 1;
			int srcX = 2 * j + 1;
			double re = src_gray_data[srcY*src_w + srcX] * 0.25;
			re += src_gray_data[(srcY - 1)*src_w + srcX] * 0.125;
			re += src_gray_data[(srcY + 1)*src_w + srcX] * 0.125;
			re += src_gray_data[srcY*src_w + srcX - 1] * 0.125;
			re += src_gray_data[srcY*src_w + srcX + 1] * 0.125;
			re += src_gray_data[(srcY - 1)*src_w + srcX + 1] * 0.0625;
			re += src_gray_data[(srcY - 1)*src_w + srcX - 1] * 0.0625;
			re += src_gray_data[(srcY + 1)*src_w + srcX - 1] * 0.0625;
			re += src_gray_data[(srcY + 1)*src_w + srcX + 1] * 0.0625;
			dst[i*dst_w + j] = re;
		}
	for (int i = 0; i < dst_h; i++)
		dst[i*dst_w + dst_w - 1] = dst[i*dst_w + dst_w - 2];
	for (int i = 0; i < dst_w; i++)
		dst[(dst_h - 1)*dst_w + i] = dst[(dst_h - 2)*dst_w + i];
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
	int temp = nh > nw ?
		nw : nh;
	if (temp > ((1 << 4) * 2 * windowsize))
	{
		maxLayer = 5;
		return;
	}
	temp = double(temp) / 2;
	while (temp > 2 * windowsize)
	{
		layer++;
		temp = double(temp) / 2;
	}
	maxLayer = layer;
}

void PyrLKTracker::buildPyramid(vector<vector<Byte>>&pyramid)
{
	for (int i = 1; i < maxLayer; i++)
	{
		pyramidSample(pyramid[i - 1], height[i - 1],
			width[i - 1], pyramid[i], height[i], width[i]);
	}
}

void PyrLKTracker::runOneFrame()
{
	vector<char> state;
	calc(state);
}

void PyrLKTracker::calc(vector<char>&state)
{
	trackPoints.resize(featurePoints.size());
	vector<double> derivativeXs;
	derivativeXs.resize((2 * windowRadius + 1)*(2 * windowRadius + 1));
	vector<double> derivativeYs;
	derivativeYs.resize((2 * windowRadius + 1)*(2 * windowRadius + 1));

	for (int i = 0; i < featurePoints.size(); i++)
	{
		//cout << "featurePoints:" << i << endl;
		double g[2] = { 0 };
		double finalopticalflow[2] = { 0 };

		for (int j = maxLayer - 1; j >= 0; j--)
		{
			//cout << "pyramidLayer:" << j << endl;
			Point2f currPoint;
			currPoint.x = featurePoints[i].x / pow(2.0, j);
			currPoint.y = featurePoints[i].y / pow(2.0, j);
			double Xleft = currPoint.x - windowRadius;
			double Xright = currPoint.x + windowRadius;
			double Yleft = currPoint.y - windowRadius;
			double Yright = currPoint.y + windowRadius;

			double gradient[4] = { 0 };
			int cnt = 0;
			for (double xx = Xleft; xx < Xright + 0.01; xx += 1.0)
				for (double yy = Yleft; yy < Yright + 0.01; yy += 1.0)
				{
					assert(xx < 1000 && yy < 1000
						&& xx >= -(double)windowRadius && yy >= -(double)windowRadius);
					double derivativeX = interpolator(prePyramid[j],
						height[j], width[j], Point2f(xx + 1.0, yy)) -
						interpolator(prePyramid[j], height[j],
							width[j], Point2f(xx - 1.0, yy));
					derivativeX /= 2.0;

					double t1 = interpolator
					(prePyramid[j], height[j], width[j], Point2f(xx, yy + 1.0));
					double t2 = interpolator(prePyramid[j], height[j],
						width[j], Point2f(xx, yy - 1.0));
					double derivativeY = (t1 - t2) / 2.0;

					derivativeXs[cnt] = derivativeX;
					derivativeYs[cnt++] = derivativeY;
					gradient[0] += derivativeX * derivativeX;
					gradient[1] += derivativeX * derivativeY;
					gradient[2] += derivativeX * derivativeY;
					gradient[3] += derivativeY * derivativeY;
				}
			double gradientInverse[4] = { 0 };
			matrixInverse(gradient, gradientInverse, 2);

			double opticalflow[2] = { 0 };
			int maxIter = 50;
			double opticalflowResidual = 1;
			int iteration = 0;
			while (iteration<maxIter&&opticalflowResidual>0.00001)
			{
				iteration++;
				double mismatch[2] = { 0 };
				cnt = 0;
				for (double xx = Xleft; xx < Xright + 0.001; xx += 1.0)
					for (double yy = Yleft; yy < Yright + 0.001; yy += 1.0)
					{
						assert(xx < 1000 && yy < 1000 && 
							xx >= -(double)windowRadius && yy >= -(double)windowRadius);
						double nextX = xx + g[0] + opticalflow[0];
						double nextY = yy + g[1] + opticalflow[1];
						double pixelDifference = (interpolator(prePyramid[j],
							height[j], width[j], Point2f(xx, yy))
							- interpolator(nextPyramid[j], height[j],
								width[j], Point2f(nextX, nextY)));
						mismatch[0] += pixelDifference*derivativeXs[cnt];
						mismatch[1] += pixelDifference*derivativeYs[cnt++];
					}
				double temp_of[2];
				matrixMul(gradientInverse, 2, 2, mismatch, 2, 1, temp_of);
				opticalflow[0] += temp_of[0];
				opticalflow[1] += temp_of[1];
				opticalflowResidual = abs(temp_of[0]) + abs(temp_of[1]);
			}
			if (j == 0)
			{
				finalopticalflow[0] = opticalflow[0];
				finalopticalflow[1] = opticalflow[1];
			}
			else
			{
				g[0] = 2 * (g[0] + opticalflow[0]);
				g[1] = 2 * (g[1] + opticalflow[1]);
			}
		}
		finalopticalflow[0] += g[0];
		finalopticalflow[1] += g[1];
		trackPoints[i].x = featurePoints[i].x + finalopticalflow[0];
		trackPoints[i].y = featurePoints[i].y + finalopticalflow[1];
	}
}

//matrix inverse
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
	//Initialization over!   
	for (int i = 0; i < dim; i++)//Process Cols   
	{
		double base = tMatrix[i*dim * 2 + i];
		if (fabs(base) < 1E-300) {
			assert(false);
		}
		for (int j = 0; j < dim; j++)//row   
		{
			if (j == i) continue;
			double times = tMatrix[j*dim * 2 + i] / base;
			for (int k = 0; k < dim * 2; k++)//col   
			{
				tMatrix[j*dim * 2 + k] = tMatrix[j*dim * 2 + k] - times*tMatrix[i*dim * 2 + k];
			}
		}
		for (int k = 0; k < dim * 2; k++) {
			tMatrix[i*dim * 2 + k] /= base;
		}
	}
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
			_pMatrix[i*dim + j] = tMatrix[i*dim * 2 + j + dim];
	}
	delete[] tMatrix;
}

bool PyrLKTracker::matrixMul(double *src1, int height1, int width1, double *src2, int height2, int width2, double *dst)
{
	int i, j, k;
	double sum = 0;
	double *first = src1;
	double *second = src2;
	double *dest = dst;
	int Step1 = width1;
	int Step2 = width2;

	if (src1 == nullptr || src2 == nullptr || dest == nullptr || height2 != width1)
		return false;

	for (j = 0; j < height1; j++)
	{
		for (i = 0; i < width2; i++)
		{
			sum = 0;
			second = src2 + i;
			for (k = 0; k < width1; k++)
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
