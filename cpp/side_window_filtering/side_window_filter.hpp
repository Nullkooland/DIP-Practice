#pragma once
#include "opencv2/core.hpp"

void swfLinear(const cv::Mat& src, cv::Mat& dst, const cv::Mat& kernel1D);
void swfMedian(const cv::Mat& src, cv::Mat& dst, int ksize);