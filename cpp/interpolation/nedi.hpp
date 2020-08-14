/**
 * @file nedi.hpp
 * @author Goose Bomb (goose_bomb@outlook.com)
 * @brief Implementation of New Edge-Directed Interpolation algorithm
 * @version 0.1
 * @date 2020-08-13
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once
#include "opencv2/core.hpp"

void NEDI(const cv::Mat& src, cv::Mat& dst, int numPass,
          float localVarianceThreshold);
