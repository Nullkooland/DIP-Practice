/**
 * @file nedi.cpp
 * @author Goose Bomb (goose_bomb@outlook.com)
 * @brief Implementation of New Edge-Directed Interpolation algorithm
 * @version 0.1
 * @date 2020-08-13
 *
 * @copyright Copyright (c) 2020
 *
 */

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "interp.hpp"
#include "nedi.hpp"

static constexpr int M = 4;

static cv::Mat calcLocalVariance(const cv::Mat& x, int ksize) {
    // Calculate E[X]
    cv::Mat xMean;
    cv::boxFilter(x, xMean, CV_32F, cv::Size(ksize, ksize));
    // Calculate E[X^2]
    cv::Mat xSquareMean;
    cv::boxFilter(x.mul(x), xSquareMean, CV_32F, cv::Size(ksize, ksize));
    // D[X] = E[X^2] - E[X]^2
    cv::Mat variance = xSquareMean - xMean.mul(xMean);
    cv::copyMakeBorder(variance, variance, 0, 1, 0, 1, cv::BORDER_CONSTANT,
                       0.0f);
    return variance;
}

static void passDiagonal(const cv::Mat& imgLowRes, cv::Mat& imgHighRes,
                         const cv::Mat& localVariance,
                         float localVarianceThreshold) {
    // Get size
    int rowsLo = imgLowRes.rows;
    int colsLo = imgLowRes.cols;
    // Extend boundaries
    cv::Mat temp;
    cv::copyMakeBorder(imgLowRes, temp, M / 2, M / 2 + 1, M / 2, M / 2 + 1,
                       cv::BORDER_REFLECT_101);

    // Run interpolation along diagonal direction ND(p)
    cv::parallel_for_(
        cv::Range(0, rowsLo * colsLo), [&](const cv::Range range) {
            for (int r = range.start; r < range.end; r++) {
                int iLo = r / colsLo;
                int jLo = r % colsLo;

                int iHi = iLo * 2 + 1;
                int jHi = jLo * 2 + 1;

                // The initial diagonal interpolation weights is
                // equal just like bilinear interpolation
                auto w = cv::Vec4f::all(0.25f);
                auto C = cv::Matx<float, M * M, 4>();
                auto y = cv::Vec<float, M * M>();

                // For edge area, calculate diagonal interpolation
                // weights using local edge-directed property
                if (localVariance.at<float>(iLo, jLo) >
                        localVarianceThreshold &&
                    localVariance.at<float>(iLo, jLo + 1) >
                        localVarianceThreshold &&
                    localVariance.at<float>(iLo + 1, jLo + 1) >
                        localVarianceThreshold &&
                    localVariance.at<float>(iLo + 1, jLo) >
                        localVarianceThreshold) {

                    int m = 0;
                    for (int k = iLo + 1; k < iLo + 1 + M; k++) {
                        for (int l = jLo + 1; l < jLo + 1 + M; l++) {
                            // Put the pixels within local window into
                            // vector y
                            y[m] = temp.at<float>(k, l);
                            // Put the ND(p) 4 diagonal neighbors of each
                            // pixel into m-th row of matrix C
                            C(m, 0) = temp.at<float>(k - 1, l - 1);
                            C(m, 1) = temp.at<float>(k - 1, l + 1);
                            C(m, 2) = temp.at<float>(k + 1, l + 1);
                            C(m, 3) = temp.at<float>(k + 1, l - 1);
                            m++;
                        }
                    }

                    // Solve for interpolation weights
                    cv::solve(C, y, w, cv::DECOMP_NORMAL | cv::DECOMP_CHOLESKY);
                }

                // Calculate the new pixel from diagonal 4 pixels
                // using interpolation weights just derived
                imgHighRes.at<float>(iHi, jHi) =
                    w[0] * temp.at<float>(iLo + M / 2, jLo + M / 2) +
                    w[1] * temp.at<float>(iLo + M / 2, jLo + M / 2 + 1) +
                    w[2] * temp.at<float>(iLo + M / 2 + 1, jLo + M / 2 + 1) +
                    w[3] * temp.at<float>(iLo + M / 2 + 1, jLo + M / 2);
            }
        });
}

static void passAxial(cv::Mat& imgHighRes, const cv::Mat& localVariance,
                      float localVarianceThreshold) {
    // Get size
    int rowsLo = imgHighRes.rows / 2;
    int colsLo = imgHighRes.cols / 2;
    // Extend boundaries
    cv::Mat temp;
    cv::copyMakeBorder(imgHighRes, temp, M + 1, M + 1, M + 1, M + 1,
                       cv::BORDER_REFLECT_101);

    // Run interpolation along vertical and horizontal direction N4(p)
    cv::parallel_for_(
        cv::Range(0, rowsLo * colsLo), [&](const cv::Range& range) {
            for (int r = range.start; r < range.end; r++) {
                int iLo = r / colsLo;
                int jLo = r % colsLo;

                int iHiEven = iLo * 2;
                int jHiEven = jLo * 2 + 1;

                int iHiOdd = iLo * 2 + 1;
                int jHiOdd = jLo * 2;

                // The initial diagonal interpolation weights is
                // equal just like bilinear interpolation
                auto wEven = cv::Vec4f::all(0.25f);
                auto CEven = cv::Matx<float, M * M, 4>();
                auto yEven = cv::Vec<float, M * M>();

                auto wOdd = cv::Vec4f::all(0.25f);
                auto COdd = cv::Matx<float, M * M, 4>();
                auto yOdd = cv::Vec<float, M * M>();

                // Build functor to process the 45 degrees rotated window
                auto processWindow = [&temp](int i, int j,
                                             cv::Matx<float, M * M, 4>& C,
                                             cv::Vec<float, M * M>& y) {
                    int m = 0;
                    for (int k = i - M + 1; k < i + M; k++) {
                        int h = M - cv::abs(k - i) - 1;
                        for (int l = j - h; l < j + h + 1; l += 2) {
                            // Put the pixels within local window into
                            // vector y
                            y[m] = temp.at<float>(k, l);
                            // Put the N4(p) 4 axial neighbors of each
                            // pixel into m-th row of matrix C
                            C(m, 0) = temp.at<float>(k - 2, l);
                            C(m, 1) = temp.at<float>(k, l + 2);
                            C(m, 2) = temp.at<float>(k + 2, l);
                            C(m, 3) = temp.at<float>(k, l - 2);
                            m++;
                        }
                    }
                };

                if (localVariance.at<float>(iLo, jLo) >
                        localVarianceThreshold &&
                    localVariance.at<float>(iLo, jLo + 1) >
                        localVarianceThreshold) {
                    // Fill in C and y of current interpolated pixel on even row
                    processWindow(iHiEven, jHiEven, CEven, yEven);
                    // Solve for interpolation weights
                    cv::solve(CEven, yEven, wEven,
                              cv::DECOMP_NORMAL | cv::DECOMP_CHOLESKY);
                }

                if (localVariance.at<float>(iLo, jLo) >
                        localVarianceThreshold &&
                    localVariance.at<float>(iLo + 1, jLo) >
                        localVarianceThreshold) {
                    // Fill in C and y of current interpolated pixel on odd row
                    processWindow(iHiOdd, jHiOdd, COdd, yOdd);
                    // Solve for interpolation weights
                    cv::solve(COdd, yOdd, wOdd,
                              cv::DECOMP_NORMAL | cv::DECOMP_CHOLESKY);
                }

                auto interpolatePixel =
                    [&imgHighRes, &temp](int i, int j, const cv::Vec4f& w) {
                        imgHighRes.at<float>(i, j) =
                            w[0] * temp.at<float>(i + M, j + M + 1) +
                            w[1] * temp.at<float>(i + M + 1, j + M + 2) +
                            w[2] * temp.at<float>(i + M + 2, j + M + 1) +
                            w[3] * temp.at<float>(i + M + 1, j + M);
                    };

                // Calculate the new pixel of even and odd rows
                // from up/right/down/left 4 pixels
                // using interpolation weights just derived
                interpolatePixel(iHiEven, jHiEven, wEven);
                interpolatePixel(iHiOdd, jHiOdd, wOdd);
            }
        });
}

void NEDI(const cv::Mat& src, cv::Mat& dst, int numPass,
          float localVarianceThreshold) {
    CV_Assert(numPass >= 1);

    int rowsLowRes = src.rows;
    int colsLowRes = src.cols;

    cv::Mat imgLowRes;
    cv::Mat imgHighRes;
    cv::Mat localVariance;

    if (src.type() == CV_32F) {
        src.copyTo(imgLowRes);
    } else if (src.type() == CV_8U) {
        src.convertTo(imgLowRes, CV_32F, 1.0 / 255.0);
    } else {
        CV_Error(cv::Error::BadDepth, "Unsupported depth");
    }

    for (size_t i = 0; i < numPass; i++) {
        // Calculate local variance
        localVariance = calcLocalVariance(imgLowRes, 3);
        // Create initial 2x interpolated image using simple nearest-neighbor
        // cv::resize(imgLowRes, imgHighRes,
        //            cv::Size(colsLowRes * 2, rowsLowRes * 2), 0, 0,
        //            cv::INTER_NEAREST);
        interpZero(imgLowRes, imgHighRes, 2);
        // First pass (diagonal)
        passDiagonal(imgLowRes, imgHighRes, localVariance,
                     localVarianceThreshold);
        // Second pass (horizontal and vertical)
        passAxial(imgHighRes, localVariance, localVarianceThreshold);
        // Update size
        rowsLowRes *= 2;
        colsLowRes *= 2;
        // Update the low-res image as 2x interpolated image we just got
        imgLowRes = imgHighRes;
    }

    // Outupt
    if (src.type() == CV_32F) {
        dst = imgHighRes;
    } else if (src.type() == CV_8U) {
        imgHighRes.convertTo(dst, CV_8U, 255.0);
    }
}