#include <array>
#include <vector>
#include <immintrin.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "side_window_filter.hpp"

using Hist8b = cv::Vec<int, 256>;

static inline int calcBGR888L1Norm(const uint8_t* a, const uint8_t* b) {
    int norm = 0;
    norm += cv::abs(a[0] - b[0]);
    norm += cv::abs(a[1] - b[1]);
    norm += cv::abs(a[2] - b[2]);
    return norm;
}

static inline uint8_t getMedianFromHistogram(const Hist8b& hist,
                                             int threshold) {
    int count = 0;
    uint8_t k = UINT8_MAX;

    do {
        count += hist[++k];
    } while (count < threshold);

    return k;
}

static void swfSelectMinError(const cv::Mat& src, cv::Mat& dst,
                              const cv::Mat& filteredBySides) {
    dst.forEach<cv::Vec3b>(
        [&src, &filteredBySides](cv::Vec3b& pixel, const int pos[]) {
            int y = pos[0];
            int x = pos[1];
            int minError = INT_MAX;
            int k = -1;

            // ||     L     ||     R     ||    U      ||...
            // || B | G | R || B | G | R || B | G | R ||... 8 x BGR888
            const uint8_t* candidates = filteredBySides.ptr<uint8_t>(y, x);
            const uint8_t* source = src.ptr<uint8_t>(y, x);

            for (int i = 0; i < 8; i++) {
                int error = calcBGR888L1Norm(source, candidates + i * 3);
                if (error < minError) {
                    minError = error;
                    k = i;
                }
            }

            pixel[0] = candidates[k * 3];
            pixel[1] = candidates[k * 3 + 1];
            pixel[2] = candidates[k * 3 + 2];
        });
}

static void sidedMedianBlur(const cv::Mat& src, cv::Mat& dst, cv::Size size,
                            cv::Point anchor) {
    dst = cv::Mat(src.size(), CV_8UC3);

    // Pad borders
    int top = anchor.y + 1;
    int bottom = size.height - anchor.y - 1;
    int left = anchor.x + 1;
    int right = size.width - anchor.x - 1;

    cv::Mat padded;
    cv::copyMakeBorder(src, padded, top, bottom, left, right,
                       cv::BORDER_REFLECT_101);

    // Fast median blur algorithm
    // See https://ieeexplore.ieee.org/document/4287006

    // Histograms of each columns
    std::vector<Hist8b> colHistsB(padded.cols);
    std::vector<Hist8b> colHistsG(padded.cols);
    std::vector<Hist8b> colHistsR(padded.cols);
    // Kernel histogram
    Hist8b kernelHistB;
    Hist8b kernelHistG;
    Hist8b kernelHistR;

    // Initialize
    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < padded.cols; x++) {
            const auto& pixel = padded.at<cv::Vec3b>(y, x);
            colHistsB[x][pixel[0]]++;
            colHistsG[x][pixel[1]]++;
            colHistsR[x][pixel[2]]++;
        }
    }

    for (int y = 1; y < size.height + 1; y++) {
        for (int x = 0; x < size.width; x++) {
            const auto& pixel = padded.at<cv::Vec3b>(y, x);
            kernelHistB[pixel[0]]++;
            kernelHistG[pixel[1]]++;
            kernelHistR[pixel[2]]++;
        }
    }

    // Sliding window
    int threashold = size.area() / 2;

    for (int y = top; y < padded.rows - bottom; y++) {
        int prevY = y - anchor.y - 1;
        int nextY = y - anchor.y + size.height - 1;

        // Update column histograms and kernel histogram at leftmost postion
        kernelHistB = 0;
        kernelHistG = 0;
        kernelHistR = 0;

        for (int x = 0; x < size.width; x++) {
            const auto& pixelToRemove = padded.at<cv::Vec3b>(prevY, x);
            const auto& pixelToAdd = padded.at<cv::Vec3b>(nextY, x);

            colHistsB[x][pixelToRemove[0]]--;
            colHistsG[x][pixelToRemove[1]]--;
            colHistsR[x][pixelToRemove[2]]--;

            colHistsB[x][pixelToAdd[0]]++;
            colHistsG[x][pixelToAdd[1]]++;
            colHistsR[x][pixelToAdd[2]]++;

            kernelHistB += colHistsB[x];
            kernelHistG += colHistsG[x];
            kernelHistR += colHistsR[x];
        }

        for (int x = left; x < padded.cols - right; x++) {
            int prevX = x - anchor.x - 1;
            int nextX = x - anchor.x + size.width - 1;

            const auto& pixelToRemove = padded.at<cv::Vec3b>(prevY, nextX);
            const auto& pixelToAdd = padded.at<cv::Vec3b>(nextY, nextX);

            colHistsB[nextX][pixelToRemove[0]]--;
            colHistsG[nextX][pixelToRemove[1]]--;
            colHistsR[nextX][pixelToRemove[2]]--;

            colHistsB[nextX][pixelToAdd[0]]++;
            colHistsG[nextX][pixelToAdd[1]]++;
            colHistsR[nextX][pixelToAdd[2]]++;

            kernelHistB += colHistsB[nextX] - colHistsB[prevX];
            kernelHistG += colHistsG[nextX] - colHistsG[prevX];
            kernelHistR += colHistsR[nextX] - colHistsR[prevX];

            auto& pixel = dst.at<cv::Vec3b>(y - top, x - left);

            pixel[0] = getMedianFromHistogram(kernelHistB, threashold);
            pixel[1] = getMedianFromHistogram(kernelHistG, threashold);
            pixel[2] = getMedianFromHistogram(kernelHistR, threashold);
        }
    }
}

void swfLinear(const cv::Mat& src, cv::Mat& dst, const cv::Mat& kernel1D) {
    if (dst.size != src.size || src.data == dst.data) {
        dst = src.clone();
    }
    // Prepare half kernels
    int ksize = kernel1D.rows;
    cv::Mat kernelL = kernel1D.clone();
    cv::Mat kernelR;

    kernelL.rowRange(cv::Range(ksize / 2 + 1, ksize)) = 0;
    kernelL /= cv::sum(kernelL);

    cv::flip(kernelL, kernelR, 0);

    // Apply side window filtering
    std::array<cv::Mat, 8> planes;
    cv::sepFilter2D(src, planes[0], CV_8UC3, kernelL, kernel1D);
    cv::sepFilter2D(src, planes[1], CV_8UC3, kernelR, kernel1D);
    cv::sepFilter2D(src, planes[2], CV_8UC3, kernel1D, kernelL);
    cv::sepFilter2D(src, planes[3], CV_8UC3, kernel1D, kernelR);
    cv::sepFilter2D(src, planes[4], CV_8UC3, kernelL, kernelL);
    cv::sepFilter2D(src, planes[5], CV_8UC3, kernelL, kernelR);
    cv::sepFilter2D(src, planes[6], CV_8UC3, kernelR, kernelL);
    cv::sepFilter2D(src, planes[7], CV_8UC3, kernelR, kernelR);

    cv::Mat filteredBySides;
    cv::merge(planes, filteredBySides);

    // Get filtering result
    swfSelectMinError(src, dst, filteredBySides);
}

void swfMedian(const cv::Mat& src, cv::Mat& dst, int ksize) {
    if (dst.size != src.size || src.data == dst.data) {
        dst = src.clone();
    }

    std::array<cv::Mat, 8> planes;

    // Apply side window filtering
    int r = ksize / 2;
    // Left
    sidedMedianBlur(src, planes[0], cv::Size(r + 1, ksize), cv::Point(r, r));
    // Right
    sidedMedianBlur(src, planes[1], cv::Size(r + 1, ksize), cv::Point(0, r));
    // Up
    sidedMedianBlur(src, planes[2], cv::Size(ksize, r + 1), cv::Point(r, r));
    // Down
    sidedMedianBlur(src, planes[3], cv::Size(ksize, r + 1), cv::Point(r, 0));
    // LeftUp
    sidedMedianBlur(src, planes[4], cv::Size(r + 1, r + 1), cv::Point(r, r));
    // LeftDown
    sidedMedianBlur(src, planes[5], cv::Size(r + 1, r + 1), cv::Point(r, 0));
    // RightUp
    sidedMedianBlur(src, planes[6], cv::Size(r + 1, r + 1), cv::Point(0, r));
    // RightDown
    sidedMedianBlur(src, planes[7], cv::Size(r + 1, r + 1), cv::Point(0, 0));

    cv::Mat filteredBySides;
    cv::merge(planes, filteredBySides);

    swfSelectMinError(src, dst, filteredBySides);
}
