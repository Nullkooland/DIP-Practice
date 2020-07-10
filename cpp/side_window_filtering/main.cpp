#include <iostream>
#include <random>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "side_window_filter.hpp"

void addSaltAndPepperNoise(cv::Mat& src, float p) {
    auto rd = std::random_device();
    auto rng_noise = cv::RNG(rd());
    auto rng_salt_or_pepper = cv::RNG(rd());

    src.forEach<cv::Vec3b>([&](cv::Vec3b& pixel, const int pos[]) {
        if ((float)rng_noise < p) {
            uint8_t noise_val = (float)rng_salt_or_pepper < 0.5f ? 0 : 255;
            pixel[0] = noise_val;
            pixel[1] = noise_val;
            pixel[2] = noise_val;
        }
    });
}

int main(int argc, char const* argv[]) {
    cv::namedWindow("Original", cv::WINDOW_KEEPRATIO);
    cv::namedWindow("Noised", cv::WINDOW_KEEPRATIO);
    cv::namedWindow("Gaussian", cv::WINDOW_KEEPRATIO);
    cv::namedWindow("Median", cv::WINDOW_KEEPRATIO);
    cv::namedWindow("SWF - Gaussian", cv::WINDOW_KEEPRATIO);
    cv::namedWindow("SWF - Median", cv::WINDOW_KEEPRATIO);

    cv::Mat src_img = cv::imread("./images/cartoon.png");
    cv::imshow("Original", src_img);

    cv::Mat gaussianNoise(src_img.size(), CV_8SC3);

    auto rd = std::random_device();
    auto rng = cv::RNG(rd());

    rng.fill(gaussianNoise, cv::RNG::NORMAL, 0, 13);

    cv::Mat noised_img;
    cv::add(src_img, gaussianNoise, noised_img, cv::noArray(), CV_8UC3);

    addSaltAndPepperNoise(noised_img, 0.075f);
    cv::imshow("Noised", noised_img);

    cv::Mat filtered_img;
    cv::GaussianBlur(noised_img, filtered_img, cv::Size(7, 7), 2.0);
    cv::imshow("Gaussian", filtered_img);

    cv::Mat gaussianKernel = cv::getGaussianKernel(7, 2.0, CV_32F);
    // cv::Mat boxKernel = cv::Mat(cv::Size(1, 7), CV_32F, 1.0f / 7.0f);
    cv::Mat swf_filtered_img;
    swfLinear(noised_img, swf_filtered_img, gaussianKernel);

    for (size_t i = 0; i < 4; i++) {
        swfLinear(swf_filtered_img, swf_filtered_img, gaussianKernel);
    }

    cv::imshow("SWF - Gaussian", swf_filtered_img);

    cv::Mat median_blur_img;
    cv::medianBlur(noised_img, median_blur_img, 7);
    cv::medianBlur(median_blur_img, median_blur_img, 3);

    cv::imshow("Median", median_blur_img);

    cv::Mat swf_median_blur_img;
    swfMedian(noised_img, swf_median_blur_img, 7);
    swfMedian(swf_median_blur_img, swf_median_blur_img, 3);

    cv::imshow("SWF - Median", swf_median_blur_img);

    cv::waitKey();
    return 0;
}
