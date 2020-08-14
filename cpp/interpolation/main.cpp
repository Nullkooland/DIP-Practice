#include <cmath>
#include <iostream>
#include <array>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "interp.hpp"
#include "nedi.hpp"

int main(int argc, char const* argv[]) {
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    // cv::namedWindow("Zero Interpolation", cv::WINDOW_AUTOSIZE);
    // cv::namedWindow("Linear Interpolation", cv::WINDOW_AUTOSIZE);
    // cv::namedWindow("Lanczos Interpolation", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("NEDI", cv::WINDOW_AUTOSIZE);

    constexpr int k = 2;

    cv::Mat src_img = cv::imread("./images/snow_leopard.png");
    cv::imshow("Original", src_img);
/*
    cv::Mat zero_interp_img;
    interpZero(src_img, zero_interp_img, k);

    cv::imshow("Zero Interpolation", zero_interp_img);

    cv::Vec<float, k * 2 - 1> linear_kernel;
    for (int i = 0; i < k - 1; i++) {
        linear_kernel[i] = static_cast<float>(i + 1) / k;
        linear_kernel[2 * k - i - 2] = linear_kernel[i];
    }

    linear_kernel[k - 1] = 1.0f;

    cv::Mat linear_interp_img;
    cv::sepFilter2D(zero_interp_img, linear_interp_img, CV_8U, linear_kernel,
                    linear_kernel);

    cv::imshow("Linear Interpolation", linear_interp_img);

    constexpr int a = 3;
    cv::Vec<float, a * k * 2 - 1> lanczos_kernel;
    for (int i = 0; i < a * k * 2 - 1; i++) {
        int x = i - a * k + 1;
        lanczos_kernel[i] = a * k * k * std::sin(M_PI * x / k) *
                            std::sin(M_PI * x / (a * k)) /
                            (M_PI * M_PI * x * x);
    }

    lanczos_kernel[a * k - 1] = 1.0f;

    cv::Mat lanczos_interp_img;
    cv::sepFilter2D(zero_interp_img, lanczos_interp_img, CV_8U, lanczos_kernel,
                    lanczos_kernel);

    cv::imshow("Lanczos Interpolation", lanczos_interp_img);
*/

    std::array<cv::Mat, 3> rgb_src;
    std::array<cv::Mat, 3> rgb_NEDI;
    cv::split(src_img, rgb_src);

    for (size_t i = 0; i < 3; i++) {
        NEDI(rgb_src[i], rgb_NEDI[i], 1, 0.0025f); 
    }

    cv::Mat img_NEDI;
    cv::merge(rgb_NEDI, img_NEDI);

    cv::imshow("NEDI", img_NEDI);

    cv::Mat linear_interp_img;
    cv::resize(src_img, linear_interp_img, cv::Size(-1, -1), 2.0, 2.0,
               cv::INTER_LINEAR);

    cv::imshow("Linear Interpolation", linear_interp_img);

    cv::waitKey();
    return 0;
}
