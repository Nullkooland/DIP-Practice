#include "opencv2/core.hpp"
#include "opencv2/gapi.hpp"
#include "opencv2/gapi/core.hpp"
#include "opencv2/gapi/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include <cstdio>

int main(int argc, char const* argv[]) {
    // Build processing pipeline

    cv::GComputationT<cv::GArray<cv::Point2f>(cv::GMat)> computation(
        [](auto input) {
            auto gray = cv::gapi::BGR2Gray(input);
            auto blurred = cv::gapi::bilateralFilter(gray, 5, 25, 5);
            return cv::gapi::goodFeaturesToTrack(blurred, 256, 0.2, 10);
        });

    // Prepare cameara video streaming
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::fprintf(stderr, "Failed to open camera!\n");
        std::exit(1);
    }

    cv::Mat frame;
    std::vector<cv::Point2f> corners;
    for (;;) {
        if (!cap.read(frame) || cv::waitKey(16) == 'q') {
            break;
        }

        // Regular version
        // int64_t tickStart = cv::getTickCount();
        // cv::Mat gray, blurred;
        // cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        // cv::bilateralFilter(gray, blurred, 5, 25, 5);
        // cv::goodFeaturesToTrack(blurred, corners, 255, 0.2, 10);
        // int64_t tickEnd = cv::getTickCount();

        // G-API version
        int64_t tickStart = cv::getTickCount();
        computation.apply(frame, corners);
        int64_t tickEnd = cv::getTickCount();

        double computationTimeSecs =
            static_cast<double>(tickEnd - tickStart) / cv::getTickFrequency();

        for (const auto& corner : corners) {
            int x = static_cast<int>(corner.x);
            int y = static_cast<int>(corner.y);
            cv::drawMarker(frame, cv::Point2i(x, y), cv::Scalar(0, 0, 255),
                           cv::MARKER_STAR, 10, 1);
        }

        char textBuffer[64];
        std::sprintf(textBuffer, "Process Time: %.2f ms, FPS: %.1f",
                     computationTimeSecs * 1e3, 1.0 / computationTimeSecs);

        cv::putText(frame, textBuffer, cv::Point(32, 32),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0),
                    2, cv::LINE_AA);
        cv::imshow("Corners", frame);
    }

    return 0;
}
