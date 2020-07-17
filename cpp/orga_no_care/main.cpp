#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <cstdio>
#include <string>

const std::string INPUT_PATH = "/Users/goose_bomb/Movies/rideon_x264.mp4";
const std::string OUTPUT_PATH = "/Users/goose_bomb/Movies/rideon_polar.mp4";

int main(int argc, char const* argv[]) {
    cv::VideoCapture cap(INPUT_PATH);
    if (!cap.isOpened()) {
        fprintf(stderr, "Failed to open %s!", INPUT_PATH.c_str());
        exit(1);
    }

    int M = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int N = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    double framerate = cap.get(cv::CAP_PROP_FPS);
    int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));

    auto fourccStr = std::string({
        static_cast<char>(fourcc),
        static_cast<char>(fourcc >> 8),
        static_cast<char>(fourcc >> 16),
        static_cast<char>(fourcc >> 24),
    });

    fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
    cv::VideoWriter writer(OUTPUT_PATH, fourcc, framerate, cv::Size(N, M));

    if (!writer.isOpened()) {
        fprintf(stderr, "Failed to open %s!", OUTPUT_PATH.c_str());
        exit(1);
    }

    cv::namedWindow("Ride on!!!", cv::WINDOW_KEEPRATIO);

    cv::Mat frame;
    cv::Mat frame_logPolar;

    while (cap.isOpened()) {
        cap >> frame;

        cv::logPolar(frame, frame_logPolar, cv::Point(N / 2, M / 2), 270,
                     cv::WARP_FILL_OUTLIERS);

        cv::imshow("Ride on!!!", frame_logPolar);

        if (cv::waitKey(16) == 'q') {
            break;
        }

        writer << frame_logPolar;
    }

    cap.release();
    writer.release();

    return 0;
}
