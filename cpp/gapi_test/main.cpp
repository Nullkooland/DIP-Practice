#include "opencv2/core.hpp"
#include "opencv2/gapi.hpp"
#include "opencv2/gapi/core.hpp"
#include "opencv2/gapi/fluid/core.hpp"
#include "opencv2/gapi/fluid/imgproc.hpp"
#include "opencv2/gapi/gcomputation.hpp"
#include "opencv2/gapi/imgproc.hpp"
#include "opencv2/gapi/streaming/cap.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include <cstdio>

int main(int argc, char const* argv[]) {
    // Build processing pipeline
    cv::GComputation pipeline([]() {
        cv::GMat input;
        cv::GMat blurred = cv::gapi::bilateralFilter(input, 15, 25, 10);
        cv::GMat gray = cv::gapi::BGR2Gray(blurred);
        auto corners = cv::gapi::goodFeaturesToTrack(gray, 256, 0.2, 10);
        return cv::GComputation(cv::GIn(input), cv::GOut(blurred, corners));
    });

    // Compile computation graph

    // Use fluid backend
    auto kernels = cv::gapi::combine(cv::gapi::core::fluid::kernels(),
                                     cv::gapi::imgproc::fluid::kernels());
    auto compiledPipeline =
        pipeline.compileStreaming(cv::compile_args(kernels));

    compiledPipeline.setSource(
        cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(1));
    compiledPipeline.start();

    cv::Mat frame;
    std::vector<cv::Point2f> corners;

    size_t frameCount = 0;
    double avgProcessTimeSecs = 0.0;

    for (;;) {

        if (cv::waitKey(10) == 'q') {
            break;
        }

        int64_t tickStart = cv::getTickCount();
        if (!compiledPipeline.pull(cv::gout(frame, corners))) {
            break;
        }
        int64_t tickEnd = cv::getTickCount();

        double computationTimeSecs =
            static_cast<double>(tickEnd - tickStart) / cv::getTickFrequency();

        frameCount++;
        avgProcessTimeSecs += computationTimeSecs;

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
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2,
                    cv::LINE_AA);
        cv::imshow("Corners G-API", frame);
    }
    compiledPipeline.stop();

    std::printf("[G-API] Average process time %.2f ms\n",
                avgProcessTimeSecs / frameCount * 1e3);

    // The traditional way
    cv::VideoCapture cap(1);

    if (!cap.isOpened()) {
        std::fprintf(stderr, "Failed to open camera!\n");
        std::exit(1);
    }

    frameCount = 0;
    avgProcessTimeSecs = 0.0;

    for (;;) {
        if (cv::waitKey(10) == 'q') {
            break;
        }

        // Regular version
        int64_t tickStart = cv::getTickCount();

        if (!cap.read(frame)) {
            break;
        }

        cv::Mat gray, blurred;
        cv::bilateralFilter(frame, blurred, 15, 25, 10);
        cv::cvtColor(blurred, gray, cv::COLOR_BGR2GRAY);
        cv::goodFeaturesToTrack(gray, corners, 256, 0.2, 10);

        int64_t tickEnd = cv::getTickCount();

        double computationTimeSecs =
            static_cast<double>(tickEnd - tickStart) / cv::getTickFrequency();

        frameCount++;
        avgProcessTimeSecs += computationTimeSecs;

        for (const auto& corner : corners) {
            int x = static_cast<int>(corner.x);
            int y = static_cast<int>(corner.y);
            cv::drawMarker(blurred, cv::Point2i(x, y), cv::Scalar(0, 0, 255),
                           cv::MARKER_STAR, 10, 1);
        }

        char textBuffer[64];
        std::sprintf(textBuffer, "Process Time: %.2f ms, FPS: %.1f",
                     computationTimeSecs * 1e3, 1.0 / computationTimeSecs);

        cv::putText(blurred, textBuffer, cv::Point(32, 32),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2,
                    cv::LINE_AA);
        cv::imshow("Corners", blurred);
    }

    std::printf("[TRAD] Average process time %.2f ms\n",
                avgProcessTimeSecs / frameCount * 1e3);

    cv::destroyAllWindows();
    return 0;
}
