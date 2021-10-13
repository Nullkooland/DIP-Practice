#include <array>
#include <cstdio>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/fluid/imgproc.hpp>
#include <opencv2/gapi/gcomputation.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/ocl/core.hpp>
#include <opencv2/gapi/ocl/imgproc.hpp>
#include <opencv2/gapi/render.hpp>
#include <opencv2/gapi/render/render.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace draw = cv::gapi::wip::draw;

namespace custom {
// clang-format off

G_API_OP(DrawCorners,
         <cv::GArray<draw::Prim>(cv::GArray<cv::Point2f>)>,
         "demo.g-api.draw-corners") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc&){
        return cv::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVDrawCorners, DrawCorners) {
   static void run(const std::vector<cv::Point2f> &corners,
                   std::vector<draw::Prim> &outPrims) {
        outPrims.clear();
        for (const auto& corner: corners) {
            int x = static_cast<int>(corner.x);
            int y = static_cast<int>(corner.y);
            auto circle = draw::Circle({x, y}, 5, CV_RGB(255, 0, 125), -1, cv::LineTypes::LINE_AA);
            outPrims.emplace_back(circle);
        }
    }
};

// clang-format on
} // namespace custom

int main(int argc, char const* argv[]) {
    // Build processing pipeline
    cv::GComputation pipeline([]() {
        cv::GMat input;
        cv::GMat blurred = cv::gapi::bilateralFilter(input, 15, 25, 10);
        cv::GMat gray = cv::gapi::BGR2Gray(blurred);

        auto corners = cv::gapi::goodFeaturesToTrack(gray, 256, 0.2, 10);
        auto prims = custom::DrawCorners::on(corners);
        cv::GMat output = draw::render3ch(blurred, prims);

        return cv::GComputation(cv::GIn(input), cv::GOut(output));
    });

    // Use OpenCL backend
    auto customKernels = cv::gapi::kernels<custom::OCVDrawCorners>();
    auto kernels = cv::gapi::combine(cv::gapi::core::ocl::kernels(),
                                     cv::gapi::imgproc::ocl::kernels(),
                                     customKernels);
    // Use fluid backend
    // auto kernels = cv::gapi::combine(cv::gapi::core::fluid::kernels(),
    //                                  cv::gapi::imgproc::fluid::kernels(),
    //                                  customKernels);

    auto compiledPipeline =
        pipeline.compileStreaming(cv::compile_args(kernels));

    compiledPipeline.setSource(
        cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(1));
    compiledPipeline.start();

    cv::Mat frame;
    std::vector<cv::Point2f> corners;
    cv::TickMeter tm;

    cv::namedWindow("Corners G-API");
    for (;;) {
        if (cv::waitKey(10) == 'q') {
            break;
        }

        tm.start();
        if (!compiledPipeline.pull(cv::gout(frame))) {
            break;
        }
        tm.stop();

        std::array<char, 64> textBuffer;
        std::sprintf(textBuffer.data(),
                     "Process Time: %.2f ms, FPS: %.1f",
                     tm.getAvgTimeMilli(),
                     tm.getFPS());

        cv::putText(frame,
                    textBuffer.data(),
                    cv::Point(32, 32),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    cv::Scalar(0, 255, 0),
                    2,
                    cv::LINE_AA);
        cv::imshow("Corners G-API", frame);
    }
    compiledPipeline.stop();

    // The traditional way
    cv::VideoCapture cap(1);

    if (!cap.isOpened()) {
        std::fprintf(stderr, "Failed to open camera!\n");
        std::exit(1);
    }

    tm.reset();

    for (;;) {
        if (cv::waitKey(10) == 'q') {
            break;
        }

        // Regular version
        tm.start();

        if (!cap.read(frame)) {
            break;
        }

        cv::Mat gray, blurred;
        cv::bilateralFilter(frame, blurred, 15, 25, 10);
        cv::cvtColor(blurred, gray, cv::COLOR_BGR2GRAY);
        cv::goodFeaturesToTrack(gray, corners, 256, 0.2, 10);

        for (const auto& corner : corners) {
            int x = static_cast<int>(corner.x);
            int y = static_cast<int>(corner.y);
            cv::drawMarker(blurred,
                           cv::Point2i(x, y),
                           cv::Scalar(0, 0, 255),
                           cv::MARKER_STAR,
                           10,
                           1);
        }

        tm.stop();

        std::array<char, 64> textBuffer;
        std::sprintf(textBuffer.data(),
                     "Process Time: %.2f ms, FPS: %.1f",
                     tm.getAvgTimeMilli(),
                     tm.getFPS());

        cv::putText(blurred,
                    textBuffer.data(),
                    cv::Point(32, 32),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    cv::Scalar(0, 255, 0),
                    2,
                    cv::LINE_AA);
        cv::imshow("Corners", blurred);
    }

    cv::destroyAllWindows();
    return 0;
}
