#include <memory>

#include "interp.hpp"
#include "opencv2/core.hpp"

void interpZero(const cv::Mat& src, cv::Mat& dst, int k) {
    if (dst.size() != src.size() * k) {
        dst = cv::Mat(src.size() * k, src.type(), cv::Scalar(0));
    }

    cv::parallel_for_(cv::Range(0, src.total()), [&](const cv::Range& range) {
        for (int r = range.start; r < range.end; r++) {
            int y = r / src.cols;
            int x = r % src.cols;

            const std::byte* srcPixel = src.ptr<std::byte>(y, x);
            std::byte* dstPixel = dst.ptr<std::byte>(y * k, x * k);

            memcpy(dstPixel, srcPixel, src.elemSize());
        }
    });
}