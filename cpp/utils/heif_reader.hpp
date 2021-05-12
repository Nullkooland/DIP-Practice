#pragma once

#include <opencv2/core.hpp>
#include <string_view>

namespace utils {
cv::Mat readHEIF(std::string_view path);
}