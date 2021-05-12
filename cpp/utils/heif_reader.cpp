#include "heif_reader.hpp"

#include <libheif/heif.h>
#include <libheif/heif_cxx.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace utils {
cv::Mat readHEIF(std::string_view path) {
    auto heifCtx = heif::Context();
    heifCtx.read_from_file(path.data());

    auto heifHandle = heifCtx.get_primary_image_handle();
    auto heifImage = heifHandle.decode_image(heif_colorspace_RGB,
                                             heif_chroma_interleaved_RGB);

    int width = heifImage.get_width(heif_channel_interleaved);
    int height = heifImage.get_height(heif_channel_interleaved);

    int stride;
    uint8_t* pixels = heifImage.get_plane(heif_channel_interleaved, &stride);

    auto src_img = cv::Mat(height, width, CV_8UC3, pixels, stride);
    cv::cvtColor(src_img, src_img, cv::COLOR_RGB2BGR);

    return src_img;
}
} // namespace utils