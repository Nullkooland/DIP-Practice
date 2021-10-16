from Foundation import NSData, NSURL, NSMutableData
import Quartz
from Quartz import CIContext, CIImage

import numpy as np
from numpy.typing import NDArray, DTypeLike


class ImageReader():
    """
    Image reader using macOS native API
    requires 'pyobjc-framework-Quartz' package
    """

    def __init__(self):
        options = {
            Quartz.kCIContextWorkingFormat: Quartz.kCIFormatRGBA8,
            Quartz.kCIContextWorkingColorSpace: Quartz.CGColorSpaceCreateWithName(
                Quartz.kCGColorSpaceSRGB)
        }

        self.context = CIContext.contextWithOptions_(options)
        if not self.context:
            raise RuntimeError("Failed to create CIContext")

    def read(self, filename: str,
             dtype: DTypeLike = np.uint8,
             ignore_alpha: bool = True,
             swapRB: bool = False) -> NDArray:
        url = NSURL.fileURLWithPath_(filename)
        options = {
            "kCIImageApplyOrientationProperty": True,
            "kCIImageCacheHint": False,
            "kCIImageCacheImmediately": False,
            "kCIImageAVDepthData": None,
        }
        ci_image = CIImage.imageWithContentsOfURL_options_(url, options)

        w = int(ci_image.extent().size.width)
        h = int(ci_image.extent().size.height)
        # colorspace = ci_image.colorSpace()
        colorspace = self.context.workingColorSpace()

        dtype = np.dtype(dtype)
        if swapRB:
            if dtype != np.uint8:
                raise TypeError(
                    f"Unsupported datatype: {dtype} in BGRA format")
            ci_format = Quartz.kCIFormatBGRA8
        elif dtype == np.uint8:
            ci_format = Quartz.kCIFormatRGBA8
        elif dtype == np.uint16:
            ci_format = Quartz.kCIFormatRGBA16
        elif dtype == np.float32:
            ci_format = Quartz.kCIFormatRGBAf
        elif dtype == np.float16:
            ci_format = Quartz.kCIFormatRGBAh
        else:
            raise TypeError(f"Unsupported datatype: {dtype} in RGBA format")

        row_bytes = 4 * w * dtype.itemsize
        buffer = NSMutableData.dataWithLength_(h * row_bytes)

        self.context.render_toBitmap_rowBytes_bounds_format_colorSpace_(
            ci_image,
            buffer,
            row_bytes,
            ci_image.extent(),
            ci_format,
            colorspace
        )

        img = np.frombuffer(buffer, dtype=dtype).reshape((h, w, 4))
        if ignore_alpha:
            img = img[..., :3].copy()

        return img


if __name__ == "__main__":
    # TEST
    reader = ImageReader()
    # img = reader.read("images/lena.heic", np.uint8)
    # img = reader.read("images/lena.heic", np.uint8, swapRB=True)
    # img = reader.read("images/lena.heic", np.uint16) // 255
    # img = reader.read("images/lena.heic", np.float32)
    # img = reader.read("images/lena.heic", np.float16).astype(np.float32)
    img = reader.read("images/opencv_logo.heic", np.uint8, ignore_alpha=False)

    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()
