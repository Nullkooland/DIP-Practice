import cv2
from utils.image_reader import ImageReader
import numpy as np

if __name__ == "__main__":
    reader = ImageReader()
    src_img = reader.read("images/cat.heic")
    h, w, c = src_img.shape

    pry_down1 = cv2.pyrDown(src_img)
    pry_down2 = cv2.pyrDown(pry_down1)

    up2x_img = cv2.resize(pry_down1, (h, w), interpolation=cv2.INTER_LANCZOS4)
    up4x_img = cv2.resize(pry_down2, (h, w), interpolation=cv2.INTER_LANCZOS4)

    gmsd = cv2.quality_QualityGMSD()
    ret, gmsd_map = gmsd.compute(src_img, up2x_img)

    print(ret)
    cv2.imshow("GMSD", gmsd_map)

    cv2.waitKey()