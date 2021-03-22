import numpy as np
import cv2
import pyheif
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# load source image
src_img = pyheif.read_as_numpy("./images/nebula.heic")
fig = plt.figure(figsize=(12, 8))
im = plt.imshow(src_img)

# convert to YCbCr color space
src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2YCrCb)
src_img = np.float32(src_img)

# split channels and compute DCT for each channel
(y, cb, cr) = cv2.split(src_img)
y_dct = cv2.dct(y)
cb_dct = cv2.dct(cb)
cr_dct = cv2.dct(cr)

ax_luma_quant = plt.axes([0.15, 0.96, 0.75, 0.02])
slider_luma_quant = Slider(ax_luma_quant,
                           "Luma Quant",
                           1,
                           500,
                           valinit=1,
                           valstep=1)

ax_chroma_quant = plt.axes([0.15, 0.92, 0.75, 0.02])
slider_chroma_quant = Slider(ax_chroma_quant,
                             "Chroma Quant",
                             1,
                             500,
                             valinit=1,
                             valstep=1)


def get_q_img():
    q_img = cv2.merge((y, cb, cr))
    q_img = np.uint8(q_img)
    q_img = cv2.cvtColor(q_img, cv2.COLOR_YCrCb2RGB)
    return q_img


def update_luma_quant(val):
    global y
    q_y_dct = np.round(y_dct / val) * val
    y = cv2.idct(q_y_dct)

    im.set_data(get_q_img())
    fig.canvas.draw_idle()


def update_chroma_quant(val):
    global cb, cr
    q_cb_dct = np.round(cb_dct / val) * val
    cb = cv2.idct(q_cb_dct)

    q_cr_dct = np.round(cr_dct / val) * val
    cr = cv2.idct(q_cr_dct)

    im.set_data(get_q_img())
    fig.canvas.draw_idle()


# Set slider value changed callback
slider_luma_quant.on_changed(update_luma_quant)
slider_chroma_quant.on_changed(update_chroma_quant)

plt.show()
plt.tight_layout()
