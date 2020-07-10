import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def estimate_noise(roi, fig, hist_ax, fit_ax):
    cv2.imshow('Flat Region', roi)
    hist = cv2.calcHist([roi], [0], None, [256], [0, 256])

    hist = cv2.normalize(hist, None, 1, 0, norm_type=cv2.NORM_L1, dtype=cv2.CV_32F)
    guess_mean = np.argmax(hist)
    mean, std = norm.fit(roi.data, loc=guess_mean)
    fit_curve = norm.pdf(np.arange(256), mean, std)

    hist_ax.set_ydata(hist)
    fit_ax.set_ydata(fit_curve)
    fig.canvas.draw_idle()

def adaptive_denoise_filter(img, std_noise, ksize):
    img = np.float32(img)
    local_mean = cv2.GaussianBlur(img, (ksize, ksize), 0)
    local_variance = cv2.GaussianBlur(img ** 2, (ksize, ksize), 0) - local_mean ** 2

    filtered_img = img - (std_noise ** 2 / (local_variance + 1e-4)) * (img - local_mean)
    filtered_img= np.clip(filtered_img, 0, 255)
    return np.uint8(filtered_img)


if __name__ == "__main__":
    # Prepare this crappy 720P camera on my MBP
    cap = cv2.VideoCapture(0)
    # fig = plt.figure('Noise Estimation', figsize=(12, 4))
    # x = np.arange(256)
    # hist_ax, fit_ax = plt.plot(x, np.zeros_like(x), x, np.zeros_like(x))
    # plt.axis([0, 256, 0, 0.25])

    # plt.show(block=False)
    # plt.tight_layout()

    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Frame', frame)

        roi = frame[0:180, 1000:1180]
        mean, std_noise = norm.fit(roi.data)

        denoised_frame = adaptive_denoise_filter(frame, std_noise, 11)
        cv2.imshow('Denoised', denoised_frame)

        # estimate_noise(roi, fig, hist_ax, fit_ax)

        if cv2.waitKey(16) & 0xFF == ord('q'):
            break;

    # plt.close(fig)
    cv2.destroyAllWindows()