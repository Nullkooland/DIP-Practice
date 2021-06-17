from typing import Iterable, List
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, ConnectionPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_keypoints(img: np.ndarray, 
                   keypoints: List[cv2.KeyPoint],
                   show_response: bool=True):
    h, w = img.shape[:2]

    if show_response:
        responses = list(map(lambda keypoint: keypoint.response, keypoints))
        norm = Normalize(0.0, max(responses))
        cm = plt.get_cmap("turbo")
    else:
        norm = None
        cm = None

    fig, ax = plt.subplots(
        1, 1, num="Keypoint", figsize=(8, 8))

    ax.imshow(img)

    for keypoint in keypoints:
        x, y = keypoint.pt
        s = keypoint.size if keypoint.size > 0.0 else 0.01 * min(h, w)

        if show_response:
            response = keypoint.response
            color_face = cm(norm(response), alpha=norm(response) * 0.3)
            color_edge = cm(norm(response))
        else:
            color_face = "None"
            color_edge = "red"

        ax.plot(x, y, ls='', marker='.', color=color_edge)

        if keypoint.angle != -1.0:
            angle = np.deg2rad(keypoint.angle)
            dx, dy = np.cos(angle) * s, np.sin(angle) * s
            ax.arrow(x, y, dx, dy, width=s * 0.05, head_width=s * 0.25,
                        head_length=s * 0.25, length_includes_head=True,
                        color=color_edge)

        circle = Circle((x, y), radius=s, facecolor=color_face, edgecolor=color_edge)
        ax.add_artist(circle)

    if show_response:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.1)
        fig.colorbar(ScalarMappable(norm, cmap=cm),
                    cax=cax, label="Response", orientation="vertical")

    plt.show()

def plot_matches(img_a: np.ndarray,
                 img_b: np.ndarray,
                 keypoints_a: List[cv2.KeyPoint],
                 keypoints_b: List[cv2.KeyPoint],
                 matches: Iterable[cv2.DMatch]):

    distances = list(map(lambda m: m.distance, matches))
    norm = Normalize(min(distances), max(distances))
    cm = plt.get_cmap("rainbow")

    fig, axs = plt.subplots(
        1, 2, num="Matches", figsize=(14, 6))

    axs[0].imshow(img_a)
    axs[1].imshow(img_b)

    for match in matches:
        keypoint_a = keypoints_a[match.queryIdx]
        keypoint_b = keypoints_b[match.trainIdx]
        distance = match.distance

        x0, y0 = keypoint_a.pt
        x1, y1 = keypoint_b.pt

        s0 = keypoint_a.size
        s1 = keypoint_a.size

        circle0 = Circle((x0, y0), radius=s0,
                         facecolor=cm(norm(distance), alpha=0.3),
                         edgecolor=cm(norm(distance)))

        circle1 = Circle((x1, y1), radius=s1,
                         facecolor=cm(norm(distance), alpha=0.3),
                         edgecolor=cm(norm(distance)))

        axs[0].add_artist(circle0)
        axs[1].add_artist(circle1)

        angle0 = np.deg2rad(keypoint_a.angle)
        angle1 = np.deg2rad(keypoint_b.angle)
        dx0, dy0 = np.cos(angle0) * s0, np.sin(angle0) * s0
        dx1, dy1 = np.cos(angle1) * s1, np.sin(angle1) * s1

        axs[0].arrow(x0, y0, dx0, dy0, width=s0 * 0.05, head_width=s0 * 0.25,
                     head_length=s0 * 0.25, length_includes_head=True,
                     color=cm(norm(distance)))

        axs[1].arrow(x1, y1, dx1, dy1, width=s1 * 0.05, head_width=s1 * 0.25,
                     head_length=s1 * 0.25, length_includes_head=True,
                     color=cm(norm(distance)))

        con = ConnectionPatch(xyA=(x0, y0), xyB=(x1, y1), coordsA="data", coordsB="data",
                              axesA=axs[0], axesB=axs[1], color=cm(norm(distance)), alpha=0.5)
        con.set_in_layout(False)
        axs[1].add_artist(con)

    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="4%", pad=0.1)
    fig.colorbar(ScalarMappable(norm, cmap=cm),
                 cax=cax, label="Distance", orientation="vertical")

    plt.show()
