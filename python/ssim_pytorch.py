import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution

    Returns:
        torch.Tensor: 1D kernel
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size//2

    g = torch.exp(-(coords**2) / (2*sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blured
        window (torch.Tensor): 1-D gauss kernel

    Returns:
        torch.Tensor: blured tensors
    """

    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def _ssim(X, Y, win, data_range=255, size_average=True, full=False, K=(0.01, 0.03), nonnegative_ssim=False):
    r""" Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid negative results.

    Returns:
        torch.Tensor: ssim results
    """
    K1, K2 = K
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq +
                                   sigma2_sq + C2)  # set alpha=beta=gamma=1
    if nonnegative_ssim:
        cs_map = F.relu(cs_map, inplace=True)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=255, size_average=True, full=False, K=(0.01, 0.03), nonnegative_ssim=False):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid negative results.

    Returns:
        torch.Tensor: ssim results
    """

    if len(X.shape) != 4:
        raise ValueError('Input images must be 4-d tensors.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    ssim_val, cs = _ssim(X, Y,
                         win=win,
                         data_range=data_range,
                         size_average=False,
                         full=True, K=K, nonnegative_ssim=nonnegative_ssim)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ms_ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=255, size_average=True, full=False, weights=None, K=(0.01, 0.03), nonnegative_ssim=False):
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if len(X.shape) != 4:
        raise ValueError('Input images must be 4-d tensors.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size-1) * (2**4), \
        "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % (
            (win_size-1) * (2**4))

    if weights is None:
        weights = torch.FloatTensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(X.device, dtype=X.dtype)

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    levels = weights.shape[0]
    mcs = []
    for _ in range(levels):
        ssim_val, cs = _ssim(X, Y,
                             win=win,
                             data_range=data_range,
                             size_average=False,
                             full=True, K=K, nonnegative_ssim=nonnegative_ssim)
        mcs.append(cs)

        padding = (X.shape[2] % 2, X.shape[3] % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)  # mcs, (level, batch)
    # weights, (level)
    # rua = torch.prod(mcs ** weights.unsqueeze(1), dim=0)
    msssim_val = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1))
                            * (ssim_val ** weights[-1]), dim=0)  # (batch, )

    if size_average:
        msssim_val = msssim_val.mean()
    return msssim_val


class SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=None, size_average=True, channel=3, K=(0.01, 0.03), nonnegative_ssim=False):
        r""" class for ssim
        Args:
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid negative results.
        """

        super(SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ssim(X, Y, win=self.win, data_range=self.data_range, size_average=self.size_average, K=self.K, nonnegative_ssim=self.nonnegative_ssim)


class MS_SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=None, size_average=True, channel=3, weights=None, K=(0.01, 0.03), nonnegative_ssim=False):
        r""" class for ms-ssim
        Args:
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid NaN results.
        """

        super(MS_SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ms_ssim(X, Y, win=self.win, size_average=self.size_average, data_range=self.data_range, weights=self.weights, K=self.K, nonnegative_ssim=self.nonnegative_ssim)


class MS_SSIM_Loss(nn.Module):
    def __init__(self, n_levels=5, L=2.0):
        super(MS_SSIM_Loss, self).__init__()

        gaussian_kernel = torch.tensor(cv2.getGaussianKernel(
            11, 1.5, cv2.CV_32F), requires_grad=False).unsqueeze(0).repeat(3, 1, 1, 1)

        self.register_buffer('gaussian_kernel_y', gaussian_kernel)
        self.register_buffer('gaussian_kernel_x',
                             gaussian_kernel.transpose(2, 3))

        self.C1 = (0.01 * L) ** 2
        self.C2 = (0.03 * L) ** 2

        weights = torch.tensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333], requires_grad=False)

        self.register_buffer('scale_weights', weights)

        self.n_levels = n_levels

        self.register_buffer('ms_cs', torch.empty(
            n_levels, requires_grad=False))

    def __get_mean_map(self, images):
        images = F.pad(images, (5, 5, 5, 5))
        out = F.conv2d(images, self.gaussian_kernel_x,
                       stride=1, padding=0, groups=3)
        out = F.conv2d(out, self.gaussian_kernel_y,
                       stride=1, padding=0, groups=3)
        return out

    def forward(self, images_A, images_B):
        for i in range(self.n_levels):
            mean_A = self.__get_mean_map(images_A)
            mean_B = self.__get_mean_map(images_B)
            mean_AB = self.__get_mean_map(images_A * images_B)
            mean_A2 = self.__get_mean_map(images_A ** 2)
            mean_B2 = self.__get_mean_map(images_B ** 2)

            variance_A = mean_A2 - mean_A ** 2
            variance_B = mean_B2 - mean_B ** 2
            convariance_AB = mean_AB - mean_A * mean_B

            cs_map = (2 * convariance_AB + self.C2) / \
                     (variance_A + variance_B + self.C2)

            self.ms_cs[i] = cs_map.mean()

            if i < self.n_levels - 1:
                images_A = F.avg_pool2d(images_A, kernel_size=2)
                images_B = F.avg_pool2d(images_B, kernel_size=2)

        l_map = (2 * mean_A * mean_B + self.C1) / \
                (mean_A ** 2 + mean_B ** 2 + self.C1)

        ms_ssim = l_map.mean() * np.dot(self.ms_cs, self.scale_weights)

        return 1.0 - ms_ssim


if __name__ == "__main__":
    msssim = MS_SSIM(data_range=1.0)

    src_img = cv2.imread('./images/spike.png')
    # src_img = cv2.resize(src_img, (128, 128), interpolation=cv2.INTER_AREA)
    src_img = np.float32(src_img) / 255.0

    blur_img = cv2.blur(src_img, (5, 5))
    blur_img += np.random.randn(*blur_img.shape) * 0.05

    cv2.imshow('Deteriorated', blur_img)
    cv2.waitKey()

    src_img = torch.from_numpy(src_img.transpose(2, 0, 1)).unsqueeze(0)
    blur_img = torch.from_numpy(blur_img.transpose(2, 0, 1)).unsqueeze(0)

    ms_ssim_loss = MS_SSIM_Loss(L=1.0)
    ms_ssim_l = ms_ssim_loss(src_img, blur_img)

    # msssim = MS_SSIM(data_range=1.0)
    # ms_ssim_g = msssim(src_img, blur_img)

    l1_norm = nn.L1Loss()
    l1_l = l1_norm(src_img, blur_img)

    print(f'MS-SSIM Loss:\n {ms_ssim_l.item():.5f}')
    print(f'L1 Loss:\n {l1_l.item():.5f}')
