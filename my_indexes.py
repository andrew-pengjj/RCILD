import numpy as np
import cv2
import scipy.io as scio
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

# author skx time 23.8.9 note:参考rctv-main的matlab版本中的psnr,ssim,ergas写的相应函数，结果ngmmet-master中的sam写的相应函数，结果在两位小数内一定可以对应

def ssim(img1, img2, K=None, window=None, L=None):
    if img1.shape != img2.shape:
        mssim = -np.inf
        ssim_map = -np.inf
        return mssim, ssim_map
    M, N = img1.shape
    if K is None or len(K) < 2:
        K = [0.01, 0.03]
    if window is None or L is None:
        if M < 11 or N < 11:
            mssim = -np.inf
            ssim_map = -np.inf
            return mssim, ssim_map
        window = cv2.getGaussianKernel(11, 1.5) @ cv2.getGaussianKernel(11, 1.5).T

        L = 255
    if len(K) != 2 or K[0] < 0 or K[1] < 0:
        mssim = -np.inf
        ssim_map = -np.inf
        return mssim, ssim_map
    window /= np.sum(window)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    f = max(1, int(np.round(min(M, N) / 256)))
    if f > 1:
        lpf = np.ones((f, f)) / (f ** 2)
        img1 = convolve2d(img1, lpf, mode='same', boundary='wrap')
        img2 = convolve2d(img2, lpf, mode='same', boundary='wrap')
        img1 = img1[::f, ::f]
        img2 = img2[::f, ::f]
    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2
    mu1 = convolve2d(img1, window, mode='valid')
    mu2 = convolve2d(img2, window, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = convolve2d(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = convolve2d(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = convolve2d(img1*img2, window, mode='valid') - mu1_mu2

    if C1 > 0 and C2 > 0:
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    else:
        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2

        ssim_map = np.ones_like(mu1)
        index = (denominator1 * denominator2) > 0
        ssim_map[index] = (numerator1[index] * numerator2[index]) / (denominator1[index] * denominator2[index])

        index = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[index] = numerator1[index] / denominator1[index]

    mssim = np.mean(ssim_map)
    return mssim, ssim_map


def SAM(T, H):
    eps = 1e-8
    SigmaTR = np.dot(T, H.T) + eps
    SigmaT2 = np.dot(T, T.T) + eps
    SigmaR2 = np.dot(H, H.T) + eps

    SAM_value = np.degrees(np.arccos(SigmaTR / np.sqrt(SigmaT2 * SigmaR2)))

    return SAM_value

def ErrRelGlobAdimSyn(x_true, x_pred):
    assert x_true.ndim == 3 and x_true.shape == x_pred.shape
    m, n, k = x_true.shape
    ergas = 0;
    for i in range(k):
        ergas = ergas + np.mean((x_true[:,:, i] - x_pred[:,:, i]) ** 2) / np.mean( x_true[:,:, i])
    ergas = 100 * np.sqrt(ergas / k)
    return ergas

def msqia(imagery1, imagery2):
    #imagery1是gt
    M, N, p = imagery1.shape

    psnrvector = np.zeros(p)
    for i in range(p):
        J = 255 * imagery1[:, :, i]
        I = 255 * imagery2[:, :, i]
        psnrvector[i] = PSNR(J,I, data_range=np.max(J))
    mpsnr = np.mean(psnrvector)

    SSIMvector = np.zeros(p)
    for i in range(p):
        J = 255 * imagery1[:, :, i]
        I = 255 * imagery2[:, :, i]
        # SSIMvector[i],_ = ssim(J.astype(np.uint8), I.astype(np.uint8))
        SSIMvector[i], _ = ssim(J, I)
    mssim = np.mean(SSIMvector)
    # mssim2 = SSIM(imagery1.astype(np.float32),imagery2.astype(np.float32),channel_axis=-1)

    sum1 = 0.0
    for i in range(M):
        for j in range(N):
            T = imagery1[i, j, :]
            T = T.ravel()
            H = imagery2[i, j, :]
            H = H.ravel()
            sum1 += SAM(T, H)
    avsam1 = sum1 / (M * N)

    ergas = ErrRelGlobAdimSyn(255 * imagery1, 255 * imagery2)

    return mpsnr, mssim, avsam1, ergas


# if __name__ == "__main__":
#     filepath = '../data/' + r'PaviaU_case.mat'  # r'Pavia_80.mat'# r'PaviaU.mat'##
#     print(filepath)
#     mat = scio.loadmat(filepath)
#     cln_hsi = mat["cln"]  # (256, 256, 191) (200.200 93)
#     noi_hsi = mat["noi_case5"]
#     std_e = mat["stde_case5"]
#     mpsnr, mssim, avsam1, ergas = msqia(cln_hsi,noi_hsi)
#     print(mpsnr, mssim, avsam1, ergas)