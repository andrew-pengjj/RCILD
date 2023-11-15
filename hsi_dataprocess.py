import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy
from my_indexes import *
# from utils import sta

def sta(img, mode):
    img = np.float32(img)
    if mode == 'all':
        ma = np.max(img)
        mi = np.min(img)
        #   return (img - mi)/(ma - mi)
        img = (img - mi) / (ma - mi)
        return img
    elif mode == 'pb':
        ma = np.max(img, axis=(0, 1))
        mi = np.min(img, axis=(0, 1))
        img = (img - mi) / (ma - mi)
        return img

    else:
        print('Undefined Mode!')
        return img

def add_sp(image,prob):
    h = image.shape[0]
    w = image.shape[1]
    output = image.copy()
    sp = h*w   # 计算图像像素点个数
    NP = int(sp*prob)   # 计算图像椒盐噪声点个数
    for i in range(NP):
        randx = np.random.randint(1, h-1)   # 生成一个 1 至 h-1 之间的随机整数
        randy = np.random.randint(1, w-1)   # 生成一个 1 至 w-1 之间的随机整数
        if np.random.random() <= 0.5:   # np.random.random()生成一个 0 至 1 之间的浮点数
            output[randx, randy] = 0
        else:
            output[randx, randy] = 1
    return output

def add_sp_noise(data_path, std_e):
    data = scipy.io.loadmat(data_path)
    cln_hsi = data['data'].astype(np.float32)
    Hei, Wid, Band = cln_hsi.shape
    noi_hsi = np.zeros([Hei,Wid,Band])
    print('add sparse noise  (%s)' % std_e)
    for ind in range(Band):
        noi_hsi[:, :, ind] = add_sp(cln_hsi[:, :, ind].copy(),std_e)
    return cln_hsi, noi_hsi

def add_gaussian(image, sigma):
    # add gaussian noise
    # image in [0,1], sigma in [0,1]
    output = image.copy()
    output = output + np.random.normal(0, sigma,image.shape)
    # output = output + np.random.randn(image.shape[0], image.shape[1])*sigma
    return output

def add_Gaussian_noise(data_path, std_list):
    data = scipy.io.loadmat(data_path)
    cln_hsi = data['data'].astype(np.float32)
    Hei, Wid, Band = cln_hsi.shape
    noi_hsi = np.zeros([Hei,Wid,Band])
    for ind in range(Band):
        noi_hsi[:, :, ind] = add_gaussian(cln_hsi[:, :, ind].copy(), std_list[ind])
    #cln_hsi = cln_hsi[0:100, 0:100, 0:90]
    #noi_hsi = noi_hsi[0:100, 0:100, 0:90]
    return cln_hsi, noi_hsi

def add_Mixture_noise(data_path, std_g_list, std_s_list):
    data = scipy.io.loadmat(data_path)
    cln_hsi = data['data'].astype(np.float32)
    Hei, Wid, Band = cln_hsi.shape
    cln_hsi = sta(cln_hsi, mode='pb')
    noi_hsi = np.zeros([Hei,Wid,Band])
    # print('add Gaussian noise  (%s)' % std_g)
    # print('add Sparse noise  (%s)' % std_s)
    for ind in range(Band):
        noi_hsi[:, :, ind] = add_sp(cln_hsi[:, :, ind].copy(), std_s_list[ind])
        noi_hsi[:, :, ind] = add_gaussian(noi_hsi[:, :, ind].copy(), std_g_list[ind])
    return cln_hsi, noi_hsi

def get_variance(InputT):
    Hei, Wid, Band = InputT.shape
    InputM = InputT.reshape(Hei*Wid, Band)
    InputM = InputM[5000:6000, :]
    listA = [i for i in range(Band)]
    std_e = np.zeros(Band)
    for ind in range(Band):
        x = InputM[:, np.delete(listA, ind)]
        y = InputM[:, [ind]]
        res = y - np.dot(x, np.dot(np.linalg.inv(np.dot(x.T,x)+np.eye(Band-1)*0.001),np.dot(x.T,y)))
        std_e[ind] = np.std(res[:,0])
    return std_e

def GW(InputT, std_e):
    Hei, Wid, Band = InputT.shape
    NorT = np.zeros([Hei, Wid, Band])
    for ind in range(Band):
        NorT[:, :, ind] = InputT[:, :, ind]/std_e[ind]
    return NorT

def IGW(InputT, std_e):
    Hei, Wid, Band = InputT.shape
    NorT = np.zeros([Hei, Wid, Band])
    for ind in range(Band):
        NorT[:, :, ind] = InputT[:, :, ind]*std_e[ind]
    return NorT

def GetNoise(datapath,noise_case,std):
    """
    """
    if noise_case == 'complex':
        print('complex')
        std_g_list = np.random.uniform(low=0.0, high=std, size=300)
        std_s_list = np.random.uniform(low=0.0, high=0.1, size=300)
        [cln_hsi, noi_hsi] = add_Mixture_noise(datapath, std_g_list, std_s_list)
        return cln_hsi, noi_hsi
    elif noise_case == 'n.i.i.d-g':
        print('n.i.i.d-g')
        std_g_list = np.random.uniform(low=0.0, high=std, size=300)
        [cln_hsi, noi_hsi] = add_Gaussian_noise(datapath, std_g_list)
        return cln_hsi, noi_hsi
    else:
        print('i.i.d-g')
        std_g_list = std*np.ones(300)
        [cln_hsi, noi_hsi] = add_Gaussian_noise(datapath, std_g_list)
        return cln_hsi, noi_hsi

if __name__ == "__main__":
    datapath = "training_data/cd_ms.mat"
    std_g_list = np.random.uniform(low=0.0, high=0.4, size=300)
    std_s_list = np.random.uniform(low=0.0, high=0.2, size=300)
    #[cln_hsi, noi_hsi] = add_Mixture_noise(datapath, std_g_list, std_s_list)
    [cln_hsi, noi_hsi] = add_Gaussian_noise(datapath, std_g_list)
    Hei, Wid, Band = noi_hsi.shape
    noi_mat = noi_hsi.reshape(Hei * Wid, Band)
    std_e = get_variance(noi_mat[1501:2500,:])
    print(np.mean(std_e), np.mean(std_g_list))
    [mpsnr, mssim, avsam1, ergas] = msqia(cln_hsi, noi_hsi)
    band = random.randint(0, 31)
    print(band)
    print(mpsnr, mssim, avsam1, ergas)
    plt.subplot(1, 2, 1)
    plt.imshow(cln_hsi[:, :, band])
    plt.subplot(1, 2, 2)
    plt.imshow(noi_hsi[:, :, band])
    noi_hsi[:, :, band]
    plt.show()
