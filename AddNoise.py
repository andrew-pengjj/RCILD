import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random

def add_gaussian(image, sigma):
    # add gaussian noise
    # image in [0,1], sigma in [0,1]
    output = image.copy()
    output = output + np.random.normal(0, sigma,image.shape)
    # output = output + np.random.randn(image.shape[0], image.shape[1])*sigma
    return output

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

def add_stripe(image, min_value, max_value):
    # add stripe
    output = image.copy()
    number = random.randint(min_value, max_value)
    col = [i for i in range(image.shape[1])]
    loc = random.sample(col, number)
    for i in range(number):
        stripe = np.random.randn(1, 1)*0.5-0.25
        output[:, loc[i]] = image[:, loc[i]] - stripe
    return output

def add_deadline(image, min_value, max_value):
    # add stripe
    output = image.copy()
    number = random.randint(min_value, max_value)
    col = [i for i in range(image.shape[1])]
    loc = random.sample(col, number)
    for i in range(number):
        output[:, loc[i]] = 0.0
    #output[:, loc] = 0
    return output

if __name__ == "__main__":
    OriImage = mpimg.imread('./data/Lena.png')
    plt.imshow(OriImage)
    plt.show()

    I1 = add_sp(OriImage, 0.3)
    plt.imshow(I1)
    plt.show()
    I11 = add_sp(OriImage[:, :, 1], 0.3)
    plt.imshow(I11)
    plt.show()

    I2 = add_stripe(OriImage, 0.15, 0.2)
    plt.imshow(I2)
    plt.show()
    I21 = add_stripe(OriImage[:, :, 1], 0.15, 0.2)
    plt.imshow(I21)
    plt.show()

    I3 = add_deadline(OriImage, 0.15, 0.2)
    plt.imshow(I2)
    plt.show()
    I31 = add_deadline(OriImage[:, :, 1], 0.15, 0.2)
    plt.imshow(I31)
    plt.show()

