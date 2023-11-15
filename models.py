import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        #layers.append(nn.ReLU(inplace=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            #layers.append(nn.ReLU(inplace=False))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out

class RCDIP(nn.Module):
    # Input Variable:
        # number_layer : The number of network layers
        # iters: the total iteration of RCDIP
        # lamb: the trade-off parameter in the regularization step
    # Output Variable:
        # cleanUV: the restore clean image
    def __init__(self, number_layer=12, iters=6, lamb=0.9):
        super(RCDIP, self).__init__()
        #self.h = HyperCNN()
        self.iters = iters
        self.lamb = lamb
        self.p = self.make_xnet(self.iters, number_layer)
    def make_xnet(self, iters, number_layer):
        layers = []
        for i in range(iters):
            layers.append(DnCNN(channels = 1, num_of_layers=number_layer))
        return nn.Sequential(*layers)
    def forward(self, noi_mat, initU, initV, Hei, Wid, order):
        us1, r = initU.shape
        vs1, _ = initV.shape
        tmp_uv = (self.lamb*torch.mm(initU, initV.t())+(1-self.lamb)*noi_mat).cuda()
        u, s, vt = torch.linalg.svd(torch.mm(tmp_uv.t(), initU), full_matrices=False)
        noiseU = torch.zeros(us1, r).cuda()
        V = torch.mm(u, vt)
        MatU = torch.mm(tmp_uv, V)
        for ind_x in range(r):
            im_noise = MatU[:, ind_x].reshape(Hei, Wid).reshape(1, 1, Hei, Wid)
            out_train = self.p[order](im_noise)
            noiseU[:, ind_x] = out_train.reshape(Hei * Wid)
        ClnU = MatU - noiseU
        cleanUV = torch.mm(ClnU, V.t())
        return cleanUV, ClnU, V