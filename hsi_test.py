import os
import argparse
from hsi_dataprocess import *
from models import *
from utils import *
from LRDIP import RCDIP_train, RCDIP_test

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--model", type=str, default='model/gaussian.pth', help="model path")
parser.add_argument("--num_of_layers", type=int, default=15, help="Number of total layers")
parser.add_argument("--noise_case", type=str, default="i.i.d-G", help='i.i.d-G, n.i.i.d-G, Complex')
parser.add_argument("--std", type=float, default=0.1, help='0.4, 0.2, 0.1, 0.05')
parser.add_argument("--rank", type=int, default=10, help='rank of data')
parser.add_argument("--Ite", type=int, default=6, help='Iteration of unfold network')#20
parser.add_argument("--gpu_id", type=int, default=0, help='Iteration of unfold network')#20
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# save_path
modelsave = opt.model
datapath = "testing_data/simu_indian.mat"
print(datapath)
print(modelsave)
print('noise_case: ', opt.noise_case, ', std: ', opt.std)
# initial model
model = RCDIP(number_layer=opt.num_of_layers, iters=opt.Ite).cuda()
model.apply(weights_init_kaiming)
if os.path.exists(modelsave):
    model.load_state_dict(torch.load(modelsave, map_location=device))

if __name__ == "__main__":
    cln_hsi, noi_hsi = GetNoise(datapath, opt.noise_case, opt.std)
    # GW
    std_e = get_variance(noi_hsi)
    #std_e = opt.std*np.ones(300)
    print(' GW variance :%.4f\n' % np.mean(std_e))
    noi_hsi = GW(noi_hsi, std_e)
    cleanUV = RCDIP_test(model, noi_hsi, opt.rank, opt.Ite)
    # IGW
    cleanUV = IGW(cleanUV, std_e)
    mpsnr, mssim, avsam1, ergas = msqia(cleanUV, cln_hsi)
    print(mpsnr, mssim)
