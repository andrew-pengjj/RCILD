import os
import argparse
from hsi_dataprocess import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
parser = argparse.ArgumentParser(description="RCDIP")
parser.add_argument("--num_of_layers", type=int, default=15, help="Number of total layers")
parser.add_argument("--model", type=str, default='model/gaussian.pth', help="model path")
parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=50, help="When to decay learning rate; should be less than epochs")# 30
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate,default=1e-3") # 1e-3
parser.add_argument("--noise_case", type=str, default="i.i.d-g", help='i.i.d-G, n.i.i.d-g, complex')
parser.add_argument("--std", type=float, default=0.4, help='0.4, 0.2, 0.1, 0.05')
parser.add_argument("--rank", type=int, default=15, help='rank of data')
parser.add_argument("--Ite", type=int, default=6, help='Iteration of unfold network')#20
parser.add_argument("--gpu_id", type=int, default=0, help='Iteration of unfold network')#20
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)

from models import *
from utils import *
from LRDIP import RCDIP_train, RCDIP_test
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device : ', device, ', gpu_id: ', opt.gpu_id)
# save_path
modelsave = opt.model

# initial model
model = RCDIP(number_layer=opt.num_of_layers, iters=opt.Ite).cuda()
model.apply(weights_init_kaiming)
if os.path.exists(modelsave):
    model.load_state_dict(torch.load(modelsave, map_location=device))

print('load model, model name = %s\n'% modelsave)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)


def Batch_Train():
    Basepath = 'training_data/'
    listA = os.listdir(Basepath)
    lenListA = len(listA)
    for dataind in range(lenListA):
        filepath = Basepath + listA[dataind]
        print('========= Load Data: ', filepath, ',  noise_case:', opt.noise_case, ' std: ', opt.std, '============\n')
        cln_hsi, noi_hsi = GetNoise(filepath, opt.noise_case, opt.std)
        # GW
        std_e = get_variance(noi_hsi)
        print(' GW variance :%.4f\n' % np.mean(std_e))
        noi_hsi = GW(noi_hsi, std_e)
        cln_hsi = GW(cln_hsi, std_e)
        RCDIP_train(model, optimizer, noi_hsi, cln_hsi, opt.rank, opt.Ite)
    torch.cuda.empty_cache()
    torch.save(model.state_dict(), modelsave)


if __name__ == "__main__":
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            lr_ = opt.lr
        else:
            lr_ = opt.lr / 10.
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_
        print('========= epoch =%d  ==========\n' % (epoch))
        Batch_Train()
