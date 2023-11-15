import time
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.SmoothL1Loss().cuda()

def RCDIP_train(model, optimizer, noi_hsi, cln_hsi, opt_rank, opt_iter):
    '''
     Input Variable:
        # model: model structure
        # optimizer: the optimizer of model
        # noi_hsi: noisy data
        # cln_hsi: clean data
        # opt_rank: the number of representative coefficient map
        # opt_iter: iteration number
     Output Variable:
        # res_hsi: restore data
    '''
    Hei, Wid, Band = noi_hsi.shape
    cln_mat = torch.FloatTensor(cln_hsi.reshape(Hei * Wid, Band)).cuda()
    noi_mat = torch.FloatTensor(noi_hsi.reshape(Hei * Wid, Band)).cuda()
    u, sigma, v = torch.linalg.svd(noi_mat, full_matrices=False)
    U = torch.mm(u[:, 0:opt_rank], torch.diag(sigma[0:opt_rank])).cuda()
    V = v[0:opt_rank, :].t().cuda()

    start = time.time()
    for i in range(opt_iter):
        clean_UV, U, V = model(noi_mat, U, V, Hei, Wid, i)
        loss = criterion(clean_UV, cln_mat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('===== iteration = %d, loss=%.4f ==========\n' % (i, loss))
        torch.cuda.empty_cache()
        #torch.save(model.state_dict(), modelsave)
        U = U.detach().data
        V = V.detach().data
    end = time.time()
    gpu_time = end-start
    print('========= Runing Time is %f ==========\n'%(gpu_time))
    return clean_UV.detach().cpu().data.numpy().reshape(Hei, Wid, Band)

def RCDIP_test(model, noi_hsi, opt_rank, opt_iter):
    '''
    Input Variable:
         model: model structure
         noi_hsi: noisy data
         opt_rank: the number of representative coefficient map
         opt_iter: iteration number
    Output Variable:
         res_hsi: restore data
    '''
    Hei, Wid, Band = noi_hsi.shape
    noi_mat = torch.FloatTensor(noi_hsi.reshape(Hei * Wid, Band)).cuda()
    u, sigma, v = torch.linalg.svd(noi_mat, full_matrices=False)
    U = torch.mm(u[:, 0:opt_rank], torch.diag(sigma[0:opt_rank])).cuda()
    V = v[0:opt_rank, :].t().cuda()

    start = time.time()
    with torch.no_grad():
        for i in range(opt_iter):
            clean_UV, U, V = model(noi_mat, U, V, Hei, Wid, i)
            torch.cuda.empty_cache()
            #torch.save(model.state_dict(), modelsave)
            U = U.detach().data
            V = V.detach().data
            clean_UV = clean_UV.detach()
    end = time.time()
    gpu_time = end-start
    print('========= Runing Time is %f ==========\n'%(gpu_time))
    return clean_UV.cpu().data.numpy().reshape(Hei, Wid, Band)