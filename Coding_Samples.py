import numpy as np
import torch
from sklearn.utils import check_random_state
from utils import *
from matplotlib import pyplot as plt
import torch.nn as nn
import pickle as pkl

#Set random seeds, devices
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True

#Initialize network
class ConvNet_CIFAR10(nn.Module):
    def __init__(self):
        super(ConvNet_CIFAR10, self).__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block =([nn.Conv2d(in_filters, out_filters, 3, 2, 1), 
                     nn.LeakyReLU(0.2, inplace=True),  
                     nn.Dropout2d(0)])
            if bn: #Batch normalization
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        self.model = nn.Sequential(
            nn.Unflatten(1,(3,32,32)),
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),)
        ds_size = 2
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 300))






    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        feature = self.adv_layer(out)
        return feature
    
def crit(mmd_val, mmd_var, loss='liu'):
    if loss=='liu': #Heuristic implemented in Liu et al. 2016
        mmd_std_temp = torch.sqrt(mmd_var+10**(-8)) #approximating the standard deviation
        return -1 * torch.div(mmd_val, mmd_std_temp)
    elif loss=='sharpe': #Optimizing mean-var, takes into account of scaling automatically
        return mmd_val - 2.0 * mmd_var 

#####Defining kernel statistics#####
def kernelize(Fea, batch_size, S, sigma, sigma0, epsilon, cst):
    '''
    Given the features and parameters, compute the kernel matrix
    '''
    X_fea = Fea[0:batch_size, :] # fetch the sample 1 (features of deep networks)
    Y_fea = Fea[batch_size:, :]
    X_org = S[0:batch_size, :] # fetch the original sample
    Y_org = S[batch_size:, :] 
    L = 1 # generalized Gaussian (if L>1)
    Dxx = Pdist2(X_fea, X_fea) # compute the pairwise distance matrix on features
    Dyy = Pdist2(Y_fea, Y_fea)
    Dxy = Pdist2(X_fea, Y_fea)
    Dxx_org = Pdist2(X_org, X_org) # compute the pairwise distance matrix on data
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)
    #computes smoothed kernel matrix
    Kx = cst*((1-epsilon) * torch.exp(-(Dxx / sigma0) - (Dxx_org / sigma))**L + 
              epsilon * torch.exp(-Dxx_org / sigma))
    Ky = cst*((1-epsilon) * torch.exp(-(Dyy / sigma0) - (Dyy_org / sigma))**L + 
              epsilon * torch.exp(-Dyy_org / sigma))
    Kxy = cst*((1-epsilon) * torch.exp(-(Dxy / sigma0) - (Dxy_org / sigma))**L + 
               epsilon * torch.exp(-Dxy_org / sigma))
    return Kx, Ky, Kxy

def mmd_var(Fea, batch_size, S, sigma, sigma0, epsilon, cst):
    '''
    Computes the mean and variance of the mmd estimator calling back to utils
    '''
    Kx, Ky, Kxy = kernelize(Fea, batch_size, S, sigma, sigma0, epsilon, cst)
    return mmd(Kx, Ky, Kxy, use_1sample_U=True, compute_variance=True)

def mmd_general(X, Y, model_u, n, sigma, sigma0, epsilon, cst, device, dtype):
    '''
    computes mmd in feature_mmd which is quicker than mmd_var without computing variance
    '''
    S = MatConvert(np.concatenate((X, Y), axis=0), device, dtype)
    Fea = model_u(S) #Combine feature models into one
    Kx, Ky, Kxy = kernelize(Fea, n, S, sigma, sigma0, epsilon, cst)
    return mmd(Kx, Ky, Kxy, use_1sample_U=True, compute_variance=False)

################################################
#Following functions are used for loading data, 
#copied from utils into this file for references
################################################
def load_diffusion_cifar_32():
    '''Load the diffusion and cifar10 data'''
    diffusion = np.load("../data/ddpm_generated_images.npy")
    cifar10 = np.load('../data/cifar_images.npy') 
    dataset_ddpm = diffusion.reshape(diffusion.shape[0], -1)
    dataset_cifar = cifar10.reshape(cifar10.shape[0], -1)
    return dataset_ddpm, dataset_cifar[:10000, :], dataset_cifar[10000:, :]

def train_data_t(n):
    '''
    Load training data in tensor format, stored in cuda. 
    n is at most 25000
    '''
    assert n <= 25000, "Data Capacity Exceeded"
    X = train_cifar[np.random.choice(train_cifar.shape[0], n, replace=False), :]
    Y = train_mixed[np.random.choice(train_mixed.shape[0], n, replace=False), :]
    return X, Y

def test_data_t(n):
    '''
    Load testing data in tensor format, stored in cuda.
    n is at most 10000
    '''
    assert n <= 10000, "Data Capacity Exceeded"
    X = test_cifar[np.random.choice(test_cifar.shape[0], n, replace=False), :]
    Y = test_mixed[np.random.choice(test_mixed.shape[0], n, replace=False), :]
    return X, Y
################################################

def train_d(n_size, learning_rate=5e-4, N_epoch=50, print_every=20, batch_size=32, loss='liu', verbose=False): 
    batches = n_size//batch_size
    print("##### Starting N_epoch=%d epochs per data trial #####"%(N_epoch))
    X, Y = train_data_t(n_size)  #Generate training data
    population_data=MatConvert(np.concatenate((X, Y), axis=0), device, dtype)
    batched_data = [(X[i*batch_size : i*batch_size + batch_size], 
                     Y[i*batch_size : i*batch_size + batch_size]) for i in range(batches)]
    batched_data = [MatConvert(np.concatenate((X, Y), axis=0), device, dtype) for (X, Y) in batched_data]
    model_u = ConvNet_CIFAR10().cuda()
    #Kernel parameters, see paper
    epsilonOPT = MatConvert(np.array([-1.0]), device, dtype)
    epsilonOPT.requires_grad = True
    sigmaOPT = MatConvert(np.array([10000.0]), device, dtype)
    sigmaOPT.requires_grad = True
    sigma0OPT = MatConvert(np.array([0.1]), device, dtype)
    sigma0OPT.requires_grad = True
    cst = MatConvert(np.ones((1,)), device, dtype) # set to 1 to meet liu etal objective
    if loss == 'liu':
        cst.requires_grad = False #Send c to fixed c=1
        optimizer_u = torch.optim.Adam(list(model_u.parameters())+[sigmaOPT], lr=learning_rate)
    elif loss == 'sharpe':
        cst.requires_grad = True #Optimize c on the run
        optimizer_u = torch.optim.Adam(list(model_u.parameters())+[sigmaOPT]+[cst], lr=learning_rate)
    for _ in range(N_epoch):
        for ind in range(batches):
                ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
                sigma = sigmaOPT ** 2
                sigma0 = sigma0OPT ** 2
                S = batched_data[ind] #fetch batch data
                modelu_output = model_u(S) 
                TEMP = mmd_var(modelu_output, batch_size, S, sigma, sigma0, ep, cst)
                STAT_u = crit(TEMP[0], TEMP[1], loss=loss) #loss function, also indicating test power
                optimizer_u.zero_grad()
                STAT_u.backward(retain_graph=True)
                optimizer_u.step()
        if _ % print_every==0 and verbose: #print loss on last batch
            print("Epoch %d Loss: "%_ + str(STAT_u))
            print("sigma: ", sigmaOPT ** 2)
            print("sigma0: ", sigma0OPT ** 2)
            print("epsilon: ", torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT)))
            print("c: ", cst)
    sigma = sigmaOPT ** 2
    sigma0 = sigma0OPT ** 2
    ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
    mmd, var=mmd_var(model_u(population_data), n_size, population_data, sigma, sigma0, ep, cst)
    if verbose:
        print("##### Finished training #####")
        print("Kernel Parameters:")
        print("sigma: ", sigma)
        print("sigma0: ", sigma0)
        print("epsilon: ", ep)
        print("c: ", cst)
        print("Population traning loss:")
        print(crit(mmd, var, loss=loss))
    return model_u, sigma, sigma0, ep , cst, X, Y


if __name__ == "__main__":
    dataset_ddpm, dataset_cifar_test, dataset_cifar_train = load_diffusion_cifar_32() 
    #cifar divided for studying covariate shifts
    mix_rate = 2 # for each corrupted data, match with mix_rate*cifar data points
    train_size, test_size = 5000, 2000
    
    # mix the dataset so that the mmd distance between hypotheses is not too large
    train_mixed = np.concatenate((dataset_ddpm[ :train_size, :], 
                                  dataset_cifar_train[ : train_size*mix_rate, :]), axis = 0)
    train_cifar = dataset_cifar_train[train_size*mix_rate : train_size*(2*mix_rate + 1), :]

    test_mixed = np.concatenate((dataset_ddpm[train_size : train_size + test_size, :], 
                                 dataset_cifar_test[: test_size*mix_rate, :]), axis = 0)
    test_cifar = dataset_cifar_test[test_size*mix_rate: test_size*(2*mix_rate + 1), :]
    
    #generate a random shuffle over datasets and convert to tensor
    train_mixed = MatConvert(np.random.shuffle(train_mixed), device, dtype)
    train_cifar = MatConvert(np.random.shuffle(train_cifar), device, dtype)
    test_mixed = MatConvert(np.random.shuffle(test_mixed), device, dtype)
    test_cifar = MatConvert(np.random.shuffle(test_cifar), device, dtype) 




    print("##### Starting LFI testing on kernel #####")
    sets=10 #sets of independent experiments
    iter_cal=100
    iter_lfi=500
    m_list=[96, 128, 160, 192, 216, 240, 256, 320, 384]
    n_size=1920
    results=[]
    for _ in range(sets):
        print("##### Starting set %d of %d #####" %(_+1, sets))
        #kernel training
        model_u, sigma, sigma0, ep, cst, X_train, Y_train=train_d(n_size,
                                                learning_rate=5e-4, N_epoch=80, print_every=20, batch_size=32)
        with torch.no_grad(): #inference phase
            print("Under this trained kernel, we run N = %d iterations of LFI: "%iter_lfi)
            for i in range(len(m_list)):
                R_v = np.zeros(iter_lfi)
                P_v = np.zeros(iter_lfi)
                print("start testing m = %d"%m_list[i])
                m_size = m_list[i]
                for k in range(iter_lfi): 
                    stats=[]
                    null_cal, _ = test_data_t(n_size) #sample calibration set for estimation of p-val using null data
                    for j in range(iter_cal): #probing null data
                        Z_temp = null_cal[np.random.choice(n_size, m_size, replace=False), :]
                        mmd_XZ = mmd_general(X_train, Z_temp, model_u, n_size, sigma, sigma0, ep, cst, device, dtype)[0] 
                        mmd_YZ = mmd_general(Y_train, Z_temp, model_u, n_size, sigma, sigma0, ep, cst, device, dtype)[0]
                        stats.append(float(mmd_XZ - mmd_YZ)) #difference of mmd distances as proxy for test statistics
                    stats = np.sort(stats)
                    thres = stats[int(0.95*iter_cal)] #threshold at 95% quantile of null stats
                    _, mixed_test = test_data_t(m_size)
                    mmd_XZ = mmd_general(X_train, mixed_test, model_u, n_size, sigma, cst, device, dtype)[0] 
                    mmd_YZ = mmd_general(Y_train, mixed_test, model_u, n_size, sigma, cst, device, dtype)[0]
                    R_v[k] = mmd_XZ - mmd_YZ > thres  #rejected or not
                    P_v[k] = np.searchsorted(stats, float(mmd_XZ - mmd_YZ), side="left")/iter_cal #p-value
                print("n, m=",str(n_size)+str('  ')+str(m_size),"--- Rejection rate: ", R_v.mean())
                print("n, m=",str(n_size)+str('  ')+str(m_size),"--- Expected p-val: ", P_v.mean())
                results.append([n_size, m_size, R_v.mean(), P_v.mean()])

    #stores results
    with open('results.pkl', 'wb') as f:
        pkl.dump(results, f)