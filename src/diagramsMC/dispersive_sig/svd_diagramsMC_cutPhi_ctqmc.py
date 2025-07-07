'''
This code is the CTQMC version of the svd_diagramsMC_cutPhi.py code, which overcomes some problems.

Note: This code can only be run in the Perturbed_DMFT/perturbation directory, as:
(mpirun -np 8) python -m diagramsMC.dispersive_sig.svd_diagramsMC_cutPhi
the parallel part in the beginning is recommended but optional.
This code is MC estimation of self-energy diagrams using the trick of svd.
This function can be directly called. and the evaluation of integrand is accelerated.
'''






from scipy import *
from scipy.interpolate import interp1d
from . import svd_weight_lib_cutPhi
from numpy import linalg
from numpy import random
import sys
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import time
from . import diag_def_cutPhifast
# import diag_def_cutPhi
# import diag_def_cutPhibackup
from mpi4py import MPI
sys.path.append('../')
import perturb_lib as lib
import fft_convolution as fft
from ..diagramsMC_lib import *
import diagramsMC.basis as basis
import copy


class params:
    def __init__(self):
        self.Nitt = 5000000   # number of MC steps in a single proc
        self.Ncout = 200000    # how often to print
        self.Nwarm = 1000     # warmup steps
        self.tmeassure = 10   # how often to meassure
        self.V0norm = 4e-2    # starting V0
        self.recomputew = 5e4/self.tmeassure # how often to check if V0 is correct
        self.per_recompute = 7 # how often to recompute fm auxiliary measuring function

@jit(nopython=True)
def geniloop(Ndimk,Ndimtau,Ndimlat):
    r = np.random.rand()
    iloop=0
    if r<0.55:
        iloop=np.random.randint(0,Ndimk)
    elif r<0.8:
        iloop=Ndimk+np.random.randint(0,Ndimtau)
    elif r<0.9:
        iloop=Ndimk+Ndimlat+np.random.randint(0,Ndimtau)
    elif r<0.95:
        iloop=Ndimk+Ndimlat+Ndimtau
    else:
        iloop=Ndimk+Ndimlat+Ndimtau+1
    return iloop


def get_gtau(U,T,nfreq,knum):
    beta=1/T
    mu=U/2
    name1='../data/files_boldc/{}_{}/Sig.out'.format(U,T)
    filename1=readDMFT(name1)
    name2='../data/files_ctqmc/{}_{}/Sig.out'.format(U,T)
    filename2=readDMFT(name2)
    name3='../data/files_boldc/{}_{}/Sig.OCA'.format(U,T)
    filename3=readDMFT(name3)
    # print(filename1)
    # print(filename2)
    if (os.path.exists(filename1)):
        filename=filename1
    elif (os.path.exists(filename2)):
        filename=filename2
        # print('reading DMFT data from {}'.format(filename))
    elif (os.path.exists(filename3)):
        filename=filename3
    else:
        print('{} cannot be found!'.format(filename))  
        return 0  
    
    sigma=np.loadtxt(filename)[:nfreq,:]
    check=sigma[-1,1]
    om=sigma[:,0]
    # anyways real part of sigA will be greater.
    if check>U/2:
        sigA=sigma[:,1]+1j*sigma[:,2]
        sigB=U-sigma[:,1]+1j*sigma[:,2]
    else:
        sigB=sigma[:,1]+1j*sigma[:,2]
        sigA=U-sigma[:,1]+1j*sigma[:,2]
    Sigma11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sigma11+=lib.ext_sig(sigA)[:,None,None,None]
    Sigma22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sigma22+=lib.ext_sig(sigB)[:,None,None,None]
    Sigma12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    z_1=lib.z4D(beta,mu,Sigma11,knum,nfreq)#z-delta
    z_2=lib.z4D(beta,mu,Sigma22,knum,nfreq)#z+delta
    G11_iom,G12_iom=lib.G_iterative(knum,z_1,z_2,Sigma12)
    G22_iom=-G11_iom.conjugate()
    G11_tau=fft.fermion_fft_diagG(knum,G11_iom,beta,sigA,mu)# currently sigma12=0
    G12_tau=fft.fast_ft_fermion(G12_iom,beta)
    G22_tau=G11_tau[::-1] 
    Gloc11_tau=np.sum(G11_tau,axis=(1,2,3))[:,None,None,None]/knum**3*np.ones((knum,knum,knum))[None,:,:,:]
    Gloc22_tau=np.sum(G22_tau,axis=(1,2,3))[:,None,None,None]/knum**3*np.ones((knum,knum,knum))[None,:,:,:]
    # Gloc11=np.sum(G11_tau,axis=(1,2,3))/knum**3
    # Gloc22=np.sum(G22_tau,axis=(1,2,3))/knum**3
    return G11_tau.real,G12_tau.real,G22_tau.real,Gloc11_tau.real,Gloc22_tau.real


def IntegrateByMetropolis_ctqmc(func,qx,p,seed,lmax,imax,iflocal=0):
    '''
    This function is the CTQMC version of the IntegrateByMetropolis_ctqmc.py code, which overcomes some problems.
    Also, this code should work under doped regime.
    '''
    ifprint=0
    #-------basic settings-----
    # time check
    time_trial=0
    time_evaluate=0
    time_accrej=0
    time_others=0
    Nacc,Nrej,Nall=0,0,0
    time_begin=time.time()
    ifrecomp=1
    np.random.seed(seed)# use the given seed
    # random.seed(0)         # make sure that we always get the same sequence of steps. If parallel. they should have different seeds.
    knum=func.knum
    taunum=func.taunum
    taufold=np.arange(func.taunum+2)
    taufold[-1]=func.taunum-1
    taufold[func.taunum]=0
    kfold=np.arange(func.knum+2)
    kfold[-1]=func.knum-1
    kfold[func.knum]=0
    # Pnorm2 = np.zeros_like(qx)  # Final results V_physical is stored in Pval
    
    Pnorm = 0.0            # V_alternative is stored in Pnorm
    Pval_sum = 0.0         # this is widetilde{V_physical}
    Pnorm_sum = 0.0        # this is widetilde{V_alternative}
    V0norm = p.V0norm      # this is V0
    Vphys=0
    dk_hist = 1.0          # we are creating histogram by adding each configuration with weight 1.
    # note: here i have both k and tau as external variable.
    Ndimk = func.Ndimk       # dimensions of the problem
    Ndimtau=func.Ndimtau
    Ndimlat=func.Ndimlat
    Pval=np.zeros_like(qx)
    inc_recompute = (p.per_recompute+0.52)/p.per_recompute # How often to self-consistently recompute
    # the wight functions g_i and h_{ij}.
    kbasisindlist=basis.gen_basisindlist(imax)
    maxbasis=10# reject all attempts of n1+n2+n3+l>10



    momentum=np.random.randint(low=0, high=knum, size=(Ndimk,3))
    # imagtime=np.random.randint(low=0, high=taunum, size=(Ndimtau,1))
    imagtime=np.random.rand(Ndimtau,1)*func.beta# continuous time
    sublatind=np.random.randint(low=1,high=3,size=(Ndimlat))# indices for each time point. 
    l=0#np.random.randint(0,lmax)# generate the external variable
    i_coeff=np.random.randint(0,func.kbasisnum)# this is the index for kspace basis.
    if iflocal:
        i_coeff=0
        sublatind=np.ones_like(sublatind)
    ti_coeff=i_coeff
    tl=l
    tmomentum=copy.deepcopy(momentum)
    timagtime=copy.deepcopy(imagtime)
    tsublatind=copy.deepcopy(sublatind)

    # myweight = svd_weight_lib_cutPhi.meassureWeight(Ndimk, Ndimtau,knum,taunum,Ndimlat)
    # to be updated. add sublatint in the update function.
    fQ = func.update(momentum,imagtime,sublatind,i_coeff,l)#, V0norm * myweight( momentum,imagtime,sublatind-1 ) # fQ=(f(X), V0*f_m(X)) sublatind consists 1 and 2 but we'd better start from 0.
    # print('starting with f=', fQ, '\nstarting momenta=', momentum,'\n starting time=',imagtime)

    Nmeassure = 0  # How many measurements we had?
    Nall_l,Nacc_l,Nall_i,Nacc_i,Nall_ind,Nacc_ind, Nall_k, Nall_t, Nacc_t, Nacc_k = 0, 0, 0, 0, 0,0,0,0,0,0
    c_recompute = 0 # when to recompute the auxiliary function?
    for itt in range(p.Nitt):   # long loop
        time0=time.time()
        # variables: k,tau,sublatind, i_coeff,l_coeff
        # iloop = int( (Ndimk+Ndimtau+Ndimlat+2) * random.rand() )   # which variable to change, iloop=0 changes external r_0
        iloop=geniloop(Ndimk,Ndimtau,Ndimlat)
        accept = False
        if (iloop >= 0) and (iloop < Ndimk):# changing internal variable k
            Nall_k += 1
            (K_new,  trialaccept) = TrialStep1_k(iloop,momentum,knum,kfold)
            
            # if iflocal:
            #     trialaccept=0
        elif (iloop >= Ndimk) and (iloop < Ndimk+Ndimtau):# changing internal variable tau
            # (tau_new, trialaccept)=TrialStep1_tau(iloop,imagtime,taunum,Ndimk,taufold)
            (tau_new, trialaccept)=TrialStep1_tau_ctqmc(iloop,taunum,Ndimk)
            Nall_t+=1
        elif (iloop >= Ndimk+Ndimtau) and (iloop < Ndimk+Ndimtau+Ndimlat):# changing sublatint. does not matter in or external variable.
            sublatind_new=3-sublatind[iloop-Ndimk-Ndimtau]#np.random.randint(2)+1#
            trialaccept=1
            Nall_ind+=1
            if iflocal:
                trialaccept=0
        elif (iloop==Ndimk+Ndimtau+Ndimlat):# changing external variable i
            i_coeffnew=np.random.randint(0,func.kbasisnum)
            Nall_i+=1
            trialaccept=1

            # if np.sum(kbasisindlist[i_coeffnew])*2+l<=maxbasis:
            #     trialaccept=0
        elif (iloop == Ndimk+Ndimtau+Ndimlat+1): # changing external variable l
            # lnew=np.random.randint(0,lmax)
            # trial_ratio=1; trialaccept=1
            (lnew, trialaccept)=Trialstep0_l(lmax)
            Nall_l+=1
            # if np.sum(kbasisindlist[i_coeff])*2+lnew<=maxbasis:
            #     trialaccept=0

        time1=time.time()
        time_trial+=(time1-time0)
        if (trialaccept): # trial step successful. We did not yet accept, just the trial step.
            if (iloop<Ndimk):# k is changed
                tmomentum= Give_new_K(momentum, K_new, iloop)
            elif (iloop<Ndimk+Ndimtau):# tau is changed
                timagtime=Give_new_tau(imagtime, tau_new, iloop,Ndimk)
            elif (iloop<Ndimk+Ndimtau+Ndimlat):
                tsublatind[iloop-Ndimk-Ndimtau]=sublatind_new 
            elif (iloop==Ndimk+Ndimtau+Ndimlat):# i is changed
                ti_coeff=i_coeffnew         
            elif (iloop==Ndimk+Ndimtau+Ndimlat+1):# l is changed
                tl=np.copy(lnew)
 

            time_beforecalc=time.time()

            fQ_new = func.update_temp(iloop,tmomentum,timagtime,tsublatind,ti_coeff,tl)#, V0norm * myweight(tmomentum,timagtime,tsublatind-1) # f_new
            time_aftercalc=time.time()
            time_evaluate+=(time_aftercalc-time_beforecalc)
            # ratio = (abs(fQ_new[0])+fQ_new[1])/(abs(fQ[0])+fQ[1]) 
            ratio=abs(fQ_new)/abs(fQ)
            # print('ratio=',ratio)

            accept = abs(ratio) > 1-random.rand() # Metropolis
            if accept: # the step succeeded
                func.metropolis_accept(iloop)
                if (iloop<Ndimk):
                    momentum[iloop] = K_new
                    Nacc_k += 1
                elif iloop<Ndimk+Ndimtau:
                    imagtime[iloop-Ndimk]=tau_new
                    Nacc_t += 1
                elif iloop<Ndimk+Ndimtau+Ndimlat:
                    sublatind[iloop-Ndimk-Ndimtau]=tsublatind[iloop-Ndimk-Ndimtau]
                    Nacc_ind+=1
                elif (iloop==Ndimk+Ndimtau+Ndimlat):
                    i_coeff=ti_coeff
                    Nacc_i+=1
                elif (iloop==Ndimk+Ndimtau+Ndimlat+1):
                    l=np.copy(tl)
                    Nacc_l+=1
                fQ = fQ_new
                Nacc+=1

            else:
                Nrej+=1
                time0=time.time()
                if (iloop<Ndimk):
                    tmomentum[iloop] = momentum[iloop]
                elif iloop<Ndimk+Ndimtau:
                    timagtime[iloop-Ndimk]=imagtime[iloop-Ndimk]
                elif iloop<Ndimk+Ndimtau+Ndimlat:
                    tsublatind[iloop-Ndimk-Ndimtau]=sublatind[iloop-Ndimk-Ndimtau]
                    # print('accept trialsublatind. new sublatind=',sublatind)
                elif (iloop==Ndimk+Ndimtau+Ndimlat):
                    ti_coeff=i_coeff
                elif (iloop==Ndimk+Ndimtau+Ndimlat+1):
                    tl=l
                    
                func.metropolis_reject(iloop)
                # print('metropolis rejected!\n')
        if (itt >= p.Nwarm and itt % p.tmeassure==0 and trialaccept==1): # below is measuring every p.tmeassure stepsand trialaccept==1
            Nmeassure += 1   # new meassurements
            Pval[l,i_coeff]+=1
            Vphys+=1/fQ

            # myweight.Add_to_K_histogram(dk_hist*Wphs, momentum,imagtime,sublatind-1)
    Pval *=  (2*lmax*func.kbasisnum/Vphys) #  Finally, the integral is I = V0 *V_physical/V_alternative.
    return Pval.real