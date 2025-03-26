import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess,math
import time
sys.path.append('../python_src/')
from mpi4py import MPI
from perturb_lib import *
import perturb_imp as imp
import fft_convolution as fft
# import pert_DMFT_energy as energy
import perm_def
from scipy.interpolate import interp1d
import diagrams
# import mpi_module
import serial_module
import copy
import logging
import XAF
sys.path.append('./diagramsMC/dispersive_sig')
import diagramsMC.basis as basis
import diagramsMC.dispersive_sig.svd_diagramsMC_cutPhi as svd_diagramsMC_cutPhi
import diagramsMC.dispersive_sig.diag_def_cutPhifast as diag_def_cutPhifast
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
"""
# Yueyi Wang. Nov 2024
This file is modified from Pert_DMFT_PM.py
which aims to calculate the xi_AFM by adding a staggered magnetic field to the action/hamiltonian.
At the limit of B->0, there are a few ideas to solve the hamiltonian under B field:
1. add a B field in the Counter Term. Then do the perturation as before. Use original DMFT Sigma^PM.
2. add a B field in the modified propagator. Then do the perturation as before. Use original DMFT Sigma^PM.
3. add a B field in the modified propagator. Use self-energy from DMFT UNDER THE B FIELD. 
    This is the most complicated one, should be the last resort. However I don't think a tiny B field will change the Sigma^PM_imp... 
    so use Sigma^PM_imp from DMFT without the external B should be fine, at leats in the linear response limit?

This code is designed to search critical behavior of magnetization. This requires A LOT of MC steps for order 4.
"""
class params:
    '''
    milnum controls the steps of MC. unit is 1 million.
    '''
    def __init__(self,milnum=5):
        self.Nitt = 1000000*milnum   # number of MC steps in a single proc
        self.Ncout = 1000000    # how often to print
        self.Nwarm = 1000     # warmup steps
        self.tmeassure = 10   # how often to meassure
        self.V0norm = 4e-2    # starting V0
        self.recomputew = 5e4/self.tmeassure # how often to check if V0 is correct
        self.per_recompute = 7 # how often to recompute fm auxiliary measuring function

def readDMFT(dir): # read DMFT Sigma and G.
    indexlist=np.arange(200)
    filefound=0
    if (os.path.exists(dir)):
        filename=dir
        filefound=1
    else:
        for i in indexlist[::-1]:
            filename=dir+'.{}'.format(i)
            # print(filename)
            if (os.path.exists(filename)):
                filefound=1
                # print('file found:',filename)
                break
            if i<10:
                # print('warning: only {} DMFT iterations. result might not be accurate!'.format(i))
                break
        

    # if filefound==0:
        # print('{} cannot be found!'.format(filename))  
    # else:
    #     print('reading DMFT data from {}'.format(filename))
    # sigma=np.loadtxt(filename)[:nfreq,:]
    # sigA=sigma[:,1]+1j*sigma[:,2]
    # sigB=sigma[:,3]+1j*sigma[:,4]
    return filename# this also works for G!

def diff_sigma(sigma11,newsigma11,sigma22,newsigma22):
    res=np.sum(np.abs(sigma11-newsigma11)+np.abs(sigma22-newsigma22))/knum**3
    return res

def read_sigimp(U,T):
    filename='./Sigma_imp/coeff_{}_{}.txt'.format(U,T)
    outarray=np.loadtxt(filename,dtype=float)
    # print('shape of outarray',np.shape(outarray))
    filenameu='./Sigma_imp/taubasis.txt'
    ut=np.loadtxt(filenameu).T
    beta=1/T
    lmax=int(outarray[0,0])
    taunum=int(outarray[1,0])
    nfreq=int(outarray[2,0])
    sigimp_1=outarray[3,0]
    cl2=outarray[:lmax,1]
    cl31=outarray[:lmax,2]
    cl32=outarray[:lmax,3]
    cl41=outarray[:lmax,4]
    cl42=outarray[:lmax,5]
    cl43=outarray[:lmax,6]
    cl44=outarray[:lmax,7]
    cl45=outarray[:lmax,8]
    Sigmaimp2=basis.restore_Gf(cl2,ut)
    Sigmaimp31=basis.restore_Gf(cl31,ut)
    Sigmaimp32=basis.restore_Gf(cl32,ut)
    Sigmaimp41=basis.restore_Gf(cl41,ut)
    Sigmaimp42=basis.restore_Gf(cl42,ut)
    Sigmaimp43=basis.restore_Gf(cl43,ut)
    Sigmaimp44=basis.restore_Gf(cl44,ut)
    Sigmaimp45=basis.restore_Gf(cl45,ut)
    # if rank==0:
    #     plt.plot(Sigmaimp44,label='sigimp44')
    #     plt.legend()
    #     plt.show()



    taulist=(np.arange(taunum+1))/taunum*beta
    ori_grid=(np.arange(nfreq*2)+0.5)/(nfreq*2)*beta
    #note: linear interpolation will generate spikes in momentum space. make sure at least use quadratic.
    ab_pts=int(0.01*taunum)
    interpolator_2 = interp1d(taulist, Sigmaimp2, kind='cubic', fill_value='extrapolate')
    interpolator_31 = interp1d(taulist, Sigmaimp31, kind='cubic', fill_value='extrapolate')
    interpolator_32 = interp1d(taulist, Sigmaimp32, kind='cubic', fill_value='extrapolate')
    interpolator_41 = interp1d(taulist, Sigmaimp41, kind='cubic', fill_value='extrapolate')#[1:taunum-1][1:taunum-1]
    interpolator_42 = interp1d(taulist, Sigmaimp42, kind='cubic', fill_value='extrapolate')#[1:taunum-1][1:taunum-1]
    interpolator_43 = interp1d(taulist[ab_pts:-ab_pts], Sigmaimp43[ab_pts:-ab_pts], kind='cubic', fill_value='extrapolate')#[1:taunum-1][1:taunum-1]
    interpolator_44 = interp1d(taulist[ab_pts:-ab_pts], Sigmaimp44[ab_pts:-ab_pts], kind='cubic', fill_value='extrapolate')#[1:taunum-1][1:taunum-1]
    interpolator_45 = interp1d(taulist, Sigmaimp45, kind='cubic', fill_value='extrapolate')#[1:taunum-1][1:taunum-1]
    Sigmaimptau2_11=interpolator_2(ori_grid)
    Sigmaimptau31_11=interpolator_31(ori_grid)
    Sigmaimptau32_11=interpolator_32(ori_grid)
    Sigmaimptau41_11=interpolator_41(ori_grid)
    Sigmaimptau42_11=interpolator_42(ori_grid)
    Sigmaimptau43_11=interpolator_43(ori_grid)
    Sigmaimptau44_11=interpolator_44(ori_grid)
    Sigmaimptau45_11=interpolator_45(ori_grid)



    Sigmaimpiom2_11=fft.fermion_ifft(Sigmaimptau2_11,beta)
    Sigmaimpiom31_11=fft.fermion_ifft(Sigmaimptau31_11,beta)
    Sigmaimpiom32_11=fft.fermion_ifft(Sigmaimptau32_11,beta)
    Sigmaimpiom41_11=fft.fermion_ifft(Sigmaimptau41_11,beta)
    Sigmaimpiom42_11=fft.fermion_ifft(Sigmaimptau42_11,beta)
    Sigmaimpiom43_11=fft.fermion_ifft(Sigmaimptau43_11,beta)
    Sigmaimpiom44_11=fft.fermion_ifft(Sigmaimptau44_11,beta)
    Sigmaimpiom45_11=fft.fermion_ifft(Sigmaimptau45_11,beta)
    return ut,lmax,taunum,nfreq,sigimp_1,Sigmaimpiom2_11,Sigmaimpiom31_11,Sigmaimpiom32_11,Sigmaimpiom41_11,Sigmaimpiom42_11,Sigmaimpiom43_11,Sigmaimpiom44_11,Sigmaimpiom45_11,Sigmaimp43,Sigmaimp44

def read_complex_numbers(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    complex_numbers = []
    for line in lines:
        line = line.strip()
        
        complex_number = complex(line)
        complex_numbers.append(complex_number)

    complex_array = np.array(complex_numbers)
    
    return complex_array


def get_mag(U,T,order,alpha,foldernum=1,ifchecksvd=0):
    knum=10
    beta=1/T
    mu=U/2
    backup=''
    # backup='backup3/'
    sigmafilename11='./Sigma_disp{}/{}{}_{}/{}_{}_{}_{}_11.dat'.format(foldernum,backup,U,T,U,T,order,alpha)
    sigmafilename11const='./Sigma_disp{}/{}{}_{}/{}_{}_{}_{}_11const.dat'.format(foldernum,backup,U,T,U,T,order,alpha)
    sigmafilename12='./Sigma_disp{}/{}{}_{}/{}_{}_{}_{}_12.dat'.format(foldernum,backup,U,T,U,T,order,alpha)
    filenameu='./Sigma_imp/taubasis.txt'
       
    ut=np.loadtxt(filenameu).T 
    taunum=np.shape(ut)[1]-1
    nfreq=500
    imax=4
    #if rank==0:
     #   print('beginning getmag')
    if (os.path.exists(sigmafilename11)==0):
        if rank==0:
            print('{} not found!'.format(sigmafilename11))
        return -2
    #if rank==0:
     #   print('file does exist!')
    if order>0:
        cli_11=np.loadtxt(sigmafilename11)
        cli_12=np.loadtxt(sigmafilename12)
        ci_11=np.loadtxt(sigmafilename11const)# for const
      #  if rank==0:
       #     print('file loaded!')
        basisnum=basis.gen_basisnum(imax)
        #kbasis=np.empty((2,basisnum,knum,knum,knum),dtype=float)
        #if rank==0:
        kbasis=basis.gen_kbasis_new(imax,knum)
        #kbasis = np.ascontiguousarray(kbasis)
        #comm.Bcast(kbasis, root=0)    
        #if rank==0:
         #   print('kbasis finished!')
        taulist=(np.arange(taunum+1))/taunum*beta
        ori_grid=(np.arange(nfreq*2)+0.5)/(nfreq*2)*beta
        Sig11tk_raw=basis.restore_tk(cli_11,ut,kbasis[0])
        Sig12tk_raw=basis.restore_tk(cli_12,ut,kbasis[1])
        interpolator_11 = interp1d(taulist, Sig11tk_raw, kind='cubic', axis=0, fill_value='extrapolate')
        interpolator_12 = interp1d(taulist, Sig12tk_raw, kind='cubic',  axis=0,fill_value='extrapolate')
        Sig11tk_full=interpolator_11(ori_grid)
        Sig12tk_full=interpolator_12(ori_grid)
        sig11const=basis.restore_k(ci_11,kbasis[0])
        Sigiom_11=fft.fast_ift_fermion(Sig11tk_full,beta)+sig11const[None,:,:,:]
        Sigiom_12=fft.fast_ift_fermion(Sig12tk_full,beta)
        #if rank==0:
         #   print('sigma generated!')

    elif order==0:# if order==0, the self-energy is not saved using the basis, since it is local.
        Sigiom_11=read_complex_numbers(sigmafilename11)[:,None,None,None]*np.ones((2*nfreq,knum,knum,knum))
        # Sigiom_11=np.loadtxt(sigmafilename11)[:,None,None,None]*np.zeros((2*nfreq,knum,knum,knum))
        Sigiom_12=np.zeros((2*nfreq,knum,knum,knum))
    Sigiom_22=U-Sigiom_11.conjugate()
    znew_1=z4D(beta,mu,Sigiom_11,knum,nfreq)
    znew_2=z4D(beta,mu,Sigiom_22,knum,nfreq)
    Gdress11_iom,Gdress12_iom=G_iterative(knum,znew_1,znew_2,Sigiom_12)
    Gdress22_iom=-Gdress11_iom.conjugate()
    nnewloc11=particlenumber4D(Gdress11_iom,beta)
    nnewloc22=particlenumber4D(Gdress22_iom,beta)
    mag=nnewloc22-nnewloc11
    #if rank==0:
     #   print('getmag finished')
    return mag#Sigiom_11,Sigiom_12,Sigiom_22,Gdress11_iom,Gdress12_iom,Gdress22_iom

def GetHighFrequency4D(CC,om):
    " Approximates CC ~  A/(i*om-C) "
    A = 1./(1/(CC[-1,:,:,:]*om[-1])).imag
    C = -A*(1./CC[-1,:,:,:]).real# before taking real part this C is purely imaginary.
    return (A, C)


def perturbation(om,SigDMFT1,SigDMFT2,U,T,B,nfreq,order,alpha,foldernum='B'):
    '''
    the main function doing iterative pertubation. 
    Input:
    SigDMFT1/2: input DMFT self energy.
    U,T: Hubbard U and temperature.
    order: max order taken into account in perturbation. 
    max order number supported:4
    maxit: max number of DMFT self consistant perturbation.
            1: single shot perturbation. 'basic'
            big number: 
    alpha: the factor which suppresses the AFM splitting
    '''
    # Here we assume there is no iterative perturbation.
    time0=time.time()
    mu=U/2
    beta=1/T
    knum=10
    lmax=1# init. later will read.
    taunum=1# init. later will read.
    nnewloc11=0
    nnewloc22=0
    Sigma11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sigma22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sigma12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    # Sigmod11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    # Sigmod12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    # Sigmod22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    G0_11_iom=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    G0_12_iom=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    G0_22_iom=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    G0_11_tau=np.zeros((2*nfreq,knum,knum,knum),dtype=float)
    G0_12_tau=np.zeros((2*nfreq,knum,knum,knum),dtype=float)
    G0_22_tau=np.zeros((2*nfreq,knum,knum,knum),dtype=float)
    # if rank ==0:
        # print("\t-----DMFT perturbation ----")
    Sigma11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sigma11+=ext_sig(SigDMFT1)[:,None,None,None]
    Sigma22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sigma22+=ext_sig(SigDMFT2)[:,None,None,None]
        # ori_Sigma11 = ext_sig(SigDMFT1)
        # ori_Sigma22=ext_sig(SigDMFT2)
        

    # this part does not change over iterations.the diagrams can be calculated before iteration.
    # DMFT GF, impurity self-energy: (which is prepared not in multi-process way)
    
#--------------Generating Gimp used in counter terms---------------
    iom=np.concatenate((om[::-1],om))*1j
    z_1=z4D(beta,mu,Sigma11,knum,nfreq)-B#z-delta
    z_2=z4D(beta,mu,Sigma22,knum,nfreq)+B#z+delta
    G11_iom,G12_iom=G_iterative(knum,z_1,z_2,Sigma12)
    G22_iom=-G11_iom.conjugate()
    G11_tau=fft.fermion_fft_diagG(knum,G11_iom,beta,SigDMFT1,mu)# currently sigma12=0
    G12_tau=fft.fast_ft_fermion(G12_iom,beta)
    G12_tau = np.ascontiguousarray(G12_tau)
    G22_tau=G11_tau[::-1] 
    G22_tau = np.ascontiguousarray(G22_tau)
    G11imp_iom=np.sum(G11_iom,axis=(1,2,3))/knum**3 # impurity GF=sum_k DMFT GF
    G22imp_iom=np.sum(G22_iom,axis=(1,2,3))/knum**3 # impurity GF=sum_k DMFT GF
    G11imp_tau=np.sum(G11_tau,axis=(1,2,3))/knum**3
    G22imp_tau=G11imp_tau[::-1] 
#-------------Generating G0=(iom+mu-epsk-Sig_PM-alpha*Sig_AFM)^-1---------

    Sig_PM=1j*Sigma11.imag+mu
    Sig_AFM=Sigma11-Sig_PM
    Sigmod11=Sig_PM+alpha*U/2*(1-ifsigAFM)+alpha*Sig_AFM*ifsigAFM# idea1: add B field here in the starting point of the perturbation.
    Sigmod22=Sig_PM-alpha*U/2*(1-ifsigAFM)-alpha*Sig_AFM*ifsigAFM
    Sigmod12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    #Note: this def means +B and +alpha polarize the system in the same way. 
    # in z we have zA=iom+mu-sigmaPM-alpha*U-B.  zB=iom+mu-sigmaPM+alpha*U+B.
    zmod_1=z4D(beta,mu,Sigmod11,knum,nfreq)-B#z-delta
    zmod_2=z4D(beta,mu,Sigmod22,knum,nfreq)+B#z+delta
    G0_11_iom,G0_12_iom=G_iterative(knum,zmod_1,zmod_2,Sigmod12)# currently sigma12=0
    G0_22_iom=-G0_11_iom.conjugate()
    G0_11_tau=fft.fermion_fft_diagG(knum,G0_11_iom,beta,Sigmod11[:,0,0,0]+B,mu).real
    G0_12_tau=fft.fast_ft_fermion(G0_12_iom,beta).real
    G0_11_tau = np.ascontiguousarray(G0_11_tau)
    G0_12_tau = np.ascontiguousarray(G0_12_tau)
    G0_22_tau=G0_11_tau[::-1].real
    G0_22_tau = np.ascontiguousarray(G0_22_tau)
    n0loc11=particlenumber4D(G0_11_iom,beta)
    n0loc22=particlenumber4D(G0_22_iom,beta)
    # G0_11imp_iom=np.sum(G0_11_iom,axis=(1,2,3))/knum**3 # impurity GF=sum_k DMFT GF
    # G0_22imp_iom=np.sum(G0_22_iom,axis=(1,2,3))/knum**3 # impurity GF=sum_k DMFT GF
    # G0_11imp_tau=np.sum(G0_11_tau,axis=(1,2,3))/knum**3
    # G0_22imp_tau=G0_11imp_tau[::-1]

#-----------counter term skeleton diagrams, IN TERMS OF IMP PROPAGATOR!!!-------------
    # instead of reading data, we just calculate them for the first 3 orders.     
    sigimp_1_11=(particlenumber1D(G22imp_iom,beta))*U# this is the original approach.
    sigimp_1_22=(particlenumber1D(G11imp_iom,beta))*U               
    sigimpPM_1=(sigimp_1_11+sigimp_1_22)/2
    sigimpAFM_1=(sigimp_1_11-sigimp_1_22)/2        
    sigimpmod_1_11=sigimpPM_1+alpha*U/2*(1-ifsigAFM)+alpha*sigimpAFM_1*ifsigAFM
    sigimpmod_1_22=sigimpPM_1-alpha*U/2*(1-ifsigAFM)-alpha*sigimpAFM_1*ifsigAFM
    #2nd
    # sigimp_2_11,sigimp_2_22=imp.pertimp_func(G11imp_iom,delta_inf,beta,U,knum,2)# 2nd order diagram in Sigma_DMFT
    sigimp_2=imp.pertimp_func_tau(G11imp_tau,beta,U,knum,2)
    sigimp_2_11=fft.fermion_ifft(sigimp_2,beta)
    # sigimp_2_11=Sigmaimpiom2_11
    sigimp_2_22=-sigimp_2_11.conjugate()
    sigimpPM_2=(sigimp_2_11+sigimp_2_22)/2
    sigimpAFM_2=(sigimp_2_11-sigimp_2_22)/2
    sigimpmod_2_11=sigimpPM_2+alpha*sigimpAFM_2*ifsigAFM
    sigimpmod_2_22=sigimpPM_2-alpha*sigimpAFM_2*ifsigAFM
    #3rd
    # sigimp_3_11=Sigmaimpiom3_11
    # sigimp_3_22=-Sigmaimpiom3_11.conjugate()
    # sigimpcheck_3_1_11,sigimpcheck_3_2_11=imp.pertimp_func(G11imp_iom,delta_inf,beta,U,knum,3)# 3rd order diagram in Sigma_DMFT    
    sigimp_31,sigimp_32=imp.pertimp_func_tau(G11imp_tau,beta,U,knum,3)
    sigimp_3_11=fft.fermion_ifft(sigimp_31+sigimp_32,beta)
    sigimp_3_22=-sigimp_3_11.conjugate()
    sigimpPM_3=(sigimp_3_11+sigimp_3_22)/2
    sigimpAFM_3=(sigimp_3_11-sigimp_3_22)/2
    sigimpmod_3_11=sigimpPM_3+alpha*sigimpAFM_3*ifsigAFM
    sigimpmod_3_22=sigimpPM_3-alpha*sigimpAFM_3*ifsigAFM    
    #4th
    # sigimp_4_11=Sigmaimpiom4_11
    # sigimp_4_22=-Sigmaimpiom4_11.conjugate()
    # sigimpPM_4=(sigimp_4_11+sigimp_4_22)/2
    # sigimpAFM_4=(sigimp_4_11-sigimp_4_22)/2
    # sigimpmod_4_11=sigimpPM_4+alpha*sigimpAFM_4*ifsigAFM
    # sigimpmod_4_22=sigimpPM_4-alpha*sigimpAFM_4*ifsigAFM   
        # test the sigmaimp read from saved files
    # if rank ==0: 
    #     print('reading sigimp finished! time={}s'.format(time.time()-time0))

        # for testing only. they behaves very nice.
        # print(sigimp_1_11,sigimp_1)
        # plt.plot(sigimp_2_11,label='BF2')
        # plt.plot(Sigmaimpiom2_11,label='restored2')
        # plt.legend()
        # plt.show()

        # plt.plot(sigimp_3_11,label='BF3')
        # plt.plot(Sigmaimpiom3_11,label='restored3')
        # plt.legend()
        # plt.show()

        # plt.plot(Sigmaimpiom4_11,label='restored4')
        # plt.legend()
        # plt.show()
    comm.Barrier()



    # if rank==0:
    #     print('rank{}: lmax before bcast:{}'.format(rank,lmax))
    #     print('rank{}: taunum before bcast:{}'.format(rank,taunum))
    # lmax = comm.bcast(lmax, root=0)
    # taunum = comm.bcast(taunum, root=0)
    # if rank==0:
    #     print('rank{}: lmax after bcast:{}'.format(rank,lmax))
    #     print('rank{}: taunum after bcast:{}'.format(rank,taunum))    
    # # print('rank={},lmax={}'.format(rank,lmax))
    # ut=np.zeros((lmax,taunum+1))
    # if rank==0:
    #     ut=ut_temp
    # print('rank{}: ut before bcast:{}  {}'.format(rank,ut.dtype,ut.shape))
    # comm.Bcast(ut, root=0)
    # print('rank{}: ut after bcast:{}   {}'.format(rank,ut.dtype,ut.shape))



    # comm.Bcast(G0_11_iom, root=0)
    # comm.Bcast(G0_12_iom, root=0)
    # comm.Bcast(G0_22_iom, root=0)
    # comm.Bcast(G0_11_tau, root=0)
    # comm.Bcast(G0_12_tau, root=0)
    # comm.Bcast(G0_22_tau, root=0)



    # ---------------------Perturbation-------------------
    sig_corr_11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    sig_corr_12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    sig_corr_22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    # if rank==0:
    #     for kx in np.arange(knum):
    #         for ky in np.arange(knum):
    #             for kz in np.arange(knum):
    #                 plt.plot(G0_11_tau[:,kx,ky,kz],label='dispersive')
    #                 # plt.plot(Sigmaimpiom44_11,label='imp')
    #                 plt.legend()
    #                 plt.show()
    imax=4
    kbasisnum=basis.gen_basisnum(imax)
    kbasis=np.empty((2,kbasisnum,knum,knum,knum),dtype=float)
    if rank==0:
        kbasis=basis.gen_kbasis_new(imax,knum)
    kbasis = np.ascontiguousarray(kbasis)
    comm.Bcast(kbasis, root=0)    
    if order>=1:
        # This G_dressed is for the iterated perturbation. But in the 1st iteration is should be G_0.
        z_1=z4D(beta,mu,Sigmod11+sig_corr_11,knum,nfreq)-B
        z_2=z4D(beta,mu,Sigmod22+sig_corr_22,knum,nfreq)+B
        Gdress11_iom,Gdress12_iom=G_iterative(knum,z_1,z_2,sig_corr_12)
        Gdress22_iom=-Gdress11_iom.conjugate()
        # nloc11=np.sum(Gdress11_iom).real/knum**3/beta+1/2
        # nloc22=np.sum(Gdress22_iom).real/knum**3/beta+1/2
        nloc11=particlenumber4D(Gdress11_iom,beta)
        nloc22=particlenumber4D(Gdress22_iom,beta)
        sig_corr1_11=(nloc22*U-sigimpmod_1_11)*np.ones((2*nfreq,knum,knum,knum),dtype=complex)
        sig_corr1_22=(nloc11*U-sigimpmod_1_22)*np.ones((2*nfreq,knum,knum,knum),dtype=complex)
        sig_corr_11=copy.deepcopy(sig_corr1_11)
        sig_corr_22=copy.deepcopy(sig_corr1_22)
        # print('sig_corr1_11:',nloc22*U-sigimpmod_1_11)
        sig_corr_12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        # if rank==0:
        #     print('1st order done! time={}s'.format(time.time()-time0))
    if order>=2:# second order correction to self energy
        sig_corr2_11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        sig_corr2_22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        sig_corr2_12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        #skeleton corrections: seems better to do serial then broadcast..
        if rank==0:
            P22_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G0_22_tau,G0_22_tau,0)
            P12_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12,G0_12_tau,G0_12_tau,1)

            Sig2_11,Sig2_12=diagrams.sig2(G0_11_tau,G0_12_tau,G0_22_tau,P22_tau,P12_tau,knum,nfreq,U,beta)
            Sig2_22=-Sig2_11.conjugate()
            # print('skeleton')
            time21=time.time()
            # non-skeleton diagrams(if not doing iterative perturbation) Since we use sig_corr in the first order we have to do non-skeletons first
            # for second order, it is everything from 1st order inserted in a hartree(which is the only diagram at 1st order)
            sigext2_11=diagrams.sig2_nonskeleton(G0_22_iom,G0_12_iom,sig_corr1_11,sig_corr1_22,knum,nfreq,U,beta)
            sigext2_22=-sigext2_11
            # print('nonskeleton')
            sig_corr2_11+=(Sig2_11-sigimpmod_2_11[:,None,None,None]+sigext2_11)
            sig_corr2_22+=(Sig2_22-sigimpmod_2_22[:,None,None,None]+sigext2_22)
            sig_corr2_12+=Sig2_12
        # print('bcast')
        comm.Bcast(sig_corr2_11, root=0)
        comm.Bcast(sig_corr2_22, root=0)
        comm.Bcast(sig_corr2_12, root=0)            
        sig_corr_11+=sig_corr2_11
        sig_corr_22+=sig_corr2_22
        sig_corr_12+=sig_corr2_12
        time22=time.time()
        # if rank==0:
        #     # print('2nd order time1={}s  time2={}s'.format(time21-time20,time22-time21))
        #     print('2nd ordertime={}s'.format(time.time()-time0))
    if order>=3: # 3rd order correction
        time30=time.time()
        sig_corr3_11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        sig_corr3_22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        sig_corr3_12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
        if rank==0:
            Q11_tau=serial_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G0_22_tau,G0_11_tau,0)#Q=G_{s',-k}(tau)*G_{s,k+q}(tau)
            Q12_tau=serial_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,12, G0_12_tau,G0_12_tau,1)# Note: G12_-k=-G12_k!
            Q22_tau=serial_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G0_11_tau,G0_22_tau,0)
            Q11_iom=fft.fast_ift_boson(Q11_tau,beta)
            Q22_iom=fft.fast_ift_boson(Q22_tau,beta)
            Q12_iom=fft.fast_ift_boson(Q12_tau,beta)
            R11_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G0_22_tau,G0_11_tau,0)#R=G_{s',k}(-tau)*G_{s,k+q}(tau)
            R12_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12, G0_12_tau,G0_12_tau,1)
            R22_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G0_11_tau,G0_22_tau,0)
            R11_iom=fft.fast_ift_boson(R11_tau,beta)
            R22_iom=fft.fast_ift_boson(R22_tau,beta)
            R12_iom=fft.fast_ift_boson(R12_tau,beta)
            Sig3_1_11,Sig3_1_12,Sig3_2_11,Sig3_2_12=diagrams.sig3(G0_11_iom,G0_12_iom,G0_11_tau,G0_12_tau,G0_22_tau,Q11_iom,Q12_iom,Q22_iom,R11_iom,R12_iom,R22_iom,knum,nfreq,U,beta)
            Sig3_11=Sig3_1_11+Sig3_2_11
            Sig3_12=Sig3_1_12+Sig3_2_12
            Sig3_22=-Sig3_11.conjugate()
            sigext3a_11=diagrams.sig3_nonskeleton_A(G0_22_iom,G0_12_iom,sig_corr2_11,sig_corr2_22,sig_corr2_12,knum,nfreq,U,beta)#second order insertion on a hartree
            sigext3a_22=-sigext3a_11#
                    # important note: I had a bug for the diagramB and I was using sig_corr1_11. and got ridiculous bugs. BE CAREFUL HERE!
                    #DO NOT USE sig_corr1_11!!!! BUT WHY? because immutable and mutable variables. then we have to use deepcopy!!!
            sigext3b_11=diagrams.sig3_nonskeleton_B(G0_22_iom,G0_12_iom,G0_11_iom,sig_corr1_11,sig_corr1_22,knum,nfreq,U,beta)
            sigext3b_22=-sigext3b_11
            # sigext3c_11=diagrams.sig2_nonskeleton(G0_22_iom,G0_12_iom,sigext2_11,sigext2_22,knum,nfreq,U,beta)
            # sigext3c_22=-sigext3c_11
                        # print('CT diagrams order3: a={:.4f} b={:.4f} c={:.4f} a+b+c={:.4f}'.format(sigext3a_11.real,sigext3b_11.real,sigext3c_11.real,sigext3a_11.real+sigext3b_11.real+sigext3c_11.real))
            time32=time.time()
            sigext3def_11,sigext3def_12=diagrams.sig3_nonskeleton_DEF(G0_11_iom,G0_12_iom,G0_11_tau,G0_12_tau,P22_tau,P12_tau,sig_corr1_11,sig_corr1_22,beta,knum,nfreq,U)
            sigext3def_22=-sigext3def_11.conjugate()
        
                    #skeleton corrections
            sig_corr3_11+=(Sig3_11+sigext3def_11-sigimpmod_3_11[:,None,None,None]+sigext3a_11+sigext3b_11)#+sigext3c_11
            sig_corr3_22+=(Sig3_22+sigext3def_22-sigimpmod_3_22[:,None,None,None]+sigext3a_22+sigext3b_22)#+sigext3c_22
            sig_corr3_12+=(Sig3_12+sigext3def_12)


        comm.Bcast(sig_corr3_11, root=0)
        comm.Bcast(sig_corr3_22, root=0)
        comm.Bcast(sig_corr3_12, root=0)            
        sig_corr_11+=sig_corr3_11
        sig_corr_22+=sig_corr3_22
        sig_corr_12+=sig_corr3_12
        # if rank==0:
            # print('3nd order timeske={}s  timeabc={}s  timedef={}s'.format(time31-time30,time32-time31,time33-time32))
            # print('3rd order done! time={}s'.format(time.time()-time0))
    # if order>=4:#Note: here 4th order is only for non-iterative version.
    #     sig_corr4_11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    #     sig_corr4_22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    #     sig_corr4_12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    #     #summon the monte carlo to get the 2 most difficult diagrams of the 4th order.
        
    #     GFs=(G0_11_tau.real,G0_12_tau.real,G0_22_tau.real)
    #     sublatind_basis=np.array([[1,0,0,0],
    #                           [0,1,0,0],
    #                           [0,0,1,0],
    #                           [0,0,0,1]])

    #     # for make the code more efficient, we'd better think when do we need more MC steps:
    #     # The derivative of Fermi-dirac distribution: \frac{df(E)}{dE} = f(E)(1 - f(E))*beta.
    #     # Assume the self-energy is a real const. At lower T and paramagnetic case, same error of Sigma will result in much larger error of magnetization.
    #     # so, at low alpha and low T, more MC steps should be used.
    #     stepnum_basics1=int(200/nprocs)
    #     stepnum_basics2=int(200/nprocs )        
    #     if alpha <=0.3:
    #         stepnum_basics1=int(300/nprocs)
    #         stepnum_basics2=int(300/nprocs)
    #     if alpha<=0.2:
    #         stepnum_basics1=int(600/nprocs)
    #         stepnum_basics2=int(600/nprocs)
    #     if alpha<=0.1:
    #         stepnum_basics1=int(1000/nprocs)
    #         stepnum_basics2=int(1000/nprocs)
    #     # if beta>4:
    #     #     stepnum_basics1*=(beta/4)
    #     #     stepnum_basics2*=(beta/4)
    #     # stepnum_basics1=int(max(stepnum_basics1,50/nprocs))
    #     # stepnum_basics2=int(max(stepnum_basics2,50/nprocs))


    #     func43=diag_def_cutPhifast.FuncNDiagNew(T,U,knum,taunum,nfreq,4,ut,kbasis,perm_def.perm43,GFs,perm_def.dep43,2)
    #     func44=diag_def_cutPhifast.FuncNDiagNew(T,U,knum,taunum,nfreq,4,ut,kbasis,perm_def.perm44,GFs,perm_def.dep44,4)
    #     # if rank==0:
    #     #     print('doing 1st MC..  time={}s step#={}M'.format(time.time()-time0,stepnum_basics1))        
    #     Sig4_3_11coeff,Sig4_3_12coeff,Sig4_3_11,Sig4_3_12,Sig4_3_11tau,Sig4_3_12tau=svd_diagramsMC_cutPhi.Summon_Integrate_Parallel_dispersive(func43,params(stepnum_basics1),imax,lmax,ut,kbasis,beta,alpha,svdcheck)
    #     # if rank==0:
    #         # print('doing 2nd MC..  time={}s step#={}M'.format(time.time()-time0,stepnum_basics2))        
    #     Sig4_4_11coeff,Sig4_4_12coeff,Sig4_4_11,Sig4_4_12,Sig4_4_11tau,Sig4_4_12tau=svd_diagramsMC_cutPhi.Summon_Integrate_Parallel_dispersive(func44,params(stepnum_basics2),imax,lmax,ut,kbasis,beta,alpha,svdcheck)
    #     if rank==0:
    #         print('MC finished!  time={}s'.format(time.time()-time0))
    #     time40=time.time()
    #     # then we evaluate ladder, rpa,....first prepare Q and R. we'll use them later.

    #     if rank==0:
    #     # P22_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G0_22_tau,G0_22_tau,0)
    #     # P12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12,G0_12_tau,G0_12_tau,1)
    #         P11_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G0_11_tau,G0_11_tau,0)
    #         P11_iom=fft.fast_ift_boson(P11_tau,beta)
    #         P22_iom=fft.fast_ift_boson(P22_tau,beta)
    #         P12_iom=fft.fast_ift_boson(P12_tau,beta)
    #         # then, generate all diagrams which are 1 insertion in 3rd order. This also have to be done in parallel way.
    #         Sig4_1_11,Sig4_1_12=diagrams.sig4_1(G0_11_tau,G0_12_tau,G0_22_tau,Q11_iom,Q12_iom,Q22_iom,knum,nfreq,U,beta)
    #         Sig4_2_11,Sig4_2_12=diagrams.sig4_2(G0_11_tau,G0_12_tau,G0_22_tau,R11_iom,R12_iom,R22_iom,knum,nfreq,U,beta)
    #         Sig4_5_11,Sig4_5_12=diagrams.sig4_5(G0_11_tau,G0_12_tau,G0_22_tau,P11_iom,P12_iom,P22_iom,knum,nfreq,U,beta)
    #         # That's all skeleton diagrams of order 4. Then we should consider the 1st order insertion in the 3rd order.

    #         print('4th order ladderrpa finished!  timeladrpa={}s'.format(time.time()-time0))


    #         Sig4_3plus_11,Sig4_3plus_12=diagrams.sig4_3plus(G0_11_iom,G0_12_iom,G0_22_iom,G0_11_tau,G0_12_tau,G0_22_tau,  sig_corr1_11,sig_corr1_22,  Q11_iom,Q12_iom,Q22_iom,R11_iom,R12_iom,R22_iom,knum,nfreq,U,beta)
    #         Sig4_2plus_11,Sig4_2plus_12=diagrams.sig4_2plus(G0_11_iom,G0_12_iom,G0_22_iom,G0_11_tau,G0_12_tau,G0_22_tau,P22_tau,P12_tau,  sig_corr1_11,sig_corr1_22,sig_corr2_11,sig_corr2_12,sig_corr2_22,knum,nfreq,U,beta)
    #         Sig4_1plus_11=diagrams.sig4_1plus(G0_11_iom,G0_12_iom,G0_22_iom,sig_corr1_11,sig_corr1_22,sig_corr2_11,sig_corr2_12,sig_corr2_22,sig_corr3_11,sig_corr3_12,sig_corr3_22,knum,nfreq,U,beta)

    #         sig_corr4_11+=(Sig4_1_11+Sig4_2_11+Sig4_3_11+Sig4_4_11+Sig4_5_11-sigimpmod_4_11[:,None,None,None]+Sig4_3plus_11+Sig4_2plus_11+Sig4_1plus_11)#
    #         sig_corr4_12+=(Sig4_1_12+Sig4_2_12+Sig4_3_12+Sig4_4_12+Sig4_5_12+Sig4_3plus_12+Sig4_2plus_12)  #       
    #         sig_corr4_22=-sig_corr4_11.conjugate()

    #         # for kx in np.arange(knum):
    #         #     for ky in np.arange(knum):
    #         #         for kz in np.arange(knum):
    #         #             plt.plot(Sig4_1_11[:,kx,ky,kz].real,label='dispersive41 real')#+Sig4_2_11[:,kx,ky,kz].real+Sig4_3_11[:,kx,ky,kz].real+Sig4_4_11[:,kx,ky,kz].real+Sig4_5_11[:,kx,ky,kz].real
    #         #             plt.plot(Sig4_1_11[:,kx,ky,kz].imag,label='dispersive41 imag')#+Sig4_2_11[:,kx,ky,kz].imag+Sig4_3_11[:,kx,ky,kz].imag+Sig4_4_11[:,kx,ky,kz].imag+Sig4_5_11[:,kx,ky,kz].imag
    #         #             plt.plot(Sigmaimpiom41_11.real,label='imp41 real')
    #         #             plt.plot(Sigmaimpiom41_11.imag,label='imp41 imag')
    #         #             plt.legend()
    #         #             plt.show()
    #         #             plt.plot(Sig4_2_11[:,kx,ky,kz].real,label='dispersive42 real')
    #         #             plt.plot(Sig4_2_11[:,kx,ky,kz].imag,label='dispersive42 imag')
    #         #             plt.plot(Sigmaimpiom42_11.real,label='imp42 real')
    #         #             plt.plot(Sigmaimpiom42_11.imag,label='imp42 imag')
    #         #             plt.legend()
    #         #             plt.show()
    #         #             plt.plot(Sig4_3_11[:,kx,ky,kz].real,label='dispersive43 11 real')
    #         #             plt.plot(Sig4_3_11[:,kx,ky,kz].imag,label='dispersive43 11 imag')                        
    #         #             plt.plot(Sigmaimpiom43_11.real,label='imp43 real')
    #         #             plt.plot(Sigmaimpiom43_11.imag,label='imp43 imag')
    #         #             plt.legend()
    #         #             plt.show()
    #         #             plt.plot(Sig4_3_11tau[:,kx,ky,kz].real,label='dispersive43tau 11 real')                   
    #         #             plt.plot(Sigmaimp43tau.real,label='imp43 real')
    #         #             plt.legend()
    #         #             plt.show()
    #         #             plt.plot(Sig4_4_11[:,kx,ky,kz].real,label='dispersive44 11 real')
    #         #             plt.plot(Sig4_4_11[:,kx,ky,kz].imag,label='dispersive44 11 imag')                        
    #         #             plt.plot(Sigmaimpiom44_11.real,label='imp44 real')
    #         #             plt.plot(Sigmaimpiom44_11.imag,label='imp44 imag')
    #         #             plt.legend()
    #         #             plt.show()
    #         #             plt.plot(Sig4_4_11tau[:,kx,ky,kz].real,label='dispersive44tau 11 real')                   
    #         #             plt.plot(Sigmaimp44tau.real,label='imp44 real')
    #         #             plt.legend()
    #         #             plt.show()
    #         #             plt.plot(Sig4_5_11[:,kx,ky,kz].real,label='dispersive45 real')
    #         #             plt.plot(Sig4_5_11[:,kx,ky,kz].imag,label='dispersive45 imag')
    #         #             plt.plot(Sigmaimpiom45_11.real,label='imp45 real')
    #         #             plt.plot(Sigmaimpiom45_11.imag,label='imp45 imag')
    #         #             plt.legend()
    #         #             plt.show()

    #     comm.Bcast(sig_corr4_11, root=0)
    #     comm.Bcast(sig_corr4_22, root=0)
    #     comm.Bcast(sig_corr4_12, root=0)                         
    #     sig_corr_11+=sig_corr4_11
    #     sig_corr_22+=sig_corr4_22
    #     sig_corr_12+=sig_corr4_12  
    #     if rank==0:
    #         print('4th order done!  time={}s'.format(time.time()-time0))

    filename_u='../data/Sigma_imp/taubasis.txt'
    ut=np.loadtxt(filename_u).T
    taunum=np.shape(ut)[1]-1
    lmax=np.shape(ut)[0]
    # print('taunum=',taunum,'lmax=',lmax)
    # getting the observables
    sigmafilename11='../data/Sigma_disp{}/{}_{}_{}/{}_{}_{}_{}_{}_11.dat'.format(foldernum,U,T,B,U,T,B,order,alpha)
    sigmafilename11const='../data/Sigma_disp{}/{}_{}_{}/{}_{}_{}_{}_{}_11const.dat'.format(foldernum,U,T,B,U,T,B,order,alpha)
    sigmafilename12='../data/Sigma_disp{}/{}_{}_{}/{}_{}_{}_{}_{}_12.dat'.format(foldernum,U,T,B,U,T,B,order,alpha)
    os.makedirs(os.path.dirname(sigmafilename11), exist_ok=True)
    # sigmafilename22='./Sigma_disp/{}_{}_{}_{}_22.dat'.format(U,T,order,alpha)  22 can be got from particle hole symmetry.
    taulist=(np.arange(taunum+1))/taunum*beta
    ori_grid=(np.arange(nfreq*2)+0.5)/(nfreq*2)*beta


    # Then we have to save the self-energy. 
    # However, according to out diagrammatics, any orders of perturbation will contribute at least a constant. which means simple fft does not work.
    if order>0:                
        # finally, according to the sigma, get the best GF.
        sigfinal11=Sigmod11+sig_corr_11
        sigfinal22=Sigmod22+sig_corr_22
        sigfinal12=sig_corr_12
        if rank==0:


            #Note: the sigfinal11 has constant and 1/omega part, which is not suitable for simple fft. 
            # to do a nice fft we have to take care of these things.
            omega=(2*np.arange(2*nfreq)+1-2*nfreq)*np.pi/beta
            sigfinal11_const=sigfinal11[-1,:,:,:].real# This is constant.
            sigfinal11_const_coeffk=basis.coeff_k(sigfinal11_const,kbasis[0])


            #Note: It's not a good idea to use FFT to treat this const part. Also, svd basis is not good at taking care of these consts.
            # As a result, it's better to create another file to save the consts using kspace basis.
            A,C=GetHighFrequency4D(sigfinal11-sigfinal11_const,omega)
            sigfinal11_om1=A[None,:,:,:]/(1j*omega[:,None,None,None]-C[None,:,:,:])
            sigfinal11_tau1=  A[None,:,:,:]*(fermi(C[None,:,:,:],beta)-1)*np.exp(-ori_grid[:,None,None,None]*C[None,:,:,:])             # analytical FT of A/(iom-C)
            sigfinal11_rest=sigfinal11-sigfinal11_const-sigfinal11_om1
            sigfinaltau11_rest=fft.fast_ft_fermion(sigfinal11_rest,beta).real
            sigfinaltau11=sigfinaltau11_rest+sigfinal11_tau1# Note: This does not include const part!!


            sigfinaltau12=fft.fast_ft_fermion(sigfinal12,beta).real
            # sigfinaltau22=fft.fast_ft_fermion(sigfinal22,beta)
            # print(np.shape(sigfinaltau11))
            sigfinaltau11_splined=interp1d(ori_grid, sigfinaltau11, kind='cubic', axis=0,fill_value='extrapolate')
            sigfinaltau12_splined=interp1d(ori_grid, sigfinaltau12, kind='cubic', axis=0,fill_value='extrapolate')
        
            cli_11=basis.coeff_tk(sigfinaltau11_splined(taulist),ut,kbasis[0])
            cli_12=basis.coeff_tk(sigfinaltau12_splined(taulist),ut,kbasis[1])
            # cli_22=basis.coeff_tk(sigfinal22,ut,kbasis)
            np.savetxt(sigmafilename11,cli_11)
            np.savetxt(sigmafilename11const,sigfinal11_const_coeffk)
            np.savetxt(sigmafilename12,cli_12)
            # np.savetxt(sigmafilename22,cli_22)

            # Sig11tk_raw=basis.restore_tk(cli_11,ut,kbasis)
            # Sig12tk_raw=basis.restore_tk(cli_12,ut,kbasis)
            # for kx in np.arange(knum):
            #     for ky in np.arange(knum):
            #         for kz in np.arange(knum):
            #             plt.plot(sigfinal11[:,kx,ky,kz].real-sigfinal11_const[kx,ky,kz].real,label='final11 real')
            #             plt.plot(sigfinal11[:,kx,ky,kz].imag,label='final11 imag')
            #             plt.plot(sigfinal22[:,kx,ky,kz].real,label='final22 real')
            #             plt.plot(sigfinal22[:,kx,ky,kz].imag,label='final22 imag')
            #             plt.legend()
            #             plt.show()
            #             plt.plot(sigfinaltau11[:,kx,ky,kz].real,label='final11 tau')
            #             plt.plot(sigfinaltau12[:,kx,ky,kz].real,label='final12 tau')
            #             plt.legend()
            #             plt.show()
            #             plt.plot(sigfinaltau11_splined(taulist)[:,kx,ky,kz].real,label='final11spline tau')
            #             plt.plot(Sig11tk_raw[:,kx,ky,kz].real,label='final11recovered tau')
            #             plt.plot(sigfinaltau12_splined(taulist)[:,kx,ky,kz].real,label='final12spline tau')
            #             plt.plot(Sig12tk_raw[:,kx,ky,kz].real,label='final12recovered tau')
            #             plt.legend()
            #             plt.show()



        znew_1=z4D(beta,mu,sigfinal11,knum,nfreq)-B
        znew_2=z4D(beta,mu,sigfinal22,knum,nfreq)+B
        Gdress11_iom,Gdress12_iom=G_iterative(knum,znew_1,znew_2,sigfinal12)
        Gdress22_iom=-Gdress11_iom.conjugate()
        # nnewloc11=np.sum(Gdress11_iom).real/knum**3/beta+1/2
        # nnewloc22=np.sum(Gdress22_iom).real/knum**3/beta+1/2
        nnewloc11=particlenumber4D(Gdress11_iom,beta)
        nnewloc22=particlenumber4D(Gdress22_iom,beta)
        # Fimp, Eimp,Fdisp,Edisp=energy.PertFreeEnergy(sigfinal11,sigfinal22,sigfinal12,U,T)
    else:
        if rank==0:
            np.savetxt(sigmafilename11,Sigmod11[:,0,0,0])
        # np.savetxt(sigmafilename12,cli_12)# 0th order is diagonal.
        # save sigmod11 and sigmod 22. do not have to use the ut basis.
        nnewloc11=n0loc11
        nnewloc22=n0loc22
        # Fimp, Eimp,Fdisp,Edisp=energy.PertFreeEnergy(Sigmod11,Sigmod22,Sigmod12,U,T)
    
    return nnewloc22-nnewloc11#,Fdisp,Edisp

def run_perturbation(U,T,B,nfreq,ordernum,alpha,foldernum='B'):
    if U>=8:# for U>=8 boldc is decent, but below that it is better to choose ctqmc.
        mode='boldc'
    else:
        mode='ctqmc'
    name1='../data/files_{}/{}_{}/Sig.out'.format(mode,U,T)
    filename1=readDMFT(name1)
    name2='../data/files_boldcB/{}_{}_{}/Sig.OCA'.format(U,T,B)
    filename2=readDMFT(name2)
    # print(filename1)
    # print(filename2)
    if (os.path.exists(filename1)):
        filename=filename1
    elif (os.path.exists(filename2)):
        filename=filename2
        # print('reading DMFT data from {}'.format(filename))
    else:
        print('{} {} cannot be found!'.format(filename1,filename2))  
        return 0  
    sigma=np.loadtxt(filename)[:nfreq,:]
    check=sigma[-1,1]
    om=sigma[:,0]
    # anyways real part of sigA will be greater. This is my convension.
    sigA=sigma[:,1]+1j*sigma[:,2]#
    sigB=U-sigma[:,1]+1j*sigma[:,2]

    mag=perturbation(om,sigA,sigB,U,T,B,nfreq,ordernum,alpha,foldernum)
    #,Fdisp,Edisp
    return mag#,Fdisp,Edisp

def launch_UT(U,T,B,maxorder,ifskip=1,foldernum='B'):
    ifit=0# 0: no iteration
    typelist=['basic','iterative']
    order_arr = np.arange(maxorder+1)[::-1]
    # alpha_arr=np.array(([0.01,0.05,0.1,0.2,0.3,0.6,1.0]))
    # alpha_arr=np.arange(10)*B/10.
    # alpha_arr=np.array(([0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08]))#
    alpha_arr=np.arange(10)/2000
    alpha_arr = np.round(alpha_arr, decimals=4)
    print(alpha_arr)
    # alpha_arr=np.array(([1.0,0.05,0.1,0.2,0.3,0.6]))
    # alpha_arr=np.array(([0.01,0.05,0.1,0.2,0.3,0.6,1.0]))
    # magarr=np.zeros((6,alpha_arr.size),dtype=float)
    # Earr=np.zeros((4,alpha_arr.size),dtype=float)
    # Farr=np.zeros((4,alpha_arr.size),dtype=float)
    # print('U={},T={}'.format(U,T))

    for order in order_arr:
        for i,alpha in enumerate(alpha_arr):
            # if rank==0:
                # print('U={},T={},order={},alpha={} started!'.format(U,T,order,alpha))
            sigmafilename11='./Sigma_disp{}/{}_{}/{}_{}_{}_{}_11.dat'.format(foldernum,U,T,U,T,order,alpha)
            sigmafilename11const='./Sigma_disp{}/{}_{}/{}_{}_{}_{}_11const.dat'.format(foldernum,U,T,U,T,order,alpha)
            sigmafilename12='./Sigma_disp{}/{}_{}/{}_{}_{}_{}_12.dat'.format(foldernum,U,T,U,T,order,alpha)
            ifhavefiles=1
            if order>=1:

                if (os.path.exists(sigmafilename11))==0 or (os.path.exists(sigmafilename11const))==0 or (os.path.exists(sigmafilename12))==0 :
                    ifhavefiles=0
            else:
                if (os.path.exists(sigmafilename11))==0  :
                    ifhavefiles=0
            if ifskip==0 or ifhavefiles==0:
                # if rank==0:
                #     print('U={},T={},order={},alpha={} started!'.format(U,T,order,alpha))                    
                run_perturbation(U,T,B,nfreq,order,alpha,foldernum)
                if rank==0:
                    print('U={},T={},order={},alpha={} finished!'.format(U,T,order,alpha))      
            # else:
            #     if rank==0:
            #         print('U={},T={},order={},alpha={} already found. Skipped!'.format(U,T,order,alpha))      
    return 0

def search_optalp(U,T,B,maxorder,opt=0):
    order_arr = np.arange(maxorder+1)
    magarr=np.zeros_like(order_arr,dtype=float)
    foldernum='B'
    magmin=999.0
    alpha=0.1*B
    step=0.1*B
    converged=0
    countstep=0
    flag=0
    ifredo=1
    print('searching: U={} T={} B={}'.format(U,T,B))
    if opt==0:  
        filepath='../data/Sigma_dispB/{}_{}_{}.txt'.format(U,T,B)
    else:
        filepath='../data/Sigma_dispB/{}_{}_{}opt.txt'.format(U,T,B)
    with open(filepath, 'w'):
        pass
    while not converged:
        
        for order in order_arr:
            sigmafilename11='../data/Sigma_disp{}/{}_{}_{}/{}_{}_{}_{}_{}_11.dat'.format(foldernum,U,T,B,U,T,B,order,alpha)
            sigmafilename11const='../data/Sigma_disp{}/{}_{}_{}/{}_{}_{}_{}_{}_11const.dat'.format(foldernum,U,T,B,U,T,B,order,alpha)
            sigmafilename12='../data/Sigma_disp{}/{}_{}_{}/{}_{}_{}_{}_{}_12.dat'.format(foldernum,U,T,B,U,T,B,order,alpha)
            check=0
            if order>0:
                if (os.path.exists(sigmafilename11))==0 or (os.path.exists(sigmafilename11const))==0 or (os.path.exists(sigmafilename12))==0:
                    check=1# file missed
            else:
                if (os.path.exists(sigmafilename11))==0:
                    check=1
            if check==0 and ifredo==0:
                # print('U={},T={},order={},alpha={} found!'.format(U,T,order,alpha))
                Sigiom_11,Sigiom_12,Sigiom_22,Gdress11_iom,Gdress12_iom,Gdress22_iom=XAF.get_sigma_and_G(U,T,B,order,alpha,foldernum)
                nnewloc11=particlenumber4D(Gdress11_iom,1/T)
                nnewloc22=particlenumber4D(Gdress22_iom,1/T)
                magarr[order]=nnewloc22-nnewloc11
            else:
                magarr[order]=run_perturbation(U,T,B,nfreq,order,alpha,foldernum)
            # print('U={},T={},order={},alpha={} finished!'.format(U,T,order,alpha))  

        mag=np.mean(magarr[1+opt:])  
        if opt==0:  
            stddev=np.std(magarr[1:])
        else:
            stddev=np.std(magarr[2:])
        print('step=',countstep,'alpha=',alpha,'mag={:.5f}'.format(mag), 'stddev={:.6f}'.format(stddev))
        with open(filepath, 'a') as f:
            f.write('{} {} {:.6f} {:.6f}\n'.format(countstep,alpha,mag,stddev))
        if countstep>0.5:
            if prvstddev<stddev:
                if flag==0:
                    alpha-=2*step
                    step/=10
                    countstep=-1
                    flag=1
                else:
                    break
        alpha+=step  
        alpha=np.round(alpha,decimals=4) 
        prvmag=copy.deepcopy(mag)
        prvstddev=copy.deepcopy(stddev)
        countstep+=1
    print('mag={:.6f}'.format(prvmag))
    return prvmag





if __name__ == "__main__":
    # some debug settings
    svdcheck=0
    ifsigAFM=0# ifsigAFM=1 means use alpha*sigma_AFM, ifsigAFM=0 means a simple splitting alpha*U splitting.
    # ifsigAFM=1 might be better for AFM DMFT solution, but for PM DMFT solution we can only use ifsigAFM=0.

    knum=10
    nfreq=500
    index=50
    
    U=8.0  
    T=0.4
    B=0.01

    search_optalp(U,T,B,3)

