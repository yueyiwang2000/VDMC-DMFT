from scipy import *
from scipy.interpolate import interp1d
# import weight_lib 
from numpy import linalg
from scipy.sparse.linalg import svds
import sys,os
import copy
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import time
sys.path.append('../')
import perturb_lib as lib
import fft_convolution as fft
from diagramsMC_lib import *

#-----------useful functions----------


def fermi_kernel(t, w, beta):
    t = np.asarray(t)
    w = np.asarray(w)

    x = beta * w / 2.0
    y = 2.0 * t / beta - 1.0

    result = np.zeros((t.size, w.size))

    mask_large_x = x > 100
    mask_small_x = x < -100
    mask_mid_x = ~mask_large_x & ~mask_small_x
    # print(mask_large_x,mask_mid_x,mask_small_x)
    result[:,mask_large_x] = np.exp(-x[None,mask_large_x] * (y[:,None] + 1.0))
    result[:,mask_small_x] = np.exp(x[None,mask_small_x] * (1.0 - y[:,None]))
    result[:,mask_mid_x] = np.exp(-x[None,mask_mid_x] * y[:,None]) / (2.0 * np.cosh(x[None,mask_mid_x]))

    return result

def svd_kernel_slow(ker,lmax):
    '''
    traditional svd. this slow version is usually better if the size of the kernel is not very large.
    ker(x,y)=U@sigma@VT
    shape of output: u=(lmax,taunum) sigma=(lmax) v=(lmax,omnum)
    '''
    U, Sigma, VT = np.linalg.svd(ker)
    # U, Sigma, VT = svds(ker, k=lmax)
    return U[:,:lmax].T,Sigma[:lmax],VT[:lmax,:]

def svd_kernel_fast(ker,lmax):
    '''
    if we are only looking for the first few sigular values, there is a faster way, especially when the size of the kernel is much greater than lmax.
    ker(x,y)=U@sigma@VT
    shape of output: u=(lmax,taunum) sigma=(lmax) v=(lmax,omnum)
    '''
    U, Sigma, VT = svds(ker, k=lmax)
    U = U[:, ::-1]
    Sigma = Sigma[::-1]
    VT = VT[::-1, :]
    return U.T,Sigma,VT

def inner_prod(Gtau,u):
    coeffs=np.sum(Gtau[None,:]*u,axis=(1)).real
    return coeffs

def restore_Gf(coeff,u):
    # inter=sym+1# sym=0 means count every coeff,1 means skip 0th, 2nd, 4th,....coeffs.because of the  symmetry.
    return np.sum(coeff[:,None]*u,axis=(0))

#------------some test functions----------
def ortho_check(u,lmax):
    '''
    check the orthogonality of different columns in u and v.
    '''
    for i in np.arange(lmax):
        for j in np.arange(lmax):
            res=np.sum(u[i,:]*u[j,:])# check u
            # res=np.sum(u[i]*u[j])# check v
            print(res)
    return 0


def svd_check():
    T=0.3
    beta=1/T
    taunum=20
    omnum=100
    lm=5
    nfreq=500
    # taulist=np.arange(taunum+1)/taunum*beta
    # omlist=(2*np.arange(omnum)+1-2*omnum)*np.pi/beta  # have to use negative freqs!
    # ker=fermi_kernel(taulist,omlist,beta)

    taulist=(np.arange(taunum+1))/taunum*beta#
    # taulist=(np.arange(taunum)+0.5)/taunum*beta#
    # taulist=(np.arange(taunum+1))/taunum*beta#
    omlist=(2*np.arange(2*nfreq)+1-2*nfreq)*np.pi/beta 
    ker1=fermi_kernel(taulist,omlist,beta)
    ut1,sig1,v1=svd_kernel_fast(ker1,lm)

    taulist2=(np.arange(taunum*2))/(taunum*2)*beta#
    ker2=fermi_kernel(taulist2,omlist,beta)
    ut2,sig2,v2=svd_kernel_fast(ker2,lm)
    # for l in np.arange(lm):
    #     plt.plot(taulist,ut1[l],label='ut1 {}'.format(l))
    #     plt.plot(taulist2,ut2[l],label='ut2 {}'.format(l))
    #     plt.legend()
    #     plt.show()
    # time2=time.time()
    # print(np.shape(sig1))
    print(np.shape(ut1))
    print(np.shape(v1))
    # print(np.shape(sig2))
    # print(np.shape(ut2))
    # print(np.shape(v2))
    # print(sig1)
    # print(sig2)
    # print('norm of ut:',np.linalg.norm(ut2[1]))
    # print(u1[1])
    # print(vt1[:,1])
    # print(u2[1])
    # print(vt2[:,1])

    # print('slow svd:{}s'.format(time1-time0))
    # print('fast svd:{}s'.format(time2-time1))

    ortho_check(ut2,lm)
    return 0

def get_GFandsig(U,T,nfreq,knum):
    # note: sigma(tau) and G(tau) are all real.
    # U=8
    # T=0.4
    # nfreq=500
    mu=U/2
    beta=1/T
    # knum=10
    #------copied from previous files-------
    name1='../../files_boldc/{}_{}/Sig.out'.format(U,T)
    filename1=readDMFT(name1)
    name2='../../files_ctqmc/{}_{}/Sig.out'.format(U,T)
    filename2=readDMFT(name2)
    # print(filename1)
    # print(filename2)
    if (os.path.exists(filename1)):
        filename=filename1
    elif (os.path.exists(filename2)):
        filename=filename2
        # print('reading DMFT data from {}'.format(filename))
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

    sigma2=sig2(G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta)
    sigma2off=sig2offdiag(G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta)
    # sigma2loc=sig2(Gloc11_tau,G12_tau,Gloc22_tau,knum,nfreq,U,beta)
    return G11_tau,G12_tau,Gloc11_tau,sigma2,sigma2off

def reproduce_check():
    U=8.0
    T=0.4
    nfreq=500
    knum=10
    beta=1/T
    taunum=100
    omnum=nfreq*2
    G11tau,G12tau,Gloctau,Sig11,Sig12=get_GFandsig(U,T,nfreq,knum)
    print('G and sigma generated!')

    time0=time.time()
    # taulist=(np.arange(taunum)+0.5)/taunum*beta
    taulist=(np.arange(taunum+1))/taunum*beta
    


    taulistdense=(np.arange(omnum)+0.5)/omnum*beta
    omlist=(2*np.arange(2*nfreq)+1-2*nfreq)*np.pi/beta  # have to use negative freqs!
    lm=10
    ker=fermi_kernel(taulist,omlist,beta)
    ut,sig,v=svd_kernel_fast(ker,lm)
    # ut,sig,v=svd_kernel_slow(ker,lm)
    for l in np.arange(lm):
        plt.plot(taulist,ut[l],label='ut {}'.format(l))
        plt.legend()
        plt.show()
    time1=time.time()
    
    # for l in np.arange(lm):
    #     plt.plot(ut[l],label='l={}'.format(l))
    # plt.legend()
    # plt.show()
    print('sigularvals:\n',sig)
    kx=1;ky=2;kz=3
    # Gtest=G11tau[:,kx,ky,kz]
    Gtest=Sig11[:,kx,ky,kz]#
    # Gtest=G12tau[:,kx,ky,kz]
    # Gtest=Sig12[:,kx,ky,kz]#    

    interpolator1 = interp1d(taulistdense, Gtest, kind='linear', axis=0, fill_value='extrapolate')
    G=interpolator1(taulist)
    coeff=inner_prod(G,ut)
    G_res=restore_Gf(coeff,ut)
    time2=time.time()
    print('coeffs:\n',coeff)
    print('time_svd=',time1-time0,'time_coeff_and_restore=',time2-time1)
    plt.plot(taulist,G.real,label='original')
    plt.plot(taulist,G_res.real,label='restored')
    plt.xlabel('tau')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # svd_check()
    reproduce_check()
    