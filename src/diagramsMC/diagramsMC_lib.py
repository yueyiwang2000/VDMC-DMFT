from scipy import *
from numpy import linalg
from numpy import random
from scipy import special
import sys,os
import numpy as np
sys.path.append('../')
import perturb_lib as lib
import fft_convolution as fft
import mpi_module
import matplotlib.pyplot as plt
from numba import jit
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

@jit(nopython=True)
def Gshift(G12,q,tau,opt,knum,taunum,epsilon=0):
    """
    opt==1 means shift with sign!
    the epsilon defines how to take care of tau=0.
    """
    qx=q[0]
    qy=q[1]
    qz=q[2]
    
    # always have antisymmetry of time domain
    # here we should anwser a question: at tau=0 which value should we pick?
    # tfactor=(-1)**((np.mod(tau, taunum)-(tau))/(taunum))#0+
    tfactor=(-1)**((np.mod(tau-epsilon, taunum)-(tau-epsilon))/(taunum))#0-
    if opt==1:#with factor
        G_12_factor=(-1)**((np.mod(qx, knum)-(qx))/knum+(np.mod(qy, knum)-(qy))/knum+(np.mod(qz, knum)-(qz))/knum)*tfactor
    else:
        G_12_factor=tfactor
    # G12_kq = G_12_factor*G12[np.mod(tau, taunum), np.mod(qx, knum), np.mod(qy, knum), np.mod(qz, knum)]#0+
    G12_kq = G_12_factor*G12[np.mod(tau-epsilon, taunum)+epsilon, np.mod(qx, knum), np.mod(qy, knum), np.mod(qz, knum)]#0-
    # print(epsilon.dtype)
    if np.abs(G12_kq.imag)>1e-4:#
        print('WARNING! non-zero imag part!',G12_kq,tau,q,G12[np.mod(tau-epsilon, taunum)+epsilon, np.mod(qx, knum), np.mod(qy, knum), np.mod(qz, knum)])
        print(np.mod(tau-epsilon, taunum)+epsilon)
    return G12_kq




@jit(nopython=True)
def Gshift_simple(G12,q,tau,opt,knum):
    """
    opt==1 means shift with sign!
    if tau must fall betw 0 and beta, use this may be faster.
    """
    qx=q[0]
    qy=q[1]
    qz=q[2]
    
    if opt==0:#without factor
        G_12_factor=1
    else:#with factor
        G_12_factor=(-1)**((np.mod(qx, knum)-(qx))/knum+(np.mod(qy, knum)-(qy))/knum+(np.mod(qz, knum)-(qz))/knum)
    G12_kq = G_12_factor*G12[tau, np.mod(qx, knum), np.mod(qy, knum), np.mod(qz, knum)]#0-
    if np.abs(G12_kq.imag)>1e-4:#
        print('WARNING! non-zero imag part!',G12_kq,tau,q,G12[tau, np.mod(qx, knum), np.mod(qy, knum), np.mod(qz, knum)])
    return G12_kq

def Gshiftloc(G12,tau,taunum):
    """
    opt==1 means shift with sign!
    """
    # always have antisymmetry of time domain
    # here we should anwser a question: at tau=0 which value should we pick?
    # tfactor=(-1)**((np.mod(tau, taunum)-(tau))/(taunum))#0+
    tfactor=(-1)**((np.mod(tau-1, taunum)-(tau-1))/(taunum))#0-

    G_12_factor=tfactor
    

    # G12_kq = G_12_factor*G12[np.mod(tau, taunum), np.mod(qx, knum), np.mod(qy, knum), np.mod(qz, knum)]#0+
    G12_kq = G_12_factor*G12[np.mod(tau-1, taunum)+1]#0-
    return G12_kq

@jit(nopython=True)
def TrialStep0_k(knum):
    # tiQ = int( random.rand()*len(qx) )               # trial iQ for qx[iQ]
    # Ka_new = qx[tiQ]                                 # |r_0| length of the vector
    K_new =np.array([int( random.rand()*knum) , int( random.rand()*knum ) , int( random.rand()*knum ) ]) # new 3D vector r_0
    accept=True    
    trial_ratio = 1.

    return (K_new, accept)

@jit(nopython=True)
def TrialStep0_tau(taunum):     
    tau_new =int( random.rand()*taunum) 
    accept=True    
    trial_ratio = 1.
    return (tau_new, accept)

@jit(nopython=True)
def TrialStep1_k(iloop,momentum,knum,kfold):
    choices = np.array([-3,-2,-1,0,1,2,3])
    dk = np.random.choice(choices, size=3)
    # trial_ratio = 1.      
    accept=1
    # choice 1: 'periodic boundary'
    K_new = np.mod(momentum[iloop,:] + dk,knum)    # K_new = K_old + dK
    # choice 2: reject those out of boundary
    # K_new = momentum[iloop,:] + dk
    # if np.any(K_new>=knum) or np.any(K_new<0):
    #     accept=0
    #choice 3: random
    # K_new=np.random.randint(0,knum,size=3)
    # accept=1
    return (K_new,  accept)


@jit(nopython=True)
def TrialStep1_tau(iloop,imagtime,taunum,Ndimk,taufold):
    stepmax=int(taunum/5)
    steps=np.arange(-1*stepmax,stepmax+1)
    # steps=np.array([-3,-2,-1,0,1,2,3])
    dtau =  np.random.choice(steps, size=1)
    accept =1
    # trial_ratio = 1. 
    #choice 1
    # tau_new = np.mod(imagtime[iloop-Ndimk,:] + dtau,taunum)
    # tau_new=taufold[imagtime[iloop-Ndimk,:] + dtau]
    tau_new=(imagtime[iloop-Ndimk,:]+ dtau)%taunum

    #choice 2, seems to be wrong
    # tau_new = imagtime[iloop-Ndimk,:] + dtau
    # if tau_new>=taunum or tau_new<0:
    #     accept=0
    #choice3
    # tau_new=np.random.randint(0,taunum)
    # accept =1
    return (tau_new, accept)


@jit(nopython=True)
def Trialstep0_l(lmax):
    lnew=np.random.randint(0,lmax)
    trial_ratio=1; trialaccept=1
    return (lnew,trialaccept)

@jit(nopython=True)
def Give_new_K(momentum, K_new, iloop):
    tmomentum = np.copy(momentum)
    tmomentum[iloop,:] = K_new  # this is trial configuration X_{new}=momentum
    return tmomentum

@jit(nopython=True)
def Give_new_tau(imagtime, tau_new, iloop,Ndimk):
    timagtime = np.copy(imagtime)
    timagtime[iloop-Ndimk,:] = tau_new  
    return timagtime


def readDMFT(dir): # read DMFT Sigma and G.
    # filename1='../files_variational/{}_{}_{}/Sig.out.{}'.format(B,U,T,index)
    # filename2='../files_variational/{}_{}_{}/Sig.OCA.{}'.format(B,U,T,index)
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


def sym_ave(Pval,knum,opt):
    '''
    This function take the average over symmetrical k points.
    '''
    if opt==1:
        power=1
    else:
        power=2
    Pval_averaged=np.zeros_like(Pval)
    for kx in np.arange(knum):
        for ky in np.arange(knum):
            for kz in np.arange(knum):
                all_sym_kpoints=lib.sym_mapping(kx,ky,kz,knum)
                count=0
                for q in all_sym_kpoints:
                    Pval_averaged[:,kx,ky,kz]+=Pval[:,q[0],q[1],q[2]]*(q[3]**power)
                    count+=1
                Pval_averaged[:,kx,ky,kz]/=count

    return Pval_averaged









# below are a few test diagrams.
def sig3(G11_tau,G22_tau,knum,nfreq,U,beta): # calculate all 3rd order diagrams

    '''
    This function packs all 3rd order diagrams. and the result is in freq space.
    '''
    # do check those 3rd order diagrams.
    Q11_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G22_tau,G11_tau,0)#Q=G_{s',-k}(tau)*G_{s,k+q}(tau)
    # Q12_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, -G12_tau,G12_tau,1)# Note: G12_-k=-G12_k!
    # Q22_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G11_tau,G22_tau,0)
    # R11_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G22_tau,G11_tau,0)#R=G_{s',k}(-tau)*G_{s,k+q}(tau)
    # R12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G12_tau,G12_tau,1)
    # R22_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G11_tau,G22_tau,0)
    # P22_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G22_tau,G22_tau,0)
    # P12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G12_tau,G12_tau,1)
    # Note1: Polarization P contains 2 propagators with same spin. But this is not the case for 3rd order.
    # Note2: Q11,Q12,R11,R12 are all symmetric in k space. see proof in '240126 third order diagram'
    #FT
    # R11_iom=fft.fast_ift_boson(R11_tau,beta)
    # R22_iom=fft.fast_ift_boson(R22_tau,beta)
    # R12_iom=fft.fast_ift_boson(R12_tau,beta)
    Q11_iom=fft.fast_ift_boson(Q11_tau,beta)
    # Q22_iom=fft.fast_ift_boson(Q22_tau,beta)
    # Q12_iom=fft.fast_ift_boson(Q12_tau,beta)
    #definitions and notations according to qualifier paper. indices are: 111,121,122,112. 

    # B_111_tau=fft.precalc_C(R11_iom,R11_iom,beta)
    # B_121_tau=fft.precalc_C(R12_iom,R12_iom,beta)
    # B_112_tau=fft.precalc_C(R11_iom,R12_iom,beta)
    # B_122_tau=fft.precalc_C(R12_iom,R22_iom,beta)
    A_111_tau=fft.precalc_C(Q11_iom,Q11_iom,beta)
    # A_121_tau=fft.precalc_C(Q12_iom,Q12_iom,beta)
    # A_112_tau=fft.precalc_C(Q11_iom,Q12_iom,beta)
    # A_122_tau=fft.precalc_C(Q12_iom,Q22_iom,beta)
    #precalcsig has the factor. (-1)*U**2/knum**3. actually factor needed is U**3. need extra -U.
    # Note: calculations below are simplified using symmetries of k, tau, and spin. for details, see '240126 third order diagram'.
    Sig3_1_111=-U*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,11,-G11_tau,A_111_tau,beta,U,0 )
    # Sig3_1_121=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,-G11_tau,A_121_tau,beta,U,0 )
    # Sig3_1_112=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,A_112_tau,beta,U,1 )# check here.
    # Sig3_1_122=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,A_122_tau,beta,U,1 )

    # Sig3_2_111=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,G22_tau,B_111_tau,beta,U,0 )
    # Sig3_2_121=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,G22_tau,B_121_tau,beta,U,0 )
    # Sig3_2_112=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,B_112_tau,beta,U,1 )
    # Sig3_2_122=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,B_122_tau,beta,U,1 )



    Sig3_11=Sig3_1_111#+Sig3_2_111+Sig3_1_121+Sig3_2_121
    # Sig3_12=Sig3_1_112+Sig3_2_112+Sig3_1_122+Sig3_2_122

    return Sig3_11#,Sig3_12

def sig3_2(G11_tau,G22_tau,knum,nfreq,U,beta): # calculate all 3rd order diagrams

    '''
    This function packs all 3rd order diagrams. and the result is in freq space.
    '''
    # do check those 3rd order diagrams.
    Q11_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G22_tau,G11_tau,0)#Q=G_{s',-k}(tau)*G_{s,k+q}(tau)
    # Q12_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, -G12_tau,G12_tau,1)# Note: G12_-k=-G12_k!
    # Q22_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G11_tau,G22_tau,0)
    R11_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G22_tau,G11_tau,0)#R=G_{s',k}(-tau)*G_{s,k+q}(tau)
    # R12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G12_tau,G12_tau,1)
    # R22_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G11_tau,G22_tau,0)
    # P22_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G22_tau,G22_tau,0)
    # P12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G12_tau,G12_tau,1)
    # Note1: Polarization P contains 2 propagators with same spin. But this is not the case for 3rd order.
    # Note2: Q11,Q12,R11,R12 are all symmetric in k space. see proof in '240126 third order diagram'
    #FT
    R11_iom=fft.fast_ift_boson(R11_tau,beta)
    # R22_iom=fft.fast_ift_boson(R22_tau,beta)
    # R12_iom=fft.fast_ift_boson(R12_tau,beta)
    Q11_iom=fft.fast_ift_boson(Q11_tau,beta)
    # Q22_iom=fft.fast_ift_boson(Q22_tau,beta)
    # Q12_iom=fft.fast_ift_boson(Q12_tau,beta)
    #definitions and notations according to qualifier paper. indices are: 111,121,122,112. 

    B_111_tau=fft.precalc_C(R11_iom,R11_iom,beta)
    # B_121_tau=fft.precalc_C(R12_iom,R12_iom,beta)
    # B_112_tau=fft.precalc_C(R11_iom,R12_iom,beta)
    # B_122_tau=fft.precalc_C(R12_iom,R22_iom,beta)
    A_111_tau=fft.precalc_C(Q11_iom,Q11_iom,beta)
    # A_121_tau=fft.precalc_C(Q12_iom,Q12_iom,beta)
    # A_112_tau=fft.precalc_C(Q11_iom,Q12_iom,beta)
    # A_122_tau=fft.precalc_C(Q12_iom,Q22_iom,beta)
    #precalcsig has the factor. (-1)*U**2/knum**3. actually factor needed is U**3. need extra -U.
    # Note: calculations below are simplified using symmetries of k, tau, and spin. for details, see '240126 third order diagram'.
    Sig3_1_111=-U*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,11,-G11_tau,A_111_tau,beta,U,0 )
    Sig3_1iom=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,-G11_tau,A_111_tau,beta,U,0 )
    # Sig3_1_121=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,-G11_tau,A_121_tau,beta,U,0 )
    # Sig3_1_112=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,A_112_tau,beta,U,1 )# check here.
    # Sig3_1_122=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,A_122_tau,beta,U,1 )

    Sig3_2_111=-U*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,11,G22_tau,B_111_tau,beta,U,0 )
    Sig3_2iom=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,G22_tau,B_111_tau,beta,U,0 )
    # Sig3_2_121=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,G22_tau,B_121_tau,beta,U,0 )
    # Sig3_2_112=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,B_112_tau,beta,U,1 )
    # Sig3_2_122=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,B_122_tau,beta,U,1 )



    Sig3_11=Sig3_2_111
    return Sig3_11#,Sig3_12





def sig3_1_122(G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta): # calculate all 3rd order diagrams

    '''
    This function packs all 3rd order diagrams. and the result is in freq space.
    '''
    Q12_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,12, G12_tau,G12_tau,1)# Note: G12_-k=-G12_k!
    Q22_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G11_tau,G22_tau,0)
    Q22_iom=fft.fast_ift_boson(Q22_tau,beta)
    Q12_iom=fft.fast_ift_boson(Q12_tau,beta)
    A_122_tau=fft.precalc_C(Q12_iom,Q22_iom,beta)
    Sig3_1_122=-U*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,12,G12_tau,A_122_tau,beta,U,1 )

    Sig3_12=Sig3_1_122#+Sig3_2_111+Sig3_1_121+Sig3_2_121

    return Sig3_12#,Sig3_12

def sig3_1_112(G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta): # calculate all 3rd order diagrams

    '''
    This function packs all 3rd order diagrams. and the result is in freq space.
    '''
    Q12_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,12, G12_tau,G12_tau,1)# Note: G12_-k=-G12_k!
    Q22_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G11_tau,G22_tau,0)
    Q11_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G22_tau,G11_tau,0)
    Q22_iom=fft.fast_ift_boson(Q22_tau,beta)
    Q12_iom=fft.fast_ift_boson(Q12_tau,beta)
    Q11_iom=fft.fast_ift_boson(Q11_tau,beta)
    A_112_tau=fft.precalc_C(Q11_iom,Q12_iom,beta)
    Sig3_1_112=-U*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,12,G12_tau,A_112_tau,beta,U,1 )

    Sig3_12=Sig3_1_112#+Sig3_2_111+Sig3_1_121+Sig3_2_121

    return Sig3_12#,Sig3_12

def sig3_1_121(G11_tau,G12_tau,knum,nfreq,U,beta): 
    Q12_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,12, G12_tau,G12_tau,1)
    Q12_iom=fft.fast_ift_boson(Q12_tau,beta)
    A_121_tau=fft.precalc_C(Q12_iom,Q12_iom,beta)
    Sig3_1_121=-U*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,11,-G11_tau,A_121_tau,beta,U,0 )
    Sig3_11=Sig3_1_121

    return Sig3_11#,Sig3_12


def sig2(G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta): # calculate all 2nd order diagrams
    '''
    This function packs all 2nd order diagrams. and the result is in freq space.
    '''
    # G11_tau=fft.fermion_fft_diagG(knum,G11_iom,beta,SigDMFT1-B,mu)
    # G12_tau=fft.fast_ft_fermion(G12_iom,beta)
    P22_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G22_tau,G22_tau,0)
    # P12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G12_tau,G12_tau,1)
    Sig2_11=mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,11, G11_tau,P22_tau,beta,U,0)
    # Sig2_12=mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12, G12_tau,P12_tau,beta,U,1)
    # Sig2_22=-Sig2_11.conjugate()
    # Sig2tau_11=mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,11, G11_tau,P22_tau,beta,U,0)
    return Sig2_11#,Sig2_12#,Sig2tau_11

def sig2offdiag(G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta): # calculate all 2nd order diagrams
    '''
    This function packs all 2nd order diagrams. and the result is in freq space.
    '''
    # G11_tau=fft.fermion_fft_diagG(knum,G11_iom,beta,SigDMFT1-B,mu)
    # G12_tau=fft.fast_ft_fermion(G12_iom,beta)
    # P22_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G22_tau,G22_tau,0)
    P12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12,G12_tau,G12_tau,1)
    # Sig2_11=mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,11, G11_tau,P22_tau,beta,U,0)
    Sig2_12=mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,12, G12_tau,P12_tau,beta,U,1)
    # Sig2_22=-Sig2_11.conjugate()
    # Sig2tau_11=mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,11, G11_tau,P22_tau,beta,U,0)
    return Sig2_12#,Sig2_12#,Sig2tau_11




def P12(G12_tau,knum,nfreq,U,beta): # calculate all 2nd order diagrams
    '''
    This function packs all 2nd order diagrams. and the result is in freq space.
    '''
    P12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12,G12_tau,G12_tau,1)
    return P12_tau

def Q12(G12_tau,knum,nfreq): 
    Q12_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,12, G12_tau,G12_tau,1)# Note: G12(-tau)=-G12(beta-tau)=G12(tau)

    return Q12_tau#,Sig3_12

def R12(G12_tau,knum,nfreq): 
    R12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12, G12_tau,G12_tau,1)# Note: G12(-tau)=-G12(beta-tau)=G12(tau)

    return R12_tau#,Sig3_12

def sig4_2(G11_tau,G22_tau,knum,nfreq,U,beta): # calculate all 3rd order diagrams

    '''
    This function packs all 3rd order diagrams. and the result is in freq space.
    '''
    # do check those 3rd order diagrams.
    Q11_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G22_tau,G11_tau,0)
    R11_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G22_tau,G11_tau,0)
    #FT
    R11_iom=fft.fast_ift_boson(R11_tau,beta)
    # R22_iom=fft.fast_ift_boson(R22_tau,beta)
    # R12_iom=fft.fast_ift_boson(R12_tau,beta)
    Q11_iom=fft.fast_ift_boson(Q11_tau,beta)
    # Q22_iom=fft.fast_ift_boson(Q22_tau,beta)
    # Q12_iom=fft.fast_ift_boson(Q12_tau,beta)
    #definitions and notations according to qualifier paper. indices are: 111,121,122,112. 

    B_111_tau=fft.precalc_ladder3(R11_iom,R11_iom,R11_iom,beta)
    # B_121_tau=fft.precalc_C(R12_iom,R12_iom,beta)
    # B_112_tau=fft.precalc_C(R11_iom,R12_iom,beta)
    # B_122_tau=fft.precalc_C(R12_iom,R22_iom,beta)
    A_111_tau=fft.precalc_ladder3(Q11_iom,Q11_iom,Q11_iom,beta)
    # A_121_tau=fft.precalc_C(Q12_iom,Q12_iom,beta)
    # A_112_tau=fft.precalc_C(Q11_iom,Q12_iom,beta)
    # A_122_tau=fft.precalc_C(Q12_iom,Q22_iom,beta)
    #precalcsig has the factor. (-1)*U**2/knum**3. actually factor needed is U**3. need extra -U.
    # Note: calculations below are simplified using symmetries of k, tau, and spin. for details, see '240126 third order diagram'.
    Sig3_1_111=U**2*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,11,-G11_tau,A_111_tau,beta,U,0 )
    # Sig3_1iom=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,-G11_tau,A_111_tau,beta,U,0 )
    # Sig3_1_121=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,-G11_tau,A_121_tau,beta,U,0 )
    # Sig3_1_112=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,A_112_tau,beta,U,1 )# check here.
    # Sig3_1_122=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,A_122_tau,beta,U,1 )

    Sig3_2_111=U**2*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,11,G22_tau,B_111_tau,beta,U,0 )
    # Sig3_2iom=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,G22_tau,B_111_tau,beta,U,0 )
    # Sig3_2_121=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,G22_tau,B_121_tau,beta,U,0 )
    # Sig3_2_112=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,B_112_tau,beta,U,1 )
    # Sig3_2_122=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,B_122_tau,beta,U,1 )



    Sig3_11=Sig3_2_111#+Sig3_1_111#+Sig3_1_121+Sig3_2_121
    # Sig3_12=Sig3_1_112+Sig3_2_112+Sig3_1_122+Sig3_2_122
    return Sig3_11#,Sig3_12



def sig4_1(G11_tau,G22_tau,knum,nfreq,U,beta): # calculate all 3rd order diagrams

    '''
    This function packs all 3rd order diagrams. and the result is in freq space.
    '''
    # do check those 3rd order diagrams.
    Q11_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G22_tau,G11_tau,0)#Q=G_{s',-k}(tau)*G_{s,k+q}(tau)
    # Q12_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, -G12_tau,G12_tau,1)# Note: G12_-k=-G12_k!
    # Q22_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G11_tau,G22_tau,0)
    # R11_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G22_tau,G11_tau,0)#R=G_{s',k}(-tau)*G_{s,k+q}(tau)
    # R12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G12_tau,G12_tau,1)
    # R22_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G11_tau,G22_tau,0)
    # P22_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G22_tau,G22_tau,0)
    # P12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G12_tau,G12_tau,1)
    # Note1: Polarization P contains 2 propagators with same spin. But this is not the case for 3rd order.
    # Note2: Q11,Q12,R11,R12 are all symmetric in k space. see proof in '240126 third order diagram'
    #FT
    # R11_iom=fft.fast_ift_boson(R11_tau,beta)
    # R22_iom=fft.fast_ift_boson(R22_tau,beta)
    # R12_iom=fft.fast_ift_boson(R12_tau,beta)
    Q11_iom=fft.fast_ift_boson(Q11_tau,beta)
    # Q22_iom=fft.fast_ift_boson(Q22_tau,beta)
    # Q12_iom=fft.fast_ift_boson(Q12_tau,beta)
    #definitions and notations according to qualifier paper. indices are: 111,121,122,112. 

    # B_111_tau=fft.precalc_C(R11_iom,R11_iom,beta)
    # B_121_tau=fft.precalc_C(R12_iom,R12_iom,beta)
    # B_112_tau=fft.precalc_C(R11_iom,R12_iom,beta)
    # B_122_tau=fft.precalc_C(R12_iom,R22_iom,beta)
    A_111_tau=fft.precalc_ladder3(Q11_iom,Q11_iom,Q11_iom,beta)
    # A_121_tau=fft.precalc_C(Q12_iom,Q12_iom,beta)
    # A_112_tau=fft.precalc_C(Q11_iom,Q12_iom,beta)
    # A_122_tau=fft.precalc_C(Q12_iom,Q22_iom,beta)
    #precalcsig has the factor. (-1)*U**2/knum**3. actually factor needed is U**3. need extra -U.
    # Note: calculations below are simplified using symmetries of k, tau, and spin. for details, see '240126 third order diagram'.
    Sig3_1_111=U**2*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,11,-G11_tau,A_111_tau,beta,U,0 )
    # Sig3_1_121=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,-G11_tau,A_121_tau,beta,U,0 )
    # Sig3_1_112=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,A_112_tau,beta,U,1 )# check here.
    # Sig3_1_122=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,A_122_tau,beta,U,1 )

    # Sig3_2_111=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,G22_tau,B_111_tau,beta,U,0 )
    # Sig3_2_121=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,G22_tau,B_121_tau,beta,U,0 )
    # Sig3_2_112=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,B_112_tau,beta,U,1 )
    # Sig3_2_122=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,B_122_tau,beta,U,1 )



    Sig3_11=Sig3_1_111#+Sig3_2_111+Sig3_1_121+Sig3_2_121
    # Sig3_12=Sig3_1_112+Sig3_2_112+Sig3_1_122+Sig3_2_122

    return Sig3_11#,Sig3_12






def allsig3_1(G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta,iftau=1):

    '''
    This function packs all 3rd order diagrams. and the result is in freq space.
    '''
    # do check those 3rd order diagrams.
    Q11_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G22_tau,G11_tau,0)#Q=G_{s',-k}(tau)*G_{s,k+q}(tau)
    Q12_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,12, G12_tau,G12_tau,1)# Note: G12_-k=-G12_k!
    Q22_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G11_tau,G22_tau,0)
    # R11_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G22_tau,G11_tau,0)#R=G_{s',k}(-tau)*G_{s,k+q}(tau)
    # R12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G12_tau,G12_tau,1)
    # R22_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G11_tau,G22_tau,0)
    # P22_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G22_tau,G22_tau,0)
    # P12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G12_tau,G12_tau,1)
    # Note1: Polarization P contains 2 propagators with same spin. But this is not the case for 3rd order.
    # Note2: Q11,Q12,R11,R12 are all symmetric in k space. see proof in '240126 third order diagram'
    #FT
    # R11_iom=fft.fast_ift_boson(R11_tau,beta)
    # R22_iom=fft.fast_ift_boson(R22_tau,beta)
    # R12_iom=fft.fast_ift_boson(R12_tau,beta)
    Q11_iom=fft.fast_ift_boson(Q11_tau,beta)
    Q22_iom=fft.fast_ift_boson(Q22_tau,beta)
    Q12_iom=fft.fast_ift_boson(Q12_tau,beta)
    #definitions and notations according to qualifier paper. indices are: 111,121,122,112. 

    # B_111_tau=fft.precalc_C(R11_iom,R11_iom,beta)
    # B_121_tau=fft.precalc_C(R12_iom,R12_iom,beta)
    # B_112_tau=fft.precalc_C(R11_iom,R12_iom,beta)
    # B_122_tau=fft.precalc_C(R12_iom,R22_iom,beta)
    A_111_tau=fft.precalc_C(Q11_iom,Q11_iom,beta)
    A_121_tau=fft.precalc_C(Q12_iom,Q12_iom,beta)
    A_112_tau=fft.precalc_C(Q11_iom,Q12_iom,beta)
    A_122_tau=fft.precalc_C(Q12_iom,Q22_iom,beta)
    #precalcsig has the factor. (-1)*U**2/knum**3. actually factor needed is U**3. need extra -U.
    # Note: calculations below are simplified using symmetries of k, tau, and spin. for details, see '240126 third order diagram'.
    if iftau==1:
        Sig3_1_111=-U*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,11,-G11_tau,A_111_tau,beta,U,0 )
        Sig3_1_121=-U*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,11,-G11_tau,A_121_tau,beta,U,0 )
        Sig3_1_112=-U*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,12,G12_tau,A_112_tau,beta,U,1 )# check here.
        Sig3_1_122=-U*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,12,G12_tau,A_122_tau,beta,U,1 )
    elif iftau==0:
        Sig3_1_111=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,-G11_tau,A_111_tau,beta,U,0 )
        Sig3_1_121=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,-G11_tau,A_121_tau,beta,U,0 )
        Sig3_1_112=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,A_112_tau,beta,U,1 )# check here.
        Sig3_1_122=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,A_122_tau,beta,U,1 )        
    # Sig3_2_111=-U*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,11,G22_tau,B_111_tau,beta,U,0 )
    # Sig3_2_121=-U*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,11,G22_tau,B_121_tau,beta,U,0 )
    # Sig3_2_112=-U*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,12,G12_tau,B_112_tau,beta,U,1 )
    # Sig3_2_122=-U*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,12,G12_tau,B_122_tau,beta,U,1 )



    Sig3_11=Sig3_1_111+Sig3_1_121
    Sig3_12=Sig3_1_112+Sig3_1_122
    return Sig3_11,Sig3_12

def allsig3_2(G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta,iftau=1):

    '''
    This function packs all 3rd order diagrams. and the result is in freq space.
    '''
    # do check those 3rd order diagrams.
    # Q11_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G22_tau,G11_tau,0)#Q=G_{s',-k}(tau)*G_{s,k+q}(tau)
    # Q12_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, -G12_tau,G12_tau,1)# Note: G12_-k=-G12_k!
    # Q22_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G11_tau,G22_tau,0)
    R11_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G22_tau,G11_tau,0)#R=G_{s',k}(-tau)*G_{s,k+q}(tau)
    R12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12, G12_tau,G12_tau,1)
    R22_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G11_tau,G22_tau,0)
    # P22_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G22_tau,G22_tau,0)
    # P12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G12_tau,G12_tau,1)
    # Note1: Polarization P contains 2 propagators with same spin. But this is not the case for 3rd order.
    # Note2: Q11,Q12,R11,R12 are all symmetric in k space. see proof in '240126 third order diagram'
    #FT
    R11_iom=fft.fast_ift_boson(R11_tau,beta)
    R22_iom=fft.fast_ift_boson(R22_tau,beta)
    R12_iom=fft.fast_ift_boson(R12_tau,beta)
    # Q11_iom=fft.fast_ift_boson(Q11_tau,beta)
    # Q22_iom=fft.fast_ift_boson(Q22_tau,beta)
    # Q12_iom=fft.fast_ift_boson(Q12_tau,beta)
    #definitions and notations according to qualifier paper. indices are: 111,121,122,112. 

    B_111_tau=fft.precalc_C(R11_iom,R11_iom,beta)
    B_121_tau=fft.precalc_C(R12_iom,R12_iom,beta)
    B_112_tau=fft.precalc_C(R11_iom,R12_iom,beta)
    B_122_tau=fft.precalc_C(R12_iom,R22_iom,beta)
    # A_111_tau=fft.precalc_C(Q11_iom,Q11_iom,beta)
    # A_121_tau=fft.precalc_C(Q12_iom,Q12_iom,beta)
    # A_112_tau=fft.precalc_C(Q11_iom,Q12_iom,beta)
    # A_122_tau=fft.precalc_C(Q12_iom,Q22_iom,beta)
    #precalcsig has the factor. (-1)*U**2/knum**3. actually factor needed is U**3. need extra -U.
    # Note: calculations below are simplified using symmetries of k, tau, and spin. for details, see '240126 third order diagram'.
    # Sig3_1_111=-U*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,11,-G11_tau,A_111_tau,beta,U,0 )
    # Sig3_1iom=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,-G11_tau,A_111_tau,beta,U,0 )
    # Sig3_1_121=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,-G11_tau,A_121_tau,beta,U,0 )
    # Sig3_1_112=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,A_112_tau,beta,U,1 )# check here.
    # Sig3_1_122=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,A_122_tau,beta,U,1 )
    if iftau==1:
        Sig3_2_111=-U*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,11,G22_tau,B_111_tau,beta,U,0 )
        Sig3_2_121=-U*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,11,G22_tau,B_121_tau,beta,U,0 )
        Sig3_2_112=-U*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,12,G12_tau,B_112_tau,beta,U,1 )
        Sig3_2_122=-U*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,12,G12_tau,B_122_tau,beta,U,1 )
    elif iftau==0:
        Sig3_2_111=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,G22_tau,B_111_tau,beta,U,0 )
        Sig3_2_121=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,G22_tau,B_121_tau,beta,U,0 )
        Sig3_2_112=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,B_112_tau,beta,U,1 )
        Sig3_2_122=-U*mpi_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,B_122_tau,beta,U,1 )

    Sig3_11=Sig3_2_111+Sig3_2_121
    Sig3_12=Sig3_2_112+Sig3_2_122
    return Sig3_11,Sig3_12

def allsig4_1(G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta,iftau=1): # calculate all 3rd order diagrams

    '''
    This function packs all 3rd order diagrams. and the result is in freq space.
    '''
    # do check those 3rd order diagrams.
    Q11_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G22_tau,G11_tau,0)#Q=G_{s',-k}(tau)*G_{s,k+q}(tau)
    Q12_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,12, G12_tau,G12_tau,1)# Note: G12_-k=-G12_k!
    Q22_tau=mpi_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G11_tau,G22_tau,0)
    Q11_iom=fft.fast_ift_boson(Q11_tau,beta)
    Q22_iom=fft.fast_ift_boson(Q22_tau,beta)
    Q12_iom=fft.fast_ift_boson(Q12_tau,beta)
    A_1111_tau=fft.precalc_ladder3(Q11_iom,Q11_iom,Q11_iom,beta)
    A_1121_tau=fft.precalc_ladder3(Q11_iom,Q12_iom,Q12_iom,beta)*2# A_1211_tau=fft.precalc_ladder3(Q12_iom,Q12_iom,Q11_iom,beta)=A1121
    A_1221_tau=fft.precalc_ladder3(Q12_iom,Q22_iom,Q12_iom,beta)

    A_1112_tau=fft.precalc_ladder3(Q11_iom,Q11_iom,Q12_iom,beta)
    A_1212_tau=fft.precalc_ladder3(Q12_iom,Q12_iom,Q12_iom,beta)
    A_1222_tau=fft.precalc_ladder3(Q12_iom,Q22_iom,Q22_iom,beta)
    A_1122_tau=fft.precalc_ladder3(Q11_iom,Q12_iom,Q22_iom,beta)
    #precalcsig has the factor. *U**3/knum**3. actually factor needed is U**3. need extra U**2.
    # Note: calculations below are simplified using symmetries of k, tau, and spin. for details, see '240126 third order diagram'.
    Sig4_1_11=U**2*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,11,-G11_tau,A_1111_tau+A_1121_tau+A_1221_tau,beta,U,0 )
    Sig4_1_12=U**2*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,12,G12_tau,A_1112_tau+A_1212_tau+A_1222_tau+A_1122_tau,beta,U,1 )
    return Sig4_1_11,Sig4_1_12

def allsig4_2(G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta,iftau=1): # calculate all 3rd order diagrams

    '''
    This function packs all 3rd order diagrams. and the result is in freq space.
    '''
    # do check those 3rd order diagrams.
    R11_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G22_tau,G11_tau,0)#R=G_{s',k}(-tau)*G_{s,k+q}(tau)
    R12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12, G12_tau,G12_tau,1)
    R22_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G11_tau,G22_tau,0)
    #FT
    R11_iom=fft.fast_ift_boson(R11_tau,beta)
    R22_iom=fft.fast_ift_boson(R22_tau,beta)
    R12_iom=fft.fast_ift_boson(R12_tau,beta)
    #definitions and notations according to qualifier paper. indices are: 111,121,122,112. 

    B_1111_tau=fft.precalc_ladder3(R11_iom,R11_iom,R11_iom,beta)
    B_1121_tau=fft.precalc_ladder3(R11_iom,R12_iom,R12_iom,beta)*2# B_1211_tau=fft.precalc_ladder3(R12_iom,R12_iom,R11_iom,beta)=B1121
    B_1221_tau=fft.precalc_ladder3(R12_iom,R22_iom,R12_iom,beta)

    B_1112_tau=fft.precalc_ladder3(R11_iom,R11_iom,R12_iom,beta)
    B_1212_tau=fft.precalc_ladder3(R12_iom,R12_iom,R12_iom,beta)
    B_1222_tau=fft.precalc_ladder3(R12_iom,R22_iom,R22_iom,beta)
    B_1122_tau=fft.precalc_ladder3(R11_iom,R12_iom,R22_iom,beta)



    Sig4_2_11=U**2*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,11,G22_tau,B_1111_tau+B_1121_tau+B_1221_tau,beta,U,0 )#
    Sig4_2_12=U**2*mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,12,G12_tau,B_1112_tau+B_1212_tau+B_1222_tau+B_1122_tau,beta,U,1 )
    return Sig4_2_11,Sig4_2_12
