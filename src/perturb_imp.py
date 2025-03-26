import matplotlib.pyplot as plt 
import numpy as np
from scipy.interpolate import interp1d
import os,sys,subprocess
import time
import hilbert
from perturb_lib import *
# import fft_convolution as conv
# this code is written to reproduce a few leading order of 

def fermion_fft(Gk,beta):
    N=np.shape(Gk)[0]
    # Gktau=np.fft.fft(Gk,axis=0)*np.exp(1j*(N-1)*np.pi*np.arange(N)/N-1j*(2*np.arange(N)-N+1)*np.pi*0.5/N)[:,None,None,None]
    Gktau=np.fft.fft(Gk*np.exp(-1j*(2*np.arange(N)-N+1)*np.pi*0.5/N),axis=0)*np.exp(1j*(N-1)*np.pi*np.arange(N)/N)/beta
    return Gktau

def fermion_ifft(Gk,beta):# same way back. in fft, we fft then shift; in ifft, we shift back then fft.
    N=np.shape(Gk)[0]
    Gkiom=np.fft.ifft(Gk*np.exp(-1j*(N-1)*np.pi*np.arange(N)/N))*np.exp(+1j*(2*np.arange(N)-N+1)*np.pi*0.5/N)*beta
    return Gkiom

def boson_fft(Pk,beta):
    N=np.shape(Pk)[0]
    Pktau=np.fft.fft(Pk*np.exp(-1j*(2*np.arange(N)-N)*np.pi*0.5/N))*np.exp(1j*(N)*np.pi*np.arange(N)/N)/beta
    return Pktau

def boson_ifft(Pk,beta):
    N=np.shape(Pk)[0]
    Pkiom=np.fft.ifft(Pk*np.exp(-1j*(N)*np.pi*np.arange(N)/N))*np.exp(+1j*(2*np.arange(N)-N)*np.pi*0.5/N)*beta
    return Pkiom

def interp_tau(P,delta_tau):
    '''
    This function is used for interpolation of functions in tau space. since sometimes we need even points and sometimes odd points to perform the correct FFT.
    delta_tau is usually chosen as +1 or -1, which means add or subtract 1 tau point.
    '''
    nf=np.shape(P)[0]
    # old and new tau grid, in unit of beta
    old_tau=(np.arange(nf)+1/2)/nf
    new_tau=(np.arange(nf+delta_tau)+1/2)/(nf+delta_tau)
    f=interp1d(old_tau,P,fill_value="extrapolate")
    Pnew=f(new_tau)
    return Pnew

def pertimp():
    T=0.01
    U=2.0
    dosfile='DOS_3D.dat'
    sigfile='{}_{}.dat'.format(U,T)
    mu=U/2.
    beta=1/T
    # read dos and generate Hilbert transformation
    x, Di = np.loadtxt(dosfile).T
    # plt.plot(x,Di)
    # plt.show()
    W = hilbert.Hilb(x,Di)

    sigma=np.loadtxt(sigfile)
    om=sigma[:,0]
    n=om.size
    print('# of frequencies: ',n)
    allom=(2*np.arange(4*n)+1-n*4)*np.pi/beta
    if om[0]/(np.pi/beta)>1.01 or om[0]/(np.pi/beta)<0.99:
        print('seems the temperature does not match!')
        return 0
    sigA_short=sigma[:,1]+1j*sigma[:,2]
    sigB_short=sigma[:,3]+1j*sigma[:,4]
    sigA=ext_sig(sigA_short)
    sigB=ext_sig(sigB_short)
    # # take a glance of my sigma!
    # plt.plot(sigA.real,label='sigA real')
    # plt.plot(sigA.imag,label='sigA imag')
    # plt.plot(sigB.real,label='sigB real')
    # plt.plot(sigB.real,label='sigB imag')
    # plt.legend()
    # plt.show()

    G_A=np.zeros_like(sigA,dtype=complex)
    G_B=np.zeros_like(sigB,dtype=complex)
    for i in np.arange(4*n):
        z_A=1j*allom[i]+mu-sigA[i]
        z_B=1j*allom[i]+mu-sigB[i]
        G_A[i]=W(z_A)
        G_B[i]=W(z_B)
    # check the generated impurity green function!
    plt.plot(G_A.real,label='G_A real')
    plt.plot(G_A.imag,label='G_A imag')
    plt.plot(G_B.real,label='G_B real')
    plt.plot(G_B.imag,label='G_B imag')
    plt.legend()
    plt.show()
    
    #calculate P. P is calculated on Boson matsubara freq points!
    P_A=np.zeros(n*2,dtype=complex)
    P_B=np.zeros(n*2,dtype=complex)
    for i in np.arange(n*2):# here, i is the index of boson matsubara freq.
        P_A[i]=T*np.sum(G_A[n:n*3]*G_A[n+i-n:n*3+i-n])
        P_B[i]=T*np.sum(G_B[n:n*3]*G_B[n+i-n:n*3+i-n])
    # take a look at P
    plt.plot(P_A.real,label='P_A real')
    plt.plot(P_A.imag,label='P_A imag')
    plt.plot(P_B.real,label='P_B real')
    plt.plot(P_B.imag,label='P_B imag')
    plt.legend()
    plt.show()
    
    #calculate sig. sig is calculate on fermion matsubara freq points!
    sigp_A=np.zeros(n*2,dtype=complex)
    sigp_B=np.zeros(n*2,dtype=complex)
    for i in np.arange(n*2):# here, i is the index of fermion matsubara freq.
        sigp_A[i]=-U**2*T*np.sum(P_B*G_A[n+i-n:n*3+i-n])
        sigp_B[i]=-U**2*T*np.sum(P_A*G_B[n+i-n:n*3+i-n])
    plt.plot(sigp_A.real,label='sigp_A real')
    plt.plot(sigp_A.imag,label='sigp_A imag')
    plt.plot(sigp_B.real,label='sigp_B real')
    plt.plot(sigp_B.imag,label='sigp_B imag')
    plt.legend()
    plt.show()
    # corrected sigma
    plt.plot(sigA_short[-1].real+sigp_A.real,label='sig_bf_pert_A real')
    plt.plot(sigp_A.imag,label='sig_bf_pert_A imag')
    plt.plot(sigB_short[-1].real+sigp_B.real,label='sig_bf_pert_B real')
    plt.plot(sigp_B.imag,label='sig_bf_pert_B imag')
    plt.plot(sigA[n:n*3].real,label='sigA real')
    plt.plot(sigA[n:n*3].imag,label='sigA imag')
    plt.plot(sigB[n:n*3].real,label='sigB real')
    plt.plot(sigB[n:n*3].real,label='sigB imag')
    plt.legend()
    plt.show()
    return 0 
# sigtest=np.array([1+1j,2+2j,3+3j,4+4j])
# print(ext_sig(sigtest))



#updated to faster fft version.
def pertimp_func(G_A,delta_inf,beta,U,knum,order=2):
    T=1/beta
    n=int(G_A.size/2)
    N=2*n
    iom= 1j*(2*np.arange(2*n)+1-2*n)*np.pi/beta
    iOm= 1j*(2*np.arange(2*n+1)-2*n)*np.pi/beta
    # delta_inf=0
    epsk=calc_disp(knum)
    eps2=epsk**2
    G_A0=np.sum((iom+delta_inf)[:,None,None,None]/(iom[:,None,None,None]**2-delta_inf**2-eps2[None,:,:,:]),axis=(1,2,3))/knum**3
    tlist=(np.arange(N)+0.5)/N*beta
    
    alpha=np.sqrt(eps2+delta_inf**2)[None,:,:,:]
    GA_tau_diff=fermion_fft(G_A-G_A0,beta)
    GA_tau_ana=np.sum(-1/2*((1+delta_inf/alpha)*np.exp(-alpha*tlist[:,None,None,None])/(1+np.exp(-alpha*beta))+
                        (1-delta_inf/alpha)*np.exp(alpha*tlist[:,None,None,None])/(1+np.exp(alpha*beta))),axis=(1,2,3))/knum**3
    GA_tau=GA_tau_ana+GA_tau_diff

    G_B=-G_A.conjugate()
    G_B0=np.sum((iom-delta_inf)[:,None,None,None]/(iom[:,None,None,None]**2-delta_inf**2-eps2[None,:,:,:]),axis=(1,2,3))/knum**3
    GB_tau_diff=fermion_fft(G_B-G_B0,beta)
    GB_tau_ana=np.sum(-1/2*((1-delta_inf/alpha)*np.exp(-alpha*tlist[:,None,None,None])/(1+np.exp(-alpha*beta))+
                        (1+delta_inf/alpha)*np.exp(alpha*tlist[:,None,None,None])/(1+np.exp(alpha*beta))),axis=(1,2,3))/knum**3
    GB_tau=GB_tau_ana+GB_tau_diff# This is wrong? the analytic estimation
    GB_tau=GA_tau[::-1]#GB_tau_ana+GB_tau_diff
    # make sure GB is correct! also, check GBtau=GAtau[::-1] between 0 and beta.

    # GA_bf=fermion_fft(G_A,beta)
    PA_tau=-GA_tau[::-1]*GA_tau
    PB_tau=-GB_tau[::-1]*GB_tau
    #calculate sig. sig is calculate on fermion matsubara freq points!
    Sigp_A=np.zeros(n*2,dtype=complex)
    Sigp_B=np.zeros(n*2,dtype=complex)
    if order ==1:
        Sigp_A=(-1)*(-1)*U*(np.sum(G_B).real/beta+1/2)# only works for half filling.
        Sigp_B=(-1)*(-1)*U*(np.sum(G_A).real/beta+1/2)
        return Sigp_A,Sigp_B
    if order==2:
        SigA_tau=PB_tau*GA_tau*(-1)*U**2
    if order ==3:# only 111 part should be cancelled:
        QA_tau=GA_tau*GB_tau
        RA_tau=-GA_tau*GA_tau#B_tau[::-1]
        n=QA_tau.size

        RA_iom=boson_ifft(RA_tau,beta)
        QA_iom=boson_ifft(QA_tau,beta)
        BA_iom=RA_iom*RA_iom
        AA_iom=QA_iom*QA_iom
        BA_tau=boson_fft(BA_iom,beta)
        AA_tau=boson_fft(AA_iom,beta)
        SigA1_tau=-AA_tau*GB_tau[::-1]*U**3
        SigA2_tau=+BA_tau*GB_tau*U**3
        Sigp_A1=fermion_ifft(SigA1_tau,beta)
        Sigp_A2=fermion_ifft(SigA2_tau,beta)
        return Sigp_A1,Sigp_A2
    Sigp_A=fermion_ifft(SigA_tau,beta)
    Sigp_B=-Sigp_A.conjugate()
    return Sigp_A,Sigp_B


def pertimp_func_tau(GA_tau,beta,U,knum,order=2):
    T=1/beta
    # n=int(G_A.size/2)
    # N=2*n
    # iom= 1j*(2*np.arange(2*n)+1-2*n)*np.pi/beta
    # iOm= 1j*(2*np.arange(2*n+1)-2*n)*np.pi/beta
    # # delta_inf=0
    # epsk=calc_disp(knum)
    # eps2=epsk**2
    # G_A0=np.sum((iom+delta_inf)[:,None,None,None]/(iom[:,None,None,None]**2-delta_inf**2-eps2[None,:,:,:]),axis=(1,2,3))/knum**3
    # tlist=(np.arange(N)+0.5)/N*beta
    
    # alpha=np.sqrt(eps2+delta_inf**2)[None,:,:,:]
    # GA_tau_diff=fermion_fft(G_A-G_A0,beta)
    # GA_tau_ana=np.sum(-1/2*((1+delta_inf/alpha)*np.exp(-alpha*tlist[:,None,None,None])/(1+np.exp(-alpha*beta))+
    #                     (1-delta_inf/alpha)*np.exp(alpha*tlist[:,None,None,None])/(1+np.exp(alpha*beta))),axis=(1,2,3))/knum**3
    # GA_tau=GA_tau_ana+GA_tau_diff

    # G_B=-G_A.conjugate()
    # G_B0=np.sum((iom-delta_inf)[:,None,None,None]/(iom[:,None,None,None]**2-delta_inf**2-eps2[None,:,:,:]),axis=(1,2,3))/knum**3
    # GB_tau_diff=fermion_fft(G_B-G_B0,beta)
    # GB_tau_ana=np.sum(-1/2*((1-delta_inf/alpha)*np.exp(-alpha*tlist[:,None,None,None])/(1+np.exp(-alpha*beta))+
    #                     (1+delta_inf/alpha)*np.exp(alpha*tlist[:,None,None,None])/(1+np.exp(alpha*beta))),axis=(1,2,3))/knum**3
    # GB_tau=GB_tau_ana+GB_tau_diff# This is wrong? the analytic estimation
    GB_tau=GA_tau[::-1]#GB_tau_ana+GB_tau_diff
    # make sure GB is correct! also, check GBtau=GAtau[::-1] between 0 and beta.

    # GA_bf=fermion_fft(G_A,beta)
    PA_tau=-GA_tau[::-1]*GA_tau
    PB_tau=-GB_tau[::-1]*GB_tau
    #calculate sig. sig is calculate on fermion matsubara freq points!
    # Sigp_A=np.zeros(n*2,dtype=complex)
    # Sigp_B=np.zeros(n*2,dtype=complex)
    # if order ==1:
    #     Sigp_A=(-1)*(-1)*U*(np.sum(G_B).real/beta+1/2)# only works for half filling.
    #     Sigp_B=(-1)*(-1)*U*(np.sum(G_A).real/beta+1/2)
    #     return Sigp_A,Sigp_B
    if order==2:
        SigA_tau=PB_tau*GA_tau*(-1)*U**2
        return SigA_tau
    if order ==3:# only 111 part should be cancelled:
        QA_tau=GA_tau*GB_tau
        RA_tau=-GB_tau[::-1]*GA_tau#-GA_tau*GA_tau#B_tau[::-1]
        n=QA_tau.size

        RA_iom=boson_ifft(RA_tau,beta)
        QA_iom=boson_ifft(QA_tau,beta)
        BA_iom=RA_iom*RA_iom
        AA_iom=QA_iom*QA_iom
        BA_tau=boson_fft(BA_iom,beta)
        AA_tau=boson_fft(AA_iom,beta)
        SigA1_tau=AA_tau*(-GB_tau[::-1])*U**3
        SigA2_tau=+BA_tau*GB_tau*U**3
        return SigA1_tau,SigA2_tau
    if order==4:
        QA_tau=GA_tau*GB_tau
        RA_tau=-GA_tau*GA_tau#B_tau[::-1]   
        RA_iom=boson_ifft(RA_tau,beta)
        QA_iom=boson_ifft(QA_tau,beta)     
        BA_iom=RA_iom*RA_iom*RA_iom
        AA_iom=QA_iom*QA_iom*QA_iom
        BA_tau=boson_fft(BA_iom,beta)
        AA_tau=boson_fft(AA_iom,beta)
        PA_iom=boson_ifft(PA_tau,beta)
        PB_iom=boson_ifft(PB_tau,beta)
        RPA_ladder_A=PB_iom*PA_iom*PB_iom
        RPA_ladder_Atau=boson_fft(RPA_ladder_A,beta)
        SigA1_tau=+AA_tau*GB_tau[::-1]*U**4
        SigA2_tau=-BA_tau*GB_tau*U**4
        SigA5_tau=-RPA_ladder_Atau*GA_tau*U**4
        #
        #
        # ladders, RPA of order 3 and 4 are to be checked by MC.
        return SigA1_tau,SigA2_tau,SigA5_tau

