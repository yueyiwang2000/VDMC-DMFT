import numpy as np
from perturb_lib import *
import time
import matplotlib.pyplot as plt
from numba import jit, complex128
import perturb_lib
# from numba.types import float64, complex128
from numba import jit
# shift in k space

def G12_shift(G12,q,knum,opt):
    """
    opt==1 means shift with sign!
    """
    qx=q[0]
    qy=q[1]
    qz=q[2]
    kind_list = np.arange(knum)
    kxind, kyind, kzind = np.meshgrid(kind_list, kind_list, kind_list, indexing='ij')
    if opt==1:#with factor
        G_12_factor=(-1)**((np.mod(kxind + qx, knum)-(kxind+qx))/knum+(np.mod(kyind + qy, knum)-(kyind+qy))/knum+(np.mod(kzind + qz, knum)-(kzind+qz))/knum)
    else:
        G_12_factor=1.
    G12_kq = G_12_factor*G12[:, np.mod(kxind + qx, knum), np.mod(kyind + qy, knum), np.mod(kzind + qz, knum)]
    return G12_kq

def alpha_shift(alpha,q,knum):
    qx=q[0]
    qy=q[1]
    qz=q[2]
    kind_list = np.arange(knum)
    kxind, kyind, kzind = np.meshgrid(kind_list, kind_list, kind_list, indexing='ij')
    alpha_kq = alpha[ np.mod(kxind + qx, knum), np.mod(kyind + qy, knum), np.mod(kzind + qz, knum)]
    return alpha_kq

# stupid functions for test:

def stupid_ft_fermion(Gk,n,taulist,fermion_om,k):
    Gktau=np.zeros(2*n,dtype=complex)
    for i in np.arange(2*n):
        tau=taulist[i]
        Gktau[i]=np.sum(Gk[:,k[0],k[1],k[2]]*np.exp(-1j*fermion_om*tau))
    return Gktau

def stupid_ift_fermion(Gktau,n,beta,fermion_om):
    G_iom=np.zeros(4*n,dtype=complex)
    taulist=np.linspace(0,beta,num=4*n+1)[:4*n]
    for i in np.arange(4*n):
        tau=taulist[i]
        G_iom[i]=np.sum(Gktau*np.exp(1j*fermion_om*tau))
    return G_iom

@jit(nopython=True)
def stupid_ift_boson(Pq_tau, beta, boson_om):
    N = len(Pq_tau)
    P_iom = np.zeros_like(boson_om, dtype=complex128)
    taulist = (np.arange(N)+0.5) / N * beta
    for i in range(len(boson_om)):
        sum_val = 0.0 + 0.0j
        for j in range(N):
            sum_val += Pq_tau[j] * np.exp(1j * boson_om[i] * taulist[j])
        P_iom[i] = sum_val / N 
    return P_iom

#-------fermion fft----------
# Note: all FFT and iFFT here follows the FT definition of matsubara freqs defined in Negele & Orland's book!
# For the details about how to take advantage of np.fft to do FT of Matsubara freqs, see note '231205 Matsubara Freq FFT'
# these 2 functions are for imp GF, which are 1D arrays.
def fermion_fft(Gk,beta):
    '''
    iom->tau
    '''
    N=np.shape(Gk)[0]
    # Gktau=np.fft.fft(Gk,axis=0)*np.exp(1j*(N-1)*np.pi*np.arange(N)/N-1j*(2*np.arange(N)-N+1)*np.pi*0.5/N)[:,None,None,None]
    Gktau=np.fft.fft(Gk*np.exp(-1j*(2*np.arange(N)-N+1)*np.pi*0.5/N),axis=0)*np.exp(1j*(N-1)*np.pi*np.arange(N)/N)/beta
    return Gktau

def fermion_ifft(Gk,beta):# same way back. in fft, we fft then shift; in ifft, we shift back then fft.
    '''
    tau->iom
    '''
    N=np.shape(Gk)[0]
    Gkiom=np.fft.ifft(Gk*np.exp(-1j*(N-1)*np.pi*np.arange(N)/N))*np.exp(+1j*(2*np.arange(N)-N+1)*np.pi*0.5/N)*beta
    return Gkiom

# these 2 functions are for k-dep GF, which are 4D arrays.
def fast_ft_fermion(Gk,beta):
    '''
    iom->tau
    '''
    N=np.shape(Gk)[0]
    Gktau=np.fft.fft(Gk*np.exp(-1j*(2*np.arange(N)-N+1)*np.pi*0.5/N)[:,None,None,None],axis=0)*np.exp(1j*(N-1)*np.pi*np.arange(N)/N)[:,None,None,None]/beta
    #the exp term inside is to move tau points on (0.5,1.5,2.5,...,N-0.5)beta/N. Outside means a shift.
    return Gktau

def fast_ift_fermion(Gk,beta):# same way back. in fft, we fft then shift; in ifft, we shift back then fft.
    '''
    tau->iom
    '''
    N=np.shape(Gk)[0]
    Gkiom=np.fft.ifft(Gk*np.exp(-1j*(N-1)*np.pi*np.arange(N)/N)[:,None,None,None],axis=0)*np.exp(+1j*(2*np.arange(N)-N+1)*np.pi*0.5/N)[:,None,None,None]*beta
    return Gkiom
#-------boson fft----------
# To modify ft_fermion version to ft_boson version:
#1. np.exp(-1j*(2*np.arange(N)-N)*np.pi*0.5/N) this means a shift exp(-iOm_n*0.5*beta/N) which avoids putting tau points at tau=0 or tau=beta.
#2. np.exp(1j*(N)*np.pi*np.arange(N)/N) gives correct matsubara freqs to the system. For sure it should be corrected to Boson freqs.
def fast_ft_boson(Pk,beta):
    '''
    iom->tau
    '''
    N=np.shape(Pk)[0]
    # Gktau=np.fft.fft(Gk,axis=0)*np.exp(1j*(N-1)*np.pi*np.arange(N)/N-1j*(2*np.arange(N)-N+1)*np.pi*0.5/N)[:,None,None,None]
    Pktau=np.fft.fft(Pk*np.exp(-1j*(2*np.arange(N)-N)*np.pi*0.5/N)[:,None,None,None],axis=0)*np.exp(1j*(N)*np.pi*np.arange(N)/N)[:,None,None,None]/beta
    return Pktau

def fast_ift_boson(Pk,beta):# same way back. in fft, we fft then shift; in ifft, we shift back then fft.
    '''
    tau->iom
    '''
    N=np.shape(Pk)[0]
    Pkiom=np.fft.ifft(Pk*np.exp(-1j*(N)*np.pi*np.arange(N)/N)[:,None,None,None],axis=0)*np.exp(+1j*(2*np.arange(N)-N)*np.pi*0.5/N)[:,None,None,None]*beta
    return Pkiom






#-----------convolution method to calculate bubbles----------- used in calculation
def fermion_fft_diagG_4D(knum, Gk,beta,EimpS):
    '''
    Diagonal elements of G has an issue when doing FFT since it scales like 1/omega. If the FFT is done in brute force, the quality of 
    G_tau will be extremely bad. The correct way is to figure out its 1/omega analytical part and do it analytically; and the rest part
    dies faster than 1/omega, which is safe to use brute-force FFT.
    compared to the old version, this is for 4D arrays: EimpS = -mu+s_oo, s_oo is the high frequency part of sigma.
    '''

    n = int(np.shape(Gk)[0] / 2)
    N=2*n

    k1,k2,k3=gen_full_kgrids(knum)
    #generate alpha, f(alpha) for freq sum
    delta_inf=np.abs(EimpS)
    # alphak=dispersion(kx,ky,kz)# another alpha for test.
    alpk=np.sqrt(dispersion(k1,k2,k3)**2+delta_inf**2)
    #generate unperturbed Green's function
    tlist=(np.arange(N)+0.5)/N*beta
    fermion_om = (2*np.arange(N)+1-N)*np.pi/beta
    Gk0=1/2*((1+delta_inf/alpk)/(1j*fermion_om[:,None,None,None]-alpk)+
             (1-delta_inf/alpk)/(1j*fermion_om[:,None,None,None]+alpk))
    Gk_tau_diff=fast_ft_fermion(Gk-Gk0,beta)
    Gk_tau_ana=-1/2*((1+delta_inf/alpk)*np.exp(-alpk*tlist[:,None,None,None])/(1+np.exp(-alpk*beta))+
                        (1-delta_inf/alpk)*np.exp(alpk*tlist[:,None,None,None])/(1+np.exp(alpk*beta)))
    Gk_tau=Gk_tau_ana+Gk_tau_diff   
    return Gk_tau


def fermion_fft_diagG(knum, Gk,beta,fullsig,mu):
    '''
    Diagonal elements of G has an issue when doing FFT since it scales like 1/omega. If the FFT is done in brute force, the quality of 
    G_tau will be extremely bad. The correct way is to figure out its 1/omega analytical part and do it analytically; and the rest part
    dies faster than 1/omega, which is safe to use brute-force FFT.
    '''

    n = int(np.shape(Gk)[0] / 2)
    N=2*n

    k1,k2,k3=gen_full_kgrids(knum)
    #generate alpha, f(alpha) for freq sum
    delta_inf=-mu+fullsig[-1].real
    # alphak=dispersion(kx,ky,kz)# another alpha for test.
    alpk=np.sqrt(dispersion(k1,k2,k3)**2+delta_inf**2)
    #generate unperturbed Green's function
    tlist=(np.arange(N)+0.5)/N*beta
    fermion_om = (2*np.arange(N)+1-N)*np.pi/beta
    if np.abs(delta_inf)<0.0001:# AVOID 0/0
        alpk=np.ones_like(alpk)
        # print('delta=',delta_inf,'take alpha=1')
    Gk0=1/2*((1+delta_inf/alpk)/(1j*fermion_om[:,None,None,None]-alpk)+
             (1-delta_inf/alpk)/(1j*fermion_om[:,None,None,None]+alpk))
    Gk_tau_diff=fast_ft_fermion(Gk-Gk0,beta)

    Gk_tau_ana=-1/2*((1+delta_inf/alpk)*np.exp(-alpk*tlist[:,None,None,None])/(1+np.exp(-alpk*beta))+
                        (1-delta_inf/alpk)*np.exp(alpk*tlist[:,None,None,None])/(1+np.exp(alpk*beta)))
    Gk_tau=Gk_tau_ana+Gk_tau_diff   
    return Gk_tau


def precalcP_fft(q, knum, G1k_tau,G2k_tau,opt):# this function deal with Gk*Gkq. they should be well sliced and shifted.
    n = int(np.shape(G1k_tau)[0] / 2)
    G2kq_tau=G12_shift(G2k_tau,q,knum,opt)
    Pq_tau=np.sum(-G1k_tau[::-1,:,:,:]*G2kq_tau,axis=(1,2,3))/knum**3
    return Pq_tau.real 

def precalcQ_fft(q, knum, G1k_tau,G2k_tau,opt):
    '''
    Q is another quantity which looks like P. In PM, P=-P' but in AFM they're different in principle.
    Q(q,tau)=sum_k'(G(k,tau)*(G(k+q,tau)))
    while
    P(q,tau)=sum_k'(G(k,-tau)*(G(k+q,tau)))
    opt==1 means the k-space symmetry of the quantity which is shifted. opt=0: Gk=G-k; opt=1: Gk=-G-k.
    '''
    n = int(np.shape(G1k_tau)[0] / 2)
    G2kq_tau=G12_shift(G2k_tau,q,knum,opt)
    Pq_tau=np.sum(G1k_tau*G2kq_tau,axis=(1,2,3))/knum**3
    return Pq_tau

# @jit(nopython=True)
def precalcsig_fft(q, knum, Gk_tau,Pq_tau,beta,U,opt):#for off-diagonal
    N=np.shape(Pq_tau)[0]
    Gkq_tau=G12_shift(Gk_tau,q,knum,opt)
    sig_tau=np.sum(Pq_tau*Gkq_tau,axis=(1,2,3))*(-1)*U**2/knum**3
    sig_iom=fermion_ifft(sig_tau,beta)
    # sig_iom=np.fft.ifft(sig_tau*np.exp(-1j*(N-1)*np.pi*np.arange(N)/N))*np.exp(+1j*(2*np.arange(N)-N+1)*np.pi*0.5/N)*beta
    return sig_iom#sig_tau

def precalcsigtau_fft(q, knum, Gk_tau,Pq_tau,beta,U,opt):#for off-diagonal
    N=np.shape(Pq_tau)[0]
    Gkq_tau=G12_shift(Gk_tau,q,knum,opt)
    sig_tau=np.sum(Pq_tau*Gkq_tau,axis=(1,2,3))*(-1)*U**2/knum**3
    # sig_iom=fermion_ifft(sig_tau,beta)
    # sig_iom=np.fft.ifft(sig_tau*np.exp(-1j*(N-1)*np.pi*np.arange(N)/N))*np.exp(+1j*(2*np.arange(N)-N+1)*np.pi*0.5/N)*beta
    return sig_tau

# def precalcsigp_fft(q, knum, Gk_tau,Pq_tau,beta,U,opt):#for off-diagonal
#     N=np.shape(Pq_tau)[0]
#     # Gk_tau=fast_ft_fermion(Gk,beta)
#     Gkq_tau=G12_shift(Gk_tau,q,knum,opt)
#     sig_tau=np.sum(-Pq_tau*Gkq_tau[::-1,:,:,:],axis=(1,2,3))*(-1)*U**2/knum**3
#     sig_iom=fermion_ifft(sig_tau,beta)
#     # sig_iom=np.fft.ifft(sig_tau*np.exp(-1j*(N-1)*np.pi*np.arange(N)/N))*np.exp(+1j*(2*np.arange(N)-N+1)*np.pi*0.5/N)
#     return sig_iom

def precalc_C(P1iom,P2iom,beta):
    # here Ps are bosonic quantities so we have to put them back on Bosonic matsubara freqs.
    Ciom=P1iom*P2iom
    Ctau=fast_ft_boson(Ciom,beta) 
    return Ctau

def precalc_ladder3(P1iom,P2iom,P3iom,beta):
    # here Ps are bosonic quantities so we have to put them back on Bosonic matsubara freqs.
    Ciom=P1iom*P2iom*P3iom
    Ctau=fast_ft_boson(Ciom,beta) 
    return Ctau


# other methods. just for test, not actually used any more.
def precalcP_bf(q, knum, n, G12,beta,opt=1):
    P_partial = np.zeros((n+1),dtype=np.complex128)
    if opt==1:
        G12_kq=G12_shift(G12,q,knum,1)
    else:
        G12_kq=G12_shift(G12,q,knum,0)
    G12_sliced = G12[int(n/2):int(3*n/2)]
    # ptest=0
    # pana=0
    # num=0
    #-------------analytical test--------
    # for kx in np.arange(knum):
    #     for ky in np.arange(knum):
    #         for kz in np.arange(knum):

    #             testk=[kx,ky,kz]
    #             # testk=[9,3,6]
    #             Omlist=np.arange(n+1)-int(n/2)
    #             dispk=dispersion(testk[0]/knum,testk[1]/knum,testk[2]/knum)
    #             dispkq=dispersion((testk[0]+q[0])/knum,(testk[1]+q[1])/knum,(testk[2]+q[2])/knum)
    #             # plt.plot(G12_sliced[:,testk[0],testk[1],testk[2]],label='num Gk')
    #             # plt.plot(G12_kq[int(n/2)+Om:int(3*n/2)+Om,testk[0],testk[1],testk[2]],label='num Gkq')
    #             # plt.show()
    #             ptest_par=np.zeros_like(Omlist,dtype=float)
    #             for i in np.arange(n+1):
    #                 ptest_par[i]=np.sum(G12_sliced[:,testk[0],testk[1],testk[2]] * G12_kq[int(n/2)+Omlist[i]:int(3*n/2)+Omlist[i],testk[0],testk[1],testk[2]])
    #             ptest+=ptest_par
    #             # print(Ptest)

    #             fk=fermi(dispk,beta)
    #             fkq=fermi(dispkq,beta)
    #             freq=1j*(Omlist*2*np.pi/beta+0.00001)
    #             pana_par=(0.25*beta*((fk-fkq)/(freq+dispk-dispkq)
    #                     +(fkq-fk)/(freq-dispk+dispkq)
    #                     -(fk+fkq-1)/(freq+dispk+dispkq)
    #                     -(1-fk-fkq)/(freq-dispk-dispkq)))
                # plt.plot(ptest_par.real,label='num.real')
                # plt.plot(pana_par.real,label='ana.real')
                # plt.plot(pana_par.imag,label='ana.imag')
                # plt.legend()
                # plt.show()
                # pana+=pana_par

                # if np.abs((pana_par-ptest_par))>10000/knum**3*np.abs(ptest_par):
                #     num+=1
                #     print(pana_par,ptest_par,testk)
                    # print(dispk,dispkq)
                    # print(fk,fkq,(fk-fkq),(freq+dispk-dispkq),(fk-fkq)/(freq-dispk+dispkq))
                    # print(pana_par,'analytical')
                    # print(ptest_par,'numerical')
    # print((pana.real-ptest)/knum**3,'q=',q)
    # print('bad points_num:',num)
    #-------------analytical test--------

    for Omind in np.arange(n+1):
        G12_kq_sliced=G12_kq[Omind:n+Omind,:,:,:]
        Gmul=G12_sliced * G12_kq_sliced# takes most time
        P_partial[Omind] = np.sum(Gmul)#.real
        # P_partial[int(n/2)-Omind]=P_partial[int(n/2)+Omind]   
    
    # plt.plot(ptest.real-P_partial.real,label='sum_num.real-P_partial.real')
    # plt.plot(pana.real-P_partial.real,label='sum_ana.real-P_partial.real')
    # plt.plot(P_partial.real,label='P_partial.real')
    # plt.legend()
    # plt.show()
    # return pana/ beta / knum ** 3
    return P_partial/ beta / knum ** 3

def p_analytical(q,knum,n,beta):
    kind_list = np.arange(knum)/knum
    kind1, kind2, kind3 = np.meshgrid(kind_list, kind_list, kind_list, indexing='ij')
    qx=q[0]/knum
    qy=q[1]/knum
    qz=q[2]/knum
    kqind1,kqind2,kqind3=kind1+qx,kind2+qy,kind3+qz
    dispk=perturb_lib.dispersion(kind1,kind2,kind3)
    dispkq=perturb_lib.dispersion(kqind1,kqind2,kqind3)
    
    boson_om=(2*np.arange(n)+2)*np.pi/beta+0.0001
    iom=1j*boson_om[:,None,None,None]
    fk=perturb_lib.fermi(dispk,beta)[None,:,:,:]
    fkq=perturb_lib.fermi(dispkq,beta)[None,:,:,:]
    P=np.zeros_like(boson_om,dtype=complex)
    P=0.25*np.sum((fk-fkq)/(iom+dispk-dispkq)
                  +(fkq-fk)/(iom-dispk+dispkq)
                  -(fk+fkq-1)/(iom+dispk+dispkq)
                  -(1-fk-fkq)/(iom-dispk-dispkq),axis=(1,2,3))/knum**3
    
    return P

def sig_analytical(k,knum,n,beta,U):
    kind_list = np.arange(knum)
    kp1, kp2, kp3 = np.meshgrid(kind_list, kind_list, kind_list, indexing='ij')
    k1=k[0]
    k2=k[1]
    k3=k[2]
    boson_om=(2*np.arange(n)+2)*np.pi/beta
    iom=1j*boson_om[:,None,None,None]
    fermion_om=(2*np.arange(n)+1)/beta*np.pi
    dispkp=perturb_lib.dispersion(kp1/knum,kp2/knum,kp3/knum)#esp_k'
    fermikp=perturb_lib.fermi(dispkp,beta)[None,:,:,:]#fermi(esp_k')
    for q1 in np.arange(knum):
        for q2 in np.arange(knum):
            for q3 in np.arange(knum):
                dispkpq=perturb_lib.dispersion((kp1+q1)/knum,(kp2+q2)/knum,(kp3+q3)/knum)#dispk'+q
                fermikpq=perturb_lib.fermi(dispkpq,beta)#fermi(esp_k'+q)
                dispkq=perturb_lib.dispersion((k1+q1)/knum,(k2+q2)/knum,(k3+q3)/knum)#dispk+q
                fermikq=perturb_lib.fermi(dispkq,beta)#fermi(esp_k+q)
                bosonic1=perturb_lib.boson(-dispkp+dispkpq)
                bosonic2=-1-bosonic1#boson(dispkp-dispkpq) #Notice that B(eps)+B(-eps)=-1.
                bosonic3=perturb_lib.boson(-dispkp-dispkpq)
                bosonic4=-1-bosonic3#boson(+dispkp+dispkpq)


    # fkq=perturb_lib.fermi(dispkq,beta)[None,:,:,:]

    return 0


#-------------test---------------
def conv_test(sigA,sigB,U,T,knum):
    mu=U/2
    beta=1/T
    z_A=perturb_lib.z(beta,mu,sigA,nfreq)
    z_B=perturb_lib.z(beta,mu,sigB,nfreq)
    n=sigA.size
    boson_om=(2*np.arange(n)+2)*np.pi/beta
    N=2*n
    allsigA=perturb_lib.ext_sig(beta,sigA)
    allsigB=perturb_lib.ext_sig(beta,sigB)
    G11=perturb_lib.G_11(knum,z_A,z_B)
    G12=perturb_lib.G_12(knum,z_A,z_B)
    # qtest=[4,5,5]
    # qtest=[1,4,3]
    # qtest=[1,1,3]
    # qtest=[0,0,0]

    #preparation for this trick.
    k1,k2,k3=gen_full_kgrids(knum)
    delta_inf=np.abs(-mu+allsigA[-1].real)
    alphak=np.sqrt(dispersion(k1,k2,k3)**2+delta_inf**2)
    # f_alphak=fermi(alphak,beta)
    max_sym_index,essential_kpoints, sym_array=calc_sym_array(knum)
    for qtest in essential_kpoints:
        opt=1
        p11_fft=stupid_ift_boson(precalcP_fft(qtest,knum,n,G12,beta,opt),beta,boson_om)[0:int(n/2)]
        # p11_fft=precalcP_fft(qtest,knum,n,G12,beta,opt)
        p11_bf=precalcP_bf(qtest,knum,n,G12,beta,opt)[int(n/2)+1:]
        p12_ana=p_analytical(qtest,knum,n,beta)

        # opt=0
        # p11_fft=precalcP_fft_diag(qtest,knum,n,G11,beta,delta_inf,alphak)
        # p11_bf=precalcP_bf(qtest,knum,n,G11,beta,opt)

        plt.plot(p11_fft[0:100].real,label='fft.real')
        plt.plot(p11_bf[0:100].real,label='bf.real')
        plt.plot(p12_ana[0:100].real,label='ana.real')
        # plt.plot(p12_ana[0:100].real/p11_bf[0:100].real,label='ratio.real')
        plt.legend()
        plt.show()
    return 0

if __name__ == "__main__":
    T=0.12
    U=3.0
    knum=8
    nfreq=1000
    index=1#index start from 1, not 0
    # sigma=np.loadtxt('./files_pert_boldc/{}_{}/Sig.OCA.{}'.format(U,T,index))[:nfreq,:]
    sigma=np.loadtxt('../files_ctqmc/{}_{}/ori_Sig.out.{}'.format(U,T,index))[:nfreq,:]

    # sigA=sigma[:,1]+1j*sigma[:,2]#sig+delta
    # sigB=sigma[:,3]+1j*sigma[:,4]#sig-delta
    sigA=np.ones_like(sigma[:,1])*U/2
    sigB=np.ones_like(sigma[:,1])*U/2#sig-delta
    conv_test(sigA,sigB,U,T,knum)