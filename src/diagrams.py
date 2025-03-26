import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess,math
import time
sys.path.append('../python_src/')
from mpi4py import MPI
from perturb_lib import *
import perturb_imp as imp
import fft_convolution as fft
# import mpi_module
import serial_module
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

'''
This file aims to pack the algorithm of diagrams. Our project sometimes requires different propagators in a diagrams.
'''





def sig1(G11_iom,G22_iom,knum,U,beta): # calculate all 1st order diagrams: 1st order, 1 fermion loop, ==> plus sign
    Sig1_11=(-1)*(-1)*U*(np.sum(G22_iom).real/knum**3/beta+1/2)# only works for half filling.
    Sig1_22=(-1)*(-1)*U*(np.sum(G11_iom).real/knum**3/beta+1/2)
    return Sig1_11,Sig1_22

def sig2(G11_tau,G12_tau,G22_tau,P22_tau,P12_tau,knum,nfreq,U,beta): # calculate all 2nd order diagrams
    '''
    This function packs all 2nd order diagrams. and the result is in freq space.
    '''
    # G11_tau=fft.fermion_fft_diagG(knum,G11_iom,beta,SigDMFT1-B,mu)
    # G12_tau=fft.fast_ft_fermion(G12_iom,beta)
    # P22_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G22_tau,G22_tau,0)
    # P12_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12,G12_tau,G12_tau,1)
    Sig2_11=serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11, G11_tau,P22_tau,beta,U,0)
    Sig2_12=serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12, G12_tau,P12_tau,beta,U,1)
    # Sig2_22=-Sig2_11.conjugate()
    # Sig2tau_11=mpi_module.bubble_mpi(fft.precalcsigtau_fft,knum,nfreq,11, G11_tau,P22_tau,beta,U,0)
    return Sig2_11,Sig2_12#,Sig2tau_11


def sig2_nonskeleton(G22_iom,G12_iom,sig1_11,sig1_22,knum,nfreq,U,beta):
    '''
    this function calculate non-skeleton diagrams of 2rd order.
    Currently this contain a tadpole insertion on a tadpole.
    '''
    GSigG22=G22_iom*sig1_22*G22_iom+G12_iom*sig1_11*G12_iom
    # plt.plot(GSigG22[:,0,0,0].real,label='real')
    # plt.plot(GSigG22[:,0,0,0].imag,label='imag')
    # plt.legend()
    # plt.show()
    sigext11=np.sum(GSigG22)*U/knum**3/beta
    return sigext11

def sig3(G11_iom,G12_iom,G11_tau,G12_tau,G22_tau,Q11_iom,Q12_iom,Q22_iom,R11_iom,R12_iom,R22_iom,knum,nfreq,U,beta): # calculate all 3rd order diagrams
    '''
    This function packs all 3rd order diagrams. and the result is in freq space.
    '''
    # do check those 3rd order diagrams.
    # Q11_tau=serial_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G22_tau,G11_tau,0)#Q=G_{s',-k}(tau)*G_{s,k+q}(tau)
    # Q12_tau=serial_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,12, G12_tau,G12_tau,1)# Note: G12_-k=-G12_k!
    # Q22_tau=serial_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G11_tau,G22_tau,0)
    # R11_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G22_tau,G11_tau,0)#R=G_{s',k}(-tau)*G_{s,k+q}(tau)
    # R12_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12, G12_tau,G12_tau,1)
    # R22_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G11_tau,G22_tau,0)
    # P22_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G22_tau,G22_tau,0)
    # P12_tau=mpi_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G12_tau,G12_tau,1)
    # Note1: Polarization P contains 2 propagators with same spin. But this is not the case for 3rd order.
    # Note2: Q11,Q12,R11,R12 are all symmetric in k space. see proof in '240126 third order diagram'
    #FT
    # R11_iom=fft.fast_ift_boson(R11_tau,beta)
    # R22_iom=fft.fast_ift_boson(R22_tau,beta)
    # R12_iom=fft.fast_ift_boson(R12_tau,beta)
    # Q11_iom=fft.fast_ift_boson(Q11_tau,beta)
    # Q22_iom=fft.fast_ift_boson(Q22_tau,beta)
    # Q12_iom=fft.fast_ift_boson(Q12_tau,beta)
    #definitions and notations according to qualifier paper. indices are: 111,121,122,112. 

    B_111_tau=fft.precalc_C(R11_iom,R11_iom,beta)
    B_121_tau=fft.precalc_C(R12_iom,R12_iom,beta)
    B_112_tau=fft.precalc_C(R11_iom,R12_iom,beta)
    B_122_tau=fft.precalc_C(R12_iom,R22_iom,beta)
    A_111_tau=fft.precalc_C(Q11_iom,Q11_iom,beta)
    A_121_tau=fft.precalc_C(Q12_iom,Q12_iom,beta)
    A_112_tau=fft.precalc_C(Q11_iom,Q12_iom,beta)
    A_122_tau=fft.precalc_C(Q12_iom,Q22_iom,beta)
    #precalcsig has the factor. (-1)*U**2/knum**3. actually factor needed is U**3. need extra -U.
    # Note: calculations below are simplified using symmetries of k, tau, and spin. for details, see '240126 third order diagram'.
    Sig3_1_111=-U*serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,-G11_tau,A_111_tau,beta,U,0 )
    Sig3_1_121=-U*serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,-G11_tau,A_121_tau,beta,U,0 )
    Sig3_1_112=-U*serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,A_112_tau,beta,U,1 )# check here.
    Sig3_1_122=-U*serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,A_122_tau,beta,U,1 )

    Sig3_2_111=-U*serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,G22_tau,B_111_tau,beta,U,0 )
    Sig3_2_121=-U*serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,G22_tau,B_121_tau,beta,U,0 )
    Sig3_2_112=-U*serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,B_112_tau,beta,U,1 )
    Sig3_2_122=-U*serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,B_122_tau,beta,U,1 )


    Sig3_1_11=Sig3_1_111+Sig3_1_121
    Sig3_1_12=Sig3_1_112+Sig3_1_122
    Sig3_2_11=Sig3_2_111+Sig3_2_121
    Sig3_2_12=Sig3_2_112+Sig3_2_122
    # Sig3_11=Sig3_2_111+Sig3_1_111+Sig3_1_121+Sig3_2_121
    # Sig3_12=Sig3_1_112+Sig3_2_112+Sig3_1_122+Sig3_2_122

    # return Sig3_11,Sig3_12
    return Sig3_1_11,Sig3_1_12,Sig3_2_11,Sig3_2_12

def sig3_nonskeleton_A(G22_iom,G12_iom,sig2iom_11,sig2iom_22,sig2iom_12,knum,nfreq,U,beta):
    '''
    this function calculate non-skeleton diagrams of 3rd order.
    Currently this contain a second order insertion on a tadpole.
    '''
    GSigG22=G22_iom*sig2iom_22*G22_iom+G12_iom*sig2iom_12*G22_iom*2+G12_iom*sig2iom_11*G12_iom
    sigext11=np.sum(GSigG22)*U/knum**3/beta
    return sigext11

def sig3_nonskeleton_B(G22_iom,G12_iom,G11_iom,sig1_11,sig1_22,knum,nfreq,U,beta):
    '''
    this diagram is 2 hartree insertion on a hartree so it is 3rd order
    '''
    GSigGSigG22=G22_iom**3*sig1_22**2 + G12_iom**2*G11_iom*sig1_11**2 + G12_iom**2*G22_iom*sig1_11*sig1_22*2
    sigext11=np.sum(GSigGSigG22)*U/knum**3/beta
    return sigext11 


def sig3_nonskeleton_DEF(G11_iom,G12_iom,G11_tau,G12_tau,P22_tau,P12_tau,sig1_11,sig1_22,beta,knum,nfreq,U):
    '''
    3 third order diagrams, which are 1 B insertion in second order diagrams.
    In this function, we treat 1 B insertion as order 1. This is kleinert's idea.
    '''
    G22_iom=-G11_iom.conjugate()
    G22_tau=G11_tau[::-1]
    # if rank ==0:
        # note: spin indecies are all up. 22= spin dn.
    GsigG11_iom=G11_iom**2*sig1_11+G12_iom**2*sig1_22
    GsigG12_iom=G12_iom*G22_iom*sig1_22+G11_iom*G12_iom*sig1_11
    GsigG22_iom=G22_iom**2*sig1_22+G12_iom**2*sig1_11
    GsigG11_tau=fft.fast_ft_fermion(GsigG11_iom,beta)# GBG scales at least as 1/omega**2.
    GsigG12_tau=fft.fast_ft_fermion(GsigG12_iom,beta)
    GsigG22_tau=fft.fast_ft_fermion(GsigG22_iom,beta)
    # should i broadcast?
    # GsigG11_tau=np.ascontiguousarray(GsigG11_tau)
    # GsigG12_tau=np.ascontiguousarray(GsigG12_tau)
    # GsigG22_tau=np.ascontiguousarray(GsigG22_tau)
    # comm.Bcast(GsigG11_tau, root=0)
    # comm.Bcast(GsigG12_tau, root=0)
    # comm.Bcast(GsigG22_tau, root=0)
    Pa22_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G22_tau,GsigG22_tau,0)
    Pa12_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12,G12_tau,GsigG12_tau,1)
    Pb22_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,GsigG22_tau,G22_tau,0)
    Pb12_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12,GsigG12_tau,G12_tau,1)
    # siga-c: sig1 insertion in a second order diagram. 
    Sigd11=serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11, G11_tau,Pa22_tau,beta,U,0)
    Sigd12=serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12, G12_tau,Pa12_tau,beta,U,1)
    Sige11=serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11, G11_tau,Pb22_tau,beta,U,0)
    Sige12=serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12, G12_tau,Pb12_tau,beta,U,1)    
    Sigf11=serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11, GsigG11_tau,P22_tau,beta,U,0)
    Sigf12=serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12, GsigG12_tau,P12_tau,beta,U,1)       
    Sig11=Sigd11+Sige11+Sigf11
    Sig12=Sigd12+Sige12+Sigf12
    return Sig11,Sig12

def sig4_1(G11_tau,G12_tau,G22_tau,Q11_iom,Q12_iom,Q22_iom,knum,nfreq,U,beta):
    '''
    The 1st skeleton diagram of order 4, which is a ladder.
    '''
    A_1111_tau=fft.precalc_ladder3(Q11_iom,Q11_iom,Q11_iom,beta)
    A_1121_tau=fft.precalc_ladder3(Q11_iom,Q12_iom,Q12_iom,beta)*2# A_1211_tau=fft.precalc_ladder3(Q12_iom,Q12_iom,Q11_iom,beta)=A1121
    A_1221_tau=fft.precalc_ladder3(Q12_iom,Q22_iom,Q12_iom,beta)

    A_1112_tau=fft.precalc_ladder3(Q11_iom,Q11_iom,Q12_iom,beta)
    A_1212_tau=fft.precalc_ladder3(Q12_iom,Q12_iom,Q12_iom,beta)
    A_1222_tau=fft.precalc_ladder3(Q12_iom,Q22_iom,Q22_iom,beta)
    A_1122_tau=fft.precalc_ladder3(Q11_iom,Q12_iom,Q22_iom,beta)
    #precalcsig has the factor. *U**3/knum**3. actually factor needed is U**3. need extra U**2.
    # Note: calculations below are simplified using symmetries of k, tau, and spin. for details, see '240126 third order diagram'.
    Sig4_1_11=U**2*serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,-G11_tau,A_1111_tau+A_1121_tau+A_1221_tau,beta,U,0 )#
    Sig4_1_12=U**2*serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,A_1112_tau+A_1212_tau+A_1222_tau+A_1122_tau,beta,U,1 )
    return Sig4_1_11,Sig4_1_12

def sig4_2(G11_tau,G12_tau,G22_tau,R11_iom,R12_iom,R22_iom,knum,nfreq,U,beta):
    '''
    The 2nd skeleton diagram of order 4, which is another ladder.
    '''
    B_1111_tau=fft.precalc_ladder3(R11_iom,R11_iom,R11_iom,beta)
    B_1121_tau=fft.precalc_ladder3(R11_iom,R12_iom,R12_iom,beta)*2# B_1211_tau=fft.precalc_ladder3(R12_iom,R12_iom,R11_iom,beta)=B1121
    B_1221_tau=fft.precalc_ladder3(R12_iom,R22_iom,R12_iom,beta)

    B_1112_tau=fft.precalc_ladder3(R11_iom,R11_iom,R12_iom,beta)
    B_1212_tau=fft.precalc_ladder3(R12_iom,R12_iom,R12_iom,beta)
    B_1222_tau=fft.precalc_ladder3(R12_iom,R22_iom,R22_iom,beta)
    B_1122_tau=fft.precalc_ladder3(R11_iom,R12_iom,R22_iom,beta)
    Sig4_2_11=U**2*serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,G22_tau,B_1111_tau+B_1121_tau+B_1221_tau,beta,U,0 )#
    Sig4_2_12=U**2*serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,B_1112_tau+B_1212_tau+B_1222_tau+B_1122_tau,beta,U,1 )
    return Sig4_2_11,Sig4_2_12

def sig4_5(G11_tau,G12_tau,G22_tau,P11_iom,P12_iom,P22_iom,knum,nfreq,U,beta):
    C_1111_tau=fft.precalc_ladder3(P12_iom,P22_iom,P12_iom,beta)
    C_1121_tau=fft.precalc_ladder3(P12_iom,P12_iom,P22_iom,beta)*2# B_1211_tau=fft.precalc_ladder3(R12_iom,R12_iom,R11_iom,beta)=B1121
    C_1221_tau=fft.precalc_ladder3(P22_iom,P11_iom,P22_iom,beta)

    C_1112_tau=fft.precalc_ladder3(P12_iom,P22_iom,P11_iom,beta)
    C_1212_tau=fft.precalc_ladder3(P22_iom,P12_iom,P11_iom,beta)
    C_1122_tau=fft.precalc_ladder3(P12_iom,P12_iom,P12_iom,beta)
    C_1222_tau=fft.precalc_ladder3(P22_iom,P11_iom,P12_iom,beta)
    Sig4_5_11=U**2*serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,G11_tau,C_1111_tau+C_1121_tau+C_1221_tau,beta,U,0 )#
    Sig4_5_12=U**2*serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,C_1112_tau+C_1212_tau+C_1222_tau+C_1122_tau,beta,U,1 )
    return Sig4_5_11,Sig4_5_12

def sig4_3plus(G11_iom,G12_iom,G22_iom,G11_tau,G12_tau,G22_tau,  Sigins11,Sigins22,  Q11_iom,Q12_iom,Q22_iom,R11_iom,R12_iom,R22_iom,knum,nfreq,U,beta):
    '''
    This is a part of order 4 diagrams. There are 10 diagrams here, which are 1 first order insertion in 3rd order skeleton diagrams
    '''
    G1G_11_iom=G11_iom*Sigins11*G11_iom+G12_iom*Sigins22*G12_iom
    G1G_12_iom=G11_iom*Sigins11*G12_iom+G12_iom*Sigins22*G22_iom
    G1G_22_iom=G22_iom*Sigins22*G22_iom+G12_iom*Sigins11*G12_iom
    G1G_11_tau=fft.fast_ft_fermion(G1G_11_iom,beta)
    G1G_12_tau=fft.fast_ft_fermion(G1G_12_iom,beta)
    G1G_22_tau=fft.fast_ft_fermion(G1G_22_iom,beta)
    Qupins_11_iom=fft.fast_ift_boson(serial_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G1G_22_tau,G11_tau,0),beta)   # up prpopagator has insertion. 
    Qupins_12_iom=fft.fast_ift_boson(serial_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,12, G1G_12_tau,G12_tau,1),beta)
    Qupins_22_iom=fft.fast_ift_boson(serial_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G1G_11_tau,G22_tau,0),beta)
    Qdnins_11_iom=fft.fast_ift_boson(serial_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G22_tau,G1G_11_tau,0),beta)   # dn prpopagator has insertion. 
    Qdnins_12_iom=fft.fast_ift_boson(serial_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,12, G12_tau,G1G_12_tau,1),beta)
    Qdnins_22_iom=fft.fast_ift_boson(serial_module.bubble_mpi(fft.precalcQ_fft,knum,nfreq,11, G11_tau,G1G_22_tau,0),beta)
    
    Rupins_11_iom=fft.fast_ift_boson(serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G1G_22_tau,G11_tau,0),beta)   # dn prpopagator has insertion. 
    Rupins_12_iom=fft.fast_ift_boson(serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12, G1G_12_tau,G12_tau,1),beta)
    Rupins_22_iom=fft.fast_ift_boson(serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G1G_11_tau,G22_tau,0),beta)
    Rdnins_11_iom=fft.fast_ift_boson(serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G22_tau,G1G_11_tau,0),beta)   # dn prpopagator has insertion. 
    Rdnins_12_iom=fft.fast_ift_boson(serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12, G12_tau,G1G_12_tau,1),beta)
    Rdnins_22_iom=fft.fast_ift_boson(serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11, G11_tau,G1G_22_tau,0),beta)
    # original ladders
    B_11_ori_tau=fft.precalc_C(R11_iom,R11_iom,beta)+fft.precalc_C(R12_iom,R12_iom,beta)#111+121,...
    B_12_ori_tau=fft.precalc_C(R11_iom,R12_iom,beta)+fft.precalc_C(R12_iom,R22_iom,beta)
    A_11_ori_tau=fft.precalc_C(Q11_iom,Q11_iom,beta)+fft.precalc_C(Q12_iom,Q12_iom,beta)
    A_12_ori_tau=fft.precalc_C(Q11_iom,Q12_iom,beta)+fft.precalc_C(Q12_iom,Q22_iom,beta)
    #ladders with 1 upper propagator inserted with a 1st order 
    #Note: here we group the 2 diagrams together.  
    B_11_upper_tau=fft.precalc_C(Rupins_11_iom,R11_iom,beta)*2+fft.precalc_C(Rupins_12_iom,R12_iom,beta)*2
    B_12_upper_tau=fft.precalc_C(Rupins_11_iom,R12_iom,beta)+fft.precalc_C(Rupins_12_iom,R22_iom,beta)+fft.precalc_C(R11_iom,Rupins_12_iom,beta)+fft.precalc_C(R12_iom,Rupins_22_iom,beta)

    A_11_upper_tau=fft.precalc_C(Qupins_11_iom,Q11_iom,beta)*2+fft.precalc_C(Qupins_12_iom,Q12_iom,beta)*2
    A_12_upper_tau=fft.precalc_C(Qupins_11_iom,Q12_iom,beta)+fft.precalc_C(Qupins_12_iom,Q22_iom,beta)+fft.precalc_C(Q11_iom,Qupins_12_iom,beta)+fft.precalc_C(Q12_iom,Qupins_22_iom,beta)
    #ladders with 1 lower propagator inserted with a 1st order 
    B_11_lower_tau=fft.precalc_C(Rdnins_11_iom,R11_iom,beta)*2+fft.precalc_C(Rdnins_12_iom,R12_iom,beta)*2
    B_12_lower_tau=fft.precalc_C(Rdnins_11_iom,R12_iom,beta)+fft.precalc_C(Rdnins_12_iom,R22_iom,beta)+fft.precalc_C(R11_iom,Rdnins_12_iom,beta)+fft.precalc_C(R12_iom,Rdnins_22_iom,beta)

    A_11_lower_tau=fft.precalc_C(Qdnins_11_iom,Q11_iom,beta)*2+fft.precalc_C(Qdnins_12_iom,Q12_iom,beta)*2
    A_12_lower_tau=fft.precalc_C(Qdnins_11_iom,Q12_iom,beta)+fft.precalc_C(Qdnins_12_iom,Q22_iom,beta)+fft.precalc_C(Q11_iom,Qdnins_12_iom,beta)+fft.precalc_C(Q12_iom,Qdnins_22_iom,beta)
    Sig4_3plus_11_iom=-U*(serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,-G1G_11_tau,A_11_ori_tau,beta,U,0 )
                          +serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,-G11_tau,A_11_upper_tau,beta,U,0 )
                          +serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,-G11_tau,A_11_lower_tau,beta,U,0 )
                          +serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,G1G_22_tau,B_11_ori_tau,beta,U,0 )
                          +serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,G22_tau,B_11_upper_tau,beta,U,0 )
                          +serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11,G22_tau,B_11_lower_tau,beta,U,0 ))
    Sig4_3plus_12_iom=-U*(serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G1G_12_tau,A_12_ori_tau,beta,U,1 )
                          +serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,A_12_upper_tau,beta,U,1 )
                          +serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,A_12_lower_tau,beta,U,1 )
                          +serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G1G_12_tau,B_12_ori_tau,beta,U,1 )
                          +serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,B_12_upper_tau,beta,U,1 )
                          +serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12,G12_tau,B_12_lower_tau,beta,U,1 ))    

    return Sig4_3plus_11_iom,Sig4_3plus_12_iom

def sig4_2plus(G11_iom,G12_iom,G22_iom,G11_tau,G12_tau,G22_tau,P22_tau,P12_tau,  Sigins1_11,Sigins1_22,Sigins2_11,Sigins2_12,Sigins2_22,knum,nfreq,U,beta):
    '''
    This function contains all '2+2' and '2+1+1' nonskeleton diagrams of 4th order. There are 9 of them
    '''
    G1G_11_iom=G11_iom*Sigins1_11*G11_iom+G12_iom*Sigins1_22*G12_iom
    G1G_12_iom=G11_iom*Sigins1_11*G12_iom+G12_iom*Sigins1_22*G22_iom
    G1G_22_iom=G22_iom*Sigins1_22*G22_iom+G12_iom*Sigins1_11*G12_iom
    G1G_11_tau=fft.fast_ft_fermion(G1G_11_iom,beta)
    G1G_12_tau=fft.fast_ft_fermion(G1G_12_iom,beta)
    G1G_22_tau=fft.fast_ft_fermion(G1G_22_iom,beta)
    # these G2G includes 1 2nd order insertion                                                                                  and also 2 1st order insertion.
    G2G_11_iom=G11_iom*Sigins2_11*G11_iom+G12_iom*Sigins2_22*G12_iom+G11_iom*Sigins2_12*G12_iom*2+                            G1G_11_iom*Sigins1_11*G11_iom+ G1G_12_iom*Sigins1_22*G12_iom
    G2G_12_iom=G11_iom*Sigins2_11*G12_iom+G12_iom*Sigins2_22*G22_iom+G11_iom*Sigins2_12*G22_iom+G12_iom*Sigins2_12*G12_iom+   G1G_11_iom*Sigins1_11*G12_iom+ G1G_12_iom*Sigins1_22*G22_iom
    G2G_22_iom=G12_iom*Sigins2_11*G12_iom+G22_iom*Sigins2_22*G22_iom+G22_iom*Sigins2_12*G12_iom*2+                            G1G_22_iom*Sigins1_22*G22_iom+ G1G_12_iom*Sigins1_11*G12_iom
    G2G_11_tau=fft.fast_ft_fermion(G2G_11_iom,beta)
    G2G_12_tau=fft.fast_ft_fermion(G2G_12_iom,beta)
    G2G_22_tau=fft.fast_ft_fermion(G2G_22_iom,beta)   
    # P22_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G22_tau,G22_tau,0)
    # P12_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12,G12_tau,G12_tau,1) 


    # here i group 2 diagrams with 2nd order inserted in P together.
    P_12_ins2_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12,G2G_12_tau,G12_tau,1)+serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12,G12_tau,G2G_12_tau,1)
    P_22_ins2_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G2G_22_tau,G22_tau,0)+serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G22_tau,G2G_22_tau,0)
    # These are 6 of the diagrams, except for the 3 diagrams has insetions in more than 1 propagators
    Sig4_2p2_11=serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11, G11_tau,P_22_ins2_tau,beta,U,0)+serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11, G2G_11_tau,P22_tau,beta,U,0)
    Sig4_2p2_12=serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12, G12_tau,P_12_ins2_tau,beta,U,1)+serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12, G2G_12_tau,P12_tau,beta,U,1)
    # insertion in 1 of the propagators of P
    P_12_ins1_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12,G1G_12_tau,G12_tau,1)+serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12,G12_tau,G1G_12_tau,1)
    P_22_ins1_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G1G_22_tau,G22_tau,0)+serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G22_tau,G1G_22_tau,0)
    # insertion in both of the propagators of P
    P_12_ins1p1_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,12,G1G_12_tau,G1G_12_tau,1)
    P_22_ins1p1_tau=serial_module.bubble_mpi(fft.precalcP_fft,knum,nfreq,11,G1G_22_tau,G1G_22_tau,0)


    Sig4_2p11_11=serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11, G1G_11_tau,P_22_ins1_tau,beta,U,0)+serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,11, G11_tau,P_22_ins1p1_tau,beta,U,0)
    Sig4_2p11_12=serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12, G1G_12_tau,P_12_ins1_tau,beta,U,1)+serial_module.bubble_mpi(fft.precalcsig_fft,knum,nfreq,12, G12_tau,P_12_ins1p1_tau,beta,U,1)
    return Sig4_2p11_11+Sig4_2p2_11,Sig4_2p11_12+Sig4_2p2_12

def sig4_1plus(G11_iom,G12_iom,G22_iom,Sigins1_11,Sigins1_22,Sigins2_11,Sigins2_12,Sigins2_22,Sigins3_11,Sigins3_12,Sigins3_22,knum,nfreq,U,beta):
    G1G_11_iom=G11_iom*Sigins1_11*G11_iom+G12_iom*Sigins1_22*G12_iom
    G1G_12_iom=G11_iom*Sigins1_11*G12_iom+G12_iom*Sigins1_22*G22_iom
    G1G_22_iom=G22_iom*Sigins1_22*G22_iom+G12_iom*Sigins1_11*G12_iom
    # 3rd order insertion: 4 kinds of insertion:3,1+2,2+1,1+1+1
    G3G_11_iom=G11_iom*Sigins3_11*G11_iom+G12_iom*Sigins3_22*G12_iom+G11_iom*Sigins3_12*G12_iom*2  
    G12G_11_iom=G1G_11_iom*Sigins2_11*G11_iom+ G1G_12_iom*Sigins2_22*G12_iom+G1G_11_iom*Sigins2_12*G12_iom+G1G_12_iom*Sigins2_12*G11_iom
    G21G_11_iom=G11_iom*Sigins2_11*G1G_11_iom+ G12_iom*Sigins2_22*G1G_12_iom+G11_iom*Sigins2_12*G1G_12_iom+G12_iom*Sigins2_12*G1G_11_iom
    G111G_11_iom=G1G_11_iom*Sigins1_11*G1G_11_iom+G1G_12_iom*Sigins1_22*G1G_12_iom
    # Note: Sig4_1plus_22=np.sum(G3G_11_iom+G12G_11_iom+G21G_11_iom+G111G_11_iom)*U/knum**3/beta. 11 is just up to a - sign.
    Sig4_1plus_11=-np.sum(G3G_11_iom+G12G_11_iom+G21G_11_iom+G111G_11_iom)*U/knum**3/beta
    return Sig4_1plus_11