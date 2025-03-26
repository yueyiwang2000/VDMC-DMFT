from scipy import *
from scipy.interpolate import interp1d
from . import imp_svd_weight_lib
from numpy import linalg
from numpy import random
import sys
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import time
from . import imp_diag_def
# import diag_def_cutPhi
# import diag_def_cutPhibackup
from mpi4py import MPI
sys.path.append('../')
import perturb_lib as lib
import fft_convolution as fft
from ..diagramsMC_lib import *
from .. import basis
import copy


'''
This code is MC estimation of self-energy diagrams using the trick of svd.
This function can be directly called. and the evaluation of integrand is accelerated.
'''
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

    
class params:
    def __init__(self):
        self.Nitt = 5000000   # number of MC steps in a single proc
        self.Ncout = 200000    # how often to print
        self.Nwarm = 1000     # warmup steps
        self.tmeassure = 10   # how often to meassure
        self.V0norm = 4e-2    # starting V0
        self.recomputew = 5e4/self.tmeassure # how often to check if V0 is correct
        self.per_recompute = 7 # how often to recompute fm auxiliary measuring function

def get_gtau(U,T):
    beta=1/T
    mu=U/2
    name1='../files_boldc/{}_{}/Sig.out'.format(U,T)
    filename1=readDMFT(name1)
    name2='../files_ctqmc/{}_{}/Sig.out'.format(U,T)
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
    # Gloc11=np.sum(G11_tau,axis=(1,2,3))/knum**3
    # Gloc22=np.sum(G22_tau,axis=(1,2,3))/knum**3
    return G11_tau.real,G12_tau.real,G22_tau.real,Gloc11_tau.real,Gloc22_tau.real

def IntegrateByMetropolis_svdimp(func, qx, p,seed,lmax):
    """ Integration by Metropolis:
          func(momentum)   -- function to integrate
          qx               -- mesh given by a user
          p                -- other parameters
        Output:
          Pval(qx)

    in the impurity version, the only variables are:
    tau, sublatind(all points in the same loop keeps the same, only 2 ways), l,i(11 or 22)
    """
    ifprint=0
    #-------basic settings-----
    # time check
    time_trial=0
    time_evaluate=0
    time_accrej=0
    time_others=0
    Nacc,Nrej=0,0
    time_begin=time.time()
    ifrecomp=1
    np.random.seed(seed)# use the given seed
    # random.seed(0)         # make sure that we always get the same sequence of steps. If parallel. they should have different seeds.
    # knum=func.knum
    taunum=func.taunum
    taufold=np.arange(func.taunum+2)
    taufold[-1]=func.taunum-1
    taufold[func.taunum]=0
    # Pnorm2 = np.zeros_like(qx)  # Final results V_physical is stored in Pval
    
    Pnorm = 0.0            # V_alternative is stored in Pnorm
    Pval_sum = 0.0         # this is widetilde{V_physical}
    Pnorm_sum = 0.0        # this is widetilde{V_alternative}
    V0norm = p.V0norm      # this is V0
    dk_hist = 1.0          # we are creating histogram by adding each configuration with weight 1.
    # note: here i have both k and tau as external variable.
    Ndimtau=func.Ndimtau
    Ndimlat=1
    Pval=np.zeros_like(qx)
    inc_recompute = (p.per_recompute+0.52)/p.per_recompute # How often to self-consistently recompute
    # the wight functions g_i and h_{ij}.




    imagtime=np.random.randint(low=0, high=taunum, size=(Ndimtau,1))
    # note: give the sublatind in the beginning!
    sublatind=copy.deepcopy(func.SubLatInd)# indices for each time point. here i would like to define the loop with node 0 to be up to be 0. 
    l=np.random.randint(0,lmax)# generate the external variable
    j_coeff=np.random.randint(0,2)# index for sublattice basis.\
    tj_coeff=j_coeff
    tl=l
    timagtime=copy.deepcopy(imagtime)
    tsublatind=copy.deepcopy(sublatind)

    myweight = imp_svd_weight_lib.meassureWeight(Ndimtau,taunum)
    # to be updated. add sublatint in the update function.
    fQ = func.update(imagtime,sublatind,l,j_coeff), V0norm * myweight(imagtime,sublatind) # fQ=(f(X), V0*f_m(X)) sublatind consists 1 and 2 but we'd better start from 0.
    # print('starting with f=', fQ, '\nstarting momenta=', momentum,'\n starting time=',imagtime)

    Nmeassure = 0  # How many measurements we had?
    Nall_q, Nall_k, Nall_w, Nacc_q, Nacc_k = 0, 0, 0, 0, 0
    c_recompute = 0 # when to recompute the auxiliary function?
    for itt in range(p.Nitt):   # long loop
        time0=time.time()
        # variables: k,tau,sublatind, i_coeff,l_coeff,j_coeff
        iloop = int( (Ndimtau+3) * random.rand() )   # ndimtau, sublatind(1), l,j
        accept = False
        if (iloop < Ndimtau):# changing internal variable tau
            (tau_new, trialaccept)=TrialStep1_tau(iloop,imagtime,taunum,0,taufold)
        elif (iloop == Ndimtau) :# changing sublatint. does not matter in or external variable.
            # rnd=np.random.randint(0,2)# 0 or 1
            # if rnd==1:
            sublatind_new=3-sublatind
            # elif rnd==0:
            #     sublatind_new=copy.deepcopy(sublatind)
            trialaccept=1
        elif (iloop == Ndimtau+1): # changing external variable l
            # lnew=np.random.randint(0,lmax)
            # trial_ratio=1; trialaccept=1
            (lnew, trialaccept)=Trialstep0_l(lmax)
        elif (iloop == Ndimtau+2): # changing external variable j
            j_coeffnew=np.random.randint(0,2)
            trialaccept=1

        time1=time.time()
        time_trial+=(time1-time0)
        if (trialaccept): # trial step successful. We did not yet accept, just the trial step.
            if (iloop<Ndimtau):# tau is changed
                timagtime=Give_new_tau(imagtime, tau_new, iloop,0)
            elif (iloop==Ndimtau):
                tsublatind=copy.deepcopy(sublatind_new)
            elif (iloop==Ndimtau+1):# l is changed
                tl=lnew
            elif (iloop==Ndimtau+2):# j is changed
                tj_coeff=j_coeffnew     

            time_beforecalc=time.time()

            fQ_new = func.update_temp(iloop,timagtime,tsublatind,tl,tj_coeff), V0norm * myweight(timagtime,tsublatind) # f_new
            time_aftercalc=time.time()
            time_evaluate+=(time_aftercalc-time_beforecalc)
            ratio = (abs(fQ_new[0])+fQ_new[1])/(abs(fQ[0])+fQ[1]) 
            # print('ratio=',ratio)
            accept = abs(ratio) > 1-random.rand() # Metropolis
            if accept: # the step succeeded
                func.metropolis_accept(iloop)
                if iloop<Ndimtau:
                    imagtime[iloop]=tau_new
                elif iloop==Ndimtau:
                    sublatind=copy.deepcopy(tsublatind)
                elif (iloop==Ndimtau+1):
                    l=tl
                elif (iloop==Ndimtau+2):
                    j_coeff=tj_coeff
                fQ = fQ_new
                Nacc+=1
                if iloop>=Ndimtau+1:#update external variable: k, l, sublatind.
                        Nacc_q += 1  # how many accepted steps of this type
                else:
                        Nacc_k += 1  # how many accepted steps of this bin
            else:
                Nrej+=1
                time0=time.time()
                if iloop<Ndimtau:
                    timagtime[iloop]=imagtime[iloop]
                elif iloop==Ndimtau:
                    tsublatind=copy.deepcopy(sublatind)
                    # print('accept trialsublatind. new sublatind=',sublatind)
                elif (iloop==Ndimtau+1):
                    tl=l
                elif (iloop==Ndimtau+2):
                    tj_coeff=j_coeff
                    
                func.metropolis_reject(iloop)
                # print('metropolis rejected!\n')

                
        time2=time.time()
        time_accrej+=(time2-time1)
        if (itt >= p.Nwarm and itt % p.tmeassure==0 and trialaccept==1): # below is measuring every p.tmeassure stepsand trialaccept==1
            Nmeassure += 1   # new meassurements
            W = abs(fQ[0])+fQ[1]             # this is the weight we are using
            f0, f1 = fQ[0]/W, fQ[1]/W        # the two measuring quantities
            # Next line needs CORRECTION for homework 
            # if last sublatind=1, means sig11; if last sublatind=2. means sig12.
            Pval[j_coeff,l]  += f0                  # V_physical : integral up to a constant
            # N[sublatind[0]-1]+=1
            # if itt<=p.Nwarm+1000:
            #     print('Pnorm2[itau,iQ]=',np.shape(Pval[itau,iQ]),np.shape(Pval),iQ)            
            # Pnorm2[l,iQ[0],iQ[1],iQ[2]]+=f1
            # if itt<=p.Nwarm+1000:
            #     print('Pnorm2[itau,iQ]=',Pnorm2[itau,iQ])     
            Pnorm     += f1                  # V_alternative : the normalization for the integral
            Pnorm_sum += f1                  # widetilde{V}_alternative, accumulated over all steps
            # Next line needs CORRECTION for homework 
            Wphs  = abs(f0)                  # widetilde{V}_{physical}, accumulated over all steps
            Pval_sum  += Wphs
            # if itt<=p.Nwarm+1000:
            #     print('f1, Pnorm, sum(Pnorm2)=',f1,Pnorm, np.sum(Pnorm2))
            # doing histogram of the simulation in terms of V_physical only.
            # While the probability for a configuration is proportional to f(X)+V0*fm(X), the histogram for
            # constructing g_i and h_{ij} is obtained from f(X) only. 
            # print('starting with f=', fQ, '\nstarting momenta=', momentum,'\n starting time=',imagtime)
            myweight.Add_to_K_histogram(dk_hist*Wphs,imagtime,sublatind)

            
            if itt>10000 and itt % (p.recomputew*p.tmeassure) == 0:
                # Now we want to check if we should recompute g_i and h_{ij}
                # P_v_P is V_physical/V_alternative*0.1
                P_v_P = Pval_sum/Pnorm_sum * 0.1 
                # We expect V_physical/V_alternative*0.1=P_v_P to be of the order of 1.
                # We do not want to change V0 too much, only if P_V_P falls utside the
                # range [0.25,4], we should correct V0.
                change_V0 = 0
                if P_v_P < 0.25 and itt < 0.3*p.Nitt:  # But P_v_P above 0.25 is fine
                    change_V0 = -1  # V0 should be reduced
                    V0norm    /= 2  # V0 is reduced by factor 2
                    Pnorm     /= 2  # V_alternative is proportional to V0, hence needs to be reduced too. 
                    Pnorm_sum /= 2  # widetilde{V}_alternative also needs to be reduced
                    # Pnorm2/=2
                if P_v_P > 4.0 and itt < 0.3*p.Nitt: # and P_v_P below 4 is also fine
                    change_V0 = 1   # V0 should be increased 
                    V0norm    *= 2  # actually increasing V0
                    Pnorm     *= 2
                    Pnorm_sum *= 2
                    # Pnorm2*=2
                if change_V0:       # V0 was changed. Report that. 
                    schange = ["V0 reduced to ", "V0 increased to"]
                    if ifprint==1:
                        print('%9.2fM P_v_P=%10.6f' % (itt/1e6, P_v_P), schange[int( (change_V0+1)/2 )], V0norm )
                    # Here we decied to drop all prior measurements if V0 is changed.
                    # We could keep them, but the convergence can be better when we drop them.
                    Pval = np.zeros(np.shape(Pval))
                    # N=np.zeros(2)
                    # Pnorm2=np.zeros(np.shape(Pval))
                    Pnorm = 0
                    Nmeasure = 0

                # about recomputing 
                # Next we should check if g_i and h_ij need to be recomputed.
                # This should not be done too often, and only in the first half of the sampling.
                if (c_recompute==0 and itt<0.5*p.Nitt and ifrecomp==1 ):
                    # At the beginning we recompute quite often, later not so often anymore
                    # as the per_recompute is increasing...
                    p.per_recompute = int(p.per_recompute*inc_recompute+0.5)
                    # We normalized f_m, hence all previous accumulated values are now of the order
                    # of 1/norm. We also normalize the new additions to histogram with similar value, 
                    # but 5-times larger than before.
                #Note: this normalization step does not matter what is the value from mtyweight.normalize_k_histogram.
                # the point is if we have many MC steps there might be some overflow issue
                # so we need to give a small factor in front of all current histograms.
                # also the future histograms should be with this factor.
                # here the factor 5 means later steps are more important since we have much better f_alter. so we give more weight to these steps.
                # # if everything is fine, dk_hist should not be so tiny and the two lines of code below will not be excecuted.
                    # dk_hist *= 5*myweight.Normalize_K_histogram()
                    # if dk_hist < 1e-8: # Once dk becomes too small, just start accumulating with weight 1.
                    #     dk_hist = 1.0
                    myweight.recompute()# Here we actually recompute g_i and h_{ij}.
                    # fQ = func(momentum,imagtime)*ut[l,imagtime[0]], V0norm * myweight( momentum,imagtime ) # And now we must recompute V0*f_m, because f_m has changed!
# update here!
                    fQ = func.update(imagtime,sublatind,l,j_coeff), V0norm * myweight(imagtime,sublatind)
                    # print('myweight( momentum,imagtime )=',myweight( momentum,imagtime ))
                    # if ifprint==1:
                    # print('{:9.2f}M recomputing rank={} f={} f_0={} dk_hist={} V0norm={}'.format(itt/1e6,rank, fQ[0],fQ[1],dk_hist,V0norm))
                    # print('p.per_recompute=',p.per_recompute)
                c_recompute += 1
                # print('c_recompute',c_recompute)
                if c_recompute>=p.per_recompute : c_recompute = 0 # counting when we will recompute next.        



        if (itt+1)% p.Ncout == 0 : # This is just printing information
            P_v_P = Pval_sum/Pnorm_sum * 0.1 # what is curent P_v_P
            ratio = (abs(fQ_new[0])+fQ_new[1])/(abs(fQ[0])+fQ[1]) # current ratio
            if ifprint==1:
                print('\nstep={}M, lj={} {},f_new={}  {}, ratio={} P_V_P={} '.format((itt+1)/1e6,l,j_coeff,fQ_new[0],fQ_new[1], ratio,P_v_P))
                # print('func.time_deepcopy,func.time_detectvar,func.time_detectGF,func.time_updateGF,func.time_sublatind,func.time_basis')
                # print(func.time_deepcopy,func.time_detectvar,func.time_detectGF,func.time_updateGF,func.time_sublatind,func.time_basis)
            if ifprint==0 and rank==0:
                print('step={}M\t rank={}\t lj={} {} , f_new={} {} P_V_P={} '.format((itt+1)/1e6,rank,l,j_coeff,fQ[0],fQ[1],P_v_P))

                # print('trial,    evaluate,    accrej,    others')
                # print(time_trial,time_evaluate,time_accrej,time_others)
                # print('k   tau   sublatind    basisk   basistau   basisind    test')
                # print(func.time_k,func.time_tau,func.time_sublatind,func.time_basist,func.time_basisk,func.time_basisind,func.time_test)

            # print('step={}M, P_val={}  Pnorm={}, f_old={} {} P_V_P={}'.format((itt+1)/1e6,Pval,Pnorm,fQ_new[0],fQ_new[1], fQ[0], fQ[1],P_v_P))
        time3=time.time()
        time_others+=(time3-time2)
# have to fix the Markov chain and then it is useful to use something like size of qx to normalize.   
    Pval *=  (2*lmax*V0norm / Pnorm) #  Finally, the integral is I = V0 *V_physical/V_alternative. *4 means we have 11 12 21 22.
    time_end=time.time()
    time_ttl=time_end-time_begin

    # if ifprint==1:
        # print('knum**3*taunum=',knum**3*taunum)
        # print('Pnorm',Pnorm)
        # print('Pnorm2 ave',np.mean(Pnorm2))
        # print('Pnorm2 sum',np.sum(Pnorm2))
    # print('time of rank {}: total={:.3f}s, trial={:.3f}s, evaluate={:.3f}s, accrej={:.3f}s, others={:.3f}s'.format(rank, time_ttl,time_trial,time_evaluate,time_accrej,time_others))
    return (Pval.real,myweight)


def Integratesvd_Parallel(func,qx,p,imax,lmax,ut,kbasis,sublatbasis):
    seed = np.random.randint(0, 10000) + rank*29
    # seed=0
    (Pval,myweight)=IntegrateByMetropolis_svdimp(func, qx, p,seed,lmax,imax,0)# choose different seeds for different proc.
    Pval = np.ascontiguousarray(Pval)
    # comm.barrier()
    # print('rank={},Pval={}'.format(rank,Pval[0,0,0]))
    # print('shape of Pval=',np.shape(Pval))
    if rank==0:
        all_Pval=np.zeros((nprocs,4,lmax,imax),dtype=Pval.dtype)
        ave_Pval=np.zeros_like(Pval)
    else:
        all_Pval=None
        ave_Pval=None
    comm.Gather(Pval,all_Pval,root=0)
    if rank==0:
        ave_Pval=np.sum(all_Pval,axis=(0))/nprocs

    return ave_Pval


def MC_test(U,T,nfreq,knum):
    # print('test started')
    mu=U/2
    beta=1/T
    # taunum=nfreq*2
    taunum=100
    lmax=12# number of svd coefficient.
    imax=8
    G11_tau,G12_tau,G22_tau,Gloc11_tau,Gloc22_tau=get_gtau(U,T)

    if rank==0:
        print('U={}, T={},knum={}, taunum={}'.format(U,T, knum,taunum))
    p = params()

    # ----------the original way of generating sigma3----------
    # Poff=P12(G12_tau,knum,nfreq,U,beta)
    # Qoff=Q12(G12_tau,knum,nfreq)
    # sigma2=sig2(G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta)
    # sigma2off=sig2offdiag(G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta)
    # sigma2loc=sig2(Gloc11_tau,G12_tau,Gloc22_tau,knum,nfreq,U,beta)
    sig4_1_11,sig4_1_12=allsig4_1(G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta)
    sig4_2_11,sig4_2_12=allsig4_2(G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta)
    sig3_1_11,sig3_1_12=allsig3_1(G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta)
    sig3_2_11,sig3_2_12=allsig3_2(G11_tau,G12_tau,G22_tau,knum,nfreq,U,beta)
    # G12_tau=np.zeros_like(G12_tau)

    GFs=(G11_tau,G12_tau,G22_tau)


    # generate basis of tau, k sublatind.
    # tau basis
    # note about parallel svd: sigular matrix u and v are not unique, there is a gauge freedom. So it is important to do svd in 1 proc and then broadcast.
    taulist=(np.arange(taunum+1))/taunum*beta#
    omlist=(2*np.arange(2*nfreq)+1-2*nfreq)*np.pi/beta 
    ker=basis.fermi_kernel(taulist,omlist,beta)
    # only do svd in rank0, than broadcast
    ut=np.empty((lmax,taunum+1),dtype=float)
    if rank==0:
        ut,sig,v=basis.svd_kernel_fast(ker,lmax)
        # if diag this is all coeffs. if offdiag, we only need 1st, 3rd, 5th,...
        print('sigular values:',sig)
    ut = np.ascontiguousarray(ut)
    comm.Bcast(ut, root=0)

    # kbasis
    kbasis=np.empty((imax,knum,knum,knum),dtype=float)
    if rank==0:
        kbasis=basis.gen_kbasis(imax,knum)
    kbasis = np.ascontiguousarray(kbasis)
    comm.Bcast(kbasis, root=0)    

    #sublattice indices.
    sublatind_basis=np.array([[1,0,0,0],
                              [0,1,0,0],
                              [0,0,1,0],
                              [0,0,0,1]])
    # sublatind_basis=np.array([[1,1,1,1],
    #                           [1,-1,1,-1],
    #                           [1,1,-1,-1],
    #                           [1,-1,-1,1]])/2# Hadamard matrix. this might be better.


    # some diagrams for testing
    perm31=np.array([2,3,4,5,0,1])
    perm32=np.array([2,5,4,1,0,3])
    perm41=np.array([2,3,4,5,6,7,0,1])
    perm42=np.array([2,7,4,1,6,3,0,5])
    #                k,k',q1,q2,tau,tau1
    dep31=np.array([[1,0,1,0,0,1],
                    [0,1,-1,0,0,1],
                    [1,0,0,1,1,-1],
                    [0,1,0,-1,1,-1],
                    [1,0,0,0,-1,0],
                    [0,1,0,0,-1,0]])
    dep32=np.array([[1,0,1,0,0,1],
                    [0,1,0,0,1,0],
                    [1,0,0,1,1,-1],
                    [0,1,1,0,0,-1],
                    [1,0,0,0,-1,0],
                    [0,1,0,1,-1,1]])    
    dep41=np.array([[1,0,1,0,0,0,0,1],
                    [0,1,-1,0,0,0,0,1],
                    [1,0,0,1,0,0,1,-1],
                    [0,1,0,-1,0,0,1,-1],
                    [1,0,0,0,1,1,-1,0],
                    [0,1,0,0,1,1,-1,0],
                    [1,0,0,0,0,-1,0,0],
                    [0,1,0,0,0,-1,0,0]])
    dep42=np.array([[1,0,1,0,0,0,0,1],
                    [0,1,0,0,0,1,0,0],
                    [1,0,0,1,0,0,1,-1],
                    [0,1,1,0,0,0,0,-1],
                    [1,0,0,0,1,1,-1,0],
                    [0,1,0,1,0,0,-1,1],
                    [1,0,0,0,0,-1,0,0],
                    [0,1,0,0,1,-1,1,0]])   
    sublatindinit4=np.array([1,2,1,2,1,2,1,2])
    # fun=diag_def_new.FuncNDiagNew(T,U,knum,taunum,nfreq,3,ut,perm31,GFs,dep31,4)# sig3_1_111
    fun=imp_diag_def.FuncNDiagNew(T,U,taunum,nfreq,4,ut,perm42,GFs,dep42,sublatindinit4,8)# sig4_2_111
    # fun1=diag_def_new240706.FuncNDiagNew(T,U,knum,taunum,nfreq,3,ut,perm31,GFs,dep31,4)
    qx=np.zeros((2,lmax),dtype=float)# first dimension means 11 12 21 22, and then svd basis u_l, then kspace basis m_i
    # qx=np.zeros(taunum)
    # (Pval_raw,myweight)=IntegrateByMetropolis_svd(fun, qx, p,0,lmax,ut,1)#serial test (func, qx, p,seed,lmax,ut,ifprint=1)
    Pval_raw=Integratesvd_Parallel(fun,qx,p,imax,lmax,ut,kbasis,sublatind_basis)#parallel test
    
    if rank==0:
        Cliab=np.sum(Pval_raw[:,None,:,:]*sublatind_basis[:,:,None,None],axis=0)# restore c_li(a,b)
        clk11_raw=basis.restore_clk(Cliab[0],kbasis)
        # clk12_raw=basis.restore_clk((Cliab[1]+Cliab[2])/2,kbasis)
        clk12_raw=basis.restore_clk(Cliab[1],kbasis)
        clk22_raw=basis.restore_clk(Cliab[3],kbasis)
        Pval11=sym_ave(clk11_raw,knum,0)# average of all k points with the same symmetry. offdiag choose 1, diag choose 0
        Pval12=sym_ave(clk12_raw,knum,1)# symmetry of the quantity. offdiag choose 1, diag choose 0

        ori_grid=(np.arange(nfreq*2)+0.5)/(nfreq*2)*beta
        # simp_grid=(np.arange(taunum+1))/taunum
        simp_grid=taulist
        # this is to show the k-dependence of GF, but i group them into different symmetries.
        k_displayed=np.zeros((knum,knum,knum))
        symgroup=0
        for kx in np.arange(10):
            for ky in np.arange(10):
                for kz in np.arange(10):
                    if k_displayed[kx,ky,kz]==0:# if this point is not checked
                        all_sym_kpoints=lib.sym_mapping(kx,ky,kz,knum).tolist()# find all equivalent points
                        unique_set = set(tuple(x) for x in all_sym_kpoints)
                        all_unique_sym_kpoints = [list(x) for x in unique_set]# but there might be a lot of duplicates.
                        symgroup+=1# they all belongs to the same symgroup.
                        # print(all_unique_sym_kpoints)
                        q=all_unique_sym_kpoints[0]

                        # about the original quantity
                        # quant11=sigma3_1_111[:,q[0],q[1],q[2]].real
                        # quant12=sigma3122[:,q[0],q[1],q[2]].real
                        quant11=sig4_2_11[:,q[0],q[1],q[2]].real
                        quant12=sig4_2_12[:,q[0],q[1],q[2]].real
                        interpolator11 = interp1d(ori_grid, quant11, kind='linear', fill_value='extrapolate')
                        interpolator12 = interp1d(ori_grid, quant12, kind='linear', fill_value='extrapolate')
                        G11=interpolator11(simp_grid)
                        G12=interpolator12(simp_grid)
                        coeff_BF11=basis.inner_prod(G11,ut)
                        coeff_BF12=basis.inner_prod(G12,ut)
                        # print('coeff_BF=',coeff_BF)
                        restored_quantityBF11=basis.restore_Gf(coeff_BF11,ut)
                        restored_quantityBF12=basis.restore_Gf(coeff_BF12,ut,1)


                        coeffs11=Pval11[:,q[0],q[1],q[2]].real
                        coeffs12=Pval12[:,q[0],q[1],q[2]].real
                        # print('coeffs for MC and BF{}: \n{}\n{}'.format(q,coeffs11,coeff_BF11))
                        plt.plot(coeff_BF11,label='BF11')
                        plt.plot(coeffs11,label="MC11")
                        plt.plot(coeff_BF12,label='BF12')
                        plt.plot(coeffs12,label="MC12")
                        plt.xlabel('nth')
                        plt.legend()
                        plt.show()

                        restored_quantity11=basis.restore_Gf(coeffs11,ut)
                        plt.plot((np.arange(nfreq*2)+0.5)/(nfreq*2)*beta ,quant11,label='BF11')
                        plt.plot(taulist ,G11,label='BF11 splined')
                        plt.plot(taulist,restored_quantityBF11,label='BF11 restored')
                        plt.plot(taulist,restored_quantity11,label='MC11')
                        # plt.plot(restored_quantityBF,label='BF')
                        plt.legend()
                        plt.title('k=[{},{},{}], factor={}, symgroup={} diagonal'.format(q[0],q[1],q[2],q[3],symgroup))
                        plt.show()
                        restored_quantity12=basis.restore_Gf(coeffs12,ut,1)
                        plt.plot((np.arange(nfreq*2)+0.5)/(nfreq*2)*beta ,quant12,label='BF12')
                        plt.plot(taulist ,G12,label='BF12 splined')
                        plt.plot(taulist,restored_quantityBF12,label='BF12 restored')
                        plt.plot(taulist,restored_quantity12,label='MC12')
                        plt.legend()
                        plt.title('k=[{},{},{}], factor={}, symgroup={} off-diagonal'.format(q[0],q[1],q[2],q[3],symgroup))
                        plt.show()
                        for q in all_unique_sym_kpoints:
                            k_displayed[q[0],q[1],q[2]]+=1
                            
                            
    return 0#Pval


def Summon_Integrate_Parallel_impurity(func,p,lmax,ut):
    '''
    This is for getting the impurity diagram. test if we try to block the hopping of sublatind and k?
    '''
    # knum=np.shape(kbasis)[1]

    qx=np.zeros((2,lmax),dtype=float)
    seed = np.random.randint(0, 10000) + rank*29
    (Pval,myweight)=IntegrateByMetropolis_svdimp(func, qx, p,seed,lmax)# choose different seeds for different proc.
    Pval = np.ascontiguousarray(Pval)
    if rank==0:
        all_Pval=np.zeros((nprocs,2,lmax),dtype=Pval.dtype)
        ave_Pval=np.zeros_like(Pval)
    else:
        all_Pval=None
        ave_Pval=None
    comm.Gather(Pval,all_Pval,root=0)
    cl_loc_11=np.zeros(lmax)
    cl_loc_22=np.zeros(lmax)
    if rank==0:
        ave_Pval=np.sum(all_Pval,axis=(0))/nprocs
        cl_loc_11=ave_Pval[0]
        cl_loc_22=ave_Pval[1]
        cl_loc_22[1::2]=-cl_loc_22[1::2]

        # why i am doing this? see note:see note: 240502 high order diagrams and monte carlo
        #my understanding is everytime cut the diagram, half are 22 and another half is 11. so we only get half of them…
        # however for some complicated diagrams this factor may disappear, or changed. 

    return (cl_loc_11+cl_loc_22)/2


if __name__ == "__main__":
    U=10.
    T=0.25
    knum=10
    nfreq=500
    MC_test(U,T,nfreq,knum)