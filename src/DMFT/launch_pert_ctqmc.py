#!/usr/bin/env python
# Author: Kristjan Haule, March 2007-2017
from scipy import * 
import os,sys,subprocess
import numpy as np
import matplotlib.pyplot as plt 
from scipy import integrate
sys.path.append('../perturbation/')
import hilbert
import perturb_lib
"""
This module runs ctqmc impurity solver for one-band model.
The executable should exist in directory params['exe']
This controls the workflow of dmft.
"""

   



def mycopy(filename, dir,it=-1):
    '''
    copy file to desired dir. it=-1 means use the original name like Gf.out without a number like Gf.out.1
    '''
    if it>=0:
        cmd = 'cp '+filename+' '+dir+filename+'.'+str(it)
    if it==-1:
        cmd = 'cp '+filename+' '+dir+filename
    subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr)  
    return 0

def CreateInputFile(params):
    " Creates input file (PARAMS) for CT-QMC solver"
    f = open('PARAMS', 'w')
    print('# Input file for continuous time quantum Monte Carlo', file=f)
    for p in params:
        print(p, params[p][0], '\t', params[p][1], file=f)
    f.close()

def DMFT_SCC(fDelta,splitting):
    """This subroutine creates Delta.inp from Gf.out for DMFT on bethe lattice: Delta=t^2*G
    If Gf.out does not exist, it creates Gf.out which corresponds to the non-interacting model
    In the latter case also creates the inpurity cix file, which contains information about
    the atomic states.
    """
    fileGf = 'Gf.out'
    # filesig='./trial_sigma/{}_{}.dat'.format(Uc,T)# use this filesig, DMFT will start repeatedly from a same trial sigma, to test if it is stable.
    filesig='Sig.out'
    #get sigma
    if (os.path.exists(filesig)): 
        sigma = np.loadtxt(filesig)
    else:# generate a trial sigma
        print("trial sigma cannot found...preparing trial sigma...")
        # return 0
        sigma=np.zeros((500,5),dtype=complex)
        for i in np.arange(500):
            omega=(2*i+1)*np.pi/params['beta'][0]
            sigma[i,0]=omega #omega
            sigma[i,1]=params['mu'][0]+splitting# Gaa,up,real
            sigma[i,2]=0# Gaa, up, imag
            sigma[i,3]=params['mu'][0]-splitting# Gaa, dn, real
            sigma[i,4]=0# Gaa, dn, imag
        print('writing trial sigma file')
        f = open(filesig, 'w')
        for i in range(np.shape(sigma)[0]):# consider the case that we may have many columns of Gf
            # print(Gf[i,0], 0.25*Gf[i,1], 0.25*Gf[i,2], file=f) # This is DMFT SCC: Delta = t**2*G (with t=1/2)
            # print(np.shape(Deltafile))
            for k in np.arange(np.shape(sigma)[1]):
                print(sigma[i,k].real,end='\t', file=f)# print in the same line
            print('', file=f)# switch to another line
        f.close()
        print("trial sigma finished",np.shape(sigma))

        f = open(params['cix'][0], 'w')
        print(icix, file=f)
        f.close()
    print('Getting Non-interacting DOS')


    # Preparing input file Delta.inp
    # print('Generating and writing Delta...')
    # f = open(fDelta, 'w')
    # Delta=calc_Delta_afm(sigma,elist,dos,params['mu'][0])
    # for i in range(np.shape(sigma)[0]):# consider the case that we may have many columns of Gf
    #     for k in np.arange(5):
    #         print(Delta[i,k],end='\t', file=f)# print in the same line
    #     print('', file=f)# switch to another line
    # f.close()
    freq_num=500
    om = sigma[:freq_num,0].real
    Sg_A = sigma[:freq_num,1]+sigma[:freq_num,2]*1j
    Sg_B = Uc-sigma[:freq_num,1]+sigma[:freq_num,2]*1j
    # Sg_B = sigma[:freq_num,3]+sigma[:freq_num,4]*1j
    # print(type(Sg_A),type(om),type(params['mu'][0]))

    #also, prepare the original delta for comparison.
    
    Dlt_A,Dlt_B ,g_A,g_B= hilbert.SCC_AFM(W, om, params['beta'][0], params['mu'][0], params['U'][0], Sg_A, Sg_B, False)
    # Dlt_A,Dlt_B =perturb_lib.Delta_DMFT( Sg_A, Sg_B,Uc,T)
    nA=perturb_lib.particlenumber1D(g_A,1/T)
    nB=perturb_lib.particlenumber1D(g_B,1/T)
    m=np.abs(nA-nB)
    print('m={:.5f}'.format(m))
    # plt.plot(ori_Dlt_A.real,label='delta11 real')
    # plt.plot(ori_Dlt_A.imag,label='delta11 imag')
    # plt.plot(ori_Dlt_A_HT.real,label='delta11_HT real')
    # plt.plot(ori_Dlt_A_HT.imag,label='delta11_HT imag')
    # plt.legend()
    # plt.grid()
    # plt.show()
    f = open(fDelta, 'w')
    for i,iom in enumerate(om):
        print(iom, Dlt_A[i].real, Dlt_A[i].imag, Dlt_B[i].real, Dlt_B[i].imag, file=f) 
    f.close()
    # cmd = 'cp Delta.inp '+dir+'ori_Delta.inp.'+str(it)
    # subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) 
    # print('Delta file is done!')
    return Dlt_A,Dlt_B,m


def Diff(fg1, fg2):
    data1 = np.loadtxt(fg1).transpose()
    data2 = np.loadtxt(fg2).transpose()
    diff = np.sum(abs(data1-data2))/(np.shape(data1)[0]*np.shape(data1)[1])
    return diff


if __name__ == '__main__':

    filedos='../perturbation/DOS_3D.dat'
    # Uc=8
    # beta=4
    splitting=0.5
    mixing=0.3
    if (len(sys.argv)==3):
        Uc=float(sys.argv[1])
        T=float(sys.argv[2])
        print('T=',T)
        print('Uc=',Uc)
        dir='../files_ctqmc/{}_{}/'.format(Uc,T)
        # dir='../files_DMFT/{}_{}/'.format(Uc,T)
    else:
        print('seems input format is not correct...')

    if os.path.exists(dir):
        print('already have this directory: ', dir)
        # CAREFULL! delete any previous files
        subprocess.call('rm '+dir+'*', shell=True) 
    else:
        print('directory does not exist... make a new one')
        cmd_newfolder='mkdir '+dir
        subprocess.call(cmd_newfolder, shell=True)
        # mpirun --mca orte_base_verbose 10 --mca plm_base_verbose 10 -np 2 ./ctqmc

    params = {"exe":   ["mpirun --mca pml ob1 --mca btl tcp,self -np 2 ./ctqmc",          "# Path to executable"],
          "U":     [Uc,                 "# Coulomb repulsion (F0)"],
          "mu":    [Uc/2.,              "# Chemical potential"],
          "beta":  [1/T,                "# Inverse temperature"],
          "M" :    [2e7,                "# Number of Monte Carlo steps"],
          "mode":  ["SH",               "# S stands for self-energy sampling, M stands for high frequency moment tail"],
          "cix":   ["one_band.imp",     "# Input file with atomic state"],
          "Delta": ["../cache_ctqmc/Delta.inp",        "# Input bath function hybridization"],
          "tsample":[200,               "# how often to record the measurements" ],
          "nom":   [80,                 "# number of Matsubara frequency points to sample"],
        "svd_lmax":[30,                 "# number of SVD functions to project the solution"],
          "aom":   [1,                  "# number of frequency points to determin high frequency tail"],
          "GlobalFlip":[1000000,         "# how often to perform global flip"],
          "fastFilesystem":[1,          "#to produce Delta.tau"],
          }
    icix="""# Cix file for cluster DMFT with CTQMC
    # cluster_size, number of states, number of baths, maximum matrix size
    1 4 2 1
    # baths, dimension, symmetry, global flip
    0       1 0 0
    1       1 1 0
    # cluster energies for unique baths, eps[k]
    0 0
    #   N   K   Sz size F^{+,dn}, F^{+,up}, Ea  S
    1   0   0    0   1   2         3        0   0
    2   1   0 -0.5   1   0         4        0   0.5
    3   1   0  0.5   1   4         0        0   0.5
    4   2   0    0   1   0         0        0   0
    # matrix elements
    1  2  1  1    1    # start-state,end-state, dim1, dim2, <2|F^{+,dn}|1>
    1  3  1  1    1    # start-state,end-state, dim1, dim2, <3|F^{+,up}|1>
    2  0  0  0
    2  4  1  1   -1    # start-state,end-state, dim1, dim2, <4|F^{+,up}|2>
    3  4  1  1    1
    3  0  0  0
    4  0  0  0
    4  0  0  0
    HB2                # Hubbard-I is used to determine high-frequency
    # UCoulomb : (m1,s1) (m2,s2) (m3,s2) (m4,s1)  Uc[m1,m2,m3,m4]
    0 0 0 0 0.0
    # number of operators needed
    0
    """
    subprocess.call('rm ctqmc.log Delta.inp Deltat.inp Gf.out PARAMS PPSigma.OCA Sig.out Delta.tau.00.000 Delta.tau.01.000 rDelta.tau.01.000 rDelta.tau.00.000 sig12.dat', shell=True) 
    subprocess.call('rm Aw.out.001 Aw.out.000 Gcoeff.dat gs_qmc.dat Sig.outB Sig.outD Sw.dat Probability.dat Gt.dat Gw.dat histogram.dat nohup_imp.out.000 ctqmc.log Delta.inp Deltat.inp Gf.out PARAMS PPSigma.OCA Sig.out Delta.tau.00.000 Delta.tau.01.000 rDelta.tau.01.000 rDelta.tau.00.000 ', shell=True) 
    for i in np.arange(10):
        subprocess.call('rm status.00{}'.format(i), shell=True)
    subprocess.call('rm status.010', shell=True)
    subprocess.call('rm status.011', shell=True)
    # Number of DMFT iterations
    Niter = 100
    diff_arr=np.zeros(Niter)
    # Creating parameters file PARAMS for qmc execution
    CreateInputFile(params)

    #get dos
    x, Di = np.loadtxt(filedos).T
    W = hilbert.Hilb(x,Di)
    print('Non-interacting DOS finished')

    cmd_params = 'cp '+'PARAMS'+' '+dir+'PARAMS'
    subprocess.call(cmd_params, shell=True)
    cmd_cix = 'cp '+params['cix'][0]+' '+dir+params['cix'][0]
    subprocess.call(cmd_cix, shell=True)
    # cmd_dos = 'cp '+filedos+' '+dir+filedos
    # subprocess.call(cmd_dos, shell=True)
    ifconv=0
    itstop=Niter
    for it in range(Niter):
        mixing=max(0.3,1-0.1*it)
        # Constructing bath Delta.inp from Green's function
        # note: first run the non-pert version and get self-energy for reference.
        # Then,calculate pert-version. then the final sig.out left is the sig function after perturbation.
        deltaA,deltaB,m=DMFT_SCC(params['Delta'][0],splitting)# DMFT without perturbation
        if it==0:
            prvdeltaA=deltaA
            prvdeltaB=deltaB
            prvm=m
        else:
            deltaA=mixing*deltaA+(1-mixing)*prvdeltaA
            deltaB=mixing*deltaB+(1-mixing)*prvdeltaB
            prvdeltaA=deltaA
            prvdeltaB=deltaB
        # freq_num=500
        # om=(2*np.arange(500)+1)*np.pi*T
        # f = open(params['Delta'][0], 'w')
        # for i,iom in enumerate(om):
        #     print(iom, deltaA[i].real, deltaA[i].imag, deltaB[i].real, deltaB[i].imag, file=f) 
        # f.close()
            # Running ctqmc
        print('Running ---- qmc itt.: ', it, '-----')
        subprocess.call(params['exe'][0], shell=True,stdout=sys.stdout,stderr=sys.stderr)

        # Some copying to store data obtained so far (at each iteration)
        mycopy('Gf.out',dir,it)
        # cmd = 'cp Gf.out '+dir+'Gf.out.'+str(it)
        # subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr)  # copying Gf
        mycopy('Sig.out',dir,it)
        # cmd = 'cp Sig.out '+dir+'Sig.out.'+str(it)
        # subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) # copying Sig
        mycopy('ctqmc.log',dir,it)
        # cmd = 'cp ctqmc.log '+dir+'ctqmc.log.'+str(it)
        # subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) # copying log file
        mycopy('delta.inp',dir,it)
        if it>0:
            diff = Diff('Gf.out', '{}Gf.out.'.format(dir)+str(it-1))
            diffm=np.abs(m-prvm)
            print('Diff={:.6f}, diffm={:.6f}'.format(diff,diffm))
            diff_arr[it-1]=diff
            if ((diffm/m<0.002) or diffm<0.0005) and ifconv==0:
                ifconv=1
                itstop=it+10
                print('converged, waiting for 10 iterations. will stop after',itstop)
                # break
            if it>=itstop:
                break
            prvm=m
    # finally, when the iteration is done, copy things again:
    mycopy('Gf.out',dir,-1)
    mycopy('Sig.out',dir,-1)
    mycopy('ctqmc.log',dir,-1)
    mycopy('delta.inp',dir,-1)
    mycopy('histogram.dat',dir,-1)
    # remove everything....
    # subprocess.call('rm Aw.out.001 Aw.out.000 sig12.dat Gcoeff.dat gs_qmc.dat Sig.outB Sig.outD Sw.dat Probability.dat Gt.dat Gw.dat histogram.dat nohup_imp.out.000 ctqmc.log Delta.inp Deltat.inp Gf.out PARAMS PPSigma.OCA Sig.out Delta.tau.00.000 Delta.tau.01.000 rDelta.tau.01.000 rDelta.tau.00.000 ori_Delta.inp', shell=True) 
    # for i in np.arange(10):
    #     subprocess.call('rm status.00{}'.format(i), shell=True)
    # subprocess.call('rm status.010', shell=True)
    # subprocess.call('rm status.011', shell=True)
        