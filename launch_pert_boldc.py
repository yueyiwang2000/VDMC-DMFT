#!/usr/bin/env python
# Author: Kristjan Haule, March 2007-2017
# use the new impurity solver, but my Hilbert.
from scipy import *
from scipy import interpolate
from numpy import *
import json
import os,sys,subprocess
import hilbert
import matplotlib.pyplot as plt 
# import perturb
#from pylab import *

"""
This module runs ctqmc impurity solver for one-band model.
The executable should exist in directory params['exe']
"""
# fileS = 'Sig.out'
#fileG = 'Gf.out'
fileS = 'Sig.OCA'
fileG = 'Gf.OCA'
fileD = 'Delta.inp'

# default of Uc and Beta. But usually these are given in command line.
Uc=10.0
beta=2.0
if (len(sys.argv)==2):
    beta=float(sys.argv[1])
    print('usually we need 2 parameters U and T.')
    print('beta=',beta)
if (len(sys.argv)==3):
    Uc=float(sys.argv[1])
    T=float(sys.argv[2])
    print('T=',T)
    print('Uc=',Uc)

if (len(sys.argv)==4):
    Uc=float(sys.argv[1])
    T=float(sys.argv[2])
    mode=float(sys.argv[3])#0: no perturbation. 1: with perturbation
    print('T=',T)
    print('Uc=',Uc)
    print('mode=',mode)
if mode ==1:
    dir='./files_pert_boldc/{}_{}/'.format(Uc,T)
elif mode==0:
    dir='./files_boldc/{}_{}/'.format(Uc,T)

if os.path.exists(dir):
    print('already have this directory: ', dir)
    # CAREFULL! delete any previous files
    subprocess.call('rm '+dir+'*', shell=True) 
else:
    print('directory does not exist... make a new one')
    cmd_newfolder='mkdir '+dir
    subprocess.call(cmd_newfolder, shell=True)
subprocess.call('rm Gf.OCA PARAMS PPSigma.OCA Sig.OCA ', shell=True)
Ms = 2e6
params={
    "exe"       : "mpirun ./boldc", # Path to executable
    "dos"       : "DOS_3D.dat",     # non-interacting DOS
    "U"         : Uc,               # Coulomb U
    "mu"        : Uc/2.,            # chemical potential
    "beta"      : 1/T,               # inverse temperature
    "Norder"    : 2,                # the maximum perturbation order
  "N_min_order" : 3,                # the minimal order at which we run MC (the rest analytic)
    "Ms"        : Ms,               # Number of Monte Carlo steps at each iteration
    "Nbath"     :  2,               # paramagnetic
    "mix"       : 0.2,              # mixing of pseudo self-energy
    "Nitt"      : 100,              # Number of iterations of the loop
    "cmpPhysG"  : False,            # Should we compute G00 at each step?
    "iseed"     :  0,               # seed for random number generator
    "V0norm"    : 0.001,            # constant to reweight higher order diagrams
    "tsample"   : 5,                # How often to make a measurement
    "svd_lmax"  : 13,               # using SVD functions up to cutoff
    "Delta"     : fileD,            # Input bath function hybridization
    "Nt"        : 300,              # How many times we try to change time before we consider change of diagram
    "Nd"        : 1,                # How many times we try to change diagram before we consider next change of time
    "converge_OCA": True,           # If histogram is not present, we coverge OCA before we start MC
    "diff_stop" : 0.01,             # condition to stop PP-loop
    "rescale"   : 0.4,              # during PP-loop we rescale last step values. But how much?
    "fastFilesystem": True,         # print extra file information
    "d_fraction": 3                 # We keep (Norder/d_fraction) propagators equal when proposing different diagram.
}


def CreateInputFile(params):
    " Creates input file (PARAMS) for bolc-ctqmc solver"
    f = open('PARAMS', 'w')
    print(json.dumps(params,indent=4,separators=(',', ': ')), file=f)
    f.close()

def DMFT_SCC(W, fDelta,opt):
    """This subroutine creates Delta.inp from Gf.out for DMFT on bethe lattice: Delta=t^2*G
    If Gf.out does not exist, it creates Gf.out which corresponds to the non-interacting model
    In the latter case also creates the inpurity cix file, which contains information about
    the atomic states.
    """
    if (os.path.exists(fileS)): # If output file exists, start from previous iteration
        # Gf = io.read_array(fileGf, columns=(0,-1), lines=(1,-1))
        # In the new Python, io.readarray is dropped and we should use loadtxt instead!
        Sf = loadtxt(fileS).T
        om = Sf[0]# matsubara freqs
        Sg_A = Sf[1,:]+Sf[2,:]*1j
        Sg_B = Sf[3,:]+Sf[4,:]*1j
    else: # otherwise start from non-interacting limit
        print('Starting from non-interacting model')
        # Sf=[]
        om = (2*arange(500)+1)*pi/params['beta']
        Sg_A=(Uc/2.+0.01)*ones_like(om)
        Sg_B=(Uc/2.-0.01)*ones_like(om)
        # for iom in om:
        #     Sf.append([iom,Sg_A,0,Sg_B,0])
        # Sf = array(Sf).T
        f = open(fileS, 'w')
        for i,iom in enumerate(om):
            print(iom, Sg_A[i].real, Sg_A[i].imag, Sg_B[i].real, Sg_B[i].imag, file=f) 
        f.close()
    
    if (abs(om[0]-pi/params['beta'])>1e-6):
        # for the case that mstsubara freqs does not math the specified temperature.
        print('It seems input '+fileS+' correspond to different temperature!')
        return 0
        # om = (2*arange(len(om))+1)*pi/params['beta']
        # for i in range(1,5):
        #     fS=interpolate.UnivariateSpline(Sf[0],Sf[i,:],s=0)
        #     Sf[i,:] = fS(om)
    

    if opt==0:
        Dlt_A,Dlt_B = hilbert.SCC_AFM(W, om, params['beta'], params['mu'], params['U'], Sg_A, Sg_B, False)
        f = open(fDelta, 'w')
        for i,iom in enumerate(om):
            print(iom, Dlt_A[i].real, Dlt_A[i].imag, Dlt_B[i].real, Dlt_B[i].imag, file=f) 
        f.close()
        # Dlt_A,Dlt_B = perturb.Delta_DMFT(Sg_A,Sg_B,Uc,T,20)
    elif opt==1:
        # Dlt_A,Dlt_B=perturb.Delta_pert_DMFT(Sg_A,Sg_B,Uc,T,10)
        cmd_pert='mpirun -np 8 python perturb_mpi.py {} {}'.format(Uc,T)
        subprocess.call(cmd_pert, shell=True)
    # Preparing input file Delta.inp

    

def Diff(fg1, fg2):
    data1 = loadtxt(fg1).T
    data2 = loadtxt(fg2).T
    n_points = where(data1[0,:]>10.0)[0][0]
    diff = sum(abs(data1[:,:n_points]-data2[:,:n_points]))/(n_points*shape(data1)[0])
    return diff

# Number of DMFT iterations
Niter = 10
diff_arr=zeros(Niter)
# Creating parameters file PARAMS for qmc execution
CreateInputFile(params)
# non0interacting DOS and its Hilbert transform
x, Di = loadtxt(params['dos']).T
# plt.plot(x,Di)
# plt.show()
W = hilbert.Hilb(x,Di)
# copy PARAMS and dos to the dir
cmd_params = 'cp '+'PARAMS'+' '+dir+'PARAMS'
subprocess.call(cmd_params, shell=True)
cmd_dos = 'cp '+params['dos']+' '+dir+params['dos']
subprocess.call(cmd_dos, shell=True)


for it in range(1,Niter+1):
    # Constructing bath Delta.inp from Green's function


    if mode ==0:
        #calculate non-perturbation version firstly.
        DMFT_SCC(W, params['Delta'],0)
        # Some copying to store data obtained so far (at each iteration)                                                                                            
        cmd = 'cp '+fileD+' '+dir+'ori_'+fileD+'.'+str(it)
        subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr)  # copying Gf
    
        # Running boldc
        print('Running ---- bold_impurity solver itt.: ', it, '-----')
    
        subprocess.call(params['exe'], shell=True,stdout=sys.stdout,stderr=sys.stderr)
    
        # Some copying to store data obtained so far (at each iteration)
        cmd = 'cp '+fileG+' '+dir+'ori_'+fileG+'.'+str(it)
        subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr)  # copying Gf
        cmd = 'cp '+fileS+' '+dir+'ori_'+fileS+'.'+str(it)
        subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) # copying Sig
        cmd = 'cp ctqmc.log '+dir+'ori_'+'ctqmc.log.'+str(it)
        subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) # copying log file
    
    
    if mode ==1:
        # then, getting perturbation version
        DMFT_SCC(W, params['Delta'],1)
        # Some copying to store data obtained so far (at each iteration)                                                                                            
        cmd = 'cp '+fileD+' '+dir+fileD+'.'+str(it)
        subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr)  # copying Gf
    
        # Running boldc
        print('Running ---- pert_bold_impurity solver itt.: ', it, '-----')
    
        subprocess.call(params['exe'], shell=True,stdout=sys.stdout,stderr=sys.stderr)
    
        # Some copying to store data obtained so far (at each iteration)
        cmd = 'cp '+fileG+' '+dir+fileG+'.'+str(it)
        subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr)  # copying Gf
        cmd = 'cp '+fileS+' '+dir+fileS+'.'+str(it)
        subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) # copying Sig
        cmd = 'cp ctqmc.log '+dir+'ctqmc.log.'+str(it)
        subprocess.call(cmd, shell=True,stdout=sys.stdout,stderr=sys.stderr) # copying log file
            # Constructing bath Delta.inp from Green's function

    
    
    
 
    if it>1:
        if mode==1:
            diff = Diff(fileG, dir+fileG+'.'+str(it-1))
        elif mode==0:
            diff = Diff(fileG, dir+'ori_'+fileG+'.'+str(it-1))
        print('Diff=', diff)
        #if (diff<3e-4 and params["Ms"]==Ms):
        #    params["Ms"] *= 3
        #    CreateInputFile(params)
        #if (diff<6e-5): break
        diff_arr[it-1]=diff
        if (diff<1e-6): break
        
diff_dir=dir+'diff.log'
f = open(diff_dir, 'w')
for iter in arange(it):
    print(diff_arr[iter], file=f) 
f.close()

# clean everything in the main directory. 
# everything should be able to be find in the specified dir.
# if needed, this line could be commented.
subprocess.call('rm ctqmc.log Delta.inp Deltat.inp Gf.OCA PARAMS PPSigma.OCA Sig.OCA uls.dat', shell=True) 