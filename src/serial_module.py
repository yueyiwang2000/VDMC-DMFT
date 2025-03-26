import os,sys,subprocess
sys.path.append('../python_src/')
import math
import numpy as np
from perturb_lib import *
import time


def bubble_mpi(operation,knum,nfreq,sym,*args):
    '''
    This function aims to give a generic method to calculate 'bubble-like' quantities in parallel. It assigns jobs to different processes.
    The most straightforward 'bubble' is the polarization function, but there will also be other examples, see my qualifier paper.
    To reduce the cost we tried to expolit the symmetry. However different quantities might have different symmetries. An example is G11 and G12 in this project.
    2 parts of parameters: one for the function itself, another is for the operation function, which are *args. 
    
    Parameters:
    operation: serial function called to do the actual calculation. They are all in fft_convolution.py
    knum: # of kpoints in each dimension. by default knum=10.
    nfreq: # of positive matsubara freqs.
    sym: symmetry of the output quantity (bubble) on k-space. sym=11: A_k=A_-k    sym=12:A_k=-A_-k. A is the result wanted from this function.
    *args: U, beta, Pk, Gk, fullsig.... typical inputs for the operation function. They are only used in operation function and packed as *args.

    Note:
    1. Some quantities may need some special care. Like some Green's functions scales like 1/omega, which means it's not well defined in imaginary time domain. 
    But this should be taken cared by the function called but not this function here.
    2. Generically this function can be used in many cases, and in each case the symmetry may vary.
    Examples are P11_k=P11_-k, P12_k=P12_-k; Sig11_k=Sig11_-k Sig12_k=-Sig12_-k,....
    3. This function does not works very fast.... only use it when the function itself take a long time.

    This is the serial version of mpi_module.
    

    '''
    N=2*nfreq
    # time_beg=time.time()
    max_sym_index,essential_kpoints, sym_array=calc_sym_array(knum)
    nprocs=1
    rank=0
    # print('beginning of bubble_mpi',nprocs)
    if sym==12:
        power=1
    elif sym==11:
        power=2
    else:
        print('please specify the symmetry correctly! 12 or 11!')
        return 0
    if knum % 2 != 0:
        print('knum should be a even number!')
        return 0
    
    # max_sym_index,essential_kpoints, sym_array=calc_sym_array(knum)
    # time1=time.time()
    if rank ==0:
        # devide max_sym_index for nproc processors
        ave, res = divmod(max_sym_index, nprocs)
        counts = [ave + 1 if p < res else ave for p in range(nprocs)]
        starts = [sum(counts[:p]) for p in range(nprocs)]
        ends = [sum(counts[:p+1]) for p in range(nprocs)]
        qpoints = [(starts[p], ends[p]) for p in range(nprocs)]
        # print(qpoints)
    else:
        qpoints=None
    # qpoints=comm.scatter(qpoints, root=0)
    # print(qpoints)

    # assign q points for different procs. and calculate
    pointsperproc=math.ceil(max_sym_index/nprocs)
    partsig = np.zeros((pointsperproc,N),dtype=np.complex128)
    # time2=time.time()
    for qind in np.arange(max_sym_index):
        q=essential_kpoints[qind]
        # partsig[qind,:]=fft_convolution.precalcsigp_fft(q,knum,Gk,Pk,beta,U,0)
        # time_begfun=time.time()
        partsig[qind,:]=operation(q,knum,*args)
        # time_endfun=time.time()
    # time3=time.time()
    # gathered_sig=np.zeros((nprocs,pointsperproc,N),dtype=np.complex128)
    # comm.Gather(partsig,gathered_sig,root=0)
    full_sig=np.zeros((N, knum, knum, knum),dtype=np.complex128)
        # P is compacted for fast connection between procs. now unpack it:
    for proc in np.arange(nprocs):
        for ind in np.arange(pointsperproc):
            qind=proc*pointsperproc+ind
            if qind < max_sym_index:
                q=essential_kpoints[qind]
                full_sig[:,q[0],q[1],q[2]]=partsig[ind,:]#gathered_sig[proc,ind,:]
        # restore k-space domain sym
                all_sym_kpoints=sym_mapping(q[0],q[1],q[2],knum)
                for kpoint in all_sym_kpoints:
                        full_sig[:,kpoint[0],kpoint[1],kpoint[2]]=full_sig[:,q[0],q[1],q[2]]*(kpoint[3]**power)
    # time4=time.time()
    # time_end=time.time()
        # print('total mpi={}s, operation={}s'.format(time_end-time_beg,time_endfun-time_begfun))
    # print('total mpi={}s, operations={} {} {} {} {}'.format(time_end-time_beg,time1-time_beg,time2-time1,time3-time2,time_end-time3,time_end-time4))
    return full_sig