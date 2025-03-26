from scipy import *
from scipy.interpolate import interp1d
# import weight_lib 
from numpy import linalg
from numpy import random
from scipy import special
import sys
import copy
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import time
sys.path.append('../')
# import perturb_lib as lib

import fft_convolution as fft
from .. import diagrams_finder
from ..diagramsMC_lib import *

def gen_sublatint_full(short):
    order=np.shape(short)[0]
    full_sublatint=np.zeros(2*order,dtype=int)
    for i in np.arange(order):
        full_sublatint[2*i]=short[i]
        full_sublatint[2*i+1]=3-short[i]# 1 will be 2, 2 will be 1
    return full_sublatint


def gen_tau_limit(depend,Ndimk,Ndimtau,ngf):
    '''
    This is function is used to generate correct limit for GFs at tau=0.
    Here an important issue is to pick correct limit, 0+ or 0-.
            #usually, each interaction has a tau index so all GFs should has the tau dependence like :G(tau_i-tau_j).
            # my definition is, if tau_i==tau_j, take 0+ if if i<j, take 0- if i>j. 
    The input is the labeling or dependence of GF.
    '''
    taulimit=np.ones(ngf,dtype=int)*2# initialize it to 2. 2=unspecified 1=0- 0=0+
    for i in np.arange(ngf):
        for j in np.arange(Ndimk,Ndimk+Ndimtau):
            if depend[i,j]==1:
                taulimit[i]=0# take 0+
                break
            if depend[i,j]==-1:
                taulimit[i]=1#take 0-
                break
    return taulimit


def gen_limit_factor(taulimit):
    count=np.zeros(2,dtype=int)
    taulimitfac=np.zeros_like(taulimit,dtype=float)
    fac0=0
    facbeta=0
    for i,ele in enumerate(taulimit):
        count[ele]+=1
    allcounts=count[0]+count[1]
    # print(count)
    for i,ele in enumerate(taulimit):
        taulimitfac[i]=allcounts/count[ele]
    # print('taulimitfac=',taulimitfac)
    return taulimitfac

def gen_kfold1D(dependk,knum):
    shapedep=np.shape(dependk)
    rownum=shapedep[0]
    # colnum=shapedep[1]
    numplus=0
    numminus=0
    for row in np.arange(rownum):
        currentminus=np.sum((np.abs(dependk[row])-dependk[row]))//2
        currentplus=np.sum((np.abs(dependk[row])+dependk[row]))//2
        if currentminus>numminus:
            numminus=currentminus
        if currentplus>numplus:
            numplus=currentplus
    kfold1D=np.arange((numplus+numminus)*knum)%knum
    return kfold1D

def gen_ksign3D(dependk,knum):
    shapedep=np.shape(dependk)
    rownum=shapedep[0]
    # colnum=shapedep[1]
    numplus=0
    numminus=0
    for row in np.arange(rownum):
        currentminus=np.sum((np.abs(dependk[row])-dependk[row]))//2
        currentplus=np.sum((np.abs(dependk[row])+dependk[row]))//2
        if currentminus>numminus:
            numminus=currentminus
        if currentplus>numplus:
            numplus=currentplus
    ksign1D=(-1.)**(np.arange((numplus+numminus)*knum)//knum)
    
    ksign1D[numplus*knum:]= (-1.)**(np.arange(-numminus*knum,0)//knum)# to support negative indices.
    ksign3D=ksign1D[:,None,None]*ksign1D[None,:,None]*ksign1D[None,None,:]
    return ksign3D


class FuncNDiagNew:
    """  
    This class is trying to define a class which suites different order of diagrams. And it starts from Sigma diagrams but not Phi.
    functions:
    init: initialization.
    call: (re)evaluate the integrand from calculated component of GFs.
    update: update some of GFs when the configuration get changed.


    For the result we should reserve some space for the trial before it is accepted. 
    Before accepting by metropolis, we should always keep the old results.

    initialization parameters:
    T: temperature
    U: Hubbard U
    knum: k space grid: knum*knum*knum in 3D space.
    nfreq: number of matsubara frequencies
    Ndimk: number of k variables. In a nth order phi diagram we need n+1 k indices to label the diagram.
    Ndimtau: number of tau variables. This should be norder-1 for sigma diagrams.
    norder: order of the diagram

    GFs: usually a tuple of (G11,G12,G22), the non-interacted propagaters used to build diagrams.

    ut: svd basis. has the shape (lmax,taunum)

    SubLatInd: SublatticeIndices of each point in Phidiagram. 1= site1 2=site2
        example: permutation of 2nd order Phi diagram should be [2,3,0,1]
        and the sublattice could be [1,2,1,2] for points [0,1,2,3]
        Remember interaction U connects 2 points with different spins so actually 1up=2dn.
        Later this will be removed since sublattice indices are included in configuration space.

    dependence: a matrix which tells which GF depend on which variable. 
        example: diagonal 2nd order should be: [[1,1,0,1]    1st GF: k+q, tau 
                                                [0,0,1,1]    2nd GF: k',tau
                                                [0,1,0,-1]   3nd GF: k'+q,-tau
                                                [1,0,0,-1]    4th GF: k, -tau 
        if only k get changed we only reevaluate the 1st and 4th GF.
        the svd basis ut(tau) should only depend of the external tau. and its dependence not included here.
        In principle this dependence can be generated through a function of permutation representation of Phi diagram but it is not done yet.
    
    
    """
    def __init__(self,T,U,knum,taunum,nfreq,norder,ut,kbasis,perm,GFs,dependence,symfactor):
        # orders and dimensions
        self.norder=norder
        self.symfactor=symfactor
        self.nGf=2*norder# number of GFs in a phi diagram.
        self.Ndimk = norder+1# number of k is always norder+1
        self.Ndimtau=norder-1# number of tau is always norder, correspond to n interactions, 1 for translational symmetry.
        self.Ndimlat=norder# number of sublattice variable in config space. 
        #parameters
        self.knum=knum
        self.taunum=taunum
        self.nfreq=nfreq
        self.depend=dependence
        self.taulimit=gen_tau_limit(self.depend,self.Ndimk,self.Ndimtau,self.nGf)
        self.taulimit_oppo=1-self.taulimit
        self.taulimitfactor=gen_limit_factor(self.taulimit)
        # self.taulimitfactor=np.ones_like(self.taulimitfactor)
        # print('taulimfac=',self.taulimitfactor)
        self.U=U
        self.beta=1/T
        self.perm=perm
        # find_loops gives ana array from 1 to maxloops. however for sigma diagrams we cut 1 line which means 1 loop less.
        nloopssig=np.max(diagrams_finder.find_loops(self.perm))-1
        # print('# of Fermion loops:',nloopssig)
        self.diag_coefficient=(-1)**nloopssig*(-self.U)**(self.norder)/self.knum**(3*(self.Ndimk-1))*(self.beta/self.taunum)**(self.Ndimtau-1)

        #configuration space maybe we don't need the 4 below?
        self.t=np.zeros(self.nGf,dtype=int)
        self.ttemp=np.zeros(self.nGf,dtype=int)
        # self.k=np.zeros((self.Ndimk,3),dtype=int)
        # self.ktemp=np.zeros((self.Ndimk,3),dtype=int)
        # t, k for each GF,
        self.tslice=np.zeros(self.nGf,dtype=int)
        self.tslicetemp=np.zeros(self.nGf,dtype=int)
        self.tsign=np.zeros(self.nGf,dtype=int)
        self.tsign_temp=np.zeros(self.nGf,dtype=int)     

        self.kslice=np.zeros((self.nGf,3),dtype=int)
        self.kslicetemp=np.zeros((self.nGf,3),dtype=int)
        self.ksign=np.zeros(self.nGf,dtype=int)
        self.ksign_temp=np.zeros(self.nGf,dtype=int)


        self.sublatint_short=np.ones(self.Ndimlat,dtype=int)
        self.SubLatInd=gen_sublatint_full(self.sublatint_short)# later it will be a part of config space. Remember interaction connects 2 points with different spins!
        self.SubLatInd_temp=self.SubLatInd
        self.ksym_array=np.zeros(self.nGf,dtype=int)# the k-space symmetry: 0 means diagonal, 1 means off-diagonal.
        self.slicedG=np.zeros(self.nGf)# the Gfs at current configuration.
        self.slicedG_temp=np.zeros(self.nGf)# the Gfs at current configuration. but this is for the trial of metropolis.
        self.Gind=np.zeros(self.nGf,dtype=int)# only has 2,3, and 4. GF are quoted from the dictiionary below.
        self.Gind_temp=np.zeros(self.nGf,dtype=int)

        
        self.ksignlist=np.array([1,-1,1,-1,-1,1,-1])
        self.Gindsignlist=np.array([1,1,1,1,-1]) # to let 11 and 22 accumulate together.
        # self.indcutlist=np.array([0,2,1,3])
        self.updatektau=np.zeros(self.nGf)
        self.updateind=np.zeros(4)# change 1 sublatind always change 4 GFs
        #basis
        self.ut=ut#svd basis

        self.kbasis=kbasis
        self.kbasisnum=np.shape(self.kbasis)[1]
        # self.sublatbasis=sublatind_basis

        self.tbasisslice=np.zeros(self.nGf)#,dtype=int
        self.tbasisslice_temp=np.zeros(self.nGf)#,dtype=int

        self.kbasisslice=np.zeros((2,self.nGf))#,dtype=int
        self.kbasisslice_temp=np.zeros((2,self.nGf))#,dtype=int

        # these quantities take care of configuration out of [0,beta):
        # all possible tau for GFs are in terms of ti-tj, where t is in [0,beta), so ti-tj is in (-beta,beta). and we have 0+ and 0- version.
        self.tsignlist=np.zeros((2,2*self.taunum),dtype=int)
        self.tsignlist[0]=(np.arange(2*self.taunum)//self.taunum)*2-1# this is tau=0+; t=0 correspond to tsym[self.taunum], (taunum)//taunum)*2-1=1.
        self.tsignlist[1]=((np.arange(2*self.taunum)-1)//self.taunum)*2-1# this is tau=0-. for t=0, ((taunum-1)//taunum)*2-1=-1.

        self.tfoldlist=np.zeros((2,2*self.taunum),dtype=int)# if ti-tj is out of [0,beta), fold it back to [0,beta)
        self.tfoldlist[0]=np.arange(2*self.taunum)%self.taunum# This will fold 0 to 0, beta to 0.
        self.tfoldlist[1]=(np.arange(2*self.taunum)-1)%self.taunum+1# this will fold 0 to beta, beta to beta.


        self.kfold1D=gen_kfold1D(self.depend,self.knum)
        self.ksign3D=gen_ksign3D(self.depend,self.knum)
        # kspace symmetry (for off-diagonal GFs).these quantities take care of configuration out of 1BZ

        #initialization of GF
        ori_grid=(np.arange(nfreq*2)+0.5)/(nfreq*2)
        simp_grid=np.arange(taunum+1)/taunum
        # we do splining for all G11 G12 G22 first. then even we change sublatind we dont have to spline again.
        self.interpolator11=interp1d(ori_grid, GFs[0], kind='linear', axis=0, fill_value='extrapolate')
        self.interpolator12=interp1d(ori_grid, GFs[1], kind='linear', axis=0, fill_value='extrapolate')
        self.interpolator22=interp1d(ori_grid, GFs[2], kind='linear', axis=0, fill_value='extrapolate')
        G11=self.interpolator11(simp_grid)
        G12=self.interpolator12(simp_grid)
        G22=self.interpolator22(simp_grid)
        # make it easier to quote the interpolators. I made keys to be 2,3,4, because 1+1=2, 1+2=2+1=3, 2+2=4 so it is easier to call them when i need diagonal or off diagonal elements. 
        self.GF={
            2: G11,
            3: G12,
            4: G22,
        }

        #time
        self.time_k=0.
        self.time_tau=0.
        self.time_sublatind=0.
        self.time_basist=0.
        self.time_basisk=0.
        self.time_basisind=0.
        self.time_test=0.


        for i in np.arange(self.nGf):# initialize Gind and ksymarray.
            self.Gind[i]=self.SubLatInd[i]+self.SubLatInd[perm[i]]
            self.ksym_array[i]=(self.SubLatInd[i]+self.SubLatInd[perm[i]])%2
        self.ksym_arraytemp=self.ksym_array

        # # initialization of slicedG
        # for i in np.arange(self.nGf):
        #     ki=np.sum(self.depend[i,:self.Ndimk,None]*self.k,axis=0)# note: k should be an array with 3 elements.
        #     taui=np.sum(self.depend[i,self.Ndimk:]*self.t)

        #     self.slicedG[i]=Gshift(self.GF[self.Gind[i]],ki,taui,self.ksym_array[i],self.knum,self.taunum,self.taulimit[i])


    def update(self,new_momentum,new_imagtime,newsublatindshort,i_coeff,l):
        '''
        the idea of this update function is to re-evaluate everything so we have everything correct in the beginning.
        it could be slow, since we won't use it frequently.
        This can also be considered as initialization.
        '''
        newsublatind=gen_sublatint_full(newsublatindshort)
        self.SubLatInd=newsublatind
        for i in np.arange(self.nGf):
            self.Gind[i]=newsublatind[i]+newsublatind[self.perm[i]]
            self.ksym_array[i]=self.Gind[i]%2
            
            # Note: here kslice is related to self. since the k for sigma and cut GF are the same.
            kraw=np.sum(self.depend[i,:self.Ndimk,None]*new_momentum,axis=0)
            self.kslice[i]=kraw%self.knum
            # Note: here tslice is related to self. but the traw define at (-beta,beta) belongs to self.
            # since cut Gt will get sigma(-t), and taking correct limit is not that easy.
            self.t[i]=np.sum(self.depend[i,self.Ndimk:]*new_imagtime.T)
            self.tslice[i]=self.tfoldlist[self.taulimit[i],self.t[i]+self.taunum]     
            self.ksign[i]=self.ksignlist[np.sum(kraw//self.knum)]
            self.tsign[i]=self.tsignlist[self.taulimit[i],self.t[i]+self.taunum]
            self.slicedG[i]=self.GF[self.Gind[i]][self.tslice[i],self.kslice[i,0],self.kslice[i,1],self.kslice[i,2]]*self.tsign[i]




            self.kbasisslice[:,i]=self.kbasis[:,i_coeff,self.kslice[i,0],self.kslice[i,1],self.kslice[i,2]]# cut Gk get sigk
            # self.tbasisslice[i]=self.ut[l,(-traw-(1-self.taulimit[i]))%self.taunum+(1-self.taulimit[i])]*self.tsign[1-self.taulimit[i],-traw+self.taunum]
            # print(self.tfoldlist[1-self.taulimit[i],-self.t[i]+self.taunum])
            self.tbasisslice[i]=self.ut[l,self.tfoldlist[1-self.taulimit[i],-self.t[i]+self.taunum] ]*self.tsignlist[1-self.taulimit[i],-self.t[i]+self.taunum]
            if self.t[i]==0:
                self.tbasisslice[i]*=self.taulimitfactor[i]

            if self.Gind[i]==3:
                self.slicedG[i]*=self.ksign[i]
            self.kbasisslice[1,:]*=self.ksign
                # but would this cancel the sign? 

            # self.slicedG[i]=Gshift(self.GF[self.Gind[i]],self.kslice[i],self.tslice[i],self.ksym_array[i],self.knum,self.taunum,self.taulimit[i])

        result=self.sum_allsigmas(self.slicedG,new_momentum,new_imagtime,newsublatind,i_coeff,l)
        self.kslicetemp=copy.deepcopy(self.kslice)
        self.ksign_temp=copy.deepcopy(self.ksign)
        
        self.tslicetemp=copy.deepcopy(self.tslice)
        self.tsign_temp=copy.deepcopy(self.tsign)

        self.kbasisslice_temp=copy.deepcopy(self.kbasisslice)
        self.tbasisslice_temp=copy.deepcopy(self.tbasisslice)
        # self.indbasisslice_temp=copy.deepcopy(self.indbasisslice)

        self.slicedG_temp=copy.deepcopy(self.slicedG)
        self.SubLatInd_temp=copy.deepcopy(self.SubLatInd)
        self.ksym_arraytemp=copy.deepcopy(self.ksym_array)
        self.Gind_temp=copy.deepcopy(self.Gind)
        #note: beta/taunum means dtau.
        # Note2: here the power of dtau is for all internal variables. However, using svd trick we also integrate out external tau.
        # the reason of not count dtau for external tau is that svd basis u(tau) is normalized without dtau: \sum_i u^2_l(tau_i)=1. No dtau here.
        # print('fQ:',self.slicedG[0],self.slicedG[1],self.slicedG[2],self.slicedG[3],self.slicedG[4],self.slicedG[5])
        return result*self.diag_coefficient#-1 in the beginning means 1 loop
    
    
    def update_temp(self,iloop,new_momentum,new_imagtime,newsublatindshort,i_coeff,l,globalflip=0):
        '''
        The idea of this fast evaluate function is that only update quantities when it's necessary.
        first, every time we only update 1 of the variables. It could be k, tau, sublatind, or basis function indices i,j,l.
        if we update.....
        1. k. just evaluate the new k for relevant GFs and redo the slices.  have to retake the k basis.
        2. tau. just evaluate the new tau for relevant GFs and redo the slices. have to retake the tau basis.
        3. sublatind. do not have to slice k or tau again, but we take the element from another GF at the same k and tau point. have to retake the sublatind basis.
        4. external variable. do not have to update the product of GFs. just re-evaluate the corresponded basis. 


       '''
        newsublatind=gen_sublatint_full(newsublatindshort)
        # which variable is changed?
        if iloop<self.Ndimk:# changing variable k 
            time2=time.time()
            # which GFs should be updated?
            self.updatektau=self.depend[:,iloop]
            indices=np.nonzero(self.updatektau)[0]
            # vectorize
            
            # rawk=np.sum(self.depend[indices,:self.Ndimk,None]*new_momentum[None,:,:],axis=1)
            rawk = np.dot(self.depend[indices, :self.Ndimk], new_momentum)
            
            self.kslicetemp[indices]=   rawk%self.knum       #self.kfold1D[rawk] 
            
            self.ksign_temp[indices]=self.ksignlist[np.sum(rawk//self.knum,axis=1)]
            # self.ksign_temp[indices]=self.ksign3D[rawk[:,0],rawk[:,1],rawk[:,2]]
            self.kbasisslice_temp[:,indices]=self.kbasis[:,i_coeff,self.kslicetemp[indices,0],self.kslicetemp[indices,1],self.kslicetemp[indices,2]]
            timet=time.time()
            for idx in indices:
                self.slicedG_temp[idx] = self.GF[self.Gind[idx]][self.tslice[idx],
                                                     self.kslicetemp[idx, 0],
                                                     self.kslicetemp[idx, 1],
                                                     self.kslicetemp[idx, 2]] * self.tsign[idx]  
            timett=time.time()                      
            gind3_indices = indices[self.Gind[indices] == 3]
            self.slicedG_temp[gind3_indices] *= self.ksign_temp[gind3_indices]
            self.kbasisslice_temp[1,indices] *= self.ksign_temp[indices]          

            oddlGindsign=np.ones((self.nGf))
            if l%2==1:
                oddlGindsign=self.Gindsignlist[self.Gind]
            result=sumslices_fast(self.nGf,self.symfactor,self.slicedG_temp,self.tbasisslice*oddlGindsign,self.kbasisslice_temp[self.Gind%2])
             
            time3=time.time()
            self.time_k+=(time3-time2)
            self.time_test+=(timett-timet)
        elif iloop< self.Ndimk+self.Ndimtau:# changing variable tau
            time2=time.time()
            self.updatektau=self.depend[:,iloop]
            indices = np.nonzero(self.updatektau)[0]
            # self.ttemp[indices] = np.sum(self.depend[indices, self.Ndimk:] * new_imagtime.T, axis=1)
            self.ttemp[indices] = np.dot(self.depend[indices, self.Ndimk:], new_imagtime).T# the time directly from matrix prod
            self.tslicetemp[indices] = self.tfoldlist[self.taulimit[indices], self.ttemp[indices] + self.taunum]# time folded in 0,beta. depend on if the GF is defined as 0+ or 0-.
            self.tsign_temp[indices] = self.tsignlist[self.taulimit[indices], self.ttemp[indices] + self.taunum]# the extra sign because of folding back into 0,beta.
            
            kslice = self.kslice[indices]
            tslicetemp = self.tslicetemp[indices]
            Gind = self.Gind[indices]
            slicedG_temp = np.array([
            self.GF[Gind[i]][tslicetemp[i], kslice[i, 0], kslice[i, 1], kslice[i, 2]]for i in range(len(indices))])
            self.slicedG_temp[indices] = slicedG_temp * self.tsign_temp[indices]# get the GF elements should be updated.

            ksign_indices = Gind == 3
            self.slicedG_temp[indices[ksign_indices]] *= self.ksign[indices[ksign_indices]]# extra sign due to folding back to 1BZ, if needed.

            tslice_for_sig = self.tfoldlist[self.taulimit_oppo[indices], -self.ttemp[indices] + self.taunum]
            tsign_for_sig = self.tsignlist[self.taulimit_oppo[indices], -self.ttemp[indices] + self.taunum]
            self.tbasisslice_temp[indices] = self.ut[l, tslice_for_sig] * tsign_for_sig#when we cut G(t) we should multiply by u(-t)

            ttemp_zero_indices = self.ttemp[indices] == 0
            #if there is a t=0, according to the definition we give 0+ or 0-. However we need both. 
            # for tind in ttemp_zero_indices:# for all those has 0 times. if only have 0+ i should give 0-, vice versa.
                

            self.tbasisslice_temp[indices[ttemp_zero_indices]] *= self.taulimitfactor[indices[ttemp_zero_indices]]


            # for i in np.nonzero(update)[0]:
            #     self.ttemp[i]=np.sum(self.depend[i,self.Ndimk:]*(new_imagtime.T))
            #     # update related propagators
            #     self.tslicetemp[i]=self.tfoldlist[self.taulimit[i],self.ttemp[i]+self.taunum]
            #     self.tsign_temp[i]=self.tsignlist[self.taulimit[i],self.ttemp[i]+self.taunum]
            #     self.slicedG_temp[i]=self.GF[self.Gind[i]][self.tslicetemp[i],self.kslice[i,0],self.kslice[i,1],self.kslice[i,2]]*self.tsign_temp[i]
            #     if self.Gind[i]==3:
            #         self.slicedG_temp[i]*=self.ksign[i]  

            #     # also, update time basis.
            #     tslice_for_sig=self.tfoldlist[self.taulimit_oppo[i],-self.ttemp[i]+self.taunum]
            #     tsign_for_sig=self.tsignlist[self.taulimit_oppo[i],-self.ttemp[i]+self.taunum]# maybe just take - sign of tsign for GF?
            #     # print(self.tbasisslice_temp)
            #     self.tbasisslice_temp[i]=self.ut[l,tslice_for_sig]*tsign_for_sig
            #     if self.ttemp[i]==0:
            #         self.tbasisslice_temp[i]*=self.taulimitfactor[i]
                # self.slicedG_temp[i]=Gshift(self.GF[self.Gind[i]],self.kslice[i],self.tslicetemp[i],self.ksym_array[i],self.knum,self.taunum,self.taulimit[i])
            
            oddlGindsign=np.ones((self.nGf))
            if l%2==1:
                oddlGindsign=self.Gindsignlist[self.Gind]
            result=sumslices_fast(self.nGf,self.symfactor,self.slicedG_temp,self.tbasisslice_temp*oddlGindsign,self.kbasisslice[self.Gind%2])
            time3=time.time()
            self.time_tau+=(time3-time2)
            
        elif iloop<self.Ndimk+self.Ndimtau+self.Ndimlat:# changing sublatind
            time4=time.time()
            shortind=iloop-self.Ndimk-self.Ndimtau
            self.updateind=np.array([shortind*2,shortind*2+1,np.where(self.perm==shortind*2)[0][0],np.where(self.perm==shortind*2+1)[0][0]])
            if globalflip==0:
                upd=self.updateind
            elif globalflip==1:
                upd=np.arange(self.nGf)
            #approach 1
            # self.Gind_temp[self.updateind]=newsublatind[self.updateind]+newsublatind[self.perm[self.updateind]]   
            # self.ksym_arraytemp[self.updateind]=(self.Gind_temp[self.updateind])%2# k space sym for new propagator. 
            # for i in self.updateind:
            #     self.slicedG_temp[i]=self.GF[self.Gind_temp[i]][self.tslice[i],self.kslice[i,0],self.kslice[i,1],self.kslice[i,2]]*self.tsign[i]
            #     if self.Gind_temp[i]==3:
            #         self.slicedG_temp[i]*=self.ksign[i]                  
            # self.indbasisslice_temp[self.updateind]=self.sublatbasis[j_coeff,(newsublatind[self.updateind]-1)+2*(newsublatind[self.perm[self.updateind]]-1)]
            
            # approach #2 this looks
            for i in upd:# changing internal variable sublatind
                self.ksym_arraytemp[i]=(newsublatind[i]+newsublatind[self.perm[i]])%2# k space sym for new propagator. 
                self.Gind_temp[i]=newsublatind[i]+newsublatind[self.perm[i]]   
                self.slicedG_temp[i]=self.GF[self.Gind_temp[i]][self.tslice[i],self.kslice[i,0],self.kslice[i,1],self.kslice[i,2]]*self.tsign[i]
                if self.Gind_temp[i]==3:
                    self.slicedG_temp[i]*=self.ksign[i]  
            oddlGindsign=np.ones((self.nGf))
            if l%2==1:
                oddlGindsign=self.Gindsignlist[self.Gind_temp]
            result=sumslices_fast(self.nGf,self.symfactor,self.slicedG_temp,self.tbasisslice*oddlGindsign,self.kbasisslice[self.Gind_temp%2])
            time5=time.time()
            self.time_sublatind+=(time5-time4)
        elif iloop==self.Ndimk+self.Ndimtau+self.Ndimlat:# changing basis for k.
            time4=time.time()
            self.kbasisslice_temp=self.kbasis[:,i_coeff,self.kslice[:,0],self.kslice[:,1],self.kslice[:,2]]
            gind3_indices = self.Gind == 3
            self.kbasisslice_temp[1,:] *= self.ksign
            # for i in np.arange(self.nGf):
                # self.kbasisslice_temp[i]=self.kbasis[i_coeff,self.kslice[i,0],self.kslice[i,1],self.kslice[i,2]]
                # if self.Gind[i]==3:
                    # self.kbasisslice_temp[i]*=self.ksign[i]
            oddlGindsign=np.ones((self.nGf))
            if l%2==1:
                oddlGindsign=self.Gindsignlist[self.Gind]
            result=sumslices_fast(self.nGf,self.symfactor,self.slicedG,self.tbasisslice*oddlGindsign,self.kbasisslice_temp[self.Gind%2])
            time5=time.time()
            self.time_basisk+=(time5-time4)
        elif iloop==self.Ndimk+self.Ndimtau+self.Ndimlat+1:# changing basis for t.
            time4=time.time()
            tslice_for_sig=self.tfoldlist[self.taulimit_oppo,-self.t+self.taunum]
            tsign_for_sig=self.tsignlist[self.taulimit_oppo,-self.t+self.taunum]
            self.tbasisslice_temp=self.ut[l,tslice_for_sig]*tsign_for_sig
            t_zero_indices = self.t == 0
            self.tbasisslice_temp[t_zero_indices] *= self.taulimitfactor[t_zero_indices]
            # for i in np.arange(self.nGf):
            #     if self.t[i]==0:
            #         self.tbasisslice_temp[i]*=self.taulimitfactor[i]
            oddlGindsign=np.ones((self.nGf))
            if l%2==1:
                oddlGindsign=self.Gindsignlist[self.Gind]
            result=sumslices_fast(self.nGf,self.symfactor,self.slicedG,self.tbasisslice_temp*oddlGindsign,self.kbasisslice[self.Gind%2])
            time5=time.time()
            self.time_basist+=(time5-time4)

        return result*self.diag_coefficient

    def metropolis_accept(self,iloop,globalflip=0):
        if iloop<self.Ndimk:
            # update=self.depend[:,iloop]
            indices=np.nonzero(self.updatektau)[0]
            self.slicedG[indices]=self.slicedG_temp[indices]
            self.kslice[indices]=self.kslicetemp[indices]
            self.ksign[indices]=self.ksign_temp[indices]
            self.kbasisslice[:,indices]=self.kbasisslice_temp[:,indices]

            # self.slicedG=copy.deepcopy(self.slicedG_temp)
            # self.kslice=copy.deepcopy(self.kslicetemp)
            # self.ksign=copy.deepcopy(self.ksign_temp)
            # self.kbasisslice=copy.deepcopy(self.kbasisslice_temp)

        elif iloop<self.Ndimk+self.Ndimtau:
            # update=self.depend[:,iloop]
            indices = np.nonzero(self.updatektau)[0]
            self.slicedG[indices]=self.slicedG_temp[indices]
            self.tslice[indices]=self.tslicetemp[indices]
            self.tsign[indices]=self.tsign_temp[indices]
            self.t[indices]=self.ttemp[indices]
            self.tbasisslice[indices]=self.tbasisslice_temp[indices]



            # self.slicedG=copy.deepcopy(self.slicedG_temp)
            # self.tslice=copy.deepcopy(self.tslicetemp)
            # self.tsign=copy.deepcopy(self.tsign_temp)
            # self.t=copy.deepcopy(self.ttemp)
            # self.tbasisslice=copy.deepcopy(self.tbasisslice_temp)
        elif iloop<self.Ndimk+self.Ndimtau+self.Ndimlat:
            # shortind=iloop-self.Ndimk-self.Ndimtau
            # indices=np.array([shortind*2,shortind*2+1,np.where(self.perm==shortind*2)[0][0],np.where(self.perm==shortind*2+1)[0][0]])
            if globalflip==0:
                self.slicedG[self.updateind]=self.slicedG_temp[self.updateind]
                self.Gind[self.updateind]=self.Gind_temp[self.updateind]
                self.SubLatInd[self.updateind]=self.SubLatInd_temp[self.updateind]
                self.ksym_array[self.updateind]=self.ksym_arraytemp[self.updateind]
            else:
                self.slicedG=copy.deepcopy(self.slicedG_temp)
                self.Gind=copy.deepcopy(self.Gind_temp)
                self.SubLatInd=copy.deepcopy(self.SubLatInd_temp)
                self.ksym_array=copy.deepcopy(self.ksym_arraytemp)


        elif iloop==self.Ndimk+self.Ndimtau+self.Ndimlat:
            self.kbasisslice=copy.deepcopy(self.kbasisslice_temp)
        elif iloop==self.Ndimk+self.Ndimtau+self.Ndimlat+1:
            self.tbasisslice=copy.deepcopy(self.tbasisslice_temp)
        return 0
    
    def metropolis_reject(self,iloop,globalflip=0):
        if iloop<self.Ndimk:
            indices=np.nonzero(self.updatektau)[0]
            self.slicedG_temp[indices]=self.slicedG[indices]
            self.kslicetemp[indices]=self.kslice[indices]
            self.ksign_temp[indices]=self.ksign[indices]
            self.kbasisslice_temp[:,indices]=self.kbasisslice[:,indices]


            # self.slicedG_temp=copy.deepcopy(self.slicedG)
            # self.kslicetemp=copy.deepcopy(self.kslice)
            # self.ksign_temp=copy.deepcopy(self.ksign)
            # self.kbasisslice_temp=copy.deepcopy(self.kbasisslice)
        elif iloop<self.Ndimk+self.Ndimtau:
            indices=np.nonzero(self.updatektau)[0]
            self.slicedG_temp[indices]=self.slicedG[indices]
            self.tslicetemp[indices]=self.tslice[indices]
            self.tsign_temp[indices]=self.tsign[indices]
            self.ttemp[indices]=self.t[indices]
            self.tbasisslice_temp[indices]=self.tbasisslice[indices]



            # self.slicedG_temp=copy.deepcopy(self.slicedG)
            # self.tslicetemp=copy.deepcopy(self.tslice)
            # self.tsign_temp=copy.deepcopy(self.tsign)
            # self.ttemp=copy.deepcopy(self.t)
            # self.tbasisslice_temp=copy.deepcopy(self.tbasisslice)            
        elif iloop<self.Ndimk+self.Ndimtau+self.Ndimlat:
            # shortind=iloop-self.Ndimk-self.Ndimtau
            # indices=np.array([shortind*2,shortind*2+1,np.where(self.perm==shortind*2)[0][0],np.where(self.perm==shortind*2+1)[0][0]])
            if globalflip==0:
                self.slicedG_temp[self.updateind]=self.slicedG[self.updateind]
                self.Gind_temp[self.updateind]=self.Gind[self.updateind]
                self.SubLatInd_temp[self.updateind]=self.SubLatInd[self.updateind]
                self.ksym_arraytemp[self.updateind]=self.ksym_array[self.updateind]
            else:
                self.slicedG_temp=copy.deepcopy(self.slicedG)
                self.Gind_temp=copy.deepcopy(self.Gind)
                self.SubLatInd_temp=copy.deepcopy(self.SubLatInd)
                self.ksym_arraytemp=copy.deepcopy(self.ksym_array)
        elif iloop==self.Ndimk+self.Ndimtau+self.Ndimlat:
            self.kbasisslice_temp=copy.deepcopy(self.kbasisslice)
        elif iloop==self.Ndimk+self.Ndimtau+self.Ndimlat+1:
            self.tbasisslice_temp=copy.deepcopy(self.tbasisslice)
        return 0        

    # @jit(nopython=True)
    def sum_allsigmas(self,slices,momentum, imagtime, sublatind,i_coeff,l):
        '''
        Given all slicedG and dependnecies, sublatind, momentum, imagtime, symfactor
        sum over all 2*norder integrands from cutting propagators of Phi.
        i,l,j are indices of space, time, sublatind basis.

        Note: if cut a GF G^ab(k,tau) in diagram Phi, we'll get sigma^ba(k,-tau)=-sigma^ba(k,beta-tau)
        so we should use u_l(-tau), m_i(k) and n_j(b,a) to get the coefficient c_ijl.
        also, if the cut G should take tau=0- limit, sigma should take opposite limit, tau=0+, vice versa.
        This is a slow version. used for initialization.
        '''
        
        mat=np.ones((self.nGf,self.nGf))*slices[None,:]
        np.fill_diagonal(mat, 1)
        res=np.prod(mat,axis=(1))
        # res = np.prod(slices[None, :] * np.ones((self.nGf, self.nGf)), axis=1)
        

        # getting the indices for the cut G. However we have to get these for the sigma
        # since the sigma has the same indices as the basis functions.
        timea=time.time()
        k = np.sum(self.depend[:, :self.Ndimk, None] * momentum[None,:,:], axis=1)# this is the k we need. should put it in 1BZ.
        # note for sigsym: if 11 and 22, mod=0, 1-0*2=1; if 12 or 21, mod=1,1-1*2=-1. ofcourse it is an array with nGf elements.
        sigsym=1-((sublatind-sublatind[self.perm])%2)*2
        kfactor=(sigsym)**((np.mod(k[:,0], self.knum)-(k[:,0]))/self.knum+(np.mod(k[:,1], self.knum)-(k[:,1]))/self.knum+(np.mod(k[:,2], self.knum)-(k[:,2]))/self.knum)
        sigk=np.mod(k,self.knum)
        tau = -1*np.sum(self.depend[:,self.Ndimk:] * imagtime.T, axis=1)
        oppo_taulimit=1-self.taulimit# have to take the correct limit.
        tfactor=(-1)**((np.mod(tau-oppo_taulimit, self.taunum)-(tau-oppo_taulimit))/(self.taunum))
        sigtau=np.mod(tau-oppo_taulimit, self.taunum)+oppo_taulimit
        
        
        zero_indices = np.where(tau == 0)[0]
        res[zero_indices]*=self.taulimitfactor[zero_indices]

        timeb=time.time()
        # Note: here t must be in (0,beta) and k is in 1BZ.
        self.tbasisslice=self.ut[l, sigtau]
        self.tbasisslice_temp=self.ut[l, sigtau]
        self.kbasisslice=self.kbasis[:,i_coeff, sigk[:, 0], sigk[:, 1], sigk[:, 2]]
        self.kbasisslice_temp=self.kbasis[:,i_coeff, sigk[:, 0], sigk[:, 1], sigk[:, 2]]

        res *= (self.ut[l, sigtau] * 
            self.kbasis[self.Gind%2,i_coeff, sigk[:, 0], sigk[:, 1], sigk[:, 2]])#
        if l%2==1:
            res*=self.Gindsignlist[self.Gind]# to accumulate G22 on G11 and use C_G22[1::2]=-C_G11[1::2]
        # return res[1]
        timec=time.time()
        # self.time_basis1+=(timeb-timea)
        # self.time_basis2+=(timec-timeb)
        return np.sum(res)/self.symfactor


@jit(nopython=True, cache=True, fastmath=True)
def custom_prod(arr,shape):
    result = np.ones(shape)
    for i in range(shape):
        result[i] = np.prod(arr[i, :])
    return result


@jit(nopython=True, cache=True, fastmath=True)
def sumslices_fast(ngf,symfac,slices,utslices,kbasisslices):
    # mat=np.ones((ngf,ngf))*slices[None,:]
    # np.fill_diagonal(mat, 1)
    mat = np.empty((ngf, ngf))
    for i in range(ngf):
        for j in range(ngf):
            if i == j:
                mat[i, j] = 1.0
            else:
                mat[i, j] = slices[j]
    res=custom_prod(mat,ngf)
    return np.sum(res*utslices*kbasisslices)/symfac


    