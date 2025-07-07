import matplotlib.pyplot as plt 
import numpy as np
import os,sys,subprocess,math
from mpi4py import MPI
from perturb_lib import *
import perturb_imp as imp
import fft_convolution as fft
from scipy.optimize import curve_fit
# import pert_DMFT_energy as energy
from scipy.interpolate import interp1d
import diagramsMC.basis as basis
import diagrams
import copy
from matplotlib.gridspec import GridSpec
# import pert_DMFT_PM_B
'''
This code is used to check the AFM susceotibility of DMFT and also perturbation.
'''


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

def mag_DMFT_B(U,T,B,opt=0):
    '''
    calculate the magnetization of DMFT under a small B field.
    '''
    mu=U/2
    beta=1/T
    knum=10
    nfreq=500

    beta=1/T
    mu=U/2
    name1='../data/files_boldcB/{}_{}_{}/Sig.out'.format(U,T,B)
    filename1=readDMFT(name1)
    name2='../data/files_boldcB/{}_{}_{}/Sig.OCA'.format(U,T,B)
    filename2=readDMFT(name2)
    name3='../data/files_ctqmc/{}_{}/Sig.out'.format(U,T)
    filename3=readDMFT(name3)
    # print(filename1)
    # print(filename2)
    if (os.path.exists(filename1)):
        filename=filename1
    elif (os.path.exists(filename2)):
        filename=filename2
    # elif (os.path.exists(filename3)):
    #     filename=filename3
    else:
        # print('these 3 filenames cannot be found:\n {} \n {} \n {}\n'.format(name1,name2,name3))  
        return 0
    # print(filename)
    
    sigma=np.loadtxt(filename)[:nfreq,:]
    check=sigma[-1,1]
    om=sigma[:,0]
    # anyways real part of sigA will be greater.
    # if check>U/2:
    sigA=sigma[:,1]+1j*sigma[:,2]
    sigB=U-sigma[:,1]+1j*sigma[:,2]
    # else:
    #     sigB=sigma[:,1]+1j*sigma[:,2]
    #     sigA=U-sigma[:,1]+1j*sigma[:,2]
    Sigma11=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sigma11+=ext_sig(sigA)[:,None,None,None]
    Sigma22=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    Sigma22+=ext_sig(sigB)[:,None,None,None]
    Sigma12=np.zeros((2*nfreq,knum,knum,knum),dtype=complex)
    z_1=z4D(beta,mu,Sigma11,knum,nfreq)-B#z-delta
    z_2=z4D(beta,mu,Sigma22,knum,nfreq)+B#z+delta
    G11_iom,G12_iom=G_iterative(knum,z_1,z_2,Sigma12)
    G22_iom=-G11_iom.conjugate()

    magdmft=np.abs(particlenumber4D(G11_iom,beta)-particlenumber4D(G22_iom,beta))
    if opt==0:
        return magdmft
    else:
        return sigA,sigB

def B_linear_check(U,T,ifplot=0):
    '''
    check if the B is small enough to give a linear response.
    '''#0.002,0.006,
    Blist=np.array([0.0,0.002,0.01,0.03,0.05,0.1,0.2,0.3])
    # Blist=np.array([0.0,0.01,0.03,0.05,0.1,0.2,0.3])
    # Blist=np.array([0.0,0.002,0.006,0.01])
    maglist=np.zeros_like(Blist)
    for iB, B in enumerate(Blist):
        maglist[iB]=mag_DMFT_B(U,T,B)
    for iB, B in enumerate(Blist):
        if np.abs(maglist[iB])>1e-6 and iB>0:
            # xi=(maglist[iB]-maglist[0])/(Blist[iB]-Blist[0])
            xi=(maglist[iB])/(Blist[iB])
            break
    if ifplot==1:
        plt.plot(Blist,maglist, marker='o',label='U={},T={} XAF={} 1/XAF={}'.format(U,T,xi,1/xi))
        plt.legend()
        plt.show()
    # print('U={},T={} done!'.format(U,T))
    return xi

def model(T, A, T0, m):
    return A * (T - T0) ** m

def log_model(T, logA, T0, m):
    return logA + m * np.log(T - T0)

def Xi_vs_T(U):
    Tlist1= np.arange(0.48, 0.5, 0.001)#(np.arange(18)+482)/1000
    
    Tlist2= np.arange(0.5, 0.55, 0.005)  #(np.arange(20)+100)/200
    Tlist=np.unique(np.concatenate((Tlist1, Tlist2)))
    Tlist = np.round(Tlist, decimals=3)
    # print('DMFT_Tlist:\n',Tlist)
    Xilist=np.zeros_like(Tlist)
    for iT,T in enumerate(Tlist):
        # print(T)
        Xilist[iT]=B_linear_check(U,T)
    # critical fit
    initial_guess = [1, 0.1, -1]
    TNind=np.argmax(Xilist)
    # print('TN={}'.format(Tlist[TNind]))
    # params1, params_covariance1 = curve_fit(model, Tlist[TNind+3:Tlist.size], Xilist[TNind+3:Tlist.size], p0=initial_guess)    
    # A_fit1, T0_fit1, m_fit1 = params1
    # Tfitlist=np.linspace(T0_fit1+0.001,Tlist[-1],300)
    # XAF_fit1 = model(Tfitlist, A_fit1, T0_fit1, m_fit1)
    
    # params2, params_covariance2 = curve_fit(model, Tlist[:TNind-1], Xilist[:TNind-1], p0=initial_guess)    
    # A_fit2, T0_fit2, m_fit2 = params2
    # XAF_fit2 = model(Tlist[:TNind], A_fit2, T0_fit2, m_fit2)


    # plt.plot(Tlist[TNind:],1/Xilist[TNind:], marker='o',label='DMFT'.format(U))
    # plt.plot(Tfitlist,XAF_fit1,linestyle='--')#,label='T0={:.5f} nu={:.5f}±{:.5f}'.format(T0_fit1,m_fit1,np.sqrt(params_covariance1[2,2]))
    # plt.text(0.5, 5, r'$T_N={:.5f} \\ \nu={:.5f} \pm {:.5f}$'.format(T0_fit1,m_fit1,np.sqrt(params_covariance1[2,2])), fontsize=12, color='black', ha='center', va='center')
    # plt.text(
    # 0.58, 5,
    # 'DMFT:'+ '\n'+r'$T_N={:.3f}$'.format(T0_fit1) + '\n' + r'$\gamma={:.3f} \pm {:.3f}$'.format(-m_fit1, np.sqrt(params_covariance1[2, 2])),
    # fontsize=12, color='black', ha='center', va='center'
    # )
    # plt.plot(Tlist[:TNind],XAF_fit2,label='fitting: T0={:.5f} nu={:.5f}±{:.5f}'.format(T0_fit2,m_fit2,np.sqrt(params_covariance2[2,2])))
    # plt.legend()
    # plt.show()    




    # plt.plot(Tlist,1/Xilist,label='U={}'.format(U))
    # plt.legend()
    # plt.ylim(0,25)
    # plt.show()    
    return Tlist[TNind:],1/Xilist[TNind:]

def check_SigPM(U,T):
    '''
    This function is designed to answer an question: when we have a small B field, does SigPM change a lot?
    '''
    Blist=np.array([0.0,0.01,0.03,0.05,0.1,0.2,0.3])
    for B in Blist:
        SigA,SigB=mag_DMFT_B(U,T,B,1)
        SigPM=(SigA+SigB)/2
        plt.plot(SigPM[:10].imag,label='B={}'.format(B))
    plt.legend()
    plt.show()
    return 0


def read_complex_numbers(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    complex_numbers = []
    for line in lines:
        line = line.strip()
        
        complex_number = complex(line)
        complex_numbers.append(complex_number)

    complex_array = np.array(complex_numbers)
    
    return complex_array

def get_sigma_and_G(U,T,B,order,alpha,foldernum='B',ifchecksvd=0):
    knum=10
    beta=1/T
    mu=U/2
    sigmafilename11='./Sigma_disp{}/{}_{}_{}/{}_{}_{}_{}_{}_11.dat'.format(foldernum,U,T,B,U,T,B,order,alpha)
    sigmafilename11const='./Sigma_disp{}/{}_{}_{}/{}_{}_{}_{}_{}_11const.dat'.format(foldernum,U,T,B,U,T,B,order,alpha)
    sigmafilename12='./Sigma_disp{}/{}_{}_{}/{}_{}_{}_{}_{}_12.dat'.format(foldernum,U,T,B,U,T,B,order,alpha)
    filenameu='./Sigma_imp/taubasis.txt'
       
    ut=np.loadtxt(filenameu).T 
    taunum=np.shape(ut)[1]-1
    nfreq=500
    imax=4
    if order>0:
        if (os.path.exists(sigmafilename11))==0 or (os.path.exists(sigmafilename11const))==0 or (os.path.exists(sigmafilename12))==0:
            print('{} does not exist!'.format(sigmafilename11))
            return None
        cli_11=np.loadtxt(sigmafilename11)
        cli_12=np.loadtxt(sigmafilename12)
        ci_11=np.loadtxt(sigmafilename11const)# for const
        basisnum=basis.gen_basisnum(imax)
        # print('imax=',imax,'shape of cli11',np.shape(cli_11))
        kbasis=np.empty((2,basisnum,knum,knum,knum),dtype=float)
        kbasis=basis.gen_kbasis_new(imax,knum)
        # print('imax=',imax)
        # if ifchecksvd==1:
        #     if rank==0:
        #         basisind=basis.gen_basisindlist(imax)
        #         basisindnum=np.shape(basisind)[0]
        #         for ikb in np.arange(basisindnum):
        #             if np.sum(basisind[ikb])%2==0:
        #                 plt.plot(cli_11[:,ikb],label='({},{},{})'.format(basisind[ikb,0],basisind[ikb,1],basisind[ikb,2]))
        #         plt.title('cli11: U={} T={} order={} alpha={}'.format(U,T,order,alpha))
        #         plt.legend()
        #         plt.show()

        taulist=(np.arange(taunum+1))/taunum*beta
        ori_grid=(np.arange(nfreq*2)+0.5)/(nfreq*2)*beta
        Sig11tk_raw=basis.restore_tk(cli_11,ut,kbasis[0])
        Sig12tk_raw=basis.restore_tk(cli_12,ut,kbasis[1])
        interpolator_11 = interp1d(taulist, Sig11tk_raw, kind='cubic', axis=0, fill_value='extrapolate')
        interpolator_12 = interp1d(taulist, Sig12tk_raw, kind='cubic',  axis=0,fill_value='extrapolate')
        Sig11tk_full=interpolator_11(ori_grid)
        Sig12tk_full=interpolator_12(ori_grid)
        sig11const=basis.restore_k(ci_11,kbasis[0])
        Sigiom_11=fft.fast_ift_fermion(Sig11tk_full,beta)+sig11const[None,:,:,:]
        Sigiom_12=fft.fast_ift_fermion(Sig12tk_full,beta)
        # for kx in np.arange(knum):
        #     for ky in np.arange(knum):
        #         for kz in np.arange(knum):
        #             # plt.plot(Sigiom_11[:,kx,ky,kz].real,label='11 real')
        #             # plt.plot(Sigiom_11[:,kx,ky,kz].imag,label='11 imag')
        #             plt.plot(Sig11tk_raw[:,kx,ky,kz],label='tau')
        #             plt.legend()
        #             plt.show()


    elif order==0:# if order==0, the self-energy is not saved using the basis, since it is local.
        if (os.path.exists(sigmafilename11))==0:
            print('{} does not exist!'.format(sigmafilename11))
            return None
        Sigiom_11=read_complex_numbers(sigmafilename11)[:,None,None,None]*np.ones((2*nfreq,knum,knum,knum))
        # Sigiom_11=np.loadtxt(sigmafilename11)[:,None,None,None]*np.zeros((2*nfreq,knum,knum,knum))
        Sigiom_12=np.zeros((2*nfreq,knum,knum,knum))
    Sigiom_22=U-Sigiom_11.conjugate()
    znew_1=z4D(beta,mu,Sigiom_11,knum,nfreq)-B
    znew_2=z4D(beta,mu,Sigiom_22,knum,nfreq)+B
    Gdress11_iom,Gdress12_iom=G_iterative(knum,znew_1,znew_2,Sigiom_12)
    Gdress22_iom=-Gdress11_iom.conjugate()
    return Sigiom_11,Sigiom_12,Sigiom_22,Gdress11_iom,Gdress12_iom,Gdress22_iom

def check_alpha(U,T,B,alphaarrraw,maxorder=3):
    

    foldernum='B'
    backup=''
    alphavalid=np.ones_like(alphaarrraw)
    for ialp,alpha in enumerate(alphaarrraw):
        for order in np.arange(maxorder+1):
            sigmafilename11='./Sigma_disp{}/{}_{}_{}/{}_{}_{}_{}_{}_11.dat'.format(foldernum,U,T,B,U,T,B,order,alpha)
            sigmafilename11const='./Sigma_disp{}/{}_{}_{}/{}_{}_{}_{}_{}_11const.dat'.format(foldernum,U,T,B,U,T,B,order,alpha)
            sigmafilename12='./Sigma_disp{}/{}_{}_{}/{}_{}_{}_{}_{}_12.dat'.format(foldernum,U,T,B,U,T,B,order,alpha)
            check=0
            if order>0:
                if (os.path.exists(sigmafilename11))==0 or (os.path.exists(sigmafilename11const))==0 or (os.path.exists(sigmafilename12))==0:
                    check=1
            else:
                if (os.path.exists(sigmafilename11))==0:
                    check=1
            if check==1:
                alphavalid[ialp]=0
    indices = np.where(alphavalid == 1)[0]
    return alphaarrraw[indices]

def mag_test(U,T,B):
    '''
    This function is for testing. To check the self-energy stored in the form of coefficients can really be restored and give the same magnetization.
    '''
    foldernum='B'
    beta=1/T
    maxorder=3
    orderstart=1
    # alpha_arr=np.array(([0.01,0.02,0.03,0.04,0.05]))/10#,0.06,0.07,0.08
    # alpha_arr0=np.array(([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08]))/10
    alpha_arr0=np.arange(200)/2000
    alpha_arr0 = np.round(alpha_arr0, decimals=4)
    alpha_arr=check_alpha(U,T,B,alpha_arr0,3)
    print(alpha_arr)
    alpha_plt=alpha_arr
    # alpha_plt=np.array(([-0.01,0.0,0.01,0.02,0.03,0.04,0.05]))#0.1,0.1,
    # alpha_arraw=(np.arange(101))/100# to make sure only integer time of 0.03 exist.
    # alpha_arraw = (np.arange(33)*3+3)/100
    # alpha_arraw = np.unique(np.concatenate(((np.arange(20)*3+3)/100, np.array([0.6,1.0]))))
    # print(alpha_arraw)
    # alpha_arr=check_alpha(U,T,alpha_arraw,4,0)
    alphanum=np.shape(alpha_arr)[0]
    mag_arr=np.zeros((alphanum,maxorder+1))
    # mag_arr2=np.zeros((alphanum,maxorder+1))

    magdmft=mag_DMFT_B(U,T,B)
    for order in np.arange(maxorder+1):
        for ialp,alpha in enumerate(alpha_arr):
            Sigiom_11,Sigiom_12,Sigiom_22,Gdress11_iom,Gdress12_iom,Gdress22_iom=get_sigma_and_G(U,T,B,order,alpha,foldernum)
            # znew_1=z4D(beta,mu,sigfinal11,knum,nfreq)
            # znew_2=z4D(beta,mu,sigfinal22,knum,nfreq)
            # Gdress11_iom,Gdress12_iom=G_iterative(knum,znew_1,znew_2,sigfinal12)
            # Gdress22_iom=-Gdress11_iom.conjugate()
            nnewloc11=particlenumber4D(Gdress11_iom,beta)
            nnewloc22=particlenumber4D(Gdress22_iom,beta)
            mag_arr[ialp,order]=nnewloc22-nnewloc11


        # ax.plot(alpha_arr, np.std(mag_arr[:,orderstart:],axis=1), marker='o', linestyle='-',label='T={}'.format(T))
        # ax.set_xlabel(r'$\alpha$')
        # ax.set_ylabel('standard deviation')
        # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        # plt.title('U={} T={}'.format(U, T))
        # plt.grid()
        # plt.savefig("../paperwriting/pert/alpha_{}_{}.png".format(U,T), dpi=1000)
        # plt.show()

    # plt.axhline(y=magdmft, color='b', linestyle='--', linewidth=2,label='DMFT')
    # plt.plot(alpha_arr, mag_arr[:,0], marker='o', linestyle='-',label='0th')
    # plt.plot(alpha_arr, mag_arr[:,1], marker='^', linestyle='-',label='1st')
    # plt.plot(alpha_arr, mag_arr[:,2], marker='s', linestyle='-',label='2nd')
    # plt.plot(alpha_arr, mag_arr[:,3], marker='p', linestyle='-',label='3rd')
    # # plt.plot(alpha_arr, mag_arr[:,4], marker='h', linestyle='-',label='4th')
    # plt.legend()
    # plt.xlabel('alpha')
    # plt.ylabel('magnetization')
    # plt.title('mag vs alpha: U={} T={}'.format(U, T))
    # plt.grid()
    # plt.show()

    data=np.zeros((alpha_arr.size,7))#alpha, DMFT, order01234, 7 columns
    data[:,0]=alpha_arr
    data[:,1]=magdmft
    for ord in np.arange(maxorder+1):
        data[:,ord+2]=mag_arr[:,ord]
    plt.figure(figsize=(4, 6))
    for ialp,alpha in enumerate(alpha_arr):
        if np.isin(alpha,alpha_plt):
            plt.plot(np.arange(maxorder+1), mag_arr[ialp,:], marker='o', linestyle='-',label='alpha={}'.format(alpha))
    # plt.legend()
    # plt.ylim(0,0.2)
    plt.xlabel('order')
    plt.ylabel('m')
    # plt.axhline(y=magdmft, color='b', linestyle='--', linewidth=2)
    plt.title('U={} T={}'.format(U, T))
    plt.tight_layout()
    # plt.text(2.5, 0.45, 'U={} T={}'.format(U, T), fontsize=12, color="black")
    plt.grid()
    # plt.savefig("../paperwriting/pert/mag_{}_{}.png".format(U,T), dpi=1000)
    plt.show()
    return 0

def X_VDMCAF(U,B,knum,steps,factor=1,ifplot=0):#,opt=0
    '''
    This function is designed to calculate the XAF from VDMC. 
    Basically, the minimal standard deviation and its corresponding magnetization is saved in txt files in Sigma_dispB.
    it seems the real data is {}_{}_{}opt.txt?

    '''
    Tlist = np.round(steps, decimals=3)  # Round to 3 decimal places
    Tlist=np.unique(Tlist)
    Xilist0=np.zeros_like(Tlist)
    Xilist1=np.zeros_like(Tlist)
    Xilist=np.zeros_like(Tlist)
    foldernum='B'+str(knum)
    for iT,T in enumerate(Tlist):
        # if opt==0:
        filepath0='../data/files_finitesize/Sigma_disp{}/{}_{}_{}.txt'.format(foldernum,U,T,B)
        # elif opt==1:
        filepath1='../data/files_finitesize/Sigma_disp{}/{}_{}_{}opt.txt'.format(foldernum,U,T,B)
        if (os.path.exists(filepath0))==1:
            data=np.loadtxt(filepath0)
            ind=np.argmin(data[:,3])
            mag=data[ind,2]
            Xilist[iT]=mag/B
        if (os.path.exists(filepath1))==1:
            data=np.loadtxt(filepath1)
            ind=np.argmin(np.abs(data[:,3]))
            mag=data[ind,2]
            Xilist1[iT]=mag/B
    Xilist=Xilist1# use xilist 0 or 1.
    Xilist_inv=1/Xilist/factor
    initial_guess = [55, 0.35, 1.4]#def model(T, A, T0, m):  return A * (T - T0) ** m
    initial_guess_log = [np.log(55), 0.35, 1.4]#def model(T, A, T0, m):  return A * (T - T0) ** m
    TNind=np.argmax(Xilist1)
    Thighind=len(Tlist)-1
    # for iT,T in enumerate(Tlist):
    #     if T>0.45:
    #         Thighind=iT
    #         break


    minadd=0# points too close to TN may not have small enough B to get correct chi.
    # to make the fit fare for all T, we need to use the spline to interpolate the log(Xi) on a equidistant grid.
    log_Xiinv=np.log(Xilist_inv[TNind:])
    # spline_log_Xiinv=interp1d(Tlist[TNind+minadd:],log_Xiinv[minadd:],kind='cubic',fill_value='extrapolate')
    # Tlist_spline=np.linspace(Tlist[TNind+minadd],Tlist[-1],1000)
    # log_Xiinv_spline=spline_log_Xiinv(Tlist_spline)
    # plt.plot(Tlist_spline,log_Xiinv_spline,label='spline')
    # plt.show()
    # print(Tlist_spline,log_Xiinv_spline)
    # params1, params_covariance1 = curve_fit(model, Tlist[TNind+5:], Xilist[TNind+5:], p0=initial_guess)    
    params1, params_covariance1 = curve_fit(log_model, Tlist, log_Xiinv, p0=initial_guess_log)
    # params1, params_covariance1 = curve_fit(log_model, Tlist[TNind+minadd:], np.log(Xilist_inv[TNind+minadd:]), p0=initial_guess_log)
    # params1, params_covariance1 = curve_fit(log_model, Tlist[TNind+minadd:TNind+maxadd], np.log(Xilist_inv[TNind+minadd:TNind+maxadd]), p0=initial_guess_log)
    m_err=np.sqrt(params_covariance1[2,2])

    A_fit1, T0_fit1, m_fit1 = params1
    print('fitting: {} * (T - {}) ** {} ± {}'.format(A_fit1, T0_fit1, m_fit1, m_err))
    # y_fit0=model(Tlist[minTind:],A_fit1,T0_fit1,m_fit1)
    y_fit0=np.exp(log_model(Tlist[TNind+minadd:],A_fit1,T0_fit1,m_fit1))
    residual=Xilist_inv[TNind+minadd:]-y_fit0
    Tfitlist=np.linspace(T0_fit1+0.001,Tlist[-1],300)
    # XAF_fit1 = model(Tfitlist, A_fit1, T0_fit1, m_fit1)
    XAF_fit1=np.exp(log_model(Tfitlist,A_fit1,T0_fit1,m_fit1))
    #
    Tneel=T0_fit1
    # Tneel=0.3645-Tneel
    print('Tneel={}, T with max chi={}'.format(Tneel,Tlist[TNind]))
    if ifplot:
        # plt.plot(Tlist[TNind+minadd:], Xilist_inv[TNind+minadd:], marker='o',markersize=8,zorder=2.5,label='VDMC',color='blue')#, fmt='', , linestyle='None'
        plt.plot(Tlist, Xilist_inv, marker='o',markersize=8,zorder=2.5,label='VDMC L={}'.format(knum))
        plt.plot(Tfitlist,XAF_fit1,linestyle=':',zorder=3,color='black')#,label='T0={:.5f} nu={:.5f}±{:.5f}'.format(T0_fit1,m_fit1,np.sqrt(params_covariance1[2,2]))
        # plt.show()
    return T0_fit1,m_fit1,A_fit1,residual,Tlist[TNind+minadd:],Xilist_inv[TNind+minadd:]

def conv_of_chi(U,B,Tlist,Llist):
    '''
    This function shows Xi as a function as system size L, at different temperatures in Tlist.
    '''

    Xi_inv=np.zeros_like(Llist,dtype=np.float64)
    # B=0.0002
    for T in Tlist:
        for iL,L in enumerate(Llist):
            foldernum='B'+str(L)
            # if opt==0:
            # filepath0='./Sigma_disp{}/{}_{}_{}.txt'.format(foldernum,U,T,B)
            # elif opt==1:
            filepath1='./Sigma_disp{}/{}_{}_{}opt.txt'.format(foldernum,U,T,B)
            # print(filepath1)
            # if (os.path.exists(filepath0))==1:
            #     data=np.loadtxt(filepath0)
            #     ind=np.argmin(data[:,3])
            #     mag=data[ind,2]
            #     Xi_inv[iL]=B/mag
            # else:
            #     print('did not find {}'.format(filepath0))
            if (os.path.exists(filepath1))==1:
                
                data=np.loadtxt(filepath1)
                ind=np.argmin(data[:,3])
                mag=data[ind,2]
                Xi_inv[iL]=B/mag
            #     print('found {}, mag={},Xi={}'.format(filepath1,mag,Xi_inv[iL]))
            # else:
            #     print('did not find {}'.format(filepath1))
        # plt.plot(Llist,(Xi_inv/Xi_inv[-1]-1),label='delta T={}'.format(np.round(T-0.36,3)))
        # print('T={}\n'.format(T),np.abs(Xi_inv/Xi_inv[-1]-1))
        # print(np.log(np.abs(Xi_inv/Xi_inv[-1]-1)))
        idx = np.where(np.abs(Xi_inv/Xi_inv[-1]-1) > 1e-12)[0]
        plt.plot(Llist[idx],(np.abs(Xi_inv[idx]/Xi_inv[-1]-1)),label=r'$T-T_N={}$'.format(np.round(T-0.358,3)))

    return 0




def plot_everything(U,B,knum,steps):
    minpt=0
    fa=9# normalization factor

    TN,m,A,residual,Tlist,Xinvlist=X_VDMCAF(U,B,knum,steps,fa)
    DMFTTlist,DMFTXinvlist=Xi_vs_T(U)
    DMFTXinvlist/=fa
    #linear fit the DMFT data: Y=ax+b
    
    params=np.polyfit(DMFTTlist[minpt:],DMFTXinvlist[minpt:],1)
    TN_DMFT=-params[1]/params[0]
    print('linear fit: {} * T + {}, TN_DMFT={}'.format(params[0],params[1],TN_DMFT))

    T_VDMC=np.linspace(TN+0.001,Tlist[-1],1000)
    X_VDMC=np.exp(log_model(T_VDMC,A,TN,m))
    X_2=np.exp(log_model(T_VDMC,A,TN,1.4))
    T_DMFT=np.linspace(TN_DMFT+0.001,DMFTTlist[-1],1000)
    X_DMFT=params[0]*T_DMFT+params[1]
    # y_fit0=np.exp(log_model(Tlist[TNind+minadd:],A_fit1,T0_fit1,m_fit1))
    fig = plt.figure(figsize=(10, 8))
    
    gs = GridSpec(3, 2, figure=fig, wspace=0.3, hspace=0.3)
    ax_top_left    = fig.add_subplot(gs[0:2, 0])  
    ax_bottom_left = fig.add_subplot(gs[2:, 0]) 
    ax_right       = fig.add_subplot(gs[:, 1])
    # ax_top_left.plot(DF[:,0]+TN,DF[:,1],marker='s',markersize=8,mfc='none',mew=1.5,zorder=3,label='DF',color='green')
    ax_top_left.plot(Tlist,Xinvlist,marker='o',markersize=10,mfc='none',mew=1.5,zorder=2.5,label='VDMC',color='red')
    ax_top_left.plot(DMFTTlist,DMFTXinvlist,marker='o',markersize=10,mfc='none',mew=1.5,zorder=2.5,label='DMFT',color='blue')
    ax_top_left.plot(T_VDMC,X_VDMC,linestyle='--',zorder=3,color='black')
    ax_top_left.plot(T_DMFT,X_DMFT,linestyle='--',zorder=3,color='black')
    ax_top_left.set_ylim(ymin=0.0)
    ax_top_left.set_xlabel('T',labelpad=0)
    ax_top_left.set_ylabel(r'$\chi^{-1}_{AF}$')
    ax_top_left.legend()
    ax_top_left.text(
        -0.12, 0.96, '(a)',
        transform=ax_top_left.transAxes,
        fontsize=12, fontweight='bold',
        va='top', ha='left'
    )

    ax_bottom_left.plot(Tlist[:20]-TN,residual[:20],marker='o',linestyle='',markersize=8,mfc='none',mew=1.5,zorder=2.5,label='VDMC',color='red')
    ax_bottom_left.set_xlabel(r'$T-T_N$')
    ax_bottom_left.set_ylabel('residual')
    ax_bottom_left.text(
        -0.12, 1.1, '(b)',
        transform=ax_bottom_left.transAxes,
        fontsize=12, fontweight='bold',
        va='top', ha='left'
    )

    ax_right.plot(Tlist-TN,Xinvlist,marker='o',linestyle='',markersize=10,mfc='none',mew=1.5,zorder=2.5,label='VDMC',color='red')
    ax_right.plot(DMFTTlist-TN_DMFT,DMFTXinvlist,marker='o',linestyle='',markersize=10,mfc='none',mew=1.5,zorder=2.5,label='DMFT',color='blue')
    X_14=np.exp(log_model(T_VDMC,A+0.15,TN,1.4))
    X_124=np.exp(log_model(T_VDMC,A,TN,1.24))
    X_19=np.exp(log_model(T_VDMC,A+1.3,TN,1.9))
    ax_right.plot(T_DMFT-TN_DMFT,X_DMFT,linestyle=':',zorder=3,color='black', label=r'$x^1$')
    ax_right.plot(T_VDMC-TN,X_124,linestyle='-.',zorder=3,color='black', label=r'$x^{1.24}$')
    ax_right.plot(T_VDMC-TN,X_14,linestyle='--',zorder=3,color='black', label=r'$x^{1.4}$')
    ax_right.plot(T_VDMC-TN,X_19,linestyle='-',zorder=3,color='black', label=r'$x^{1.9}$')
    ax_right.set_xlabel(r'$T-T_N$')
    ax_right.set_ylabel(r'$\chi^{-1}_{AF}$',labelpad=-30)
    ax_right.set_xscale('log')
    ax_right.set_yscale('log')
    ax_right.set_ylim(ymin=0.002,ymax=0.3)
    ax_right.set_xlim(xmin=0.002)
    ax_right.legend()
    ax_right.text(
        -0.12, 0.96, '(c)',
        transform=ax_right.transAxes,
        fontsize=12, fontweight='bold',
        va='top', ha='left'
    )
    # save as pdf
    # plt.savefig('Fig3.pdf', dpi=1000, bbox_inches="tight")
    plt.show()





if __name__ == "__main__":
    sizefont=15# default:12
    plt.rc('font', size=sizefont) 
    plt.rc('axes', titlesize=sizefont) 
    plt.rc('axes', labelsize=sizefont) 
    plt.rc('xtick', labelsize=sizefont)
    plt.rc('ytick', labelsize=sizefont)
    plt.rc('legend', fontsize=12)
    # give the size of the plot.
    # plt.figure(figsize=(5, 6))
    U=11.0
    B=0.002
    # B=0.0002
    steps = np.concatenate([
        np.arange(0.36, 0.38, 0.001),  # Very dense. 30 points.
        np.arange(0.38, 0.41, 0.003),   # Medium density. 10 points.
        np.arange(0.41, 0.435, 0.005)   # Sparser towards 0.45. 8 points.
    ])
    # steps = np.arange(0.36, 0.399, 0.001)
    # B_linear_check(10.0,0.6,1)
    # check_SigPM(10.0,0.44)
    # mag_test(10.0,0.36,0.01)
    # Xi_vs_T(10.0)
    # Llist=np.array([2,4,6,8,10,12,14,16,18,20],dtype=int)
    # conv_of_chi(11.0,0.0002,np.array([0.362]),np.array([4,6,8,10,12,14,16,18,20],dtype=int))
    # conv_of_chi(11.0,0.0002,np.array([0.365,0.37]),np.array([2,4,6,8,10,12,14,16,18,20],dtype=int))#
    # conv_of_chi(11.0,0.0002,np.array([0.39]),np.array([2,4,6,8,10,12,14],dtype=int))#
    # conv_of_chi(11.0,0.002,np.array([0.41,0.45]),np.array([2,4,6,8,10],dtype=int))#
    # plt.xlabel('L')
    # plt.ylabel('relative error')
    # plt.tight_layout()
    # plt.yscale('log')
    # plt.legend()
    # plt.savefig('FigS1.pdf', dpi=1000, bbox_inches="tight")
    # plt.show()

    plot_everything(11.0,0.002,10,steps)


