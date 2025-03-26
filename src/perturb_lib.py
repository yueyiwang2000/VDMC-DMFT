import numpy as np
# import matplotlib.pyplot as plt
# from memory_profiler import profile
"""
This file contains simple python functions might be use in other perturbation files.
"""
def fermi(eps,beta):
    return 1/(np.exp(beta*eps)+1)

def boson(eps,beta):
    return 1/(np.exp(beta*eps)-1)

def gen_full_kgrids(knum,a=1):
    kall=np.linspace(0,1,num=knum+1)
    # kroll=np.roll(kall,1)
    # kave=(kall+kroll)/2
    klist=kall[:knum] 
    # print('klist=',klist)
    k1, k2, k3 = np.meshgrid(klist, klist, klist, indexing='ij')
    return k1,k2,k3

def dispersion(k1,k2,k3,a=1,t=1):
    kx=(-k1+k2+k3)*np.pi/a
    ky=(k1-k2+k3)*np.pi/a
    kz=(k1+k2-k3)*np.pi/a
    e_k=-2*t*np.cos(kx*a)-2*t*np.cos(ky*a)-2*t*np.cos(kz*a)
    return e_k

def calc_eps2_ave(knum,a=1):
    k1, k2, k3 = gen_full_kgrids(knum)
    eps2=dispersion(k1, k2, k3)**2
    eps2_ave=np.sum(eps2)/knum**3
    return eps2_ave

def calc_disp(knum,a=1):
    k1, k2, k3 = gen_full_kgrids(knum)
    disp=dispersion(k1, k2, k3)
    return disp

def calc_expikdel(knum):
    k1,k2,k3=gen_full_kgrids(knum)
    phase=1j*(k1+k2-k3)*np.pi# Delta=0.5*(a1+a2-a3), ai*ki=2Pi, ai*ki=0
    factor=np.exp(phase)
    return factor

def z(beta,mu,sig,nfreq):
    # sometimes we want values of G beyond the range of n matsubara points. try to do a simple estimation for even higher freqs:
    # n=sig.size
    om=(2*np.arange(2*nfreq)+1-2*nfreq)*np.pi/beta
    # allsig=ext_sig(sig)
    z=om*1j+mu-sig
    return z

def ext_sig(sig):
    n=sig.size
    # print(lenom)
    allsig=np.zeros(2*n,dtype=complex)
    allsig[1*n:2*n]=sig
    allsig[:1*n]=allsig[2*n:n-1:-1].conjugate()
    return allsig

def ext_g(g):
    n=g.size
    # print(lenom)
    allg=np.zeros(2*n,dtype=complex)
    allg[1*n:2*n]=g
    allg[:1*n]=allg[2*n:n-1:-1].conjugate()
    return allg

# @profile
def z4D(beta,mu,sig,knum,n,extramu=0):
    # sometimes we want values of G beyond the range of n matsubara points. try to do a simple estimation for even higher freqs:
    om=(2*np.arange(2*n)+1-2*n)*np.pi/beta
    z=om[:,None,None,None]*1j-sig+mu
    return z

def ext_sig4D(sig,knum,n):
    allsig=np.zeros((2*n,knum,knum,knum),dtype=complex)
    allsig[1*n:2*n,:,:,:]=sig
    allsig[:1*n,:,:,:]=allsig[2*n:n-1:-1,:,:,:].conjugate()
    return allsig



def calc_sym_point(in_k123,mat,knum):
    S=np.array([[-1,1,1],
                [1,-1,1],
                [1,1,-1]],dtype=int)#(nx,ny,nz)^T=S@(n1,n2,n3)^T
    S_inv=np.linalg.inv(S)
    out_kpoint=(S_inv@mat@S@in_k123).astype(int)
    out_kpoint_mod=np.mod(out_kpoint,knum)
    sgn=(-1)**(np.sum(out_kpoint_mod-out_kpoint)/knum)
    output=np.vstack((out_kpoint_mod,np.array([sgn])))
    # print(output)
    # return out__.astype(int)
    return output.astype(int)

def sym_mapping(k1ind,k2ind,k3ind,knum=10):
    '''
    generally, both G, P, sigma has the same k space symmetry.
    But hence we have a BCC instead of simple cubic, we have to use linear algebra to figure out the symmetry.
    define symmetry matrices in x,y,z basis:
    x_inverse=np.diag([-1,1,1])#x->-x
    y_inverse=np.diag([1,-1,1])#y->-y
    z_inverse=np.diag([1,1,-1])#z->-z    
    Specifically, this takes care of the off diagonal symmetry P12_k=P12_-k,
    and also some kind of reciprocal symmetry like P12_{G-k}=-P12_{k}, where G is some reciprocal vector, which is the origin of last component of sym_points.
    '''

    xy_swap=np.array([[0,1,0],
                      [1,0,0],
                      [0,0,1]],dtype=int)# (kx,ky,kz)=(ky,kx,kz), and so on.
    xz_swap=np.array([[0,0,1],
                      [0,1,0],
                      [1,0,0]],dtype=int)
    yz_swap=np.array([[1,0,0],
                      [0,0,1],
                      [0,1,0]],dtype=int)
    # S=np.array([[-1,1,1],
    #             [1,-1,1],
    #             [1,1,-1]],dtype=int)#(nx,ny,nz)^T=S@(n1,n2,n3)^T
    # S_inv=np.linalg.inv(S)
    input_k123=np.array([[k1ind],[k2ind],[k3ind]])
    sym_points=np.array([[k1ind],[k2ind],[k3ind],[1]])# the last element means the sign.
    # ------test code------
    # print('input_k123\n',input_k123)
    # input_kxyz=S@input_k123
    # print('input_kxyz, unit:pi/knum/a\n',input_kxyz)
    # output_kxyz=x_inverse@input_kxyz
    # print('output_kxyz, unit:pi/knum/a\n',output_kxyz)
    # output_k=(S_inv@output_kxyz).astype(int)
    # shifted_output_k=np.mod(output_k,knum)
    # shifted_output_k=np.mod((S_inv@x_inverse@S@input_k123).astype(int),knum)
    # ------end of test code------
    # I think I have another 47 elements in this group.... I have to list all of them.
    # try to apply the generator in the group.
    for xinv in [-1,1]:
        for yinv in [-1,1]:
            for zinv in [-1,1]:
                inv=np.diag([xinv,yinv,zinv])
                # print(xinv,yinv,zinv)
                sym_points=np.concatenate([sym_points,calc_sym_point(input_k123,inv,knum)],axis=1)#xyz
                sym_points=np.concatenate([sym_points,calc_sym_point(input_k123,inv@xy_swap,knum)],axis=1)#yxz
                sym_points=np.concatenate([sym_points,calc_sym_point(input_k123,inv@xz_swap,knum)],axis=1)#zyx 
                sym_points=np.concatenate([sym_points,calc_sym_point(input_k123,inv@yz_swap,knum)],axis=1)#xzy
                # sym_points=np.concatenate([sym_points,calc_sym_point(input_k123,xy_swap@inv,knum)],axis=1)
                # sym_points=np.concatenate([sym_points,calc_sym_point(input_k123,xz_swap@inv,knum)],axis=1)
                # sym_points=np.concatenate([sym_points,calc_sym_point(input_k123,yz_swap@inv,knum)],axis=1)
                sym_points=np.concatenate([sym_points,calc_sym_point(input_k123,inv@xy_swap@xz_swap,knum)],axis=1)#zxy
                sym_points=np.concatenate([sym_points,calc_sym_point(input_k123,inv@xz_swap@xy_swap,knum)],axis=1)#yzx
    # print('shifted_k123\n',shifted_output_k)
    
    output_sym_points=sym_points.T
    # print('output_sym_points\n',output_sym_points)
    return output_sym_points

def calc_sym_array(knum):
    sym_array=np.zeros((knum,knum,knum))
    sym_index=0
    essential_kpoints=np.empty((0,3))
    for i in np.arange(knum):
        for j in np.arange(knum):
            for k in np.arange(knum):
                if sym_array[i,j,k]==0:
                    new_kpoint=np.array([[i,j,k]])
                    # np.concatenate((essential_kpoints,[i,j,k]),axis=0)
                    essential_kpoints=np.vstack((essential_kpoints,new_kpoint))
                    sym_index=sym_index+1
                    sym_points=sym_mapping(i,j,k,knum)
                    for point in sym_points:
                        sym_array[point[0],point[1],point[2]]=sym_index*point[3]
    # print('max_sym_index',sym_index)
    # print('essential_kpoints\n',essential_kpoints.astype(int))
    # print(sym_array)
    return sym_index,essential_kpoints.astype(int),sym_array


def G_11(knum,z_A,z_B,a=1):# and, G_22=-G_diag_11.conj
    n=z_A.size
    k1,k2,k3=gen_full_kgrids(knum,a)
    G_diag_A=np.zeros((n,knum,knum,knum),dtype=np.complex128)
    zazb=z_A*z_B
    G_diag_A = z_B / (zazb - dispersion(k1,k2,k3)**2)
    return G_diag_A



def G_22(knum,z_A,z_B,a=1):# and, G_22=-G_diag_11.conj
    n=z_A.size
    k1,k2,k3=gen_full_kgrids(knum,a)
    G_diag_A=np.zeros((n,knum,knum,knum),dtype=np.complex128)
    zazb=z_A*z_B
    G_diag_A = z_A[:, None, None, None] / (zazb[:, None, None, None] - dispersion(k1,k2,k3)**2)
    return G_diag_A


# G12 is real, or effectively can be treated as real. We are gonna to define it as a real array to accelerate the calculation.
def G_12(knum,z_A,z_B,a=1):
    k1,k2,k3=gen_full_kgrids(knum,a)
    n=z_A.size
    G_offdiag=np.zeros((n,knum,knum,knum))
    zazb=z_A*z_B
    dis=dispersion(k1, k2, k3)
    G_offdiag = dis / (zazb[:, None, None, None].real - dis**2)
    return G_offdiag


def G_11_iterative(knum,z_A,z_B,sigma12):# and, G_22=-G_diag_11.conj
    n=z_A.size
    k1,k2,k3=gen_full_kgrids(knum)
    G_diag_A=np.zeros((n,knum,knum,knum),dtype=np.complex128)
    zazb=z_A*z_B
    G_diag_A = z_B / (zazb - (dispersion(k1,k2,k3)[None,:,:,:]+sigma12)**2)
    return G_diag_A

# @profile
def G_12_iterative(knum,z_A,z_B,sigma12):
    k1,k2,k3=gen_full_kgrids(knum)
    n=z_A.size
    G_offdiag=np.zeros((n,knum,knum,knum))
    zazb=z_A*z_B
    dis_eff=dispersion(k1, k2, k3)[None,:,:,:]+sigma12
    G_offdiag = dis_eff / (zazb[:, None, None, None].real - dis_eff**2)
    return G_offdiag


# @profile
def G_iterative(knum,z_A,z_B,sigma12):
    n=z_A.size
    k1,k2,k3=gen_full_kgrids(knum)
    G_diag_A=np.zeros((n,knum,knum,knum),dtype=np.complex128)
    zazb=z_A*z_B
    dis_eff=dispersion(k1, k2, k3)[None,:,:,:]+sigma12
    denom=1/(zazb - dis_eff**2)
    G_11 = z_B *denom
    G_12 = dis_eff *denom
    return G_11,G_12

def Delta_DMFT(sig1,sig2,U,T,knum=10,a=1):
    mu=U/2
    beta=1/T
    n=sig1.size
    if sig1[-1].real>sig2[-1].real:
        sigA=sig1
        sigB=U-sig1.real+1j*sig1.imag
    else:
        sigA=sig2
        sigB=U-sig2.real+1j*sig2.imag
    om= (2*np.arange(n)+1)*np.pi/beta
    iom=1j*om
    Sigma11=np.zeros((2*n,knum,knum,knum),dtype=complex)
    Sigma11+=ext_sig(sigA)[:,None,None,None]
    Sigma22=np.zeros((2*n,knum,knum,knum),dtype=complex)
    Sigma22+=ext_sig(sigB)[:,None,None,None]
    z_1=z4D(beta,mu,Sigma11,knum,n)#z-delta
    z_2=z4D(beta,mu,Sigma22,knum,n)#z+delta
    G11_iom=G_11(knum,z_1,z_2)
    G11imp_iom=np.sum(G11_iom,axis=(1,2,3))/knum**3
    G22imp_iom=-G11imp_iom.conjugate()
    Gimp_inv_11=1/G11imp_iom
    Gimp_inv_22=1/G22imp_iom
    Delta_11=iom+mu-sigA-Gimp_inv_11[n:]
    Delta_22=iom+mu-sigB-Gimp_inv_22[n:]
    return Delta_11,Delta_22

def particlenumber1D(G_iom,beta):
    n=np.shape(G_iom)[0]
    om=(2*np.arange(n)+1-n)*np.pi/beta
    constinf=(1j*om[-1]-1/G_iom[-1]).real
    Gana=1/(1j*om-constinf)
    particlenumber=(fermi(constinf,beta)+np.sum(G_iom-Gana)/beta).real
    return particlenumber

def particlenumber4D(G_iom,beta):
    n=np.shape(G_iom)[0]
    knum=np.shape(G_iom)[1]
    om=(2*np.arange(n)+1-n)*np.pi/beta
    constinf=(1j*om[-1,None,None,None]-1/G_iom[-1]).real
    Gana=1/(1j*om[:,None,None,None]-constinf)
    particlenumber=((np.sum(fermi(constinf,beta))+np.sum(G_iom-Gana)/beta)/knum**3).real
    return particlenumber