#!/usr/bin/env python

from scipy import *
from scipy import integrate, interpolate
from pylab import *
import cmath
import sys

par={
    'dos' : 'DOS_3D.dat',
    'minSubtract' : 0.1,
    'intervalSubstract' : 1.,
    'precision' : 1e-10
    }

class Hilb:
    def __init__(self, x, Di):
        self.x = x
        self.Di = Di
        D_norm = integrate.trapezoid(Di,x=x)
        self.Di *= 1/D_norm# normalization
        self.dh = zeros(len(x))
        self.dh[1:-1] = 0.5*(x[2:]-x[:-2])
        self.dh[0]=0.5*(x[1]-x[0])
        self.dh[-1]=0.5*(x[-1]-x[-2])
        
    def __call__(self, z):
        D0, a, b, ia,ib = 0, 0, 0, 0, 0
        if (abs(z.imag) < par['minSubtract'] and z.real>self.x[1] and z.real<self.x[-1]):
            fD = interpolate.interp1d(self.x,self.Di)
            D0 = fD(z.real)
            icc = where( logical_and(z.real-par['intervalSubstract']<self.x,self.x<z.real+par['intervalSubstract']) )[0]
            ia,ib = icc[0],icc[-1]
            a = 0.5*(self.x[ia-1]+self.x[ia])
            b = 0.5*(self.x[ib-1]+self.x[ib])
            if ia==0: a=self.x[0]
        #print('icc=', icc)
        #print('D0=', D0, 'ia=', ia, 'ib=', ib, 'a=', a, 'b=', b)
        if D0>0:
            r1 = sum( self.Di[:ia]*self.dh[:ia]/(z-self.x[:ia]) )
            r2 = sum( (self.Di[ia:ib]-D0)*self.dh[ia:ib]/(z-self.x[ia:ib]) ) + D0*(log(z-a)-log(z-b))
            r3 = sum( self.Di[ib:]*self.dh[ib:]/(z-self.x[ib:]) )
            #print('r1=', r1, 'r2=', r2, 'r3=', r3)
            return r1+r2+r3
        else:
            return sum( self.Di*self.dh/(z-self.x) )

def SCC_Para(W, om, beta, mu, U, Sg, Real, delta=0.01):
    if Real:
        om = linspace(-10,10,100)
        yr = zeros(len(om),dtype=complex)
        z_ = om+mu-Sg + 1j*delta
    else:
        om = (2*arange(100)+1)*pi/beta
        z_ = om*1j+mu-Sg
    yr = zeros(len(om),dtype=complex)
    for it in range(len(z_)):
        G = W(z_[it])
        Dlt = z_[it] - 1/G
        yr[it] = Dlt
    return yr

def SCC_AFM(W, om, beta, mu, U, Sg_A, Sg_B, Real, delta=0.01):
    if Real:
        z_A = om+mu-Sg_A + 1j*delta
        z_B = om+mu-Sg_B + 1j*delta
    else:
        z_A = om*1j+mu-Sg_A
        z_B = om*1j+mu-Sg_B
    
    yr_A = zeros(len(om),dtype=complex)
    yr_B = zeros(len(om),dtype=complex)
    g_A = zeros(len(om),dtype=complex)
    g_B = zeros(len(om),dtype=complex)
    for it in range(len(z_A)):
        # equivalent to : z = sqrt(z_A[it]*z_B[it])
        r_A,p_A = cmath.polar(z_A[it])
        r_B,p_B = cmath.polar(z_B[it])
        z = cmath.rect(sqrt(r_A*r_B), (p_A+p_B)/2.)  # z = sqrt(z_A[it]*z_B[it])
        # print(z)
        w = W(z)
        G_A = z_B[it]/z * w
        G_B = z_A[it]/z * w
        g_A[it]=G_A
        g_B[it]=G_B
        Dlt_A = z_A[it] - 1/G_A
        Dlt_B = z_B[it] - 1/G_B
        yr_A[it] = Dlt_A
        yr_B[it] = Dlt_B
    # plot(om, g_A.real, label='GA_real')
    # plot(om, g_A.imag, label='GA_imag')
    # plot(om, g_B.real, label='GB_real')
    # plot(om, g_B.imag, label='GB_imag')
    
    # legend(loc='best')
    # grid()
    # show()
    return (yr_A, yr_B,g_A,g_B)

    
if __name__ == '__main__':
    # read DOS and normalize it
    x, Di = loadtxt(par['dos']).T
    W = Hilb(x,Di)
    
    Real=False
    T=0.01
    beta=1/T
    mu=1.0
    U=2.0
    sigma=np.loadtxt('{}_{}.dat'.format(U,T))[:500,:]
    # sigma=np.loadtxt('test_new_sig.imp')[:500,:]
    Sg_AA=sigma[:,1]+1j*sigma[:,2]
    Sg_BB=sigma[:,3]+1j*sigma[:,4]
    # Sg_AA=U/2.+0.1
    # Sg_BB=U/2.-0.1
    # Sg = U/2.
    
    if Real:
        om = linspace(-10,10,500)
    else:
        om = (2*arange(500)+1)*pi/beta
    
    #Dlt = SCC_Para(W, om, beta, mu, U, Sg, Real)
    #plot(om, Dlt.real, label='real')
    #plot(om, Dlt.imag, label='imag')

    Dlt_A,Dlt_B = SCC_AFM(W, om, beta, mu, U, Sg_AA, Sg_BB, Real)
    plot(om, Dlt_A.real, label='Dlt_A_real')
    plot(om, Dlt_A.imag, label='Dlt_A_imag')
    plot(om, Dlt_B.real, label='Dlt_B_real')
    plot(om, Dlt_B.imag, label='Dlt_B_imag')
    
    legend(loc='best')
    grid()
    show()
    sys.exit(0)
    
