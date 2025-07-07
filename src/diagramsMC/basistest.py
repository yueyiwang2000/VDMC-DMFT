import numpy as np
from scipy.interpolate import interp1d
from scipy.special import roots_legendre
from numpy.polynomial.legendre import Legendre

def tau_gauss_legendre(N, beta):
    """
    Gauss–Legendre 节点映射到 [0, β]
    """
    x, _ = roots_legendre(N)
    return 0.5 * beta * (x + 1)

def tau_gauss_lobatto(N, beta):
    """
    Gauss–Lobatto–Legendre 节点（含端点）映射到 [0, β]
    """
    if N < 2:
        raise ValueError("N must be >= 2 for Gauss-Lobatto")
    P = Legendre.basis(N - 1)
    x_int = P.deriv().roots()  # P_{N-1}' 的根
    x = np.concatenate(([-1.0], x_int, [1.0]))
    x.sort()
    return 0.5 * beta * (x + 1)

def tau_logistic(N, beta, p=2):
    """
    对称 Logistic 映射：端点聚集，可调节 p 参数
    """
    u = np.linspace(0, 1, N)
    return beta * (u**p / (u**p + (1 - u)**p))

def tau_tanh_sinh(N, beta, lam=1.0, Y=5.0):
    """
    双指数 (tanh–sinh) 变换：极端端点聚集
    """
    y = np.linspace(-Y, Y, N)
    return 0.5 * beta * (1 + np.tanh(lam * y))

def fermi_kernel(t, w, beta):
    t = np.asarray(t)
    w = np.asarray(w)

    x = beta * w / 2.0
    y = 2.0 * t / beta - 1.0

    result = np.zeros((t.size, w.size))

    mask_large_x = x > 100
    mask_small_x = x < -100
    mask_mid_x = ~mask_large_x & ~mask_small_x
    # print(mask_large_x,mask_mid_x,mask_small_x)
    result[:,mask_large_x] = np.exp(-x[None,mask_large_x] * (y[:,None] + 1.0))
    result[:,mask_small_x] = np.exp(x[None,mask_small_x] * (1.0 - y[:,None]))
    result[:,mask_mid_x] = np.exp(-x[None,mask_mid_x] * y[:,None]) / (2.0 * np.cosh(x[None,mask_mid_x]))

    return result


def generate_stable_basis(beta, N_tau_nu, N_omega, L, N_tau_uni,opt='chebyshev'):
    """
    生成两步 SVD 稳定基底：
    1. 在非均匀 Chebyshev 时间格点和 Matsubara 频率格点上做第一次 SVD；
    2. 将得到的前 L 模基函数插值到均匀时间格点，再做第二次 SVD。
    
    参数:
        beta (float): 逆温度 β
        N_tau_nu (int): 非均匀时间格点数量
        N_omega (int): Matsubara 频率格点数量
        L (int): 保留的模式数量
        N_tau_uni (int): 均匀时间格点数量
    
    返回:
        dict 包含：
        - tau_nonuni: 非均匀时间格点 (shape: N_tau_nu)
        - omega: Matsubara 频率格点 (shape: N_omega)
        - basis1: 第一次 SVD 左奇异向量前 L 列 (shape: N_tau_nu x L)
        - sv1: 第一次 SVD 奇异值前 L (shape: L)
        - tau_uniform: 均匀时间格点 (shape: N_tau_uni)
        - basis2: 第二次 SVD 左奇异向量前 L 列 (shape: N_tau_uni x L)
        - sv2: 第二次 SVD 奇异值前 L (shape: L)
    """
    # 1. 非均匀 Chebyshev 时间格点 [0, β)
    if opt=='chebyshev':
        j = np.arange(N_tau_nu)
        tau_nonuni = 0.5 * beta * (1 + np.cos((2*j + 1) / (2 * N_tau_nu) * np.pi))
    elif opt=='gauss_legendre':
        tau_nonuni = tau_gauss_legendre(N_tau_nu, beta)
    elif opt=='gauss_lobatto':
        tau_nonuni = tau_gauss_lobatto(N_tau_nu, beta)
    elif opt=='logistic':
        tau_nonuni = tau_logistic(N_tau_nu, beta)
    elif opt=='tanh_sinh':
        tau_nonuni = tau_tanh_sinh(N_tau_nu, beta)
    # 2. Fermionic Matsubara 频率 (仅正频率示例)
    n = np.arange(2*N_omega)-N_omega
    omega = (2*n + 1) * np.pi / beta
    omlist=(2*np.arange(2*N_omega)+1-2*N_omega)*np.pi/beta 
    # 3. 构造核矩阵 K_{ij} = e^{-τ_i ω_j} / (1 + e^{-β ω_j})
    K=fermi_kernel(tau_nonuni,omlist,beta)
    # K = np.exp(-np.outer(tau_nonuni, omega)) / (1 + np.exp(-beta * omega))
    
    # 4. 第一次 SVD
    U, S, Vh = np.linalg.svd(K, full_matrices=False)
    U_L = U[:, :L]
    S_L = S[:L]
    
    
    return {
        "tau_nonuni": tau_nonuni,
        "omega": omega,
        "basis1": U_L,
        "sv1": S_L,
    }

# 示例调用
beta = 10.0
N_tau_nu = 500
N_omega = 500
L = 10
N_tau_uni = 500

result = generate_stable_basis(beta, N_tau_nu, N_omega, L, N_tau_uni,opt='gauss_lobatto')# options: chebyshev, gauss_legendre, gauss_lobatto, logistic, tanh_sinh
print("第一次 SVD 奇异值:", result["sv1"])
print("第二次 SVD 奇异值:", result["sv2"])

import matplotlib.pyplot as plt
#plot all basis functions
plt.figure(figsize=(10, 5))
for i in range(L):
    plt.plot(result["basis1"][:, i], label=f"Basis Function {i+1}")
    plt.legend()
    plt.show()

#basis2
# plt.figure(figsize=(10, 5))
# for i in range(L):
#     plt.plot(result["basis2"][:, i], label=f"Basis Function {i+1}")
#     plt.legend()
#     plt.show()
