from math import pi, sqrt
from scipy.special import dawsn
from scipy.optimize import minimize
import numpy as np

def is_PD(A):

    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

# Find the nearest positive-definite matrix
def nearest_PD(A):
    
    # based on 
    # N.J. Higham (1988) https://doi.org/10.1016/0024-3795(88)90223-6
    # and 
    # https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    B = (A + A.T)/2
    _, Sigma_mat, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(Sigma_mat), V))

    A_nPD = (B + H) / 2
    A_symm = (A_nPD + A_nPD.T) / 2

    k = 1
    I = np.eye(A_symm.shape[0])

    while not is_PD(A_symm):
        eps = np.spacing(np.linalg.norm(A_symm))

        # MATLAB's 'chol' accepts matrices with eigenvalue = 0, numpy does not not. 
        # So where the matlab implementation uses 'eps(mineig)', we use the above definition.

        min_eig = min(0, np.min(np.real(np.linalg.eigvals(A_symm))))
        A_symm += I * (-min_eig * k**2 + eps)
        k += 1

    return A_symm


# Part 1 - DRT kernel
# this function computes the DRT kernel 
# according to the band-limited formulas given in the article
def k_DRT(omega, omega_prime, sigma_DRT, tau_max, block):

    if block == 're':
        # DRT kernel - real part (even)
        if omega == omega_prime:
            out_val = 0.5*(tau_max/(1+(tau_max*omega)**2)\
                            + np.arctan(tau_max*omega)/omega)
        else:
            num_out_val = omega*np.arctan(tau_max*omega)\
                            - omega_prime*np.arctan(tau_max*omega_prime)
            den_out_val = omega**2-omega_prime**2
            out_val = num_out_val/den_out_val

    elif block == 'im':
        # DRT kernel - imaginary part (odd)
        if omega == omega_prime:
            out_val = 0.5*(-tau_max/(1+(tau_max*omega)**2)\
                            + np.arctan(tau_max*omega)/omega)
        else:
            num_out_val = omega*np.arctan(tau_max*omega_prime)\
                            - omega_prime*np.arctan(tau_max*omega)
            den_out_val = omega**2-omega_prime**2
            out_val = num_out_val/den_out_val

    elif block == 're-im':
        # DRT kernel - re-im
        if omega == omega_prime:
            out_val = -tau_max**2*omega/(2.+2.*(tau_max*omega)**2)
        else:
            arg_log_num = 1+(tau_max*omega)**2 
            arg_log_den = 1+(tau_max*omega_prime)**2
            num_out_val = -omega_prime*(np.log(arg_log_num)-np.log(arg_log_den))
            den_out_val = 2*(omega**2-omega_prime**2)
            out_val = num_out_val/den_out_val

    elif block == 'im-re':
        # DRT kernel - im-re
        if omega == omega_prime:
            out_val = -tau_max**2*omega/(2.+2.*(tau_max*omega)**2)
        else:
            arg_log_num = 1+(tau_max*omega)**2 
            arg_log_den = 1+(tau_max*omega_prime)**2
            num_out_val = -omega*(np.log(arg_log_num)-np.log(arg_log_den))
            den_out_val = 2*(omega**2-omega_prime**2)
            out_val = num_out_val/den_out_val

    else:
        out_val = 0.0

    out_val = (sigma_DRT**2)*out_val
    return out_val

# Part 2 - Stationary-based kernel
# this part is for the stationary-based kernel
# we start by defining the stationary-based kernel
def k_0(x, ker_opts):

    # parameters:
    sigma_SB = ker_opts['sigma_SB'] # prefactor
    ell = ker_opts['ell']  # lengthscale
    SB_ker_type = ker_opts['SB_ker_type'] # type of kernel used
    
    a = 1./(sqrt(2)*ell)*x
    # inverse quadratic kernel
    if SB_ker_type == 'IQ':
        out_val = 1/(1.+a**2)

    # squared exponential kernel
    elif SB_ker_type == 'SE':
        out_val = np.exp(-a**2)

    # multiply by prefactor
    out_val = (sigma_SB**2)*out_val

    return out_val

# HT of the stationary kernel
def k_0_H(x, ker_opts):

    # parameters:
    sigma_SB = ker_opts['sigma_SB'] # prefactor
    ell = ker_opts['ell']  # lengthscale
    SB_ker_type = ker_opts['SB_ker_type'] # type of kernel used

    a = 1./(sqrt(2)*ell)*x

    # inverse quadratic kernel 
    if SB_ker_type == 'IQ':
        out_val = -a/(1.+a**2)

    # squared exponential kernel
    elif SB_ker_type == 'SE':
        out_val = 2./sqrt(pi)*dawsn(a)          

    # multiply by prefactor
    out_val = (sigma_SB**2)*out_val

    return out_val

# stationary-based kernel using the formulas from the article
def k_SB(omega, omega_prime, ker_opts, block):

    if block == 're':
        # from main article: real part is an even function
        out_val = k_0(omega-omega_prime, ker_opts) + k_0(omega+omega_prime, ker_opts)

    elif block == 'im':
        # from main article: imaginary part is an odd function
        out_val = k_0(omega-omega_prime, ker_opts) - k_0(omega+omega_prime, ker_opts)

    elif block == 're-im':
        # from main article: re_im is obtained from HT of k_re w.r.t. omega_prime
        out_val = -k_0_H(omega_prime-omega, ker_opts) - k_0_H(omega+omega_prime, ker_opts)

    elif block == 'im-re':
        out_val = -k_0_H(omega-omega_prime, ker_opts) - k_0_H(omega+omega_prime, ker_opts)

    else:
        out_val = 0

    return out_val

# Part 3 - matrix K
def mat_K(omega_m_vec, omega_n_vec, ker_opts, block):

    # generally we are going to take
    # k = k_DRT + k_SB
    # but we may want to turn on or off one of the two components
    # this is done by activating these two switches
    DRT_switch = ker_opts['DRT'] # DRT kernel
    SB_switch = ker_opts['SB'] # Stationary-based kernel

    # we will need to select either diagonal or off-diagonal submatrices
    # diagonal: re or im
    # off-diagonal: re-im (upper) or im-re (lower)

    # value of the sigma_DRT
    sigma_DRT = ker_opts['sigma_DRT']
    tau_max = ker_opts['tau_max'] # this accounts for the band limitations
    
    # size of the matrix
    N_m_freqs = omega_m_vec.size
    N_n_freqs = omega_n_vec.size

    K_mat = np.zeros([N_m_freqs, N_n_freqs])

    for m in range(0, N_m_freqs):
        for n in range(0, N_n_freqs):

            K_loc = 0.0

            # add DRT kernel if DRT switch is on
            if DRT_switch:
                k_DRT_loc = k_DRT(omega_m_vec[m], omega_n_vec[n], sigma_DRT, tau_max, block)
                K_loc += k_DRT_loc

            # add SB kernel if SB switch is on
            if SB_switch:
                k_SB_loc = k_SB(omega_m_vec[m], omega_n_vec[n], ker_opts, block)
                K_loc += k_SB_loc

            K_mat[m, n] = K_loc

    return K_mat

# Part 4 - marginal likelihood
def NMLL_fct(theta, u, omega_vec, ker_opts_in, type_data):

    # update all required options
    sigma_n = theta[0]
    sigma_DRT = theta[1]
    sigma_SB = theta[2]
    ell = theta[3]

    # this overloads the parameters to a new dictionary
    # so that the external values will not be modified
    ker_opts = ker_opts_in.copy()
    ker_opts['sigma_SB'] = sigma_SB  # prefactor stationary-based kernel
    ker_opts['sigma_DRT'] = sigma_DRT  # prefactor DRT
    ker_opts['ell'] = ell  # lengthscale

    # put together the kernel + error covariance + added model
    N_freqs = omega_vec.size

    if type_data == 'im':
        sigma_L = theta[4]
        K_im = mat_K(omega_vec, omega_vec, ker_opts, type_data)
        Sigma = (sigma_n**2)*np.eye(N_freqs)
        K_full = K_im + Sigma + (sigma_L**2)*np.outer(omega_vec, omega_vec)

    elif type_data == 're':
        sigma_R = theta[4]     
        K_re = mat_K(omega_vec, omega_vec, ker_opts, type_data)
        Sigma = (sigma_n**2)*np.eye(N_freqs)
        K_full = K_re + Sigma + (sigma_R**2)*np.ones_like(Sigma)
    
    else:
        sigma_R = theta[4]
        sigma_L = theta[5]

        K_full = np.zeros((2*N_freqs, 2*N_freqs))
        Sigma = (sigma_n**2)*np.eye(N_freqs)

        K_re = mat_K(omega_vec, omega_vec, ker_opts, 're')
        K_re_im = mat_K(omega_vec, omega_vec, ker_opts, 're-im')
        K_im_re = mat_K(omega_vec, omega_vec, ker_opts, 'im-re')
        K_im = mat_K(omega_vec, omega_vec, ker_opts, 'im')

        K_full[:N_freqs, :N_freqs] = K_re + Sigma + (sigma_R**2)*np.ones_like(Sigma)
        K_full[:N_freqs, N_freqs:] = K_re_im
        K_full[N_freqs:, :N_freqs] = K_im_re
        K_full[N_freqs:, N_freqs:] = K_im + Sigma + (sigma_L**2)*np.outer(omega_vec, omega_vec)


    # Cholesky-decompose K_full
    # begin FC - added 
    if not is_PD(K_full):
        K_full = nearest_PD(K_full)
    
    # end FC - added

    # Cholesky-decompose K_full
    L = np.linalg.cholesky(K_full)

    # solve for alpha
    alpha = np.linalg.solve(L, u)
    alpha = np.linalg.solve(L.T, alpha)

    # output NMLL
    return 0.5*np.dot(u, alpha) + np.sum(np.log(np.diag(L)))

def compute_mu_sigma(theta, ker_opts, inv_K_full, freq_virt_vec, freq_vec, Z_exp_all):
 
    # retrive vals from theta
    sigma_n, sigma_DRT, sigma_SB, ell, sigma_R, sigma_L = theta

    # number of frequencies
    N_freqs = freq_vec.size

    # number of angular frequencies
    N_virt_freqs = freq_virt_vec.size
    
    # angular frequencies
    omega_vec = 2.*pi*freq_vec
    omega_virt_vec = 2.*pi*freq_virt_vec

    mu_re_virt_vec = np.zeros_like(omega_virt_vec)
    sigma_re_virt_vec = np.zeros_like(omega_virt_vec)

    mu_im_virt_vec = np.zeros_like(omega_virt_vec)
    sigma_im_virt_vec = np.zeros_like(omega_virt_vec)

    for index, omega_virt in enumerate(omega_virt_vec):

        # print('iter = ', index+1, '/', N_virt_freqs)
        omega_virt_np = np.array([omega_virt])

        # k_virt_virt
        k_virt_virt_re = mat_K(omega_virt_np, omega_virt_np, ker_opts, 're').flatten() + (sigma_R**2)
        k_virt_virt_im = mat_K(omega_virt_np, omega_virt_np, ker_opts, 'im').flatten() + (sigma_L**2)*omega_virt_np**2

        # k_virt    
        k_virt_re_re = mat_K(omega_virt_np, omega_vec, ker_opts, 're').flatten() + (sigma_R**2)*np.ones(N_freqs)
        k_virt_re_im = mat_K(omega_virt_np, omega_vec, ker_opts, 're-im').flatten()
        k_virt_im_re = mat_K(omega_virt_np, omega_vec, ker_opts, 'im-re').flatten()
        k_virt_im_im = mat_K(omega_virt_np, omega_vec, ker_opts, 'im').flatten() + (sigma_L**2)*omega_vec*omega_virt_np

        # k_virt_re
        k_virt_re = np.zeros(2*N_freqs)
        k_virt_re[:N_freqs] = k_virt_re_re
        k_virt_re[N_freqs:] = k_virt_re_im

        # k_virt_im
        k_virt_im = np.zeros(2*N_freqs)
        k_virt_im[:N_freqs] = k_virt_im_re
        k_virt_im[N_freqs:] = k_virt_im_im

        mu_re_virt_vec[index] = k_virt_re@(inv_K_full@Z_exp_all)
        sigma_re_virt_vec[index] = np.sqrt(k_virt_virt_re - k_virt_re@(inv_K_full@k_virt_re))

        mu_im_virt_vec[index] = k_virt_im@(inv_K_full@Z_exp_all)
        sigma_im_virt_vec[index] = np.sqrt(k_virt_virt_im - k_virt_im@(inv_K_full@k_virt_im))
    
    return mu_re_virt_vec, mu_im_virt_vec, sigma_re_virt_vec, sigma_im_virt_vec

def compute_K_inv(theta, ker_opts, freq_vec):

    # retrive vals from theta
    sigma_n, sigma_DRT, sigma_SB, ell, sigma_R, sigma_L = theta
    
    # number of freqs
    N_freqs = freq_vec.size

    # angular frequencies
    omega_vec = 2.*pi*freq_vec

    # build blocks
    K_re = mat_K(omega_vec, omega_vec, ker_opts, 're')
    K_im = mat_K(omega_vec, omega_vec, ker_opts, 'im')
    K_re_im = mat_K(omega_vec, omega_vec, ker_opts, 're-im')
    K_im_re = mat_K(omega_vec, omega_vec, ker_opts, 'im-re')

    # noise shift
    Sigma = sigma_n**2*np.eye(N_freqs)

    # declare full matrix
    K_full = np.zeros((2*N_freqs, 2*N_freqs))
    K_full[:N_freqs, :N_freqs] = K_re + Sigma + (sigma_R**2)*np.ones(N_freqs)
    K_full[:N_freqs, N_freqs:] = K_re_im
    K_full[N_freqs:, :N_freqs] = K_im_re
    K_full[N_freqs:, N_freqs:] = K_im + Sigma + (sigma_L**2)*np.outer(omega_vec, omega_vec)

    # if not positive definite, then shift
    if not is_PD(K_full):
        K_full = nearest_PD(K_full)

    # cholesky factorization
    L = np.linalg.cholesky(K_full)

    # covariance matrix
    inv_L = np.linalg.inv(L)
    inv_K_full = np.dot(inv_L.T, inv_L)

    return inv_K_full

def compute_ALM(log10_freq_virt, theta, ker_opts, freq_vec, inv_K_full):

    sigma_n, sigma_DRT, sigma_SB, ell, sigma_R, sigma_L = theta

    N_freqs = freq_vec.size    
    #
    omega_vec = 2.*pi*freq_vec
    omega_virt = 2.*pi*(10**log10_freq_virt)

    omega_virt_np = np.array([omega_virt])

    # k_virt_virt
    k_virt_virt_re = mat_K(omega_virt_np, omega_virt_np, ker_opts, 're').flatten() + (sigma_R**2)
    k_virt_virt_re_im = mat_K(omega_virt_np, omega_virt_np, ker_opts, 're-im').flatten()
    k_virt_virt_im_re = mat_K(omega_virt_np, omega_virt_np, ker_opts, 'im-re').flatten()
    k_virt_virt_im = mat_K(omega_virt_np, omega_virt_np, ker_opts, 'im').flatten() + (sigma_L**2)*omega_virt_np**2

    k_virt_virt = np.zeros((2, 2))
    k_virt_virt[0, 0] = k_virt_virt_re
    k_virt_virt[0, 1] = k_virt_virt_re_im
    k_virt_virt[1, 0] = k_virt_virt_im_re
    k_virt_virt[1, 1] = k_virt_virt_im

    # k_virt    
    k_virt_re_re = mat_K(omega_virt_np, omega_vec, ker_opts, 're').flatten() + (sigma_R**2)*np.ones(N_freqs)
    k_virt_re_im = mat_K(omega_virt_np, omega_vec, ker_opts, 're-im').flatten()
    k_virt_im_re = mat_K(omega_virt_np, omega_vec, ker_opts, 'im-re').flatten()
    k_virt_im_im = mat_K(omega_virt_np, omega_vec, ker_opts, 'im').flatten() + (sigma_L**2)*omega_vec*omega_virt_np

    # k_virt_re
    k_virt = np.zeros((2, 2*N_freqs))
    k_virt[0, :N_freqs] = k_virt_re_re
    k_virt[0, N_freqs:] = k_virt_re_im
    k_virt[1, :N_freqs] = k_virt_im_re
    k_virt[1, N_freqs:] = k_virt_im_im
    
    # print('shape: k_virt_virt = ', k_virt_virt.shape, '; k_virt = ', k_virt.shape, '; inv_K_full = ', inv_K_full.shape)

    covariance = k_virt_virt - k_virt@(inv_K_full@k_virt.T)
    indicator = np.linalg.det(covariance)
    # indicator = np.trace(covariance)
    # indicator = np.linalg.eigvals(covariance).max()
    
    sigma_tot = -indicator
    
    return sigma_tot


def compute_ALC(log10_freq_Np1, theta, ker_opts, freq_vec_N, log10_freq_vec_int):

    sigma_n, sigma_DRT, sigma_SB, ell, sigma_R, sigma_L = theta

    # freq_Np1 from log10
    freq_Np1 = 10**log10_freq_Np1

    # search where to insert
    index = np.searchsorted(freq_vec_N, freq_Np1)
    # update frequency vector
    freq_vec_Np1 = np.insert(freq_vec_N, index, freq_Np1)
    # create omega vector
    omega_vec_Np1 = 2.0*pi*freq_vec_Np1

    # number of freqs
    N_freqs_Np1 = freq_vec_Np1.size

    # compute K_inv from the new freq_vec 
    inv_K_Np1 = compute_K_inv(theta, ker_opts, freq_vec_Np1)

    indicator = np.zeros_like(log10_freq_vec_int)
    
    for iter, log10_freq_int in enumerate(log10_freq_vec_int):

        # omega
        omega_int = 2.0*pi*(10**log10_freq_int)
        omega_int_np = np.array([omega_int])

        # k_int_int
        k_int_int_re = mat_K(omega_int_np, omega_int_np, ker_opts, 're').flatten() + (sigma_R**2)
        k_int_int_re_im = mat_K(omega_int_np, omega_int_np, ker_opts, 're-im').flatten()
        k_int_int_im_re = mat_K(omega_int_np, omega_int_np, ker_opts, 'im-re').flatten()
        k_int_int_im = mat_K(omega_int_np, omega_int_np, ker_opts, 'im').flatten() + (sigma_L**2)*omega_int_np**2

        k_int_int = np.zeros((2, 2))
        k_int_int[0, 0] = k_int_int_re
        k_int_int[0, 1] = k_int_int_re_im
        k_int_int[1, 0] = k_int_int_im_re
        k_int_int[1, 1] = k_int_int_im
        
        # k_int    
        k_int_re_re = mat_K(omega_int_np, omega_vec_Np1, ker_opts, 're').flatten() + (sigma_R**2)*np.ones(N_freqs_Np1)
        k_int_re_im = mat_K(omega_int_np, omega_vec_Np1, ker_opts, 're-im').flatten()
        k_int_im_re = mat_K(omega_int_np, omega_vec_Np1, ker_opts, 'im-re').flatten()
        k_int_im_im = mat_K(omega_int_np, omega_vec_Np1, ker_opts, 'im').flatten() + (sigma_L**2)*omega_vec_Np1*omega_int_np

        k_int = np.zeros((2, 2*N_freqs_Np1))
        k_int[0, :N_freqs_Np1] = k_int_re_re
        k_int[0, N_freqs_Np1:] = k_int_re_im
        k_int[1, :N_freqs_Np1] = k_int_im_re
        k_int[1, N_freqs_Np1:] = k_int_im_im

        # print('shape: k_int_int = ', k_int_int.shape, '; k_int = ', k_int.shape, '; inv_K_Np1 = ', inv_K_Np1.shape)
        covariance = k_int_int - k_int@(inv_K_Np1@k_int.T)
        indicator[iter] = np.linalg.det(covariance)
        # indicator[iter] = np.trace(covariance)
        # indicator[iter] = np.linalg.eigvals(covariance).max()

    ALC_out = np.trapz(indicator, x=log10_freq_vec_int)
    
    return ALC_out

def compute_opt_theta(theta_0, ker_opts_0, freq_vec, Z_exp_all, type_data = 'all'):

    omega_vec = 2.*pi*freq_vec

    if type_data == 'all':
        def print_results(theta):
            print('%.4E, %.4E, %.4E, %.4E, %.4E, %.6E; evidence = %.8E'%(theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], NMLL_fct(theta, Z_exp_all, omega_vec, ker_opts_0, type_data)))
    else:
        def print_results(theta):
            print('%.4E, %.4E, %.4E, %.4E, %.6E; evidence = %.8E'%(theta[0], theta[1], theta[2], theta[3], theta[4], NMLL_fct(theta, Z_exp_all, omega_vec, ker_opts_0, type_data)))
 

    res = minimize(NMLL_fct, theta_0, args=(Z_exp_all, omega_vec, ker_opts_0, type_data), method='Powell', \
                        callback=print_results, options={'disp': True, 'xtol': 1E-6, 'ftol': 1E-6})

    theta = res.x

    if type_data == 'all':
        sigma_n, sigma_DRT, sigma_SB, ell, sigma_R, sigma_L = theta
    else:
        sigma_n, sigma_DRT, sigma_SB, ell, sigma_L = theta

    ker_opts = ker_opts_0.copy()

    ker_opts['sigma_SB'] = sigma_SB
    ker_opts['ell'] = ell
    ker_opts['sigma_DRT'] = sigma_DRT

    return theta, ker_opts

def update_exp(freq_new, Z_exp_new, freq_vec_N, Z_exp_Np):

    # search where to insert
    index = np.searchsorted(freq_vec_N, freq_new)

    # update frequency vector
    freq_vec_Np1 = np.insert(freq_vec_N, index, freq_new)


    N_freqs_Np1 = freq_vec_Np1.size
    # update Z_exp
    Z_exp_Np1 = np.insert(Z_exp_Np, index, Z_exp_new)
    Z_exp_all_Np1 = np.zeros(2*N_freqs_Np1)
    Z_exp_all_Np1[:N_freqs_Np1] = Z_exp_Np1.real
    Z_exp_all_Np1[N_freqs_Np1:] = Z_exp_Np1.imag

    return freq_vec_Np1, Z_exp_Np1, Z_exp_all_Np1

def res_score(res, band):
    # count the points fallen in the band
    count = np.zeros(3)
    for k in range(3):
        count[k] = np.sum(np.logical_and(res < (k+1)*band, res > -(k+1)*band))
        
    return count/len(res)