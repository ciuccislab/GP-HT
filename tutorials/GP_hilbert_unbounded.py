from math import pi, sqrt
from scipy.special import dawsn 
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

def k_DCT(omega, omega_prime, sigma_DCT, tau_min, block):

    if block == 're':
        # DCT kernel - real part (even)
        if omega == omega_prime:
            out_val = 1./4.*omega*(pi + \
                            2.*tau_min*omega/(1+(tau_min*omega)**2)-\
                            2.*np.arctan(tau_min*omega))
        else:
            num_out_val = omega*omega_prime *\
                            (pi*(omega-omega_prime)+\
                            2*omega_prime*np.arctan(tau_min*omega)-\
                            2*omega*np.arctan(tau_min*omega_prime))
            den_out_val = 2*(omega**2-omega_prime**2)
            out_val = num_out_val/den_out_val

    elif block == 'im':
        # DCT kernel - imaginary part (odd)
        if omega == omega_prime:
            out_val = 1./4.*omega*(pi -\
                                   2.*tau_min*omega/(1+(tau_min*omega)**2) - \
                                   2.*np.arctan(tau_min*omega))
        else:
            num_out_val = omega*omega_prime *\
                            (pi*(omega-omega_prime)-\
                            2*omega*np.arctan(tau_min*omega)+\
                            2*omega_prime*np.arctan(tau_min*omega_prime))
            den_out_val = 2*(omega**2-omega_prime**2)
            out_val = num_out_val/den_out_val

    elif block == 're-im':
        # DCT kernel - re-im
        if omega == omega_prime:
            out_val = 0.5*omega/(1.+(tau_min*omega)**2)
        else:
            arg_log_num = omega**2*(1+(tau_min*omega_prime)**2)
            arg_log_den = omega_prime**2*(1+(tau_min*omega)**2)
            num_out_val = omega**2*omega_prime*(np.log(arg_log_num)-np.log(arg_log_den))
            den_out_val = 2*(omega**2-omega_prime**2)
            out_val = num_out_val/den_out_val

    elif block == 'im-re':
        # DCT kernel - im-re
        if omega == omega_prime:
            out_val = 0.5*omega/(1.+(tau_min*omega)**2)
        else:
            arg_log_num = omega**2*(1+(tau_min*omega_prime)**2)
            arg_log_den = omega_prime**2*(1+(tau_min*omega)**2)
            num_out_val = omega_prime**2*omega*(np.log(arg_log_num)-np.log(arg_log_den))
            den_out_val = 2*(omega**2-omega_prime**2)
            out_val = num_out_val/den_out_val

    else:
        out_val = 0.0

    out_val = (sigma_DCT**2)*out_val
    return out_val

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

def k_SB(omega, omega_prime, ker_opts, block):

    if block == 're':
        # from main article: real part is an even function
        out_val = k_0(omega-omega_prime, ker_opts) + k_0(omega+omega_prime, ker_opts)
    elif block == 'im':
        # from main article: imaginary part is an odd function
        out_val = k_0(omega-omega_prime, ker_opts) - k_0(omega+omega_prime, ker_opts)
    elif block == 're-im':
        # from main article: re_im is obtained from HT of k_re w.r.t. omega_prime
        out_val = k_0_H(omega_prime-omega, ker_opts) + k_0_H(omega+omega_prime, ker_opts)
    elif block == 'im-re':
        out_val = k_0_H(omega-omega_prime, ker_opts) + k_0_H(omega+omega_prime, ker_opts)
    else:
        out_val = 0

    return out_val

def mat_K(omega_m_vec, omega_n_vec, ker_opts, block):

    # generally we are going to take
    # k = k_SB + k_DCT 
    # but we may want to turn on or off one of the two components
    # this is done by activating these two switches
    DCT_switch = ker_opts['DCT'] # DCT kernel
    SB_switch = ker_opts['SB'] # Stationary-derived kernel

    # we will need to select either diagonal or off-diagonal submatrices
    # diagonal: re or im
    # off-diagonal: re-im (upper) or im-re (lower)

    # value of the sigma_DCT
    sigma_DCT = ker_opts['sigma_DCT']

    # value of the tau_min
    tau_min = ker_opts['tau_min']

    # size of the matrix
    N_m_freqs = omega_m_vec.size
    N_n_freqs = omega_n_vec.size

    K_mat = np.zeros([N_m_freqs, N_n_freqs])

    for m in range(0, N_m_freqs):
        for n in range(0, N_n_freqs):

            K_loc = 0.0

            if DCT_switch:
                k_DCT_loc = k_DCT(
                    omega_m_vec[m], omega_n_vec[n], sigma_DCT, tau_min, block)
                K_loc += k_DCT_loc
            
            if SB_switch:
                k_SB_loc = k_SB(omega_m_vec[m], omega_n_vec[n], ker_opts, block)
                K_loc += k_SB_loc

            K_mat[m, n] = K_loc

    return K_mat

def NMLL_fct(theta, u, omega_vec, ker_opts_in, type_data='im'):
        
    sigma_n = theta[0]
    sigma_DCT = theta[1]
    sigma_SB = theta[2]
    ell = theta[3]

    # this overloads the parameters to a new dictionary
    # so that the external values will not be redefined
    ker_opts = ker_opts_in.copy()
    ker_opts['sigma_SB'] = sigma_SB # prefactor stationary-based kernel
    ker_opts['sigma_DCT'] = sigma_DCT # prefactor DCT
    ker_opts['ell'] = ell  # lengthscale

    N_freqs = omega_vec.size

    K_im = mat_K(omega_vec, omega_vec, ker_opts, type_data)
    Sigma = (sigma_n**2)*np.eye(N_freqs)
    K_full = K_im + Sigma

    # begin FC - added 
    if not is_PD(K_full):
        K_full = nearest_PD(K_full)

    # end FC - added     
     
    # Cholesky
    L = np.linalg.cholesky(K_full)
     
    # solve for alpha
    alpha = np.linalg.solve(L, u)
    alpha = np.linalg.solve(L.T, alpha)
    return 0.5*np.dot(u,alpha) + np.sum(np.log(np.diag(L)))

def res_score(res, band):
    # count the points fallen in the band
    count = np.zeros(3)
    for k in range(3):
        count[k] = np.sum(np.logical_and(res < (k+1)*band, res > -(k+1)*band))
        
    return count/len(res)