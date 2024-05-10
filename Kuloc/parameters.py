import numpy as np
from astropy.cosmology import Planck18, z_at_value

#--------# POSSIS BNS
def POSSIS_BNS_parameters():
    M_dyn_list = np.linspace(0.001,0.02,10)
    M_pm_list = np.linspace(0.01,0.13,10)
    phi_list = np.linspace(0,90,10)
    M_dyn_, M_pm_, phi_= np.meshgrid(M_dyn_list,M_pm_list,phi_list)

    M_dyn_flat = M_dyn_.flatten()
    M_pm_flat = M_pm_.flatten()
    phi_flat = phi_.flatten()
    param_flat = np.array([M_dyn_flat,M_pm_flat,phi_flat]).T
    return param_flat

def POSSIS_NSBH_parameters():
    M_dyn_list = np.linspace(0.01,0.09,20)
    M_pm_list = np.linspace(0.01,0.09,20)
    M_dyn_, M_pm_ = np.meshgrid(M_dyn_list,M_pm_list)

    M_dyn_flat = M_dyn_.flatten()
    M_pm_flat = M_pm_.flatten()
    param_flat = np.array([M_dyn_flat,M_pm_flat]).T
    return param_flat


def BNS_parameters():
    Mc_list = np.linspace(1,2,10)
    q_list = np.linspace(0.6,1,10)
    lambda_list = np.linspace(110,500,10)
    Mtov_list = np.linspace(2,2.3,10)
    Mc_,q_,lambda_,Mtov_ = np.meshgrid(Mc_list,q_list,lambda_list,Mtov_list)

    Mc_flat = Mc_.flatten()
    q_flat = q_.flatten()
    lambda_flat = lambda_.flatten()

    Mtov_flat = Mtov_.flatten()
    eta_flat = np.array([0.2 for i in range(len(Mc_flat))]).T
    param_bns_flat = np.array([Mc_flat,q_flat,lambda_flat,Mtov_flat,eta_flat]).T
    return param_bns_flat

def Kasen_parameters():
    logMej = np.linspace(-3,-1,10)
    vej = np.linspace(0.03,0.3,10)
    logX = np.linspace(-9,-1,10)
    return None

def Kasen_2comp_parameters():
    Mej_blue = 10**np.linspace(-3,-1,10)
    Mej_red = 10**np.linspace(-3,-1,10)
    X_blue = np.array([1e-3,1e-2,1e-1])
    X_red = np.array([1e-7,1e-6,1e-5,1e-4])
    Mej_blue_, Mej_red_, X_blue_, X_red_ = np.meshgrid(Mej_blue,Mej_red,X_blue,X_red)
    Mej_blue_flat, Mej_red_flat, X_blue_flat, X_red_flat \
        = Mej_blue_.flatten(), Mej_red_.flatten(), X_blue_.flatten(), X_red_.flatten()
    vej_blue = np.array([0.2 for i in range(len(Mej_blue_flat))]).T
    vej_red = np.array([0.1 for i in range(len(Mej_blue_flat))]).T
    param_flat = np.array([Mej_blue_flat,vej_blue,X_blue_flat,Mej_red_flat,vej_red,X_red_flat]).T
    return param_flat

def random_parameters(redshifts, model,r_v=2., ebv_rate=0.11,**kwargs):
    # Amplitude
    amp = []
    for z in redshifts:
        #amp.append(10**(-0.4*Planck18.distmod(z).value))
        d_l = Planck18.luminosity_distance(z).value * 1e5
        amp.append(d_l**-2)

    return {
        'amplitude': np.array(amp),
        'hostr_v': r_v * np.ones(len(redshifts)),
        'hostebv': np.random.exponential(ebv_rate, len(redshifts))
        }

def random_parameters_ang(redshifts, model,r_v=2., ebv_rate=0.11,**kwargs):
    # Amplitude
    amp = []
    for z in redshifts:
        amp.append(10**(-0.4*Planck18.distmod(z).value))
    
    theta = np.arccos(np.random.random(len(redshifts))) / np.pi * 180

    return {
        'amplitude': np.array(amp),
        'theta': theta, 
        'hostr_v': r_v * np.ones(len(redshifts)),
        'hostebv': np.random.exponential(ebv_rate, len(redshifts))
        }