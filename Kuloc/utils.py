"""
Aujust
aujust@mail.ustc.edu.cn
"""
import argparse
import simsurvey
import multiprocess as mp
from concurrent.futures import ProcessPoolExecutor as executor
from random import choices
import sys, os
import glob
import copy
import healpy as hp 
from scipy.interpolate import RectBivariateSpline as Spline2d
from .Pointings_request import Pointings_request
import requests
from bs4 import BeautifulSoup
from astropy.time import Time
from .limited_mag import getlim
import numpy as np
import pandas as pd
import sncosmo
import astropy.units as u
import astropy.constants as c
import pickle
from .parameters import *
sys.path.append('/home/Aujust/data/Kilonova/KN_Inference/')
import KN_Inference as knif

sfd98_dir = '/home/Aujust/data/Kilonova/Constraint/data/sfd98'

def Possismodel(param_,model,phase):
    #M_dyn,M_wind,phi
    wave = np.linspace(1e2,9.99e4,500)
    cos_theta_list = np.linspace(0,1,11)
    #This is kilonovanet model
    flux = []
    for ii, cos_theta in enumerate(cos_theta_list):
        param = np.concatenate((param_,[cos_theta]))
        flux.append(model.predict_spectra(param,phase)[0].T/(4*np.pi*pc10**2))

    return phase, wave, cos_theta_list, np.array(flux).T

def Possismodel_tf(param,model,phase,wave=np.linspace(1e2,9.99e4,500)):
    #M_dyn,M_wind,phi
    #param_ = copy.deepcopy(param)
    #param_ = np.array([item for item in param])
    
    #m_dyn,m_pm = param[0],param[1]
    #param_ = np.array([m_dyn,m_pm,phi])
    #param_[0:2] = np.log10(param_[0:2])
    #wave = np.linspace(1e2,9.99e4,500)
    m_dyn,m_pm,phi = param[0],param[1],param[2]
    param_ = np.array([m_dyn,m_pm,phi])
    param_[0:2] = np.log10(param_[0:2])
    cos_theta_list = np.linspace(0,1,11)

    flux = []
    
    for ii, cos_theta in enumerate(cos_theta_list):
        param = np.concatenate((param_,[cos_theta]))
        flux.append(model.calc_spectra(tt=phase,param_list=param,wave=wave)[-1])
    return phase, wave, cos_theta_list, np.array(flux).T

def Possismodel_NSBH_tf(param,model,phase,wave=np.linspace(1e2,9.99e4,500)[:100]):
    #M_dyn,M_wind,phi
    #param_ = copy.deepcopy(param)
    #param_ = np.array([item for item in param])
    
    #m_dyn,m_pm = param[0],param[1]
    #param_ = np.array([m_dyn,m_pm,phi])
    #param_[0:2] = np.log10(param_[0:2])
    #wave = np.linspace(1e2,9.99e4,500)
    m_dyn,m_pm = param[0],param[1]
    param_ = np.array([m_dyn,m_pm])
    param_ = np.log10(param_)
    cos_theta_list = np.linspace(0,1,11)

    flux = []
    
    for ii, cos_theta in enumerate(cos_theta_list):
        param = np.concatenate((param_,[cos_theta]))
        flux.append(model.calc_spectra(tt=phase,param_list=param,wave=wave)[-1])
    return phase, wave, cos_theta_list, np.array(flux).T


def Possismodel_tf_angular(param,model,phase,wave=np.linspace(1e2,9.99e4,500)):
    #M_dyn,M_wind,phi
    #param_ = copy.deepcopy(param)
    m_dyn,m_pm,phi,cos_theta = param[0],param[1],param[2],param[3]
    param_ = np.array([m_dyn,m_pm,phi,cos_theta])
    param_[0:2] = np.log10(param_[0:2])

    flux = model.calc_spectra(tt=phase,param_list=param_,wave=wave)[-1]
    return phase, wave, np.array(flux).T


def Possismodel_angular(m_dyn,m_pm,phi,cos_theta,model,phase):

    wave = np.linspace(1e2,9.99e4,500)
    #This is kilonovanet model
    param = np.array([m_dyn,m_pm,phi,cos_theta])
    flux = model.predict_spectra(param,phase)[0].T/(4*np.pi*pc10**2)
    return phase, wave, np.array(flux).T

def Kasenmoel(param,model,phase,wave):
    wave = np.linspace(1e2,9.99e4,500)
    pass

def Kasenmodel_2comp(param,model,phase,wave):
    out = model.calc_spectra(tt=np.linspace(0,7,100),param_list=param,wave=wave) #Kasen2comp
    phase, wave, flux = out
    return phase, wave, flux

def load_plan(plan_dir):
    if os.path.exists(plan_dir):
        print('Load plan form file.')
        with open(plan_dir,'rb') as handle:
            plan = pickle.load(handle)
        handle.close()
    else:
        print('No matched plan file, fetching plan from TreasureMap. ')
        Z = {
            'graceid':event_name,
            'instruments':[telescope]
        }

        pr = Pointings_request(Z)
        plan = pr.plan
        with open(plan_dir,'wb') as handle:
            pickle.dump(plan,handle)
            handle.close()
    return plan

def load_skymap(skymap_dir,multiorder=False):
    if multiorder:
        m, meta = read_sky_map(skymap_dir,nest=False,distances=True)
        prob, distmu, distsigma = m[0], m[1], m[2]
    else:
        prob, distmu,distsigma,distnorm = hp.fitsfunc.read_map(skymap_dir,field=[0,1,2,3])
        
    skymap = {
                'prob':prob,
                'distmu':distmu,
                'distsigma':distsigma
                }
    return skymap


def load_filter_list():
    all = True
    filt_list = ['sdssu','sdssg','sdssr','sdssi','sdssz','desg','desr','desi','desz','desy','f435w','f475w','f555w','f606w','f625w',
        'f775w','nicf110w','nicf160w','f098m','f105w','f110w','f125w','f127m','f139m','f140w','f153m','f160w','f218w','f225w',
        'f275w','f300x','f336w','f350lp','f390w','f689m','f763m','f845m','f438w','uvf555w','uvf475w',
        'uvf606w','uvf625w','uvf775w','uvf814w','uvf850lp','cspb','csphs','csphd','cspjs','cspjd','cspv3009',
        'cspv3014','cspv9844','cspys','cspyd','cspg','cspi','cspk','cspr','cspu','f070w','f090w','f115w','f150w',
        'f200w','f277w','f356w','f444w','f140m','f162m','f182m','f210m','f250m','f300m','f335m','f360m','f410m','f430m',
        'f460m','f480m','lsstu','lsstg','lsstr','lssti','lsstz','lssty','keplercam::us','keplercam::b','keplercam::v','keplercam::v',
        'keplercam::r','keplercam::i','4shooter2::us','4shooter2::b','4shooter2::v','4shooter2::r','4shooter2::i','f062','f087',
        'f106','f129','f158','f184','f213','f146','ztfg','ztfr','ztfi','uvot::b','uvot::u','uvot::uvm2','uvot::uvw1','uvot::uvw2',
        'uvot::v','uvot::white','ps1::open','ps1::g','ps1::r','ps1::i','ps1::z','ps1::y','ps1::w','atlasc','atlaso','2massJ',
        '2massH','2massKs','wfst_u','wfst_g','wfst_r','wfst_i','wfst_z','wfst_w'
    ]
    for filt in filt_list:
        try:
            _x = sncosmo.get_bandpass(filt)
        except:
            print('Fail for '+filt)
            if all:
                all = False
    if all:
        print('Load all filters successfully!')
    return filt_list

def load_wfst_bands():
    add_bands = ['u','g','r','i','w','z']
    wfst_bands = ['wfst_'+i for i in add_bands]
    try:
        for add_band in add_bands:
            data = np.loadtxt('/transmission/WFST_WFST.'+add_band+'_AB.dat')
            wavelength = data[:,0]
            trans = data[:,1]
            band = sncosmo.Bandpass(wavelength, trans, name='wfst_'+add_band)
            sncosmo.register(band, 'wfst_'+add_band)
    except:
        pass
    return wfst_bands

def mab2flux(mab):
    #erg s^-1 cm^-2
    return 10**(-(mab+48.6)/2.5)

def flux2mab(f):
    #erg s^-1 cm^-2
    return -2.5*np.log10(f)-48.6

def sumab(mab_list):
    _flux_all = 0
    for mab in mab_list:
        _flux_all += mab2flux(mab)
    return flux2mab(_flux_all)

def lim_mag(survey_file):
    default_maglim = {
        'g':23.35,
        'r':22.95,
        'i':22.59
    }
    bands_index = {'g':1,'r':2,'i':3}
    survey_file['maglim'] = [getlim(int(survey_file['exposure_time'].iloc[i]),bgsky=22.0,n_frame=1,airmass=survey_file['airmass'].iloc[i],sig=5)[0][bands_index[survey_file['filt'].iloc[i]]] for i in range(len(survey_file.index))]
    #survey_file['maglim'] = [default_maglim[survey_file.loc[i,'filt']]+1.25*np.log10(survey_file.loc[i,'exposure_time']/30) for i in range(len(survey_file.index))]
    return survey_file

def survey2plan(survey_dir,fields_dir,save_dir=None,GW_trigger=None):
    wfst_survey = pd.read_csv(survey_dir)
    fields_file = np.loadtxt(fields_dir)
    if GW_trigger:
        mjd_strat = GW_trigger
    else:
        mjd_start = wfst_survey['observ_time'].min()-0.1

    wfst_fields = dict()
    wfst_fields['field_id'] = fields_file[:,0].astype(int)
    wfst_fields['ra'] = fields_file[:,1]
    wfst_fields['dec'] = fields_file[:,2]

    wfst_survey['band'] = ['wfst_'+wfst_survey.loc[i,'filt'] for i in range(len(wfst_survey['filt'].index))]
    if 'maglim' in list(wfst_survey.columns):
        pass
    else:
        wfst_survey = lim_mag(wfst_survey)
    wfst_survey['time'] = wfst_survey['observ_time']
    wfst_survey['field'] = wfst_survey['field_id']
    wfst_survey['phase'] = wfst_survey['time']-mjd_start
    wfst_survey = wfst_survey.loc[:,['time','field','band','maglim','phase']]
    if save_dir:
        with open(save_dir,'wb') as handle:
            pickle.dump(wfst_survey,handle)
            handle.close()
    return wfst_survey

def event_mjd(event_id):
    'event_id[str]: ID should be conicide with IDs in GraceDb and return the trigger time in MJD'
    url = 'https://gracedb.ligo.org/superevents/{}/view/#event-information'.format(event_id)
    response = requests.get(url)
    soup = BeautifulSoup(response.text,'html.parser')
    html_str = str(soup.find_all('time')[1])
    soupt = BeautifulSoup(html_str,'html.parser')
    return Time(soupt.time['utc'][:-4],format='iso',scale='utc').mjd

class TimeSeriesSource(sncosmo.Source):
    """A single-component spectral time series model.
    The spectral flux density of this model is given by
    .. math::
       F(t, \\lambda) = A \\times M(t, \\lambda)
    where _M_ is the flux defined on a grid in phase and wavelength
    and _A_ (amplitude) is the single free parameter of the model. The
    amplitude _A_ is a simple unitless scaling factor applied to
    whatever flux values are used to initialize the
    ``TimeSeriesSource``. Therefore, the _A_ parameter has no
    intrinsic meaning. It can only be interpreted in conjunction with
    the model values. Thus, it is meaningless to compare the _A_
    parameter between two different ``TimeSeriesSource`` instances with
    different model data.
    Parameters
    ----------
    phase : `~numpy.ndarray`
        Phases in days.
    wave : `~numpy.ndarray`
        Wavelengths in Angstroms.
    flux : `~numpy.ndarray`
        Model spectral flux density in erg / s / cm^2 / Angstrom.
        Must have shape ``(num_phases, num_wave)``.
    zero_before : bool, optional
        If True, flux at phases before minimum phase will be zeroed. The
        default is False, in which case the flux at such phases will be equal
        to the flux at the minimum phase (``flux[0, :]`` in the input array).
    time_spline_degree : int, optional
        Degree of the spline used for interpolation in the time (phase)
        direction. By default this is set to 3 (i.e. cubic spline). For models
        that are defined with sparse time grids this can lead to large
        interpolation uncertainties and negative fluxes. If this is a problem,
        set time_spline_degree to 1 to use linear interpolation instead.
    name : str, optional
        Name of the model. Default is `None`.
    version : str, optional
        Version of the model. Default is `None`.
    """

    _param_names = ['amplitude']
    param_names_latex = ['A']

    def __init__(self, phase, wave, flux, zero_before=False,
                 time_spline_degree=3, name=None, version=None):
        self.name = name
        self.version = version
        self._phase = phase
        self._wave = wave
        self._parameters = np.array([1.])
        self._model_flux = Spline2d(phase, wave, flux, kx=time_spline_degree,
                                    ky=3)
        self._zero_before = zero_before

    def _flux(self, phase, wave):
        f = self._parameters[0] * self._model_flux(phase, wave)
        if self._zero_before:
            mask = np.atleast_1d(phase) < self.minphase()
            f[mask, :] = 0.
        return f





# AngularTimeSeriesSource classdefined to create an angle dependent time serie source.
class AngularTimeSeriesSource(sncosmo.Source):
    """A single-component spectral time series model.
        The spectral flux density of this model is given by
        .. math::
        F(t, \lambda) = A \\times M(t, \lambda)
        where _M_ is the flux defined on a grid in phase and wavelength
        and _A_ (amplitude) is the single free parameter of the model. The
        amplitude _A_ is a simple unitless scaling factor applied to
        whatever flux values are used to initialize the
        ``TimeSeriesSource``. Therefore, the _A_ parameter has no
        intrinsic meaning. It can only be interpreted in conjunction with
        the model values. Thus, it is meaningless to compare the _A_
        parameter between two different ``TimeSeriesSource`` instances with
        different model data.
        Parameters
        ----------
        phase : `~numpy.ndarray`
        Phases in days.
        wave : `~numpy.ndarray`
        Wavelengths in Angstroms.
        cos_theta: `~numpy.ndarray`
        Cosine of
        flux : `~numpy.ndarray`
        Model spectral flux density in erg / s / cm^2 / Angstrom.
        Must have shape ``(num_phases, num_wave, num_cos_theta)``.
        zero_before : bool, optional
        If True, flux at phases before minimum phase will be zeroed. The
        default is False, in which case the flux at such phases will be equal
        to the flux at the minimum phase (``flux[0, :]`` in the input array).
        name : str, optional
        Name of the model. Default is `None`.
        version : str, optional
        Version of the model. Default is `None`.
        """

    _param_names = ['amplitude', 'theta']
    param_names_latex = ['A', r'\theta']

    def __init__(self, phase, wave, cos_theta, flux, zero_before=True, zero_after=True, name=None,
                 version=None):
        self.name = name
        self.version = version
        self._phase = phase
        self._wave = wave
        self._cos_theta = cos_theta
        self._flux_array = flux
        self._parameters = np.array([1., 0.])
        self._current_theta = 0.
        self._zero_before = zero_before
        self._zero_after = zero_after
        self._set_theta()

    def _set_theta(self):
        logflux_ = np.zeros(self._flux_array.shape[:2])
        for k in range(len(self._phase)):
            adding = 1e-10 # Here we are adding 1e-10 to avoid problems with null values
            f_tmp = Spline2d(self._wave, self._cos_theta, np.log(self._flux_array[k]+adding),
                             kx=1, ky=1)
            logflux_[k] = f_tmp(self._wave, np.cos(self._parameters[1]*np.pi/180)).T

        self._model_flux = Spline2d(self._phase, self._wave, logflux_, kx=1, ky=1)

        self._current_theta = self._parameters[1]

    def _flux(self, phase, wave):
        if self._current_theta != self._parameters[1]:
            self._set_theta()
        f = self._parameters[0] * (np.exp(self._model_flux(phase, wave)))
        if self._zero_before:
            mask = np.atleast_1d(phase) < self.minphase()
            f[mask, :] = 0.
        if self._zero_after:
            mask = np.atleast_1d(phase) > self.maxphase()
            f[mask, :] = 0.
        return f


#==============================================================================#
bands_lam = {'ztfg':4783,'ztfr':6417,'ztfi':7867,
             'wfst_u':3641.35,
             'wfst_g':4691.74,
             'wfst_r':6158.74,
             'wfst_i':7435.86,
             'wfst_z':8562.26}





#--------------------------------------#
#               Constant               #
#--------------------------------------#

day2sec = u.day.cgs.scale
MPC_CGS = u.Mpc.cgs.scale
C_CGS = c.c.cgs.value
M_SUN_CGS = c.M_sun.cgs.value
G_CGS = c.G.cgs.value
Jy = u.Jy.cgs.scale
ANG_CGS = u.Angstrom.cgs.scale
pi = np.pi
pc10 = 10 * u.pc.cgs.scale
SB_CGS = c.sigma_sb.cgs.value
H_CGS = c.h.cgs.value
K_B_CGS = c.k_B.cgs.value
KM_CGS = u.km.cgs.scale