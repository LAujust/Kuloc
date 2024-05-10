"""
Aujust
aujust@mail.ustc.edu.cn
"""

from .utils import *


class Tr_Generator(object):
    """
    event_name[str]:        GW event name consistent with gracedb
    group[str]:             the GW alert analysis pipline
    spec_model[object]      KN_Inference model
    tr_dir[str]             path direction to save transients
    skymap_dir[str]         path direction to read skymap
    skymap[str]             skymap file
    parameters[str]         parameters as input of spec model
    flux_model[func]        a function to calculate phase, wave and flux (see ref. in simsurvey)
    lcsimul_func[func]      lc simulation function, see simsurvey
    ang_depend[bool]        angular dependence
    """
    
    def __init__(self,**kwargs):
        self.event_name = kwargs.get('event_name',None)
        self.group = kwargs.get('group','bayestar')
        self.spec_model = kwargs.get('spec_model',None)
        self.tr_dir = kwargs.get('tr_dir',os.getcwd())
        self.skymap_dir = kwargs.get('skymap_dir',None)
        self.skymap = kwargs.get('skymap',None)
        self.parameters = kwargs.get('parameters',None)
        self.fluxmodel = kwargs.get('fluxmodel',None)
        self.ang_depend = kwargs.get('ang_depend',True)
        
        self.setup()
        
    def setup(self):
        #load skymap
        if self.skymap is None and self.skymap_dir is not None:
            self.skymap = self.load_skymap()
            
        #fetch trigger time
        self.mjd_start = event_mjd(self.event_name)
        self.mjd_range = (self.mjd_start-0.1,self.mjd_start+0.1)
        
        if self.ang_depend:
            self.sourcemodel = AngularTimeSeriesSource
        else:
            self.sourcemodel = TimeSeriesSource
            
        #test the knif model
        
        'SIMULATION SETTINGS'
        self.n = 3000
        self.zmin, self.zmax = 0,1
        self.ratefunc = lambda z: 5e-1*(1+z)
        
    def load_skymap(self):
        prob, distmu,distsigma,distnorm = hp.fitsfunc.read_map(self.skymap_dir,field=[0,1,2,3])
        skymap = {
                    'prob':prob,
                    'distmu':distmu,
                    'distsigma':distsigma
                    }
        return skymap
    
    def generate_tr(self,n=3000,npool=1,ratefunc=lambda z: 5e-1*(1+z),zrange=None,clean=True):
        self.zmin, self.zmax = zrange[0], zrange[1]
        self.ratefunc = ratefunc
        self.n = n
        self.npool = npool
        with mp.Pool(self.npool) as p:
            result= p.map(self._generate_tr,self.parameters)
            p.close()
            p.join()

        self.tr_pool = copy.deepcopy(result)

        if not os.path.exists(self.tr_dir):
            os.mkdir(self.tr_dir)
        else:
            if clean:
                files = glob.glob(self.tr_dir+'*')
                for f in files:
                    os.remove(f)
                print('Clean tr folder.')
            print('Saveing trs.')
            for i,tr in enumerate(self.tr_pool):
                prefix_list = ['{:3f}'.format(m) for m in self.parameters[i,:]]
                prefix_name = '_'.join(prefix_list) + '.pkl'
                tr = self.tr_pool[i]
                tr.save(self.tr_dir+prefix_name)
        print('Done for {}'.format(event_name))
        
        return self.store_setting()
        
    
    def _generate_tr(self,param_):
        out = self.fluxmodel(param_,model=self.spec_model,phase=np.linspace(0,7,100))  #Ejecta Model
        phase, wave, cos_theta, flux = out
        source = self.sourcemodel(phase, wave, cos_theta, flux)
        model = sncosmo.Model(source=source,effects=[sncosmo.CCM89Dust(), sncosmo.CCM89Dust()], effect_names=['host', 'MW'], effect_frames=['rest', 'obs'])
        transientprop = dict(lcmodel=model, lcsimul_func=self.lcsimul_func)
        tr = simsurvey.get_transient_generator([0,0.1],
                                                        ntransient=int(n),
                                                        ratefunc=self.ratefunc,
                                                        sfd98_dir=sfd98_dir,
                                                        transientprop=transientprop,
                                                        mjd_range=self.mjd_range,
                                                        skymap=self.skymap
                                                        )
        print(param_)
        return tr
    
    def store_setting(self):
        SIMULATION_SETTINGS = ['n','ratefunc','zmin','zmax']
        settings = dict()
        for item in SIMULATION_SETTINGS:
            settings[item] = eval('self.'+item)
        
        return settings
        
        



def load_skymap(skymap_dir):
    prob, distmu,distsigma,distnorm = hp.fitsfunc.read_map(skymap_dir,field=[0,1,2,3])
    skymap = {
                'prob':prob,
                'distmu':distmu,
                'distsigma':distsigma
                }
    return skymap




event_name = 'S240422ed'
telescope = 'DECam' # ZTF/DECam
group = 'bayestar' # LALInference / bayestar
svd_path = '/home/Aujust/data/Kilonova/GPR/NN/'
model_name = 'Bulla_bhns_spectra'
model_type = 'tensorflow'
tr_dir = '/home/Aujust/data/Kilonova/Constraint/tr/{}_{}/'.format(event_name,model_name)
plan_dir = '/home/Aujust/data/Kilonova/Constraint/plans/{}_{}_plan.pkl'.format(event_name,telescope)
skymap_dir = '/home/Aujust/data/Kilonova/Constraint/Skymaps/NSBH/{}_{}.fits'.format(group,event_name)
npool = 50
n = 3000
clean = True
    
#plan = load_plan(plan_dir=plan_dir)
# skymap = load_skymap(skymap_dir=skymap_dir)
# param_flat = POSSIS_NSBH_parameters()
# spec_model = knif.model.KilonovaGRB(model_type=model_type,model_dir=svd_path,model_name=model_name)
# print('Generating Transients File with {} processes.'.format(npool))
# mjd_start = event_mjd(event_name)
# mjd_range = (mjd_start-0.1,mjd_start+0.1)
    
# with mp.Pool(npool) as p:
#         #result = p.map(cal_effcy_p,transientpprop_list)
#     result= p.map(generate_tr_,param_flat)
#     p.close()
#     p.join()

# tr_pool = copy.deepcopy(result)

# if not os.path.exists(tr_dir):
#     os.mkdir(tr_dir)
# else:
#     if clean:
#         files = glob.glob(tr_dir+'*')
#         for f in files:
#             os.remove(f)
#         print('Clean tr folder.')
#     print('Saveing trs.')
#     for i,tr in enumerate(tr_pool):
#         prefix_list = ['{:3f}'.format(m) for m in param_flat[i,:]]
#         prefix_name = '_'.join(prefix_list) + '.pkl'
#         tr = tr_pool[i]
#         tr.save(tr_dir+prefix_name)
        
     
#print('Done for {}'.format(event_name))