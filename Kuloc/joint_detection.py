"""
Aujust
aujust@mail.ustc.edu.cn
"""
from .utils import *
from tqdm import tqdm


def POSSIS_NSBH_parameters():
    M_dyn_list = np.linspace(0.01,0.09,20)
    M_pm_list = np.linspace(0.01,0.09,20)
    M_dyn_, M_pm_ = np.meshgrid(M_dyn_list,M_pm_list)

    M_dyn_flat = M_dyn_.flatten()
    M_pm_flat = M_pm_.flatten()
    param_flat = np.array([M_dyn_flat,M_pm_flat]).T
    return param_flat

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

def load_skymap(skymap_dir):
    prob, distmu,distsigma,distnorm = hp.fitsfunc.read_map(skymap_dir,field=[0,1,2,3])
    skymap = {
                'prob':prob,
                'distmu':distmu,
                'distsigma':distsigma
                }
    return skymap

def cal_lcs(param_):
    prefix_list = ['{:3f}'.format(i) for i in param_[:]]
    prefix_name = '_'.join(prefix_list)
    dir = tr_dir + prefix_name + '.pkl'
    out = Possismodel_NSBH_tf(param_,model=spec_model,phase=np.linspace(0,7,100))  #Ejecta Model
    phase, wave, cos_theta, flux = out
    source = AngularTimeSeriesSource(phase, wave, cos_theta, flux)
    model = sncosmo.Model(source=source,effects=[sncosmo.CCM89Dust(), sncosmo.CCM89Dust()], effect_names=['host', 'MW'], effect_frames=['rest', 'obs'])
    transientprop = dict(lcmodel=model, lcsimul_func=random_parameters_ang)
    tr = simsurvey.get_transient_generator(
        [0,0.1],
        ratefunc=lambda z: 5e-1*(1+z),
        sfd98_dir=sfd98_dir,
        load=dir,
        transientprop=transientprop)
    survey = simsurvey.SimulSurvey(generator=tr, plan=plan, n_det=1, threshold=5., sourcenoise=True)
    lcs = survey.get_lightcurves(progress_bar=True)
    print(prefix_name)
    
    return lcs

if __name__ == '__main__':
    event_name = 'S240422ed'
    telescopes = ['ZTF','DECam'] # ZTF/DECam
    group = 'bayestar' # LALInference / bayestar
    svd_path = '/home/Aujust/data/Kilonova/GPR/NN/'
    model_name = 'Bulla_bhns_spectra'
    model_type = 'tensorflow'
    spec_model = knif.model.KilonovaGRB(model_type=model_type,model_dir=svd_path,model_name=model_name)
    tr_dir = '/home/Aujust/data/Kilonova/Constraint/tr/{}_{}/'.format(event_name,model_name)
    plan_root = '/home/Aujust/data/Kilonova/Constraint/plans/'
    skymap_dir = '/home/Aujust/data/Kilonova/Constraint/Skymaps/NSBH/{}_{}.fits'.format(group,event_name)
    npool = 50
    n = 3000
    
    param_flat = np.array(POSSIS_NSBH_parameters())
    skymap = load_skymap(skymap_dir)
    detection_stat = {'_'.join(['{:3f}'.format(m) for m in param_flat[i,:]]):pd.DataFrame(dict(detection=[0 for i in range(n)]),index=[k for k in range(n)]) for i in range(param_flat.shape[0])}
    for key in list(detection_stat.keys()):
        detection_stat[key].attrs['telescopes'] = []
    
    
    for telescope in telescopes:
        print('Calculating effcymap for {} survey.'.format(telescope))
        plan = load_plan(plan_root+'{}_{}_plan.pkl'.format(event_name,telescope))
        with mp.Pool(npool) as p:
            result= p.map(cal_lcs,param_flat)
            p.close()
            p.join()
        
        print('Stastitics for {} survey.'.format(telescope))
        with tqdm(total=param_flat.shape[0],desc='Manimupated parameters: ') as pbar:
            for i in range(param_flat.shape[0]):
                param_ = param_flat[i,:]
                prefix_list = ['{:3f}'.format(i) for i in param_[:]]
                prefix_name = '_'.join(prefix_list)
                lcs = result[i]
                
                detection_df = detection_stat[prefix_name]
                try:
                    for i in range(len(lcs.meta)):
                        idx = lcs[i].meta['idx_orig']
                        detection_df.iloc[idx] += len(lcs[i]['flux'])
                        
                    detection_df.attrs['telescopes'].append(telescope)
                    detection_stat[prefix_name] = detection_df
                except:
                    continue
                pbar.update(1)
            
            
            
        # with tqdm(total=param_flat.shape[0],desc='Manimupated parameters: ') as pbar:
        #     for i in range(param_flat.shape[0]):
        #         prefix_list = ['{:3f}'.format(i) for i in param_flat[i,:]]
        #         prefix_name = '_'.join(prefix_list)
        #         lcs = cal_lcs(tr_dir+prefix_name+'.pkl')
        #         detection_df = detection_stat[prefix_name]
        #         if len(lcs.meta) > 0:
        #             for i in range(len(lcs.meta)):
        #                 idx = lcs[i].meta['idx_orig']
        #                 detection_df.iloc[idx] += len(lcs[i]['flux'])
        #             detection_stat[prefix_name] = detection_df
        #         else:
        #             continue
        #         pbar.update(1)
                
    print(detection_stat)
    with open('./data/test_stat_df.pkl','wb') as handle:
        pickle.dump(detection_stat,handle)
        handle.close()
        
