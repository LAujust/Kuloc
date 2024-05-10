"""
Aujust
aujust@mail.ustc.edu.cn
"""
from .utils import *

def POSSIS_NSBH_parameters():
    M_dyn_list = np.linspace(0.01,0.09,20)
    M_pm_list = np.linspace(0.01,0.09,20)
    M_dyn_, M_pm_ = np.meshgrid(M_dyn_list,M_pm_list)

    M_dyn_flat = M_dyn_.flatten()
    M_pm_flat = M_pm_.flatten()
    param_flat = np.array([M_dyn_flat,M_pm_flat]).T
    return param_flat

def get_effcy(param_,ndet=2,n=3000):
    prefix_list = ['{:3f}'.format(i) for i in param_[:]]
    prefix_name = '_'.join(prefix_list)
    df = detection_stat[prefix_name]   
    effcy = df[df['detection']>=ndet].count().to_numpy()[0]/n
    return effcy

if __name__ == '__main__':
    n = 3000
    param_flat = POSSIS_NSBH_parameters()
    with open('./data/test_stat_df.pkl','rb') as handle:
        detection_stat = pickle.load(handle)
        handle.close()
        
    i = 3
    print(param_flat[i,:],get_effcy(param_flat[i,:],ndet=2,n=n))