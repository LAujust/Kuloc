import numpy as np
import Kuloc
import sys
sys.path.append('/home/Aujust/data/Kilonova/KN_Inference/')
import KN_Inference as knif

svd_path = '/home/Aujust/data/Kilonova/GPR/NN/'
model_name = 'Bulla_bhns_spectra'
model_type = 'tensorflow'
spec_model = knif.model.Kilonova(model_name=model_name,model_type=model_type,model_dir=svd_path)
Z = {
    'event_name':'S240422ed',
    'group':'bayestar',
    'spec_model':spec_model,
    'parameters':Kuloc.parameters.POSSIS_NSBH_parameters()
}
Tr_Gen = Kuloc.tr_generator.Tr_Generator(**Z)

print(Tr_Gen.store_setting())

print("Done")