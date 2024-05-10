'''
Requesting Pointings Data from open-source program TreasureMap
'''
import requests
import json
import pandas as pd
import numpy as np
import simsurvey
from astropy.time import Time
import re

pattern = r"[-+]?(?:\d*\.*\d+)"
BASE = 'https://treasuremap.space/api/v1'
api_token = "rUjQzTsfAv1W18i79Zch8Ah9cm3qNvzbkraOoQ"
instru_filt_map = {
    12:{'b':'uvot::','u':'uvot::','v':'uvot::','uvm1':'uvot::','uvm2':'uvot::','uvw2':'uvot::','white':'uvot::','height':0.283,'width':0.283}, #Swift
    38:{'r':'des','z':'des','height':1.6,'width':1.8}, #DECam
    47:{'g':'ztf','r':'ztf','i':'ztf','height':7.465,'width':9.295}, #ZTF
    68:{'g':'sdss','V':'keplercam::','i':'csp','r':'csp','u':'csp','height':0.487,'width':0.485},
    78:{'q':'sdss','u':'sdss','i':'sdss','height':1.75,'width':1.75},
    79:{'q':'lsst','u':'sdss','i':'sdss','height':1.75,'width':1.75},
    71:{'g':'sdss','r':'sdss','B':'keplercam::','V':'keplercam::','height':4.9,'width':3.7}, #GOTO-prototype
    93:{'q':'sdss','g':'sdss','r':'sdss','B':'keplercam::','V':'keplercam::','height':4.9,'width':3.7} #GOTO-prototype
}


class Pointings_request(object):
    def __init__(self,kwgs,TARGET='pointings'):
        self.graceid = kwgs['graceid']
        self.instrument = kwgs.get('instrument',None)
        if self.instrument is None:
            self.instruments = kwgs.get('instruments',None)
            params = {
            "api_token":api_token,
            "instruments":self.instruments,
            "graceid":self.graceid,
            "status":"completed"
            }
        else:
            params = {
            "api_token":api_token,
            "instrument":self.instrument,
            "graceid":self.graceid,
            "status":"completed"
            }

        url = "{}/{}".format(BASE, TARGET)
        r = requests.get(url=url, json=params)
        print("There are %s pointings" % len(json.loads(r.text)))

        #print the first
        self.pointings = json.loads(r.text)
        self.pointing_pointings()

    def pd_pointings(self):
        data = {}
        for point in self.pointings:
            for key,value in point.items():
                if key in data:
                    data[key].append(value)
                else:
                    data[key] = [value]
        df = pd.DataFrame(data)
        return df

    def pointing_pointings(self,zp=26):
        df = self.pd_pointings()
        self.df = df
        df['band'] = [instru_filt_map[df['instrumentid'].iloc[i]][df['band'].iloc[i]]+df['band'].iloc[i].lower() for i in range(len(df.index))]
        obs = {'time': [], 'ra': [],'dec': [], 'band': [], 'maglim': [], 'skynoise': [], 'comment': [], 'zp': []} #TreasureMap didn't provide fields
        obs['time'] = Time(list(df['time']), format='isot').mjd
        obs['band'] = df['band'].values
        obs['maglim'] = df['depth'].values
        obs['zp'] = [zp for i in range(len(df.index))]
        obs['skynoise'] = 10**(-0.4 * (np.array(obs['maglim']) - zp)) / 5
        obs['comment'] = ['' for i in range(len(df.index))]
        
        'extract position'
        position = str(df['position'])
        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', position)
        numbers = [float(num) for num in numbers]
        ra, dec = numbers[0], numbers[1]
        obs['ra'] = ra
        obs['dec'] = dec

        pos = []
        for j in range(len(df.index)):
            matches = re.findall(pattern,df['position'].iloc[j])
            pos.append([float(match) for match in matches])
        pos = np.array(pos)
        obs['ra'],obs['dec'] = pos[:,0],pos[:,1]

        plan = simsurvey.SurveyPlan(time=obs['time'],
                                band=obs['band'],
                                ra=obs['ra'],
                                dec=obs['dec'],
                                skynoise=obs['skynoise'],
                                obs_ccd=None,
                                zp=obs['zp'],
                                comment=obs['comment'],
                                height=instru_filt_map[df['instrumentid'].iloc[0]]['height'],
                                width=instru_filt_map[df['instrumentid'].iloc[0]]['width']
                                )
        self.plan = plan


