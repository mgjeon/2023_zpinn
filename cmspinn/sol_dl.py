# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_solar_data_download.ipynb.

# %% auto 0
__all__ = ['hmi_downloader']

# %% ../nbs/04_solar_data_download.ipynb 2
import os
import drms
from datetime import datetime, timedelta

# %% ../nbs/04_solar_data_download.ipynb 3
class hmi_downloader:
    def __init__(self, jsoc_email):
        jsoc_email = jsoc_email
        self.client = drms.Client(email=jsoc_email, verbose=True)

    def find_harpnum(self, start_time):
        ar_mapping = self.client.query('hmi.Mharp_720s[][%sZ]' % start_time.isoformat('_', timespec='seconds'),
                                       key=['NOAA_AR', 'HARPNUM'])
        return ar_mapping
    
    def setup_download_dir(self, download_dir, noaa_num, harp_num):
        self.download_dir = os.path.join(download_dir, f'NOAA{noaa_num}_HARP{harp_num}')
        self.harp_num = harp_num
        os.makedirs(self.download_dir, exist_ok=True)

    def setup_time_interval(self, start_time, duration):
        self.start_time = start_time
        self.duration = duration
        duration_hour = eval(duration.replace('h', ''))
        self.end_time = start_time + timedelta(hours=duration_hour)
    
    def download_hmi(self, segments = 'Br, Bp, Bt', series='sharp_cea_720s'):

        ds = 'hmi.%s[%d][%s/%s]{%s}' % \
                (series, self.harp_num, self.start_time.isoformat('_', timespec='seconds'), self.duration, segments)

        hmi_dir = os.path.join(self.download_dir, 'hmi')
        os.makedirs(hmi_dir, exist_ok=True)

        r = self.client.export(ds, protocol='fits')
        r.wait()
        download_result = r.download(hmi_dir)
        return download_result
    
    def download_aia(self, wavelength='171'):
        ds = f'aia.lev1_euv_12s[{self.start_time.isoformat("_", timespec="seconds")} \
            / {(self.end_time - self.start_time).total_seconds()}s@60s][{wavelength}]{{image}}'
        
        aia_dir = os.path.join(self.download_dir, f'aia/{wavelength}')
        os.makedirs(aia_dir, exist_ok=True)

        r = self.client.export(ds, protocol='fits')
        r.wait()
        download_result = r.download(aia_dir)
        return download_result    
