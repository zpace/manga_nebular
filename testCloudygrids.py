from importer import *

import sys, os
sys.path.append('/usr/data/minhas/zpace/stellarmass_pca')
import read_results as from_pca
pca_basedir = '/usr/data/minhas2/zpace/CSPs/CSPs_CKC14_MaNGA_20181026-1'

import numpy as np
import matplotlib.pyplot as plt
import pymc3

import manga_tools as m

import metallicity
import pi_grid
import abundances

from extinction import fitzpatrick99
from astropy.cosmology import WMAP9 as cosmo

cloudy_fsps_grid = pi_grid.load_CloudyFSPS_grid(
    linenames_fname='./data/cloudyFSPS/linenames.dat',
    data_fname='./data/cloudyFSPS/ZAU_ND_mist.lines',
    yaml_cfg_fname='./data/cloudyFSPS/cloudyFSPS.yaml',
    elines_tab_key='CloudyFSPS-name', elines_table=pi_grid.elines_table,
    lines_used=pi_grid.default_lines)

cloudy_fsps_grid.learnspace_GP()

elines = pi_grid.elines_table.copy()
elines.add_index('name')
line_ls = elines.loc[cloudy_fsps_grid.observable_names]['lvac']

ntest = 1
'''
logZ_real = np.random.uniform(*cloudy_fsps_grid.range('logZ'), ntest)
logU_real = np.random.uniform(*cloudy_fsps_grid.range('logU'), ntest)
age_real = np.random.uniform(*cloudy_fsps_grid.range('Age'), ntest)
AV_real = np.random.exponential(1., ntest)
logQH_real = np.random.uniform(48.5, 51., ntest)

linelums_real = 10.**logQH_real[:, None] * cloudy_fsps_grid.predict(
    np.stack([logZ_real, logU_real, age_real], axis=0))
extinction_at_AV1 = fitzpatrick99(wave=line_ls, a_v=1., r_v=3.1)
A_lambda = np.outer(AV_real, extinction_at_AV1)
atten = 10.**(-0.4 * A_lambda)
zdist = .0155
distmod = (4. * np.pi * cosmo.luminosity_distance(zdist)**2.).to('cm2').value
linefluxes_real = linelums_real * atten / distmod / 1.0e-17

snr = np.random.uniform(2., 50., linefluxes_real.shape)
real_unc = linefluxes_real / snr
unc_factor = np.e
linefluxes_noise = real_unc * np.random.randn(*linefluxes_real.shape)
linefluxes_obs = linefluxes_real + linefluxes_noise
obs_unc = real_unc / unc_factor
mask_obs = np.any(linefluxes_obs < 0., axis=1)
print(linefluxes_obs.shape)
'''

'''
fakemodel, faketrace = metallicity.find_ism_params(
    grid=cloudy_fsps_grid, dustlaw=fitzpatrick99,
    line_obs=[linefluxes_obs[~mask_obs], obs_unc[~mask_obs], mask_obs[~mask_obs]],
    line_ls=line_ls, drpall_row={'nsa_zdist': zdist})
'''

#####

drpall = m.load_drpall(metallicity.mpl_v)
drpall.add_index('plateifu')
drpall_row = drpall.loc['9497-9101']
plate, ifu = drpall_row['plateifu'].split('-')

el = metallicity.Elines.DAP_from_plateifu(
    plate, ifu, mpl_v, 'SPX-GAU-MILESHC', data_colname='MPL-6-name',
    lines_used=cloudy_fsps_grid.observable_names, elines_table=elines)

pcares = from_pca.PCAOutput.from_plateifu(
    basedir=os.path.join(pca_basedir, 'results'), plate=plate, ifu=ifu)

#'''
model, trace, f, unc, Rreff = metallicity.find_ism_params(
    grid=cloudy_fsps_grid, dustlaw=fitzpatrick99,
    obs=el, pca_result=pcares, line_ls=line_ls, drpall_row=drpall_row,
    nrad=5, m_at_rad=3, rlim=[0.5, 2.])
#'''

model.profile(model.logpt).summary()
