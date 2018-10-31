from importer import *

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
line_ls = elines.loc[pi_grid.default_lines]['lvac']

drpall = m.load_drpall(metallicity.mpl_v)
drpall.add_index('plateifu')
drpall_row = drpall.loc['8454-12703']
plate, ifu = drpall_row['plateifu'].split('-')

mel = metallicity.ObservedEmissionLines.from_DAP_MAPS(
    plate=plate, ifu=ifu, kind='SPX-GAU-MILESHC', mpl_v=metallicity.mpl_v,
    elines_table=pi_grid.elines_table, lines_used=pi_grid.default_lines,
    maskbits=[30], Ha_EW_t=10.)

obs = mel.get_good_obs()

logZ_real = np.random.uniform(-2.5, 0.5, 100)
logU_real = np.random.uniform(-4., -1., 100)
age_real = np.random.uniform(0.5, 10., 100)
AV_real = np.random.exponential(1., 100)
logQH_real = np.random.uniform(48.5, 50., 100)
linelums_real = 10.**logQH_real * cloudy_fsps_grid.predict(
    np.stack([logZ_real, logU_real, age_real], axis=0)).T
extinction_at_AV1 = fitzpatrick99(wave=line_ls, a_v=1., r_v=3.1)
A_lambda = np.outer(AV_real, extinction_at_AV1)
atten = 10.**(-0.4 * A_lambda)
zdist = .0155
distmod = (4. * np.pi * cosmo.luminosity_distance(zdist)**2.).to('cm2').value
linefluxes_real = linelums_real.T * atten / distmod / 1.0e-17

snr = 2.
real_unc = linefluxes_real / snr
unc_factor = np.e
linefluxes_noise = real_unc * np.random.randn(*linefluxes_real.shape)
linefluxes_obs = linefluxes_real + linefluxes_noise
obs_unc = real_unc / unc_factor

'''
fakemodel, faketrace = metallicity.find_ism_params(
    grid=cloudy_fsps_grid, dustlaw=fitzpatrick99,
    line_obs=[linefluxes_obs, obs_unc, None], line_ls=line_ls, drpall_row={'nsa_zdist': zdist})
'''

#'''
model, trace = metallicity.find_ism_params(
    grid=cloudy_fsps_grid, dustlaw=fitzpatrick99,
    line_obs=obs, line_ls=line_ls, drpall_row=drpall_row)
#'''
