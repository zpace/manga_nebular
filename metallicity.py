'''
ways to fit gas-phase metallicity
'''
import numpy as np
import matplotlib.pyplot as plt
import theano

from astropy import units, constants
from astropy.cosmology import WMAP9 as cosmo
from astropy import nddata
from astropy import units as u, constants as c
from astropy import table as t

import extinction

from importer import *

import manga_tools as m
import manga_elines as mel
import bpt
import gp_grid
from pymc3_tricks import *

import pymc3

def find_ism_params(grid, dustlaw, obs, pca_result, line_ls, drpall_row, Zsol=.0142,
                    nrad=30, m_at_rad=5, rlim=None):
    '''
    run a pymc3 grid on a whole galaxy

    - grid_covs, grid_alphas: yields from pre-GP-trained photoionization grid
    - dustlaw:
    - line_obs: tuple of flux, uncertainty, and mask
    - line_ls:
    - drpall_row:
    '''
    # access results from pca to get priors on tauV*mu and tauV*(1-mu)
    pca_results_good = ~np.logical_or(pca_result.mask, pca_result.badPDF())
    tauV_mu_loc, tauV_mu_sd = pca_result.to_normaldist('tau_V mu')
    tauV_1mmu_loc, tauV_1mmu_sd = pca_result.to_normaldist('tau_V (1 - mu)')
    logQH_loc, logQH_sd = pca_result.to_normaldist('logQH')

    # good spaxels must be good in both PCA results and emlines measurements
    goodspax = np.logical_and(obs.spaxels_good_to_run(), pca_results_good)
    print(goodspax.sum(), 'spaxels')
    # access emission-line measurements, and pick the good ones
    f = np.column_stack([obs.line_flux[k][goodspax] for k in obs.lines_used])
    unc = np.column_stack([obs.line_unc[k].array[goodspax] for k in obs.lines_used])

    # filter PCA measurements of tauV mu and tauV (1 - mu)
    tauV_mu_loc, tauV_mu_sd = \
        tauV_mu_loc[goodspax].astype(np.float32), tauV_mu_sd[goodspax].astype(np.float32)
    tauV_1mmu_loc, tauV_1mmu_sd = \
        tauV_1mmu_loc[goodspax].astype(np.float32), tauV_1mmu_sd[goodspax].astype(np.float32)
    logQH_loc, logQH_sd = \
        logQH_loc[goodspax].astype(np.float32), logQH_sd[goodspax].astype(np.float32)

    # radius in Re units
    Rreff = obs.hdulist['SPX_ELLCOO'].data[1, ...][goodspax].astype(np.float32)

    #'''
    if type(rlim) is list:
        Rtargets = np.linspace(rlim[0], rlim[1], nrad)
    else:
        Rtargets = np.linspace(Rreff.min(), Rreff.max(), nrad)
    meas_ixs = np.unique(
        np.argsort(np.abs(Rreff[None, :] - Rtargets[:, None]), axis=1)[:, :m_at_rad])
    print(meas_ixs)

    Rreff, f, unc = Rreff[meas_ixs], f[meas_ixs], unc[meas_ixs]
    tauV_mu_loc, tauV_mu_sd = tauV_mu_loc[meas_ixs], tauV_mu_sd[meas_ixs]
    tauV_1mmu_loc, tauV_1mmu_sd = tauV_1mmu_loc[meas_ixs], tauV_1mmu_sd[meas_ixs]
    logQH_loc, logQH_sd = logQH_loc[meas_ixs], logQH_sd[meas_ixs]
    #'''

    # distance, for absolute-scaling purposes
    zdist = drpall_row['nsa_zdist']
    four_pi_r2 = (4. * np.pi * cosmo.luminosity_distance(zdist)**2.).to(units.cm**2).value

    *obs_shape_, nlines = f.shape
    obs_shape = tuple(obs_shape_)
    print('in galaxy: {} measurements of {} lines'.format(obs_shape, nlines))

    with pymc3.Model() as model:
        #'''
        # gaussian process on radius determines logZ
        ls_logZ = pymc3.Gamma('ls-logZ', alpha=3., beta=3., testval=1.) # effectively [0.5, 3] Re
        gp_eta = pymc3.HalfCauchy('eta', beta=.5, testval=.25)
        cov_r = gp_eta**2. * pymc3.gp.cov.ExpQuad(input_dim=1, ls=ls_logZ)
        logZ_gp = pymc3.gp.Latent(cov_func=cov_r)

        # draw from GP
        logZ_rad = logZ_gp.prior('logZ-r', X=Rreff[:, None])
        logZ_gp_rad_sigma = pymc3.HalfCauchy('logZ-rad-sigma', beta=.2)
        logZ = pymc3.Bound(pymc3.Normal, *grid.range('logZ'))(
            'logZ', mu=logZ_rad, sd=logZ_gp_rad_sigma, shape=obs_shape, testval=-.1)
        #'''

        # priors
        ## first on photoionization model
        #logZ = pymc3.Uniform('logZ', *grid.range('logZ'), shape=obs_shape, testval=0.)
        Z = Zsol * 10.**logZ
        logU = pymc3.Bound(pymc3.Normal, *grid.range('logU'))(
            'logU', mu=-2., sd=5., shape=obs_shape, testval=-2.)
        age = pymc3.Bound(pymc3.Normal, *grid.range('Age'))(
            'age', mu=5., sd=10., shape=obs_shape, testval=2.5)
        #xid = theano.shared(0.46)

        # dust laws come from PCA fits
        tauV_mu_norm = pymc3.Bound(pymc3.Normal, lower=-tauV_mu_loc / tauV_mu_sd)(
            'tauV mu norm', mu=0, sd=1., shape=obs_shape, testval=0.)
        tauV_mu = pymc3.Deterministic(
            'tauV mu', tauV_mu_loc + tauV_mu_sd * tauV_mu_norm)
        tauV_1mmu_norm = pymc3.Bound(pymc3.Normal, lower=-tauV_1mmu_loc / tauV_1mmu_sd)(
            'tauV 1mmu norm', mu=0, sd=1., shape=obs_shape, testval=0.)
        tauV_1mmu = pymc3.Deterministic(
            'tauV 1mmu', tauV_1mmu_loc + tauV_1mmu_sd * tauV_1mmu_norm)

        #tauV = tauV_mu + tauV_1mmu
        #logGMSD = pymc3.Deterministic(
        #    'logGMSD', theano.tensor.log10(0.2 * tauV / (xid * Z)))

        grid_params = theano.tensor.stack([logZ, logU, age], axis=0)

        # the attenuation power-laws
        dense_powerlaw = theano.shared((line_ls.quantity.value.astype('float32') / 5500)**-1.3)
        diffuse_powerlaw = theano.shared((line_ls.quantity.value.astype('float32') / 5500)**-0.7)

        transmission = pymc3.math.exp(
            -(theano.tensor.outer(tauV_1mmu, dense_powerlaw) + \
              theano.tensor.outer(tauV_mu, diffuse_powerlaw)))

        # dim lines based on distance
        distmod = theano.shared(four_pi_r2)
        one_e17 = theano.shared(1.0e17)
        obsnorm = one_e17 / distmod

        # next on normalization of emission line strengths
        logQHnorm = pymc3.Normal(
            'logQHnorm', mu=0., sd=1., testval=0., shape=obs_shape)
        logQH = pymc3.Deterministic(
            'logQH', logQH_loc + logQH_sd * logQHnorm)

        eff_QH = pymc3.Kumaraswamy('effQH', a=3., b=3., shape=obs_shape, testval=0.66)

        linelumnorm = theano.tensor.outer(
            eff_QH * 10**logQH, grid.observable_norms_t.astype('float32'))

        norm = obsnorm * linelumnorm * transmission

        for i, (name, alpha, cov) in enumerate(zip(grid.observable_names, grid.alphas,
                                                   grid.covs)):
            pymc3.StudentT('-'.join(('obsflux', name)), nu=1.,
                           mu=((gp_grid.gp_predictt(
                                    cov, alpha, grid.X0, grid_params) + 1.) * norm[:, i]),
                           sd=unc[:, i], observed=f[:, i])

        model_graph = pymc3.model_to_graphviz()
        model_graph.format = 'svg'
        model_graph.render()

        step, start = densemass_sample(
            model, cores=1, chains=1,
            nstart=200, nburn=200, ntune=5000)

        try:
            nchains = 10
            trace = pymc3.sample(step=step, start=start * nchains,
                                 draws=500, tune=500, burn=500,
                                 cores=1, chains=nchains,
                                 nuts_kwargs=dict(target_accept=.95),
                                 init='adapt_diag')
        except Exception as e:
            print(e)
            trace = None

    return model, trace, f, unc, Rreff

elinekind_to_snrth = {'X-bright': 6., 'X-dim': 4., 'Y': 3., 'Z-bright': 5., 'Z-dim': 2.}

class Elines(mel.MaNGAElines):
    '''
    container that holds and retrieves emission-line observations
    '''

    def __init__(self, hdulist, elines_table, data_colname, lines_used, *args, **kwargs):
        super().__init__(hdulist)

        self.elines_table = elines_table
        self.data_colname = data_colname
        self.lines_used = lines_used

        self.fulldata = {k: self.get_qty(
                                qty='GFLUX',
                                key=self.elines_table.loc[k][data_colname],
                                sn_th=elinekind_to_snrth[self.elines_table.loc[k]['kind']],
                                maskbits=kwargs.get('maskbits', [30]))
                         for k in self.lines_used}

        self.line_flux = {k: self.fulldata[k].data
                          for k in self.lines_used}

        self.line_unc = {k: self.fulldata[k].uncertainty
                         for k in self.lines_used}

        self.line_mask = {k: self.fulldata[k].mask
                          for k in self.lines_used}

        self.line_unit = {k: self.fulldata[k].unit
                          for k in self.lines_used}

    def is_dig_dom(self, Ha_EW_t=5. * u.AA):
        Ha_EW = self.get_qty(qty='SEW', key='Ha-6564', sn_th=0., maskbits=[])
        Ha_EW_q = Ha_EW.data * u.Unit(Ha_EW.unit)
        return Ha_EW_q < Ha_EW_t

    @property
    def not_sf_dom(self):

        # mask based on Kauffmann+03 (NII/Ha) / (OIII/Hb)
        not_sf_dom_Ka03 = bpt.KaKe.Ka03_NII().classify(
            forbidden=self.line_flux['[NII]-6584'],
            Ha=self.line_flux['H-alpha'],
            Oiii=self.line_flux['[OIII]-5007'],
            Hb=self.line_flux['H-beta'])

        not_sf_dom_oi = bpt.KaKe.OI().classify(
            forbidden=self.line_flux['[OI]-6300'],
            Ha=self.line_flux['H-alpha'],
            Oiii=self.line_flux['[OIII]-5007'],
            Hb=self.line_flux['H-beta'])

        not_sf_dom = np.logical_or.reduce((not_sf_dom_Ka03, not_sf_dom_oi))

        return not_sf_dom

    def grid_is_valid(self, Ha_EW_t=5. * u.AA):
        return ~np.logical_or(self.is_dig_dom(Ha_EW_t), self.not_sf_dom)

    @property
    def all_lines_good(self):
        return ~np.logical_or.reduce(tuple(m for m in self.line_mask.values()))

    def spaxels_good_to_run(self, Ha_EW_t=5. * u.AA):
        return self.grid_is_valid(Ha_EW_t) * self.all_lines_good
