'''
ways to fit gas-phase metallicity
'''
import numpy as np
import matplotlib.pyplot as plt
import theano

from astropy import units, constants
from astropy.cosmology import WMAP9 as cosmo
from astropy import nddata
from astropy import table as t

import extinction

from importer import *
import manga_tools as m
import bpt

import pymc3

pi_grid_defaults = {
    'logOH12': 8.69,
    'xid': .2,
    'logU': -2.,
    'logsfr': -10.,
    'logdens': 2.
}

def find_ism_params(grid, dustlaw, line_obs, line_ls, drpall_row):
    '''
    run a pymc3 grid on a whole galaxy

    - grid_covs, grid_alphas: yields from pre-GP-trained photoionization grid
    - dustlaw:
    - line_obs: tuple of flux, uncertainty, and mask
    - line_ls:
    - drpall_row:
    '''

    zdist = drpall_row['nsa_zdist']
    four_pi_r2 = (4. * np.pi * cosmo.luminosity_distance(zdist)**2.).to(units.cm**2).value

    # get observations
    f, unc, _ = line_obs
    snr_order = np.argsort((f / unc).sum(axis=1))[::-1]
    f, unc = f[snr_order], unc[snr_order]
    f, unc = f[:2], unc[:2]
    print(f / unc)

    *obs_shape_, nlines = f.shape
    obs_shape = tuple(obs_shape_)
    print('in galaxy: {} measurements of {} lines'.format(obs_shape, nlines))

    with pymc3.Model() as model:

        # priors
        ## first on photoionization model
        logZ = pymc3.Uniform('logZ', *grid.range('logZ'), shape=obs_shape)
        logU = pymc3.Uniform('logU', *grid.range('logU'), shape=obs_shape)
        age = pymc3.Uniform('age', *grid.range('Age'), shape=obs_shape)

        grid_params = theano.tensor.stack([logZ, logU, age], axis=0)

        # next on normalization of emission line strengths
        logQH = pymc3.Normal('logQH', mu=49., sd=3., shape=obs_shape, testval=49.)
        linelumsperqh = grid.predictt(grid_params)
        linelums = linelumsperqh * 10**logQH[:, None]

        ## next on dust model
        extinction_at_AV1 = theano.shared(  #  shape (nlines, )
            dustlaw(wave=line_ls, a_v=1., r_v=3.1))
        AV = pymc3.Exponential(  #  shape (*obs_shape, )
            'AV', 3., shape=obs_shape, testval=1.)  #  extinction in V-band
        twopointfive = theano.shared(2.5)
        A_lambda = theano.tensor.outer(AV, extinction_at_AV1)
        atten = 10**(-A_lambda / twopointfive)

        # dim lines based on distance
        distmod = theano.shared(four_pi_r2)
        one_e_minus17 = theano.shared(1.0e-17)
        linefluxes = pymc3.Deterministic(
            'linefluxes', linelums * atten / distmod / one_e_minus17)

        #ln_unc_underestimate_factor = pymc3.Uniform(
        #    'ln-unc-underestimate', -10., 10., testval=0.)
        linefluxes_obs = pymc3.Normal(
            'fluxes-obs', mu=linefluxes,
            sd=unc, # * theano.tensor.exp(ln_unc_underestimate_factor),
            observed=f)

        trace = pymc3.sample(draws=500, tune=500, cores=1, chains=1)

    return model, trace


class ObservedEmissionLines(object):
    '''
    container that holds emission-line observations
    '''
    def __init__(self, names, flux, flux_unc, channelmasks, allmasks, snr, elines_table,
                 **kwargs):
        self.names = names
        self.flux = flux
        self.flux_unc = flux_unc
        self.cov = np.stack([np.diag(unc**2.) for unc in flux_unc])
        self.channelmasks = channelmasks
        self.allmasks = allmasks
        self.snr = snr

        self.shape = (len(self.flux), ) + self.flux[0].shape

        self.snr_threshs = np.array([3., 3., 3., 2., 1., 2., 1., 1., 1.])

        self.props = kwargs

    def goodgalaxy(self, goodfrac_th=.6):
        nspax = (self.snr > 0.).sum()

        ngood = np.all(np.stack(
                (self.snr > 0., ) + tuple(~m for m in self.allmasks.values())),
            axis=0).sum()

        return (ngood / nspax) > goodfrac_th

    @property
    def spax_good(self):
        good = np.all(np.stack(
                (self.snr > 0., ) + tuple(~m for m in self.allmasks.values())),
            axis=0)

        return good

    def get_good_obs(self):
        f = np.column_stack([ch[self.spax_good] for ch in self.flux])
        unc = np.column_stack([ch[self.spax_good] for ch in self.flux_unc])
        mask = ~np.column_stack([ch[self.spax_good] for ch in self.channelmasks])
        line_snr = (f / unc) * mask

        lines_good = np.all(line_snr > self.snr_threshs, axis=1)

        f, unc, mask = f[lines_good], unc[lines_good], mask[lines_good]

        return f, unc, mask

    @classmethod
    def from_DAP_MAPS(cls, plate, ifu, kind, mpl_v, elines_table, lines_used,
                      maskbits=[30], Ha_EW_t=10.):
        # load DAP data
        dap = m.load_dap_maps(plate, ifu, kind=kind, mpl_v=mpl_v)
        dap_fluxes = dap['EMLINE_GFLUX']
        dap_fluxes_ivar = dap[dap_fluxes.header['ERRDATA']]
        dap_fluxes_mask = dap[dap_fluxes.header['QUALDATA']]
        # figure out in which channel each emission line name lives
        key2channel = m.make_key2channel(
            dap_fluxes, axis=0, start=1, channel_key_start='C')

        # load table that helps us translate
        elines_table = elines_table.copy(copy_data=True)
        elines_table.add_index('name')
        mpl_v_colname = '{}-name'.format(mpl_v)

        # retriever object
        retriever = ChannelByNameRetriever(elines_table, key2channel, mpl_v_colname)

        # get flux, flux stdev, and flux mask for all lines
        flux = [retriever(dap_fluxes, name) for name in lines_used]
        flux_std = [1. / np.sqrt(retriever(dap_fluxes_ivar, name)) for name in lines_used]
        flux_mask = [m.mask_from_maskbits(
                         retriever(dap_fluxes_mask, name), maskbits) for name in lines_used]

        # mask based on Ha EW
        is_dig = retriever(dap['EMLINE_SEW'], 'H-alpha') < Ha_EW_t

        # mask based on Kauffmann+03 (NII/Ha) / (OIII/Hb)
        not_sf_dom = bpt.KaKe.Ka03_NII().classify(
            forbidden=retriever(dap_fluxes, '[NII]-6584'),
            Ha=retriever(dap_fluxes, 'H-alpha'),
            Oiii=retriever(dap_fluxes, '[OIII]-5007'),
            Hb=retriever(dap_fluxes, 'H-beta'))

        snr = dap['BIN_SNR'].data

        dap.close()

        return cls(names=lines_used, flux=flux, flux_unc=flux_std, channelmasks=flux_mask,
                   allmasks={'is_dig': is_dig, 'not_sf_dom': not_sf_dom}, snr=snr,
                   elines_table=elines_table)

    @classmethod
    def from_Pipe3D(cls, plate, ifu, mpl_v, names):
        raise NotImplementedError

class ChannelByNameRetriever(object):
    def __init__(self, elines_table, key2channel, data_colname):
        self.elines_table = elines_table
        self.key2channel = key2channel
        self.data_colname = data_colname

    def __call__(self, hdu, common_linename):
        data_linename = self.elines_table.loc[common_linename][self.data_colname]
        return hdu.data[self.key2channel[data_linename], ...]

