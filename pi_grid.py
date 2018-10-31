import numpy as np
import scipy.interpolate as spint
import theano

from ruamel.yaml import YAML, RoundTripLoader
from pathlib import Path

import abundances

from astropy import units
from astropy import table as t

import gp_grid

elines_table = t.Table.read('./data/elines.dat', format='ascii.ecsv')

default_lines = ['H-alpha', 'H-beta', 'H-gamma', '[OIII]-5007', '[OII]-3726', '[NII]-6584',
                 '[OI]-6300', '[SII]-6717', '[SII]-6731']

class PhotoionizationGrid(object):
    '''
    generic photoioinization grid object, applicable to any number of axes
    '''

    def __init__(self, param_names, observable_names, observable_predictions,
                 abund, grid_yaml_cfg):

        '''
        Parameters
        ----------
        param_names : list (`len` n)
            strings which define parameter ("axes") of grid
        observable_names : list (`len` m)
            strings which define observables of grid
        observable_predictions : list (`len` m) of arrays
            arrays which define the predictions of each of `observable_names`.
            each array in `observable_predictions` must have `shape` compatible
            with the total grid shape
        '''
        self.param_names = param_names
        self.nparams = len(self.param_names)
        self.grid_yaml_cfg = grid_yaml_cfg
        self.observable_names = observable_names
        self.observable_norms = np.array(list(map(np.mean, observable_predictions)))
        self.observable_norms_t = theano.shared(self.observable_norms)
        self.observable_predictions = observable_predictions
        self.norm_pred = [p / n - 1. for p, n in zip(observable_predictions,
                                                     self.observable_norms)]

        self.parameter_sparsegrids = tuple(
            np.array(self.grid_yaml_cfg['gridprops'][name]['vals'])
            for name in param_names)

        self.param_label = [self.grid_yaml_cfg['gridprops'][p]['dist']['TeX']
                            for p in self.param_names]

    def range(self, paramname):
        gridvals = np.array(self.grid_yaml_cfg['gridprops'][paramname]['vals'])
        return (gridvals.min(), gridvals.max())

    def predict(self, X1):
        pass

    def predictt(self, X1):
        pass

    def __str__(self):
        s = '\n'.join(['{} ({}): {}'.format(n, len(vals), vals)
                      for n, vals in zip(
                          self.param_names, self.parameter_sparsegrids)])
        return s

class GPPhotoionizationGrid(PhotoionizationGrid):
    def learnspace_GP(self):
        self.X0, self.covs, self.alphas = gp_grid.gp_learn_emline_space(
            self.parameter_sparsegrids, self.norm_pred)

    def predict(self, X1):
        results_cols = tuple(gp_grid.gp_predict(cov, alpha, self.X0, X1)
                             for cov, alpha in zip(self.covs, self.alphas))
        return (np.column_stack(results_cols) + 1) * self.observable_norms

    def predictt(self, X1):
        results_cols = tuple(gp_grid.gp_predictt(cov, alpha, self.X0, X1)
                             for cov, alpha in zip(self.covs, self.alphas))
        return ((theano.tensor.stack(results_cols, axis=1) + 1) * self.observable_norms_t)

class LinInterpPhotoionizationGrid(PhotoionizationGrid):
    def learnspace_interp(self):
        self.interpolator = spint.RegularGridInterpolator(
            points=tuple(self.parameter_sparsegrids),
            values=np.stack(self.observable_predictions, axis=-1), method='linear')

    def predict(self, X1):
        return self.interpolator(X1)

    def predictt(self, X1):
        # find adjacent nodes
        # assign weights
        #
        pass

def load_CloudyFSPS_grid(linenames_fname, data_fname, yaml_cfg_fname,
                         elines_tab_key='CloudyFSPS-name', elines_table=elines_table,
                         lines_used='all', native_obsunits='Lsun'):

    import astropy.io.ascii as ascii_reader

    # load yaml config file for this photoionization grid
    yaml = YAML(typ='rt')
    yaml_cfg = yaml.load(Path(yaml_cfg_fname))

    # initialize gas-phase abundances
    abund = abundances.AbundanceSet(
        solar_Z=yaml_cfg['abundprops']['Zsol'],
        solar_logOH12=yaml_cfg['abundprops']['logOH12sol'])

    # copy elines table
    elines_table = elines_table.copy(copy_data=True)
    elines_table.add_index('CloudyFSPS-name')

    # read in line names in correct order
    with open(linenames_fname, 'r') as f:
        linenames = [line.rstrip('\n') for line in f.readlines()]

    # read in photoionization grid data
    with open(data_fname, 'r') as f:
        alldata = [line.rstrip('\n') for line in f.readlines()]

    # line 0 defines size of grid
    hdr = alldata[0].lstrip('#').split(' ')
    # how many of each param
    grid_info = [(k, v) for v, k in zip(hdr[::2], hdr[1::2])
                 if k not in ['rows', 'cols']]
    # line wavelengths in AA
    waves = alldata[1].split(' ')
    points_data = alldata[2::2]
    predictions_data = alldata[3::2]

    # eline flux predictions and grid point values alternate
    grid_points = ascii_reader.read(
        points_data, delimiter=' ', format='basic', data_start=0,
        names=[p[0] for p in grid_info])
    grid_predictions = ascii_reader.read(
        predictions_data, delimiter=' ', format='basic', data_start=0,
        names=linenames)

    grid_points = grid_points[list(yaml_cfg['gridprops'].keys())]
    grid_points_names = grid_points.colnames

    # which lines are used? if all, just propagate forward the column names
    # from predictions grid
    if lines_used == 'all':
        lines_used = grid_predictions.colnames
    else:
        pass

    # transform grid-specific observable name into commonly-used name
    # or eliminate line if not used
    for grid_colname in grid_predictions.colnames:
        # try to find common line name in table. if not present, mark for deletion
        try:
            common_linename = elines_table.loc[grid_colname]['name']
        except:
            line_not_in_master_table = True
            common_linename = ''
        else:
            line_not_in_master_table = False
            #print(grid_colname, common_linename)

        # if line name not included, or if not present: delete
        # otherwise: rename it to common name
        if (common_linename not in lines_used) or line_not_in_master_table:
            del grid_predictions[grid_colname]
        else:
            grid_predictions.rename_column(grid_colname, common_linename)

    #re-order parameter grid-point table so it's easy to collapse into a grid
    grid_row_order = grid_points.argsort(grid_points_names)
    grid_points, grid_predictions = (grid_points[grid_row_order],
                                     grid_predictions[grid_row_order])

    # number of nodes in each parameter grid
    gridshape = tuple(yaml_cfg['gridprops'][name]['n'] for name in grid_points_names)

    # reshape parameter and observable grid
    parameter_sampledgrids = [grid_points[name].data.reshape(gridshape)
                              for name in grid_points_names]
    parameter_sparsegrids = [np.array(yaml_cfg['gridprops'][name]['vals'])
                             for name in grid_points_names]
    observable_names = grid_predictions.colnames
    observable_grids = [((grid_predictions[name].data * \
                          units.Unit(native_obsunits)).to('erg/s').value).reshape(gridshape)
                        for name in observable_names]

    return GPPhotoionizationGrid(
        param_names=grid_points_names, observable_names=observable_names,
        observable_predictions=observable_grids,
        abund=abund, grid_yaml_cfg=yaml_cfg)

