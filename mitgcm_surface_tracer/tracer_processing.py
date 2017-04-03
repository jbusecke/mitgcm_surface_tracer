from __future__ import print_function
# from future.utils import iteritems
import time
import numpy as np
import xarray as xr
import dask.array as da
import xgcm
from xmitgcm import open_mdsdataset
from xarrayutils.utils import aggregate, aggregate_w_nanmean
from xarrayutils.xmitgcm_utils import gradient_sq_amplitude
from xarrayutils.xmitgcm_utils import matching_coords
from xarrayutils.xmitgcm_utils import laplacian
from xarrayutils.build_grids import grid_aggregate
from .utils import readbin, paramReadout, dirCheck


class tracer_engine:
    """Make is easier doing many operations on the same
    grid."""

    def __init__(self, ddir, koc_kappa=63, koc_interval=20,
                 validmaskpath=None, makedirs=False):
        # Predefined inputs
        # Start time of offline velocities [YYYY,MM,DD,HH,MM,SS]
        self.start_time = [1993,    1,    1,    0,    0,    0]
        self.ref_date = '-'.join([str(x) for x in self.start_time[0:3]])
        # simple parsed inputs
        self.ddir = dirCheck(ddir, makedirs)
        # diagnosed small scale diff (k_num in Abernathey et al. 2013)
        self.koc_kappa = koc_kappa
        # The interval for coarsening the KOC data
        self.koc_interval = koc_interval
        self.validmaskpath = validmaskpath

        # Infos from run directory (assumes that data* files are found in ddir)
        self.modelparameters = paramReadout(self.ddir)
        self.tracernum = np.arange(
            int(self.modelparameters['data.ptracers/PTRACERS_numInUse']))+1
        self.dt_model = int(float(self.modelparameters['data/deltaTtracer']))
        self.total_time_model = int(float(
            self.modelparameters['data/nTimeSteps']))*self.dt_model

    def reset_cut_mask(self, iters, tr_num, cut_time):
        total_time = self.total_time_model
        reset_frq = int(self.modelparameters[
                            'data.ptracers/PTRACERS_resetFreq('+str(tr_num)+')'
                            ])
        reset_pha = int(self.modelparameters[
            'data.ptracers/PTRACERS_resetPhase('+str(tr_num)+')'])
        dt_model = self.dt_model

        mask, _, _ = reset_cut(reset_frq, reset_pha,
                               total_time, dt_model,
                               iters, tr_num, cut_time)
        return mask

    def dataset_readin(self, prefix, directory=None, iters='all'):
        if directory is None:
            directory = self.ddir
        ds = open_mdsdataset(directory, prefix=prefix,
                             delta_t=self.dt_model, ref_date=self.ref_date,
                             iters=iters, swap_dims=False)
        return ds

    def KOC(self, tr_num, directory=None, interval=None,
            spin_up_months=3, iters='all', debug=False):

        if interval is None:
            interval = self.koc_interval

        ds_mean = self.dataset_readin(['tracer_diags'],
                                      iters=iters,
                                      directory=directory)
        # Right now this is not used. To save time I will deactivate the
        # snapshot tracer output and put the mean into the routine
        # ds_snap = self.dataset_readin(['tracer_snapshots'],iters=iters,
        #                                 directory=directory)

        bins = [('j', self.koc_interval), ('i', self.koc_interval)]
        KOC, N, D, R, RC = KOC_Full(ds_mean, ds_mean, self.validmaskpath,
                                    tr_num, bins, kappa=self.koc_kappa)

        val_idx = self.reset_cut_mask(ds_mean.iter.data, int(tr_num),
                                      spin_up_months*30*24*60*60)

        ds = xr.Dataset({'KOC': KOC,
                         'Numerator': N,
                         'Denominator': D,
                         'AveTracer': RC})
        ds.coords['valid_index'] = (['time'], val_idx)
        return ds, R

    def KOC_combined(self, directory=None, interval=None,
                     spin_up_months=3, iters='all', debug=False):
        """ Calculates the KOC results for all tracer
        Calculates the results for each tracer and merges them together
        Keyword arguments:
        """

        pre_combined_ds = []
        pre_combined_R = []
        for tr in range(len(self.tracernum)):
            tr_str = '0'+str(tr+1)
            temp_ds, temp_R = self.KOC(tr_str, directory=directory,
                                       spin_up_months=spin_up_months,
                                       iters=iters, interval=interval,
                                       debug=debug)
            pre_combined_ds.append(temp_ds)
            pre_combined_R.append(temp_R)

        KOC = xr.concat(pre_combined_ds, 'tracernum')
        KOC.coords['tracernum'] = (['tracernum'], np.array([1, 2]))
        rawKOC = xr.concat(pre_combined_R, 'tracernum')
        rawKOC.coords['tracernum'] = (['tracernum'], np.array([1, 2]))
        return KOC, rawKOC


def reset_cut(reset_frq, reset_pha, total_time,
              dt_model, iters, tr_num, cut_time):
    """
    determine the timing of reset and define cut index

    Based on the information in the modelparameters this routine translates
    the reset time in second to iterations and constructs an index, matching
    a passed array which can then be used as a mask

    Input:  reset_frq - frequency of reset in seconds
            reset_pha - phase of reset in seconds
            total_time - total time of model run in seconds
            dt_model - timestep of model in seconds
            iters - numpy array of iterations on which index is constructed
            tr_num = str of tracernumber to be considered
                (based on data.ptracers)
            cut_time = [in seconds] time after (before; if negative number)
            the reset that should be masked by
    """

    tr_num = str(tr_num)
    reset_time = np.arange(reset_pha, total_time + 1, reset_frq)

    # iteration 0 is always considered a reset
    if reset_time[0] is not 0:
        reset_time = np.concatenate((np.array([0]), reset_time))

    # ceil the values if reset times dont divide without remainder
    # That way for snapshots the reset is evaluating the first snapshot
    # after the reset and for averages it ensures that the 'reset average'
    # contains the actual reset time
    reset_iters = np.ceil(reset_time/float(dt_model))

    # translate cut time to iters (round down)
    cut = np.ceil(cut_time/float(dt_model))
    mask = np.ones_like(iters)
    for ii in reset_iters:
        if cut_time < 0:
            idx = np.logical_and(iters > (ii+cut), iters <= ii)
        else:
            idx = np.logical_and(iters >= ii, iters < (ii+cut))
        mask[idx] = 0
    return mask, reset_iters, reset_time


def custom_coarse(a, area, bins, mask):
    a = a.where(mask)
    a_coarse = aggregate_w_nanmean(a, area, bins)
    return a_coarse


def KOC_Full(snap, mean, validfile, tr_num, bins, kappa=63,
             debug=False, method='LT'):
    # !!! totally hacky...this needs to be replaced.
    axis_bins = [('X', bins[0][1]), ('Y', bins[0][1])]
    #
    area = mean.rA
    area_sum = aggregate(area, bins, func=np.sum)
    # Masking #
    # landmask
    landmask_w = mean.hFacW.data != 0
    landmask_s = mean.hFacS.data != 0
    landmask_c = mean.hFacC != 0
    landmask = np.logical_and(np.logical_and(landmask_w, landmask_s),
                              landmask_c)

    # validmask (update with landmask)
    validmask = xr.DataArray(readbin(validfile, area.shape) == 0,
                             dims=area.dims, coords=area.coords)

    mask = np.logical_and(validmask, landmask)

    # Correct the false grid dimensions for the diagnoses output
    # !!! the uvel dims are taken...if there is no data var 'UVEL' or 'VVEL'
    # This will fail..it should be removed in the future

    if 'i' in mean['DXSqTr'+tr_num].dims:
        mean['DXSqTr'+tr_num] = xr.DataArray(mean['DXSqTr'+tr_num].data,
                                             dims=mean.UVEL.dims,
                                             coords=mean.UVEL.coords)

    if 'j' in mean['DYSqTr'+tr_num].dims:
        mean['DYSqTr'+tr_num] = xr.DataArray(mean['DYSqTr'+tr_num].data,
                                             dims=mean.VVEL.dims,
                                             coords=mean.VVEL.coords)
    if 'DXSqTr'+tr_num in snap.data_vars.keys():
        if 'i' in snap['DXSqTr'+tr_num].dims:
            snap['DXSqTr'+tr_num] = xr.DataArray(snap['DXSqTr'+tr_num].data,
                                                 dims=snap.UVEL.dims,
                                                 coords=snap.UVEL.coords)
    if 'DYSqTr'+tr_num in snap.data_vars.keys():
        if 'j' in snap['DYSqTr'+tr_num].dims:
            snap['DYSqTr'+tr_num] = xr.DataArray(snap['DYSqTr'+tr_num].data,
                                                 dims=snap.VVEL.dims,
                                                 coords=snap.VVEL.coords)

    required_fields = ['DXSqTr'+tr_num, 'DYSqTr'+tr_num, 'TRAC'+tr_num]

    if method == 'L':
        if not np.array([a in snap.keys() for a in required_fields]).all():
            raise RuntimeError(['mean dataset does not have all required \
                                   variables (']+snap.keys()+[')'])
    elif method == 'T' or method == 'LT':
        if not np.array([a in mean.keys() for a in required_fields]).all():
            raise RuntimeError(['mean dataset does not have all required \
                                   variables (']+mean.keys()+[')'])

    # snapshots averaged in space
    if method == 'L':
        data = snap
        grid = xgcm.Grid(data)
        grid_coarse = xgcm.Grid(grid_aggregate(grid._ds, axis_bins))
        # Numerator
        q = data['TRAC'+tr_num]
        q_grad_sq = gradient_sq_amplitude(grid, q)
        q_grad_sq_coarse = custom_coarse(q_grad_sq, area, bins, mask)
        n = q_grad_sq_coarse
        # Denominator
        q_coarse = custom_coarse(q, area, bins, mask)
        q_coarse_grad_sq = gradient_sq_amplitude(grid_coarse, q_coarse)
        d = q_coarse_grad_sq
    elif method == 'T':
        data = mean
        grid = xgcm.Grid(data)
        grid_coarse = xgcm.Grid(grid_aggregate(grid._ds, axis_bins))
        # Numerator
        q_gradx_sq_mean = data['DXSqTr'+tr_num]
        q_grady_sq_mean = data['DYSqTr'+tr_num]
        q_grad_sq_mean = grid.interp(q_gradx_sq_mean, 'X') + \
            grid.interp(q_grady_sq_mean, 'Y')
        n = q_grad_sq_mean
        # !!! this is not the right way to do it but its the same way ryan did
        n = custom_coarse(n, area, bins, mask)
        # Denominator
        q_mean = data['TRAC'+tr_num]
        q_mean_grad_sq = gradient_sq_amplitude(grid, q_mean)
        d = q_mean_grad_sq
        # !!! this is not the right way to do it but its the same way ryan did
        d = custom_coarse(d, area, bins, mask)
    elif method == 'LT':
        data = mean
        grid = xgcm.Grid(data)
        grid_coarse = xgcm.Grid(grid_aggregate(grid._ds, axis_bins))
        # Numerator
        q_gradx_sq_mean = data['DXSqTr'+tr_num]
        q_grady_sq_mean = data['DYSqTr'+tr_num]

        q_grad_sq_mean = grid.interp(q_gradx_sq_mean, 'X') + \
            grid.interp(q_grady_sq_mean, 'Y')
        n = custom_coarse(q_grad_sq_mean, area, bins, mask)
        # Denominator
        q_mean = data['TRAC'+tr_num]
        q_mean_coarse = custom_coarse(q_mean, area, bins, mask)
        q_mean_grad_sq = gradient_sq_amplitude(grid_coarse, q_mean_coarse)
        d = q_mean_grad_sq

    # Calculate the gradient criterion
    crit_q_mean = custom_coarse(data['TRAC'+tr_num], area, bins, mask)
    crit_q_sq_mean = custom_coarse(data['TRACSQ'+tr_num], area, bins, mask)
    crit_dict = gradient_criterion(grid_coarse, crit_q_mean, crit_q_sq_mean)
    crit = crit_dict['crit']

    # Export the 'raw tracer fields' ###
    raw = data['TRAC'+tr_num]
    raw_coarse = custom_coarse(raw, area, bins, mask)

    # Final edits for output
    koc = n/d*kappa

    # Count of aggregated valid cells per output pixel.
    mask.data = da.from_array(mask.data, mask.data.shape)
    mask_count = aggregate(mask, bins, func=np.sum)

    # replace with coarse grid coords
    co = matching_coords(grid_coarse._ds, koc.dims)
    # !!! this is not necessarily enough (this needs to be automated to chose
    # only the corrds with all dims matching i,g,time)

    d = xr.DataArray(d.data, coords=co, dims=d.dims)
    n = xr.DataArray(n.data, coords=co, dims=n.dims)
    koc = xr.DataArray(koc.data, coords=co, dims=koc.dims)
    raw_coarse = xr.DataArray(raw_coarse.data, coords=co, dims=raw_coarse.dims)

    d.coords['area'] = area_sum
    n.coords['area'] = area_sum
    koc.coords['area'] = area_sum
    raw_coarse.coords['area'] = area_sum

    d.coords['mask_count'] = mask_count
    n.coords['mask_count'] = mask_count
    koc.coords['mask_count'] = mask_count
    raw_coarse.coords['mask_count'] = mask_count

    d.coords['gradient_criterion'] = crit
    n.coords['gradient_criterion'] = crit
    koc.coords['gradient_criterion'] = crit
    raw_coarse.coords['gradient_criterion'] = crit

    raw.coords['landmask'] = landmask
    raw.coords['validmask'] = validmask
    raw.coords['mask'] = mask

    return koc, n, d, raw, raw_coarse


def gradient_criterion(grid, q_mean, q_sq_mean):
    lap_q = laplacian(grid, q_mean)
    grad_q = gradient_sq_amplitude(grid, q_mean)
    q_prime_sq_mean = q_sq_mean-(q_mean**2)
    # Perhaps this needs to padded with zeros where q_prime_sq_mean<0
    phi = q_prime_sq_mean.where(q_prime_sq_mean > 0)/2
    D = lap_q/grad_q
    crit = D*np.sqrt(phi)
    # Notes
    # - swap_dims needs to be deactivated in xmitgcm/open_mdsdataset
    return {'crit': crit,
            'lap_q': lap_q,
            'grad_q': grad_q,
            'q_prime_sq_mean': q_prime_sq_mean,
            'phi': phi,
            'D': D}


def main(ddir, odir, validmaskpath,
         koc_interval=20, kappa=63, iters='all', spin_up_time=3):
    # spin_up_time in months

    # default value for kappa=63
    # This is discussed in Abernathey 2013
    # Their value from table B1 for a ptracer diff= 25 m^2/s

    print('data_dir:'+str(ddir))
    print('out_dir:'+str(odir))
    print('validmaskpath:'+str(validmaskpath))

    print('### Initialize core class ###')
    TrCore = tracer_engine(ddir, koc_kappa=kappa, validmaskpath=validmaskpath,
                           koc_interval=koc_interval, makedirs=True)

    print('CALCULATE OSBORN-COX DIFFUSIVITY')
    start_time = time.time()
    KOC, raw = TrCore.KOC_combined(spin_up_months=spin_up_time, iters=iters)
    print("--- %s seconds ---" % (time.time() - start_time))

    print('SAVE TO FILE')
    start_time = time.time()
    KOC.to_netcdf(odir+'/'+'KOC_FINAL.nc')
    raw.to_netcdf(odir+'/'+'KOC_RAW.nc')
    print("--- %s seconds ---" % (time.time() - start_time))
