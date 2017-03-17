from __future__ import print_function
from future.utils import iteritems
import os
import time
import re
import numpy as np
import xarray as xr
import dask.array as da
from dask.diagnostics import ProgressBar
from xmitgcm import open_mdsdataset
from xarrayutils.utils import aggregate,aggregate_w_nanmean
from xarrayutils.build_grids import grid_aggregate
from xarrayutils.xmitgcm_utils import gradient,matching_coords
from .utils import readbin, paramReadout, dirCheck

class tracer_engine:
    """Make is easier doing many operations on the same
    grid."""

    def __init__(self,ddir,koc_kappa=63,odir=None,mdir=None,
    griddir=None,koc_interval=20,
    validmaskpath=None,makedirs=False):
        # Predefined inputs
        # Start time of offline velocities [YYYY,MM,DD,HH,MM,SS]
        self.start_time = [1993,    1,    1,    0,    0,    0]
        self.ref_date = '-'.join([str(x) for x in self.start_time[0:3]])
        # simple parsed inputs
        self.ddir = dirCheck(ddir,makedirs)
        # diagnosed small scale diff (k_num in Abernathey et al. 2013)
        self.koc_kappa = koc_kappa
        # The interval for coarsening the KOC data
        self.koc_interval = koc_interval
        self.validmaskpath = validmaskpath

        # Read infos from run directory (this assumes that data* files are found in ddir)
        self.modelparameters = paramReadout(self.ddir)
        self.tracernum  = np.arange(int(self.modelparameters['data.ptracers/PTRACERS_numInUse']))+1
        self.dt_model = int(float(self.modelparameters['data/deltaTtracer']))
        self.total_time_model = int(float(self.modelparameters['data/nTimeSteps']))*self.dt_model


        self.grid = open_mdsdataset(self.ddir,prefix=['tracer_diags'],
                                    delta_t=self.dt_model,
                                    ref_date=self.ref_date,
                                    iters=None)

        # Internal calculated inputs
        self.dx         = self.grid['dxC'].data
        self.dy         = self.grid['dyC'].data
        self.area       = self.grid['rA'].data
        self.x          = self.grid['XC'].data
        self.y          = self.grid['YC'].data
        self.depth      = self.grid['hFacC'].data
        self.landmask   = self.depth==0


        # optional input with defaults


        if odir == None:
            odir = ddir+'/output'
        self.odir = dirCheck(odir,makedirs)

        if mdir == None:
            mdir = ddir+'/movies'
        self.mdir = dirCheck(mdir,makedirs)

        if griddir == None:
            griddir = ddir
        self.griddir = dirCheck(griddir,makedirs)

    def reset_cut_mask(self,iters,tr_num,cut_time):
        total_time = self.total_time_model
        reset_frq  = int(self.modelparameters[
                            'data.ptracers/PTRACERS_resetFreq('+str(tr_num)+')'
                            ])
        reset_pha  = int(self.modelparameters[
                            'data.ptracers/PTRACERS_resetPhase('+str(tr_num)+')'
                            ])
        dt_model = self.dt_model

        mask,_,_ = reset_cut(reset_frq,reset_pha,
                            total_time,dt_model,
                            iters,tr_num,cut_time)
        return mask

    def dataset_readin(self,prefix,directory=None,iters='all'):
        if directory == None:
            directory = self.ddir
        ds = open_mdsdataset(directory,prefix=prefix,\
        delta_t=self.dt_model,ref_date=self.ref_date,iters=iters,swap_dims=False)
        return ds

    def KOC(self,tr_num,directory=None,interval=None,\
            spin_up_months=3,iters='all',debug=False):

        if interval==None:
            interval=self.koc_interval

        ds_mean = self.dataset_readin(['tracer_diags'],iters=iters,
                                        directory=directory)
        ds_snap = self.dataset_readin(['tracer_snapshots'],iters=iters,
                                        directory=directory)

        bins = [('j',self.koc_interval),('i',self.koc_interval)]
        KOC,N,D,R,RC= KOC_Full(ds_snap,ds_mean,self.validmaskpath,tr_num,\
                                bins,\
                                kappa=self.koc_kappa)

        val_idx = self.reset_cut_mask(ds_mean.iter.data,
                                            int(tr_num),
                                            spin_up_months*30*24*60*60)

        ds = xr.Dataset({'KOC':KOC,'Numerator':N,'Denominator':D,'AveTracer':RC})
        ds.coords['valid_index'] = (['time'], val_idx)

        return ds,R

    def KOC_combined(self,directory=None,interval=None,\
                    spin_up_months=3,iters='all',debug=False):
        """ Calculates the KOC results for all tracer

        Calculates the results for each tracer and merges them together

        Keyword arguments:

        """

        pre_combined_ds      = []
        pre_combined_R       = []
        for tr in range(len(self.tracernum)):
            tr_str = '0'+str(tr+1)
            temp_ds,temp_R = self.KOC(tr_str,directory=directory,\
                                    spin_up_months=spin_up_months,\
                                    iters=iters,interval=interval,debug=debug)
            pre_combined_ds.append(temp_ds)
            pre_combined_R.append(temp_R)

        KOC    = xr.concat(pre_combined_ds,'tracernum')
        KOC.coords['tracernum'] = (['tracernum'], np.array([1,2]))
        rawKOC = xr.concat(pre_combined_R,'tracernum')
        rawKOC.coords['tracernum'] = (['tracernum'], np.array([1,2]))
        return KOC,rawKOC

def reset_cut(reset_frq,reset_pha,total_time,dt_model,iters,tr_num,cut_time):
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
            tr_num = str of tracernumber to be considered (based on data.ptracers)
            cut_time = [in seconds] time after (before; when negative number is passed)
            the reset that should be masked by
    """

    tr_num = str(tr_num)
    reset_time = np.arange(reset_pha,total_time+1,reset_frq)

    # iteration 0 is always considered a reset
    if reset_time[0]!=0:
        reset_time = np.concatenate((np.array([0]),reset_time))

    # ceil the values if reset times dont divide without remainder
    # That way for snapshots the reset is evaluating the first snapshot
    # after the reset and for averages it ensures that the 'reset average'
    # contains the actual reset time
    reset_iters = np.ceil(reset_time/float(dt_model))

    #translate cut time to iters (round down)
    cut = np.ceil(cut_time/float(dt_model))
    mask = np.ones_like(iters)
    for ii in reset_iters:
        if cut_time<0:
            idx = np.logical_and(iters>(ii+cut),iters<=ii)
        else:
            idx = np.logical_and(iters>=ii,iters<(ii+cut))
        mask[idx] = 0
    return mask,reset_iters,reset_time

def custom_coarse(a,area,bins,mask):
    a  = a.where(mask)
    a_coarse  = aggregate_w_nanmean(a,area,bins)
    return a_coarse


def KOC_Full(snap,mean,validfile,tr_num,bins,kappa=63,\
                debug=False,method='LT'):
    area        = mean.rA
    area_sum    = aggregate(area,bins,func=np.sum)
    #### Masking ####
    ## landmask
    landmask_w = mean.hFacW.data != 0
    landmask_s = mean.hFacS.data != 0
    landmask_c = mean.hFacC !=0
    landmask   = np.logical_and(np.logical_and(landmask_w,landmask_s),landmask_c)

    ## validmask (update with landmask)
    validmask = xr.DataArray(readbin(validfile,area.shape)==0,\
                  dims=area.dims,coords=area.coords)

    mask = np.logical_and(validmask,landmask)

    # Correct the false grid dimensions for the diagnoses output
    # !!! the uvel dims are taken...if there is no data var 'UVEL' or 'VVEL'
    # This will fail..it should be removed in the future

    if 'i' in mean['DXSqTr'+tr_num].dims:
        mean['DXSqTr'+tr_num] = xr.DataArray(mean['DXSqTr'+tr_num].data,\
                                       dims=mean.UVEL.dims,coords=mean.UVEL.coords)

    if 'j' in mean['DYSqTr'+tr_num].dims:
        mean['DYSqTr'+tr_num] = xr.DataArray(mean['DYSqTr'+tr_num].data,\
                                       dims=mean.VVEL.dims,coords=mean.VVEL.coords)
    if 'DXSqTr'+tr_num in snap.data_vars.keys():
        if 'i' in snap['DXSqTr'+tr_num].dims:
            snap['DXSqTr'+tr_num] = xr.DataArray(snap['DXSqTr'+tr_num].data,\
                                       dims=snap.UVEL.dims,coords=snap.UVEL.coords)
    if 'DYSqTr'+tr_num in snap.data_vars.keys():
        if 'j' in snap['DYSqTr'+tr_num].dims:
            snap['DYSqTr'+tr_num] = xr.DataArray(snap['DYSqTr'+tr_num].data,\
                                       dims=snap.VVEL.dims,coords=snap.VVEL.coords)

    required_fields = ['DXSqTr'+tr_num,'DYSqTr'+tr_num,'TRAC'+tr_num]

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
        data             = snap
        grid             = data.drop(data.data_vars.keys())
        grid_coarse      = grid_aggregate(grid,bins)
        #Numerator
        q                = data['TRAC'+tr_num]
        q_gradx,q_grady  = gradient(grid,q,recenter=True)
        q_grad_sq        = q_gradx**2 + q_grady**2
        q_grad_sq_coarse = custom_coarse(q_grad_sq,area,bins,mask)
        n                = q_grad_sq_coarse
        #Denominator
        q_coarse         = custom_coarse(q,area,bins,mask)
        q_coarse_gradx,q_coarse_grady \
                         = gradient(grid_coarse,q_coarse,recenter=True)
        q_coarse_grad_sq = q_coarse_gradx**2+q_coarse_grady**2
        d                = q_coarse_grad_sq
    elif method == 'T':
        data             = mean
        grid             = data.drop(data.data_vars.keys())
        grid_coarse      = grid_aggregate(grid,bins)
        #Numerator
        q_gradx_sq_mean  = data['DXSqTr'+tr_num]
        q_grady_sq_mean  = data['DYSqTr'+tr_num]
        q_grad_sq_mean   = interpolateGtoC(grid,q_gradx_sq_mean,dim='x') + \
                            interpolateGtoC(grid,q_grady_sq_mean,dim='y')
        n                = q_grad_sq_mean
        # !!! this is not the right way to do it but its the same way ryan did it
        n                = custom_coarse(n,area,bins,mask)
        #Denominator
        q_mean           = data['TRAC'+tr_num]
        q_mean_gradx,q_mean_grady  = gradient(grid,q_mean,recenter=True)
        q_mean_grad_sq   = q_mean_gradx**2 + q_mean_grady**2
        d                = q_mean_grad_sq
        # !!! this is not the right way to do it but its the same way ryan did it
        d                = custom_coarse(d,area,bins,mask)
    elif method =='LT':
        data             = mean
        grid             = data.drop(data.data_vars.keys())
        grid_coarse      = grid_aggregate(grid,bins)
        #Numerator
        q_gradx_sq_mean  = data['DXSqTr'+tr_num]
        q_grady_sq_mean  = data['DYSqTr'+tr_num]

        q_grad_sq_mean   = interpolateGtoC(grid,q_gradx_sq_mean,dim='x') + \
                            interpolateGtoC(grid,q_grady_sq_mean,dim='y')
        n                = custom_coarse(q_grad_sq_mean,area,bins,mask)
        #Denominator
        q_mean           = data['TRAC'+tr_num]
        q_mean_coarse    = custom_coarse(q_mean,area,bins,mask)
        q_mean_coarse_gradx,q_mean_coarse_grady  \
                         = gradient(grid_coarse,q_mean_coarse,recenter=True)
        q_mean_grad_sq   = q_mean_coarse_gradx**2 + q_mean_coarse_grady**2
        d                = q_mean_grad_sq

    ### Export the 'raw tracer fields' ###
    raw              = data['TRAC'+tr_num]
    raw_coarse       = custom_coarse(raw,area,bins,mask)

    # Final edits for output
    koc = n/d*kappa

    #Count of aggregated valid cells per output pixel.
    mask.data = da.from_array(mask.data,mask.data.shape)
    mask_count = aggregate(mask,bins,func=np.sum)

    #replace with coarse grid coords
    co     = matching_coords(grid_coarse,koc.dims)
    # !!! this is not necessarily enough (this needs to be automated to chose only the
    # corrds with all dims matching i,g,time)

    d          = xr.DataArray(d.data,coords = co,dims = d.dims)
    n          = xr.DataArray(n.data,coords = co,dims = n.dims)
    koc        = xr.DataArray(koc.data,coords = co,dims = koc.dims)
    raw_coarse = xr.DataArray(raw_coarse.data,coords = co,dims = raw_coarse.dims)

    d.coords['area']                  = area_sum
    n.coords['area']                  = area_sum
    koc.coords['area']                = area_sum
    raw_coarse.coords['area']         = area_sum

    d.coords['mask_count']            = mask_count
    n.coords['mask_count']            = mask_count
    koc.coords['mask_count']          = mask_count
    raw_coarse.coords['mask_count']   = mask_count

    raw.coords['landmask']            = landmask
    raw.coords['validmask']           = validmask
    raw.coords['mask']                = mask

    return koc,n,d,raw,raw_coarse

def main(ddir,odir,validmaskpath,
    koc_interval=20,kappa=63,iters='all',spin_up_time = 3):
    # spin_up_time in months

    # default value for kappa=63
    #This is discussed in Abernathey 2013
    #Their value from table B1 for a ptracer diff= 25 m^2/s

    print('data_dir:'+str(ddir))
    print('out_dir:'+str(odir))
    print('validmaskpath:'+str(validmaskpath))

    print('### Initialize core class ###')
    TrCore = tracer_engine(ddir,koc_kappa=kappa,odir=odir,
                            validmaskpath=validmaskpath,
                            koc_interval=koc_interval,makedirs=True)

    print('CALCULATE OSBORN-COX DIFFUSIVITY')
    start_time = time.time()
    KOC,raw = TrCore.KOC_combined(spin_up_months=spin_up_time,iters=iters)
    print("--- %s seconds ---" % (time.time() - start_time))


    print('SAVE TO FILE')
    start_time = time.time()
    KOC.to_netcdf(TrCore.odir+'/'+'KOC_FINAL.nc')
    raw.to_netcdf(TrCore.odir+'/'+'KOC_RAW.nc')
    print("--- %s seconds ---" % (time.time() - start_time))

# maybe run this with the current dir as ddir and otherwise just defaults?
# if __name__ == "__main__":
