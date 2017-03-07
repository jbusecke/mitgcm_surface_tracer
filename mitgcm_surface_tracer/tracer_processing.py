from __future__ import print_function
from future.utils import iteritems
import os
import time
import re
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from xmitgcm import open_mdsdataset
import xarrayutils as xut
from .utils import readbin, paramReadout, dirCheck


class tracer_engine:
    """Make is easier doing many operations on the same
    grid."""

    def __init__(self,ddir,initpath,koc_kappa=63,odir=None,mdir=None,
    griddir=None,koc_interval=20,
    validmaskpath=None,makedirs=False):
        # Predefined inputs
        # Start time of offline velocities [YYYY,MM,DD,HH,MM,SS]
        self.start_time = [1993,    1,    1,    0,    0,    0]
        self.ref_date = '-'.join([str(x) for x in self.start_time[0:3]])
        # simple parsed inputs
        self.ddir = dirCheck(ddir,makedirs)
        self.initpath = initpath
        # diagnosed small scale diff (k_num in Abernathey et al. 2013)
        self.koc_kappa = koc_kappa
        # The interval for coarsening the KOC data
        self.koc_interval = koc_interval
        self.validmaskpath = validmaskpath

        self.modelparameters = paramReadout(self.ddir)
        self.tracernum  = int(self.modelparameters['data.ptracers/PTRACERS_numInUse'])+1

        (steps_tracer_diags,self.ti_tracer_diags,self.dt_model) = \
        timeStepsfromMITgcm(self.modelparameters,
        'data.diagnostics/frequency(1)')

        # This is what it should be based on the setup files
        print(str(len(steps_tracer_diags))+' steps from model setup for tracer_diags')

        self.grid = open_mdsdataset(self.ddir,prefix=['tracer_diags'],delta_t=self.dt_model,ref_date=self.ref_date,iters=None)

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


        # a) !!!!This can possibly be thrown out after refactoring

        # Compute the data file names and corresponding steplists
        diagname = ['tracer_diags','tracer_snapshots']
        diags_steplist = []
        for i,t in enumerate(diagname):
            diags_steplist.append(np.array(filelist(ddir,diagname[i])))
            print(str(len(diags_steplist[i]))+' steps found for '+diagname[i])
        self.diagname = diagname
        self.diags_steplist = diags_steplist

        # a) !!!!Until here

    def dataset_readin(self,prefix,directory=None,iters='all'):
        if directory == None:
            directory = self.ddir
        ds = open_mdsdataset(directory,prefix=prefix,\
        delta_t=self.dt_model,ref_date=self.ref_date,iters=iters,swap_dims=False)
        return ds

    def KOC(self,tr_num,directory=None,interval=None,low_grad_crit=0.05,\
            spin_up_months=3,iters='all',debug=False):

        if interval==None:
            interval=self.koc_interval

        ds_mean = self.dataset_readin(['tracer_diags'],iters=iters,directory=directory)
        ds_snap = self.dataset_readin(['tracer_snapshots'],iters=iters,directory=directory)

        bins = [('j',self.koc_interval),('i',self.koc_interval)]
        KOC,N,D,R,RC= KOC_Full(ds_snap,ds_mean,self.validmaskpath,self.initpath,tr_num,\
            bins,low_gradient_perc=low_grad_crit,\
            kappa=self.koc_kappa)

        val_idx,_ = validity_index(self.modelparameters,int(tr_num),
            KOC.data,spin_up_months,0,axis=0,timestyle='diagnostic')

        ds = xr.Dataset({'KOC':KOC,'Numerator':N,'Denominator':D,'AveTracer':RC})
        ds.coords['valid_index'] = (['time'], val_idx)

        return ds,R

    def KOC_combined(self,directory=None,interval=None,low_grad_crit=0.05,\
                    spin_up_months=3,iters='all',debug=False):
        """ Calculates the KOC results for all tracer

        Calculates the results for each tracer and merges them together

        Keyword arguments:

        """
        #!!! Performance... I think this still doesnt stream as dask array...
        pre_combined_ds      = []
        pre_combined_R       = []
        for tr in range(len(self.diagname)):
            tr_str = '0'+str(tr+1)
            print('CALCULATE OSBORN-COX DIFFUSIVITY for '+tr_str)
            temp_ds,temp_R = self.KOC(tr_str,directory=directory,\
                                    spin_up_months=spin_up_months,\
                                    low_grad_crit=low_grad_crit,\
                                    iters=iters,interval=interval,debug=debug)
            pre_combined_ds.append(temp_ds)
            pre_combined_R.append(temp_R)

        KOC    = xr.concat(pre_combined_ds,'tracernum')
        KOC.coords['tracernum'] = (['tracernum'], np.array([1,2]))
        rawKOC = xr.concat(pre_combined_R,'tracernum')
        rawKOC.coords['tracernum'] = (['tracernum'], np.array([1,2]))

        return KOC,rawKOC

def KOC_Full(snap,mean,validfile,initfile,tr_num,bins,kappa=63,\
            low_gradient_perc=0.1,debug=False,method='LT'):
    func        = np.sum #!!! with this i make all the masking pretty much
    #obsolete but I had some crazy strong outliers around the small islands
    area        = snap.rA
    #### Masking ####
    valid_mask = xr.DataArray(readbin(validfile,area.shape)==0,\
                  dims=area.dims,coords=area.coords)
    area        = area.where(snap.hFacC!=0).where(valid_mask)
    area_sum    = xut.aggregate(area,bins,func=func)

    # Ok this needs some thorough invesigation, but is probably of minor importance (e.g. only near the coast)
    # I mask out the area. Since all values are at some point multiplied with area (area weigthed ave)
    # this should take care of the masking. A problem arises when the gradient operations create
    # additional nans in the data, which are not covered in area and area_sum
    # that could potentially throw of the area weighting around missing values
    # HOWEVER I think this might not matter since pretty much everywhere the pixel
    # near land are masked out by the validmask anyways.

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

    # snapshots averaged in space
    if method == 'L':
        data             = snap
        grid             = data.drop(data.data_vars.keys())
        grid_coarse      = xut.xmitgcm_utils.grid_aggregate(grid,bins)
        #Numerator
        q                = data['TRAC'+tr_num]
        q_gradx,q_grady  = xut.xmitgcm_utils.gradient(grid,q,recenter=True)
        q_grad_sq        = q_gradx**2 + q_grady**2
        q_grad_sq_coarse = xut.aggregate(q_grad_sq*area,bins,func=func)/area_sum
        n                = q_grad_sq_coarse
        #Denominator
        q_coarse         = xut.aggregate(q*area,bins,func=func)/area_sum
        q_coarse_gradx,q_coarse_grady \
                         = xut.xmitgcm_utils.gradient(grid_coarse,q_coarse,recenter=True)
        q_coarse_grad_sq = q_coarse_gradx**2+q_coarse_grady**2
        d                = q_coarse_grad_sq
    elif method == 'T':
        data             = mean
        grid             = data.drop(data.data_vars.keys())
        grid_coarse      = xut.xmitgcm_utils.grid_aggregate(grid,bins)
        #Numerator
        q_gradx_sq_mean  = data['DXSqTr'+tr_num]
        q_grady_sq_mean  = data['DYSqTr'+tr_num]
        q_grad_sq_mean   = xut.xmitgcm_utils.interpolateGtoC(grid,q_gradx_sq_mean,dim='x') + \
                            xut.xmitgcm_utils.interpolateGtoC(grid,q_grady_sq_mean,dim='y')
        n                = q_grad_sq_mean
        # !!! this is not the right way to do it but its the same way ryan did it
        n                = xut.aggregate(n*area,bins,func=func)/area_sum
        #Denominator
        q_mean           = data['TRAC'+tr_num]
        q_mean_gradx,q_mean_grady  = xut.xmitgcm_utils.gradient(grid,q_mean,recenter=True)
        q_mean_grad_sq   = q_mean_gradx**2 + q_mean_grady**2
        d                = q_mean_grad_sq
        # !!! this is not the right way to do it but its the same way ryan did it
        d                = xut.aggregate(d*area,bins,func=func)/area_sum
    elif method =='LT':
        data             = mean
        grid             = data.drop(data.data_vars.keys())
        grid_coarse      = xut.xmitgcm_utils.grid_aggregate(grid,bins)
        #Numerator
        q_gradx_sq_mean  = data['DXSqTr'+tr_num]
        q_grady_sq_mean  = data['DYSqTr'+tr_num]
        q_grad_sq_mean   = xut.xmitgcm_utils.interpolateGtoC(grid,q_gradx_sq_mean,dim='x') + \
                            xut.xmitgcm_utils.interpolateGtoC(grid,q_grady_sq_mean,dim='y')
        n                = xut.aggregate(q_grad_sq_mean*area,bins,func=func)/area_sum
        #Denominator
        q_mean           = data['TRAC'+tr_num]
        q_mean_coarse    = xut.aggregate(q_mean*area,bins,func=func)/area_sum
        q_mean_coarse_gradx,q_mean_coarse_grady  \
                         = xut.xmitgcm_utils.gradient(grid_coarse,q_mean_coarse,recenter=True)
        q_mean_grad_sq   = q_mean_coarse_gradx**2 + q_mean_coarse_grady**2
        d                = q_mean_grad_sq

    ### Export the 'raw tracer fields' ###
    raw              = data['TRAC'+tr_num].where(grid.hFacC!=0)
    raw_coarse       = xut.aggregate(raw*area,bins,func=func)/area_sum


    # #Read initial conditions to determine which gradient is 'too low'
    q_init = xr.DataArray(readbin(initfile,grid.XC.shape),\
                          dims=grid.XC.dims,coords=grid.XC.coords)
    q_init_gradx,q_init_grady = xut.xmitgcm_utils.gradient(mean,q_init.where(valid_mask),recenter=True)
    q_init_grad_sq = q_init_gradx**2 + q_init_grady**2

    #!!! this needs some work...
    low_cut = q_init_grad_sq.data.flatten()
    low_cut = low_cut[~np.isnan(low_cut)]
    low_cut = np.percentile(low_cut,low_gradient_perc*100)



    #mask out validmask and low gradient cut
    d   = d.where(d>low_cut)

    # Final edits for output
    koc = n/d*kappa

    #replace with coarse grid coords
    co     = xut.xmitgcm_utils.matching_coords(grid_coarse,koc.dims)
    # !!! this is not necessarily enough (this needs to be automated to chose only the
    # corrds with all dims matching i,g,time)

    d          = xr.DataArray(d.data,coords = co,dims = d.dims)
    n          = xr.DataArray(n.data,coords = co,dims = n.dims)
    koc        = xr.DataArray(koc.data,coords = co,dims = koc.dims)
    raw_coarse = xr.DataArray(raw_coarse.data,coords = co,dims = raw_coarse.dims)

    d.coords['weighted_area']          = area_sum
    n.coords['weighted_area']          = area_sum
    koc.coords['weighted_area']        = area_sum
    raw_coarse.coords['weighted_area'] = area_sum
    return koc,n,d,raw,raw_coarse

def timeStepsfromMITgcm(params,targetvariable):
    '''Read the time parameters from the data* files.
    Gives the steps and time in seconds'''
    dt_model = int(float(params['data/deltaTtracer']))
    dt_total = int(float(params['data/nTimeSteps']))
    dt_target = int(float(params[targetvariable]))
    if dt_target==0:
        time_steps = np.array([])
        time = np.array([])
    else:
        time_steps = np.arange(0,dt_total+1,(dt_target/dt_model))
        time = time_steps*dt_model
    return time_steps,time,dt_model

# # !!! Possibly erase (this can be determined from xmitgcm input)
def filelist(dir,name):
    files = os.listdir(dir)
    sorted = []
    for f in files:
        a = re.search(r''+name+'',f)
        b = re.search(r'.meta',f)
        if a and b:
            b = int(re.sub(r'\D',"",f)[3:])
            sorted.append(b)
    return sorted

# I might have to simplify this with the xmitgcm output
def validity_index(mod_in,tracern,data_in,sp_up,ed_ct,timestyle='snapshot',axis=1,debug=False):
    """
    Input:
    mod_in: model parameter input dict (see paramReadout)
    tracern: tracernumber, used to extract right timing parameters from mod_in
    data_in: input data
    sp_up: spin up time to be eliminated from each reset (in months)
    ed_ct: end cut of. no of samples to be ignored before the reset. This avoids filtering issues due to centered difference scheme
        adjust according to the data processing
    timestyle: Parameter specifying whether the inputs are time averages or snapshots
    axis: axis of the input array to operate on
    """
    # !!! THis will have to be adjusted for the KOC
    # determine reset index from

    if timestyle=='snapshot':
        dt = int(mod_in['data.ptracers/PTRACERS_dumpFreq'][:-2])
    elif timestyle=='diagnostic':
        dt = int(mod_in['data.diagnostics/frequency(1)'][:-2])
    else:
        raise RuntimeError('"method" input not recognized only defined for diagnostics and snapshot')

    delta_t  = int(mod_in['data.ptracers/PTRACERS_resetFreq('+str(tracern)+')'][:-1])/dt
    t_0      = int(mod_in['data.ptracers/PTRACERS_resetPhase('+str(tracern)+')'][:-1])/dt
    total_t  = data_in.shape[axis]
    st_ct    = int(np.ceil((np.array([sp_up])*30*24*60*60)/dt))
    #!!! should I put this to ceil?

    if debug:
        print('delta_t',delta_t)
        print('total_t',total_t)
        print('t_0',t_0)
        print('dt',dt)

    if total_t<t_0:
        raise RuntimeError('not enough timesteps calculated to have one reset period, all would be cut off')

    rs_id = range(t_0,total_t,delta_t)
    # Fill the first index with 0 if its not already set
    if rs_id[0]!=0:
        rs_id.insert(0,0)

    rs_idx = np.zeros(total_t)
    for ii,aa in enumerate(rs_id):
        if aa-ed_ct<0:
            rs_idx[0:aa+st_ct]=1
        elif (aa+st_ct)>len(rs_idx):
            rs_idx[aa-ed_ct:]=1
        else:
            rs_idx[aa-ed_ct:aa+st_ct]=1

    rs_idx = rs_idx==0

    if debug:
        print('dt',dt)
        print('delta_t',delta_t)
        print('t_0',t_0)
        print('total_t',total_t)
        print('st_ct',st_ct)
        print('ed_ct',ed_ct)
    return rs_idx,rs_id

def main(ddir,initpath,odir,validmaskpath,
    koc_interval=20,kappa=63,iters='all',low_grad=0.05,spin_up_time = 3):
    # spin_up_time in months

    # default value for kappa=63
    #This is discussed in Abernathey 2013
    #Their value from table B1 for a ptracer diff= 25 m^2/s

    print('data_dir:'+str(ddir))
    print('Initial Condition Path:'+str(initpath))
    print('out_dir:'+str(odir))
    print('validmaskpath:'+str(validmaskpath))

    print('### Initialize core class ###')
    TrCore = tracer_engine(ddir,initpath,koc_kappa=kappa,odir=odir,
                validmaskpath=validmaskpath,koc_interval=koc_interval,makedirs=True)

    ##################
    # Osborn-Cox Diffusivity
    # (better) processing
    ##################
    start_time = time.time()
    KOC,raw = TrCore.KOC_combined(spin_up_months=spin_up_time,\
                                    low_grad_crit=low_grad,iters=iters)

    KOC.to_netcdf(TrCore.odir+'/'+'KOC_FINAL.nc')
    raw.to_netcdf(TrCore.odir+'/'+'KOC_RAW.nc')

    print("--- %s seconds ---" % (time.time() - start_time))

# maybe run this with the current dir as ddir and otherwise just defaults?
# if __name__ == "__main__":
