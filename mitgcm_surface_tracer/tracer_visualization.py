import matplotlib
import datetime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from .tracer_processing import tracer_engine
from .utils import dirCheck
matplotlib.use('Agg')


def QC_reset_plot(ds_di, ds_sn, tr_engine, cut_time, tr_num, ylim=None):
    """Produces passive tracer reset QC plots.
    PARAMETERS
    ----------
    ds_di : xarray.dataset
        diagnostics (time averaged) dataset from the tracer experiment
        read with xmitgcm from standard setup (see package documentation).
    ds_sn : xarray.dataset
        as above but snapshots in time
    tr_engine: mitgcm_surface_tracer/tracer_processing.tracer_engine
        Tracer computation engine constucted from matching standard setup
    cut_time : float32
        time (in seconds) after reset which are masked out.
        mitgcm_surface_tracer/tracer_processing.reset_cut for details
    tr_num : int
        tracernum (starting with 1) as defined in data.ptracer of the mitgcm
        setup
    ylim : numpy.array optional, Plot y-axis limits, if not defined
        determined from data range
    """
    plt.figure(figsize=[20, 3*len(tr_num)])
    for ii in tr_num:
        mask_di, reset_iter_di, _ = \
            tr_engine.reset_cut_mask(ds_di.iter.data, ii, cut_time)
        mask_sn, reset_iter_sn, reset_time = \
            tr_engine.reset_cut_mask(ds_sn.iter.data, ii, cut_time)
        plt.subplot(len(tr_num), 1, ii)

        if 'TRAC%02d' % ii in ds_di.keys():
            di_sq = ds_di['TRAC%02d' % ii]**2
            data_di = ((di_sq)*ds_di.rA).sum(['i', 'j'])
            data_di.plot(color='C0', linestyle='--')
            data_di.where(mask_di == 1).plot(color='C0')

            matches = data_di.iter.compute().searchsorted(reset_iter_di)
            data_di[matches].plot(color='C0', marker='o', linestyle='')
            legend_di = ['mean_full', 'mean_valid', 'mean reset_idx']
        else:
            legend_di = []

        if 'TRAC%02d' % ii in ds_sn.keys():
            sn_sq = ds_sn['TRAC%02d' % ii] ** 2
            # area integral
            data_sn = ((sn_sq)*ds_sn.rA).sum(['i', 'j'])
            data_sn.plot(color='C1', linestyle='--')
            data_sn.where(mask_sn == 1).plot(color='C1')

            matches = data_sn.iter.compute().searchsorted(reset_iter_sn)
            data_sn[matches].plot(color='C1', marker='o', linestyle='')
            legend_sn = ['snapshots', 'snapshots valid', 'snapshots reset_idx']
        else:
            legend_sn = []

        if ylim is None:
            yl = np.array([data_sn.min(), data_sn.max()])
            val_range = np.diff(yl)
            yl = [yl[0]-(val_range/3), yl[1]+(val_range/3)]
        else:
            yl = ylim
        plt.gca().set_ylim(yl)
        plt.legend(legend_di+legend_sn+['exact reset'], loc=8, ncol=8)
        plt.ylabel('sum tracer squared')

        reset_date = [(datetime.datetime(1993, 1, 1, 0, 0, 0) +
                       datetime.timedelta(seconds=a)) for a in reset_time]
        for rr in reset_date:
            plt.axvline(x=rr, color='0.5')


def QC_global_timeseries_plot(ds, non_log_axis=['AveTracer'], perc=0.25):
    """Produces global median time series with of all data variables of ds.
    PARAMETERS
    ----------
    ds : xarray.dataset
        dataset from the tracer experiment read with xmitgcm from standard
        setup (see package documentation).
    non_log_axis : list of strings which data names which should be plotten on
        linear axis.
        All others will be plotted on log axis. Defaults to 'AveTracer'
    perc = float percentile plotted as shading around the median. Area between
        perc and 1-perc is shaded
    """
    h = []
    plt.figure(figsize=[20, 15])
    for pp, ff in enumerate(ds.data_vars):
        plt.subplot(len(ds.data_vars), 1, pp+1)
        for tt in ds.tracernum:
            y = ds[ff].sel(tracernum=tt)
            y_mean = (y).median(['i', 'j'])
            y_upper = y.quantile(1-perc, dim=['i', 'j'])
            y_lower = y.quantile(perc, dim=['i', 'j'])
            co = 'C'+str(tt.data-1)
            plt.fill_between(y.time.data, y_lower.data, y_upper.data,
                             color=co, alpha=0.15)
            h.append(y_mean.plot(color=co))
            if ff not in non_log_axis:
                plt.gca().set_yscale('log')
            plt.title('Median timeseries with ' +
                      str(int(perc*100)) +
                      'th and '+str(int((1-perc)*100)) +
                      'th percentile shaded')
        plt.legend(['Trac' + str(x.data) for x in ds.tracernum])


def QC_crit_plot(ds, crit=0.5):
    clim = [100, 1e4]
    norm = matplotlib.colors.SymLogNorm(10)

    plt.figure(figsize=[20, 20])
    ax = plt.subplot(3, 2, 1)
    ds.KOC.mean(['time', 'tracernum']).plot(norm=norm,
                                            vmin=clim[0],
                                            vmax=clim[1],
                                            ax=ax)
    plt.title('raw KOC')

    ax = plt.subplot(3, 2, 2)
    ds.KOC.where(abs(ds.gradient_criterion) > 0.5).\
        mean(['time', 'tracernum']).\
        plot(norm=norm, vmin=clim[0], vmax=clim[1], ax=ax)
    plt.title('KOC with invalid crit')

    ax = plt.subplot(3, 2, 3)
    ds.KOC.where(abs(ds.gradient_criterion) <= 0.5).\
        mean(['time', 'tracernum']).\
        plot(norm=norm, vmin=clim[0], vmax=clim[1], ax=ax)
    plt.title('KOC with valid crit')

    ax = plt.subplot(3, 2, 4)
    count = ds.KOC.where(abs(ds.gradient_criterion) <= 0.5).\
        count(['time', 'tracernum']).where(ds.mask_count > 0)
    total_count = float(len(ds.time))
    (count/total_count*100.0).plot(vmin=0, vmax=100, ax=ax)
    plt.title('Percentage of valid points in time')

    ax = plt.subplot(3, 2, 5)
    x = abs(ds.gradient_criterion.data.flatten())
    y = ds.KOC.data.flatten()
    idx = np.logical_and(np.logical_and(x > 1e-4, x <= 1e2),
                         np.logical_and(y > 10, y <= 2e4))

    heat = ax.hexbin(x[idx], y[idx],
                     xscale='log',
                     bins='log',
                     cmap=plt.cm.Reds)

    ax.axvline(0.5)
    plt.colorbar(heat, label='log(occurence)', ax=ax)
    plt.title('Heatmap KOC vs crit')
    plt.xlabel('Criterion')
    plt.ylabel('KOC [m^2/s]')

    ax = plt.subplot(3, 2, 6)
    bins = np.arange(-0.1, 2.1, 0.1)
    data = abs(ds.gradient_criterion)
    data = data.where(~np.isinf(data))
    data.plot.hist(bins=bins, histtype='step', cumulative=True, normed=1)
    plt.gca().set_xlim([0, 2])
    plt.axvline(0.5)
    plt.title('Cumulative Normalized Histogram')
    plt.xlabel('Criterion')


def QC_mask_plot(ds, map_perc=0.5):
    clim = [100, 1e5]
    norm = matplotlib.colors.SymLogNorm(10)
    max_mask_count = ds.mask_count.max()
    p = max_mask_count*map_perc

    plt.figure(figsize=[20, 20])
    ax = plt.subplot(3, 2, 1)
    ds.KOC.mean(['time', 'tracernum']).plot(norm=norm,
                                            vmin=clim[0],
                                            vmax=clim[1],
                                            ax=ax)
    plt.title('raw KOC')

    ax = plt.subplot(3, 2, 2)
    ds.KOC.where(ds.mask_count > p).\
        mean(['time', 'tracernum']).\
        plot(norm=norm, vmin=clim[0], vmax=clim[1], ax=ax)
    plt.title('KOC after landmask')

    ax = plt.subplot(3, 2, 3)
    ds.AveTracer.mean(['time', 'tracernum']).plot(robust=True, ax=ax)
    plt.title('raw KOC')

    ax = plt.subplot(3, 2, 4)
    ds.AveTracer.where(ds.mask_count > p).\
        mean(['time', 'tracernum']).\
        plot(robust=True, ax=ax)
    plt.title('KOC after landmask')

    ax = plt.subplot(3, 2, 5)
    masked = ds.AveTracer.where(ds.mask_count > p)
    zonal_mean = ds.AveTracer.where(ds.mask_count > p).mean('i')
    (masked-zonal_mean).mean(['time', 'tracernum']).plot(robust=True, ax=ax)
    plt.title('Masked KOC - zonal mean masked KOC')

    ax = plt.subplot(3, 2, 6)
    y_dummy = ds.KOC*0+1
    x = (ds.mask_count*y_dummy).data.flatten()
    y = ds.KOC.data.flatten()
    idx = np.logical_and(y > 10, y <= 2e4)

    heat = ax.hexbin(x[idx], y[idx],
                     bins='log',
                     cmap=plt.cm.Reds)

    ax.axvline(p)
    plt.colorbar(heat, label='log(occurence)', ax=ax)
    plt.title('Heatmap KOC vs mask_count')
    plt.xlabel('Criterion')
    plt.ylabel('KOC [m^2/s]')


def QC_validtime_plot(ds):
    clim = [100, 1e5]
    norm = matplotlib.colors.SymLogNorm(10)
    raw = ds.KOC.mean(['time', 'tracernum'])
    valid = ds.where(ds.valid_index).KOC.mean(['time', 'tracernum'])

    plt.figure(figsize=[20, 20])
    ax = plt.subplot(2, 2, 1)
    raw.plot(norm=norm, vmin=clim[0], vmax=clim[1], ax=ax)
    plt.title('raw KOC')

    ax = plt.subplot(2, 2, 2)
    valid.plot(norm=norm, vmin=clim[0], vmax=clim[1], ax=ax)
    plt.title('valid KOC')

    ax = plt.subplot(2, 2, 3)
    (raw-valid).plot(norm=norm, robust=True, ax=ax)
    plt.title('Difference raw-valid KOC')


def main(ddir, pdir, spin_up_time=3):
    """wrapper for all QC plots.

    PARAMETERS
    ----------
    ddir : path
        mitgcm run directory
    pdir : path
        output directory for plots
    spin_up_time = float
        spint up time in months, which will be eliminated
    """
    print('### Read in data ###')
    pdir = dirCheck(pdir, True)
    odir = ddir+'/output'
    Tr = tracer_engine(ddir)
    ds_di = Tr.dataset_readin(['tracer_diags'])
    ds_sn = Tr.dataset_readin(['tracer_snapshots'])
    ds_fi = xr.open_dataset(odir+'/KOC_FINAL.nc')

    # Should I be able to read this from a file?
    cut_time = spin_up_time*30*24*60*60
    tr_num = Tr.tracernum

    print('### reset QC ###')
    QC_reset_plot(ds_di, ds_sn, Tr, cut_time, tr_num)
    plt.gcf().savefig(pdir+'/QC_reset.png')
    plt.close()

    print('### Timeseries full QC ###')
    QC_global_timeseries_plot(ds_fi)
    plt.gcf().savefig(pdir+'/QC_global_timeseries_full.png')
    plt.close()

    print('### Timeseries valid QC ###')
    QC_global_timeseries_plot(ds_fi.where(ds_fi.valid_index))
    plt.gcf().savefig(pdir+'/QC_timeseries_valid.png')
    plt.close()

    print('### Valid time map QC ###')
    QC_validtime_plot(ds_fi)
    plt.gcf().savefig(pdir+'/QC_map_valid.png')
    plt.close()

    print('### Criterion QC ###')
    QC_crit_plot(ds_fi)
    plt.gcf().savefig(pdir+'/QC_crit.png')
    plt.close()

    print('### Landmask QC ###')
    QC_mask_plot(ds_fi)
    plt.gcf().savefig(pdir+'/QC_landmask.png')
    plt.close()

    # print 'TEST Calc combined'
    # KOC = xr.concat(pre_combined_KOC,'tracernum')
    # print 'TEST Calc Mean'
    # MEAN = KOC.KOC_valid.mean('tracernum')
    # print 'TEST Calc RMSE'
    # RMSE = (KOC.KOC_valid-MEAN)**2
    # RMSE = np.sqrt(RMSE.mean(dim=('tracernum','time')))
    # print 'TEST Calc Final'
    # FINAL = xr.Dataset({'KOC':MEAN,'RMSE':RMSE})
    #
    # # dummy = [TrCore.KOC('0'+str(a+1),directory=ddir,spin_up_months=spin_up_time,\
    # #                 low_grad_crit=low_grad,iters=iters,interval=TrCore.koc_interval,debug=False) for a in range(len(TrCore.diagname))]
# def MovieRegionalContours(ax,data,masks,refs,boxes=True,sref_shift = 0.2,linew=[10,25,60],linecolors=['k','k','b']):
#     for region_idx in range(len(masks)-1):
#         sref = refs[region_idx]
#         mask = masks[region_idx]
#         f_cut = ma.MaskedArray(data,mask=mask)
#         levels = np.array([-2,-1,0])*sref_shift + sref
#         ax.contour(f_cut,levels=levels,linewidths=linew,colors=linecolors)
#         if boxes:
#             ax.contour(mask,linestyles='--',linewidths=linew[1],colors=[tuple(i*0.2  for i in (1,1,1))])
#         plt.show()
#
#         def Movies(self):
#         for tt,tname in enumerate(self.tracername):
#             for ff,step in enumerate(self.tracer_steplist[tt]):
#                 f = mit.rdmds(self.ddir+tname, step)
#                 f = ma.MaskedArray(f,mask= f==0)
#                 dpi_factor = 1
#                 fig,ax = MovieFrame(f,dpi_factor=dpi_factor)
#                 MovieRegionalContours(ax,f,self.Mask,self.S_ref)
#                 ppdir = dirCheck(self.mdir+'/STANDARD/'+tname)
#                 fig.savefig(ppdir+'frame_%04d.png' % ff,dpi=dpi_factor)
#                 plt.close()
#                 if ff%200==0:
#                     print tname+' Printing Frame('+str(ff)+')'
#
# def MovieFrame(data,bg_color=None,clim =[32,38],w = 1280,h = 720,dpi_factor=1,cmap=None):
#             fig = plt.figure(frameon=False)
#             fig.set_size_inches(w/dpi_factor,h/dpi_factor)
#             ax = plt.Axes(fig, [0., 0., 1., 1.])
#             ax.set_axis_off()
#             fig.add_axes(ax)
#             if not cmap:
#                 cmap = plt.cm.RdYlBu_r
#             if not bg_color:
#                 bg_color=np.array([1,1,1])*0.1
#             cmap.set_bad(bg_color, 1)
#             ax.set_aspect(1, anchor = 'C')
#             hsss = ax.imshow(data,cmap=cmap,clim=clim,aspect='auto')
#             ax.invert_yaxis()
#             plt.show()
#             return fig,ax
#
# def KOCstackQC(datas,param,tr_num,norm='mean/std',debug=False):
#     print 'stacking input size',datas.shape
#     count = 0
#     for yy in range(datas.shape[1]):
#         for xx in range(datas.shape[2]):
#             pp = datas[:,yy,xx]
#
#             # subtract time median to get anomaly
#             pp_mean = np.nanmedian(pp)
#             pp_std = np.nanstd(pp)
#             if norm=='mean/std':
#                 pp = (pp-pp_mean)/pp_std
#             elif norm=='mean':
#                 pp = pp-pp_mean
#             elif norm=='std':
#                 pp = pp/pp_std
#
#             _,idx_b = ValidityIndex(param,
#                 tr_num,pp,0,0,timestyle='diagnostic',axis=0)
#
#             stacked = np.squeeze(stackdata(pp,idx_b))
#
#             # only retain the median of each stack
#             if len(stacked.shape)>1:
#                 stacked = np.nanmedian(stacked,axis=1)
#
#             stacked = stacked[np.newaxis,:]
#
#             if yy==0 and xx==0:
#                 out = stacked
#             else:
#                 out = np.concatenate((out,stacked),axis=0)
#     if debug:
#         print 'idx',len(idx_b),idx_b
#         print 'stacking output size',out.shape
#
#     out = np.transpose(out)
#     return out
#
# def KOC_QCPlots(combined_KOC,pdir,param,debug=False):
#     ## loop over the tracer numbers
#     for ii in range(2):
#         tr_num    = str(ii+1)
#         stack     = KOCstackQC(combined_KOC[ii].KOC_raw.data,param,tr_num)
#         stack_cut = KOCstackQC(combined_KOC[ii].KOC_valid.data,param,tr_num)
#         idx = np.all(np.isnan(stack_cut),axis=1)
#         stack_cut = stack.copy()
#         stack_cut[idx,:] = np.nan
#         if debug:
#             print 'stack',stack.shape
#
#         fig = plt.figure()
#         plt.title('Stacked KOC results')
#         plt.ylabel('normalized diffusivity')
#         plt.xlabel('samples')
#         plt.plot(np.nanmean(stack,axis=1),'--r')
#         plt.plot(np.nanstd(stack,axis=1),'--b')
#         plt.plot(np.nanmean(stack_cut,axis=1),'r')
#         plt.plot(np.nanstd(stack_cut,axis=1),'b')
#         plt.legend(['mean of all grid points','std of all grid points'],loc=4)
#         plt.show()
#         fig.savefig(pdir+'/Tracer_'+tr_num+'normalized_diffusivity.pdf')
#
#         # estimate ther error due to reset as the difference between the two
#         if len(combined_KOC)==2:
#             error = combined_KOC[1].KOC_raw-combined_KOC[0].KOC_raw
#             error_cut = combined_KOC[1].KOC_valid-combined_KOC[0].KOC_valid
#         else:
#             raise RuntimeError('more then 2 tracers...not supported')
#
#         stack_err = KOCstackQC(abs(error.data),param,tr_num,norm='none')
#         stack_err_cut = stack_err.copy()
#         stack_err_cut[idx,:] = np.nan
#
#         fig = plt.figure()
#         plt.plot(np.nanmean(stack_err,axis=1),'--r')
#         plt.plot(np.nanmean(stack_err_cut,axis=1),'r')
#         plt.ylabel('mean absolute error between tracers')
#         plt.xlabel('samples')
#         plt.show()
#         fig.savefig(pdir+'/Tracer_'+tr_num+'mean_error.pdf')
#
#         fig = plt.figure()
#         plt.plot(np.nanstd(stack_err,axis=1),'--b')
#         plt.plot(np.nanstd(stack_err_cut,axis=1),'b')
#         plt.ylabel('std absolute error between tracers')
#         plt.xlabel('samples')
#         plt.show()
#         fig.savefig(pdir+'/Tracer_'+tr_num+'std_error.pdf')
#
#
#         fig = plt.figure()
#         combined_KOC[ii].KOC_raw.mean(dim='time').plot()
#         plt.title('raw results tracer '+str(ii))
#         plt.show()
#         fig.savefig(pdir+'/Tracer_'+tr_num+'raw_map.pdf')
#
#         ### possibly add the raw rmse but not now
