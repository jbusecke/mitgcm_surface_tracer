    # !!! needs to be integrated into the TracerViz routine
    # KOC_QCPlots(pre_combined_KOC,TrCore.pdir,TrCore.modelparameters)


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
def MovieRegionalContours(ax,data,masks,refs,boxes=True,sref_shift = 0.2,linew=[10,25,60],linecolors=['k','k','b']):
    for region_idx in range(len(masks)-1):
        sref = refs[region_idx]
        mask = masks[region_idx]
        f_cut = ma.MaskedArray(data,mask=mask)
        levels = np.array([-2,-1,0])*sref_shift + sref
        ax.contour(f_cut,levels=levels,linewidths=linew,colors=linecolors)
        if boxes:
            ax.contour(mask,linestyles='--',linewidths=linew[1],colors=[tuple(i*0.2  for i in (1,1,1))])
        plt.show()

        def Movies(self):
        for tt,tname in enumerate(self.tracername):
            for ff,step in enumerate(self.tracer_steplist[tt]):
                f = mit.rdmds(self.ddir+tname, step)
                f = ma.MaskedArray(f,mask= f==0)
                dpi_factor = 1
                fig,ax = MovieFrame(f,dpi_factor=dpi_factor)
                MovieRegionalContours(ax,f,self.Mask,self.S_ref)
                ppdir = dirCheck(self.mdir+'/STANDARD/'+tname)
                fig.savefig(ppdir+'frame_%04d.png' % ff,dpi=dpi_factor)
                plt.close()
                if ff%200==0:
                    print tname+' Printing Frame('+str(ff)+')'


def MovieFrame(data,bg_color=None,clim =[32,38],w = 1280,h = 720,dpi_factor=1,cmap=None):
            fig = plt.figure(frameon=False)
            fig.set_size_inches(w/dpi_factor,h/dpi_factor)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            if not cmap:
                cmap = plt.cm.RdYlBu_r
            if not bg_color:
                bg_color=np.array([1,1,1])*0.1
            cmap.set_bad(bg_color, 1)
            ax.set_aspect(1, anchor = 'C')
            hsss = ax.imshow(data,cmap=cmap,clim=clim,aspect='auto')
            ax.invert_yaxis()
            plt.show()
            return fig,ax


def KOCstackQC(datas,param,tr_num,norm='mean/std',debug=False):
    print 'stacking input size',datas.shape
    count = 0
    for yy in range(datas.shape[1]):
        for xx in range(datas.shape[2]):
            pp = datas[:,yy,xx]

            # subtract time median to get anomaly
            pp_mean = np.nanmedian(pp)
            pp_std = np.nanstd(pp)
            if norm=='mean/std':
                pp = (pp-pp_mean)/pp_std
            elif norm=='mean':
                pp = pp-pp_mean
            elif norm=='std':
                pp = pp/pp_std

            _,idx_b = ValidityIndex(param,
                tr_num,pp,0,0,timestyle='diagnostic',axis=0)

            stacked = np.squeeze(stackdata(pp,idx_b))

            # only retain the median of each stack
            if len(stacked.shape)>1:
                stacked = np.nanmedian(stacked,axis=1)

            stacked = stacked[np.newaxis,:]

            if yy==0 and xx==0:
                out = stacked
            else:
                out = np.concatenate((out,stacked),axis=0)
    if debug:
        print 'idx',len(idx_b),idx_b
        print 'stacking output size',out.shape

    out = np.transpose(out)
    return out

def KOC_QCPlots(combined_KOC,pdir,param,debug=False):
    ## loop over the tracer numbers
    for ii in range(2):
        tr_num    = str(ii+1)
        stack     = KOCstackQC(combined_KOC[ii].KOC_raw.data,param,tr_num)
        stack_cut = KOCstackQC(combined_KOC[ii].KOC_valid.data,param,tr_num)
        idx = np.all(np.isnan(stack_cut),axis=1)
        stack_cut = stack.copy()
        stack_cut[idx,:] = np.nan
        if debug:
            print 'stack',stack.shape

        fig = plt.figure()
        plt.title('Stacked KOC results')
        plt.ylabel('normalized diffusivity')
        plt.xlabel('samples')
        plt.plot(np.nanmean(stack,axis=1),'--r')
        plt.plot(np.nanstd(stack,axis=1),'--b')
        plt.plot(np.nanmean(stack_cut,axis=1),'r')
        plt.plot(np.nanstd(stack_cut,axis=1),'b')
        plt.legend(['mean of all grid points','std of all grid points'],loc=4)
        plt.show()
        fig.savefig(pdir+'/Tracer_'+tr_num+'normalized_diffusivity.pdf')

        # estimate ther error due to reset as the difference between the two
        if len(combined_KOC)==2:
            error = combined_KOC[1].KOC_raw-combined_KOC[0].KOC_raw
            error_cut = combined_KOC[1].KOC_valid-combined_KOC[0].KOC_valid
        else:
            raise RuntimeError('more then 2 tracers...not supported')

        stack_err = KOCstackQC(abs(error.data),param,tr_num,norm='none')
        stack_err_cut = stack_err.copy()
        stack_err_cut[idx,:] = np.nan

        fig = plt.figure()
        plt.plot(np.nanmean(stack_err,axis=1),'--r')
        plt.plot(np.nanmean(stack_err_cut,axis=1),'r')
        plt.ylabel('mean absolute error between tracers')
        plt.xlabel('samples')
        plt.show()
        fig.savefig(pdir+'/Tracer_'+tr_num+'mean_error.pdf')

        fig = plt.figure()
        plt.plot(np.nanstd(stack_err,axis=1),'--b')
        plt.plot(np.nanstd(stack_err_cut,axis=1),'b')
        plt.ylabel('std absolute error between tracers')
        plt.xlabel('samples')
        plt.show()
        fig.savefig(pdir+'/Tracer_'+tr_num+'std_error.pdf')


        fig = plt.figure()
        combined_KOC[ii].KOC_raw.mean(dim='time').plot()
        plt.title('raw results tracer '+str(ii))
        plt.show()
        fig.savefig(pdir+'/Tracer_'+tr_num+'raw_map.pdf')

        ### possibly add the raw rmse but not now
