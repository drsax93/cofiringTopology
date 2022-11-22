import sys
import numpy as np
import pandas as pd # data manipulation and analysis
from scipy import stats
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns 


#########################################
# NESTED SUBPLOTS

def nested_subplots(fig=None, r1=(1,2), r2=(2,1), hspace=.2, wspace=.2):
    # input fig if an existing figure is to be used
    # r1 is the ratio for the first subplot
    # r2 is the ratio for the second subplot
    # set common figure - first subplot frame
    if not fig: fig = plt.figure(figsize=(8,4))
    gs0 = gridspec.GridSpec(r1[0], r1[1], figure=fig, hspace=hspace, wspace=wspace)
    axs = []
    for gsi in gs0:
        # create nested subplots 1
        gs00 = gridspec.GridSpecFromSubplotSpec(r2[0], r2[1], subplot_spec=gsi)
        axs_ = []
        for axi,gsii in enumerate(gs00):
            ax_ = fig.add_subplot(gs00[axi, :])
            axs_.append(ax_)
        axs.append(axs_)
    # return 2 pairs of axes
    return axs


#########################################
# SWARMPLOT

def swarmplot(df, indeces, ax, vertical=1, spread=3, trend=1, operation=np.mean,
              paired=False, SWARM=1, swarmPlot_kw={}, trendPlot_kw={},
              color_palette=None):
    ### PLOTTING STYLE PARAMETERS
    if not color_palette: color_palette = sns.set_palette('bright',100)
    nCols = 0 # total number of groups and samples
    for l in indeces: nCols+=len(l)
    ns = len(df)
    # process keyword args
    try: swarmPlot_kw['label'] # swarmplot style
    except: swarmPlot_kw['label'] = 'Swarm plot'
    try: swarmPlot_kw['m']
    except: swarmPlot_kw['m'] = '.'
    try: swarmPlot_kw['mfc']
    except: swarmPlot_kw['mfc'] = color_palette
    try: swarmPlot_kw['err_width']
    except: swarmPlot_kw['err_width'] = 2
    try: swarmPlot_kw['alpha']
    except: swarmPlot_kw['alpha'] = .5
    try: swarmPlot_kw['xticks']
    except: swarmPlot_kw['xticks'] = True
    if trend: # trend line style
        try: trendPlot_kw['color']
        except: trendPlot_kw['color'] = [.5,.5,.5]
        try: trendPlot_kw['style']
        except: trendPlot_kw['style'] = '-'
        try: trendPlot_kw['alpha']
        except: trendPlot_kw['alpha'] = .5
                
    # initialise storing and looping variables
    xticks = []; xlab = []
    x_offset = 0
    x_ind = 0
    
    # Work
    for index in indeces: # loop over lists of groups (multiple controls)
        nC = len(index)              
        ym = []; yy = []; xm = []; xx = []
        # marker style option
        if len(swarmPlot_kw['m'])>1:
            markers = swarmPlot_kw['m']
        else: markers = [swarmPlot_kw['m']]*nCols
        # facecolor options
        if len(swarmPlot_kw['mfc'])>1:
            mfc = swarmPlot_kw['mfc']
        else: mfc = [swarmPlot_kw['mfc']]*nCols
        # loop over groups
        for n,i in enumerate(index): # loop over groups
            y_ = df[i]; y_ = y_[~np.isnan(y_)] # take nans out
            if not n and paired: nss = len(y_) # obtain number of ref samples
            xlab.append('N=%s'%y_.shape[0])
            if SWARM: # swarmplot - obtain envelope of histogram
                mag_y,_ = np.histogram(y_,bins=10)
                mag_y = np.interp(np.linspace(0, 10, len(y_)),
                                  range(0, len(mag_y)), mag_y)
                mag_y = mag_y / mag_y.max()# / 10 * np.log(len(y_))
            else: # scatter
                mag_y = 1
            rand_ = np.random.randn(len(y_))
            rand_ /= np.abs(rand_).max()*1.7
            off_x = mag_y/(.5*spread) * rand_ # random scattering amplitudes
            # plot
            x_ = (x_offset+n+.1) * np.ones(len(y_)) + off_x # x coords for scattering
            xticks.append(np.mean(x_))
            if mfc[x_ind+n]=='none':
                ax.plot(x_, y_, markers[n+x_ind],
                    markersize=swarmPlot_kw['s'],
                    color=color_palette[n+x_ind],
                    marker=markers[n+x_ind],
                    mfc=mfc[x_ind+n],
                    alpha=swarmPlot_kw['alpha'])
            else:
                ax.plot(x_, y_, markers[n+x_ind],
                    markersize=swarmPlot_kw['s'],
                    color=color_palette[n+x_ind],
                    marker=markers[n+x_ind],
                    alpha=swarmPlot_kw['alpha'])
            # plot STD bar
            if vertical:
                off_center = 1/15 * np.std(y_) # half width of the white space between std lines
                ax.vlines([x_offset+n + np.max(off_x)+0.2]*2,
                          [np.mean(y_)-np.std(y_), np.mean(y_)+off_center],
                          [np.mean(y_)-off_center, np.mean(y_)+np.std(y_)],
                         color=color_palette[n+x_ind],
                         linewidth=swarmPlot_kw['err_width'])
            # store variables for trend plot
            ym.append(operation(y_)); xm.append(x_.mean())
            xx.append(x_); yy.append(y_)
        x_offset+=n+1.1
        x_ind+=n+1
        
        if paired and trend: # paired plot
            for n in range(nss):
                x2p = [xx[i][n] for i in range(nC)]
                y2p = [yy[i][n] for i in range(nC)]
                ax.plot(x2p, y2p, color=trendPlot_kw['color'],
                       linestyle=trendPlot_kw['style'],
                       alpha=trendPlot_kw['alpha'])
        elif trend: # plot trend line
            ax.plot(xm, ym, color=trendPlot_kw['color'],
                   linestyle=trendPlot_kw['style'],
                   alpha=trendPlot_kw['alpha'])
    # set axis label and lims
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlab)
    ax.set_ylabel(swarmPlot_kw['label'])
    miny = np.nanmin(df); maxy = np.nanmax(df)
    eps = (maxy - miny)/10
    ax.set_ylim(miny-eps, maxy+eps)
    sns.despine(ax=ax)
    if vertical:
        sns.despine(ax=ax, bottom=True)
        if not swarmPlot_kw['xticks']: ax.set_xticks([])
    return ax


#########################################
# BOOTSTRAP ESTIMATION PLOT

def bootstrap(x, nsh=10000, operation=np.mean):
    mean = []
    x_ = x; x_ = x_[~np.isnan(x_)]
    for n in range(nsh):
        xm = np.random.choice(x_,len(x_))
        mean.append(operation(xm))
    return np.asarray(mean)

def confInt(x,interval):
    # Calculate confident interval given by the extremes in 'interval' of array 'x'
    lenx = len(x[~np.isnan(x)])
    mean = np.nanmean(x)
    SEM = stats.sem(x, nan_policy='omit'); # Standard Error of the Mean
    ts = stats.t.ppf((1+interval)/2, lenx-1); # T-Score
    CI = ts*SEM # Confidence Intervals
    return mean-CI, mean+CI

def jackknife_indexes(data):
    # Taken without modification from scikits.bootstrap package.
    """
    From the scikits.bootstrap package.
    Given an array, returns a list of arrays where each array is a set of
    jackknife indexes.
    For a given set of data Y, the jackknife sample J[i] is defined as the
    data set Y with the ith data point deleted.
    """

    base = np.arange(0,len(data))
    return (np.delete(base,i) for i in base)

def bca(data, alphas, statarray, statfunction, ostat, reps):
    '''
    Taken from DABEST:
    Subroutine called to calculate the BCa statistics.
    Borrowed heavily from scikits.bootstrap code.
    '''
    import warnings

    # The bias correction value.
    s0 = 1.0*np.sum(statarray < ostat, axis = 0) / reps
    z0 = stats.norm.ppf(s0)

    # Statistics of the jackknife distribution
    jackindexes = jackknife_indexes(data[0])
    jstat = [statfunction(*(x[indexes] for x in data))
            for indexes in jackindexes]
    jmean = np.mean(jstat,axis = 0)

    # Acceleration value
    a = np.divide(np.sum( (jmean - jstat)**3, axis = 0 ),
        ( 6.0 * np.sum( (jmean - jstat)**2, axis = 0)**1.5 )
        )
    if np.any(np.isnan(a)):
        nanind = np.nonzero(np.isnan(a))
        warnings.warn("Some acceleration values were undefined."
        "This is almost certainly because all values"
        "for the statistic were equal. Affected"
        "confidence intervals will have zero width and"
        "may be inaccurate (indexes: {})".format(nanind))
    zs = z0 + stats.norm.ppf(alphas).reshape(alphas.shape+(1,)*z0.ndim)
    avals = stats.norm.cdf(z0 + zs/(1-a*zs))
    nvals = np.round((reps-1)*avals)
    nvals = np.nan_to_num(nvals).astype('int')
    # obtain bca p value by running correction for various alphas
    ns = []
    for a_ in np.linspace(1e-10,1-1e-10,reps)[::-1]:
        zs = z0 + stats.norm.ppf(a_).reshape(a_.shape+(1,)*z0.ndim)
        avals = stats.norm.cdf(z0 + zs/(1-a*zs))
        nvals_ = np.round((reps-1)*avals)
        ns.append(np.nan_to_num(nvals_).astype('int'))
    ci_alphas = np.asarray([statarray[ind] for ind in ns])
    if np.all(ci_alphas>0): p_bca = 0
    else: p_bca = 1 - np.where(ci_alphas<0)[0][0] / reps
    return nvals , p_bca

def bootstrap_plot(df, indeces, ax, operation=np.mean, nsh=10000, vertical=1,
                    paired=False, BCA=True, nbins=100, ci=.95, spread=5, SMOOTH=[1,3],
                   bootPlot_kw={}, color_palette=None, lbl_rot=0):
    ### PLOTTING STYLE PARAMETERS
    nCols = 0 # total number of groups and samples
    col_ids = []
    for l in indeces:
        nCols+=len(l)
        col_ids.extend(l)
    ns = len(df)
    # process keyword args
    if not color_palette: color_palette = sns.set_palette('bright',100)
    if operation==np.mean:
        bootPlot_kw['label'] = 'Mean $\Delta$'
    elif operation==np.median:
        bootPlot_kw['label'] = 'Median $\Delta$'
    else:
        bootPlot_kw['label'] = 'Other difference'
    try:
        bootPlot_kw['ci_size']
    except:
        bootPlot_kw['ci_size'] = 4  # size of black dot
    try:
        bootPlot_kw['ci_width']
    except:
        bootPlot_kw['ci_width'] = 2 # width of ci line
    try:
        bootPlot_kw['ref_width']
    except:
        bootPlot_kw['ref_width'] = 2 # width of ref line
    try:
        bootPlot_kw['ref_style']
    except:
        bootPlot_kw['ref_style'] = '--' # style of ref line
    
    # SEt global params and initialise variables
    alphas = np.array([(1-ci)/2., 1-(1-ci)/2.]) # conf interval
    x_offset = 0;  # x-axis offset for multiple controls
    x_ind = 1 # index for colors etc
    min_bc = []; max_bc = [] # mins and max values for y axis lims
    m_b = []; ci_b = [] # mean and ci of bootstrapped difference distribution
    p = [] # p-value estimated from bootstrap resampling
    xticks = []
    
    # Work
    for index in indeces: # loop over lists of groups (multiple controls)
        nC = len(index)
        # plot control sample
        ref =  df[index[0]]
        # obtain stats for ref distribution
        ref = np.asarray(ref[~np.isnan(ref)])
        ref = np.asarray(ref[~np.isinf(ref)])
        m_ref = bootstrap(ref, nsh=nsh, operation=operation) # bootstrap
        if paired: offset = 0
        else: offset = m_ref.mean()
        # obtain stats for ref distribution
        xticks.append(x_offset+.1)
        ax.plot(x_offset+.1, 0, 'ko', markersize=bootPlot_kw['ci_size'])
        start = x_offset+.1; fin = x_offset + nC-1 + 1/spread
        ax.hlines(0, start, fin, linewidth=bootPlot_kw['ref_width'],
                  linestyle=bootPlot_kw['ref_style'], color='k')
        x_offset+=1
        m_b.append([]); ci_b.append([]); p.append([])
        for n, i in enumerate(index[1:]): # loop over test groups
            y_ = df[i]
            y_ = np.asarray(y_[~np.isnan(y_)]) # exclude possible nans if unpaired analysis
            y_ = np.asarray(y_[~np.isinf(y_)]) # exclude possible infs (from log transforms)
            if paired:
                m_ = bootstrap(y_-m_ref.mean(), nsh=nsh, operation=operation) # paired diff bootstrap
            else:
                m_ = bootstrap(y_, nsh=nsh, operation=operation) # bootstrap
            m_h = np.histogram(m_, bins=nbins)
            # obtain the centres of the hist bins
            m_binCentres = []
            for mn in range(len(m_h[0])): 
                m_binCentres.append(np.mean([m_h[1][mn+1], m_h[1][mn]]))
            m_binCentres = np.asarray(m_binCentres)
            min_bc.append(np.min(m_binCentres)-offset)
            max_bc.append(np.max(m_binCentres)-offset)
            if SMOOTH[0]: # smooth dist wit gaussian
                m_h = gaussian_filter1d(m_h[0], SMOOTH[1])
            # obtain normalised dist to fit the swarmplot spread
            if len(index)>2: # if only two groups to be compared
                m_pdf = m_h / (np.max(m_h) * (1.2*spread))
            else:
                m_pdf = m_h / (np.max(m_h) * (.9*spread))
            m_pdf[0] = 0; m_pdf[-1] = 0 # make sure distribution touches CI line
            # find confident interval - take samples from sorted dist
            ci_ind = np.round((nsh - nsh*ci)/2).astype(int)
            m_sort = np.sort(m_)
            CI = np.array([m_sort[ci_ind],m_sort[-ci_ind]]) - offset
            # obtain bias corrected estimate
            if paired: bootdiff = m_
            else: bootdiff = m_ - m_ref.mean()
            if BCA:
                tdata = (bootdiff, )
                bootsort = bootdiff.copy()
                bootsort.sort()
                summ_stat = operation(y_) - m_ref.mean() # simple difference
                bca_ind,p_bca = bca(tdata, alphas, bootsort,
                           operation, summ_stat, nsh)
                CI_ = bootsort[bca_ind]
                ci_ratio = np.abs(np.diff(CI_)) / np.abs(np.diff(CI))
                if ci_ratio==0: # bca correction failed
                    CI_ = CI
                    ci_ratio = 1
            else:
                CI_ = CI
                ci_ratio = 1
            # obtain p value of bootstrap resample wo bca
            ratio_ = np.sum(bootdiff>0) / nsh # ratio of how many samples are below the ref
            if ratio_>.5: p_ = 1 - ratio_
            else: p_ = ratio_
            # Plot distribution
            m_binCentres = (m_binCentres - m_binCentres.mean()) *\
                             np.sqrt(ci_ratio) + m_binCentres.mean() # scale wrt bca
            # m_pdf = m_pdf * np.sqrt(ci_ratio)
            m_b[-1].append(m_.mean()-offset)
            ci_b[-1].append([CI_[0], CI_[1]])
            p[-1].append([p_,p_bca])
            xticks.append(n+x_offset+.1)
            ax.plot(n+x_offset+.1, m_b[-1][-1], 'ko',
                    markersize=bootPlot_kw['ci_size']) # plot black dot
            ax.fill(m_pdf + n+x_offset+.1, m_binCentres-offset, color=color_palette[n+x_ind]) # plot dist
            ax.vlines(n+x_offset+.1, CI_[0], CI_[1], color='k',
                      linewidth=bootPlot_kw['ci_width']) # plot CI
            n_off = n # save reference for offset
        # increase offsets / counters
        try: x_offset+=n_off+1.1; x_ind+=n_off+2
        except: x_offset+=1.1; x_ind+=1
    # labels and axes lims
    ax.set_xticks(xticks)
    ax.set_xticklabels(col_ids, rotation=lbl_rot)
    ax.set_ylabel(bootPlot_kw['label'])
    miny = np.min([-0.005, np.min(min_bc)])
    maxy = np.max([0.005, np.max(max_bc)])
    eps = (maxy - miny)/10
    ax.set_ylim(miny-eps, maxy+eps)
    if vertical:
        sns.despine(ax=ax, trim=True)
    else:
        sns.despine(ax=ax, left=True, right=False)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        
    return ax, m_b, ci_b, p



#########################################
# MAIN FUNCTION THAT PUTS THE TWO TOGETHER

def estimation_plot(input_, indeces=None, vertical=1, EXC=0, trend=1, spread=3, paired=False,
                    operation=np.mean, SWARM=1, nsh=5000, ci=.95, nbins=100, BCA=True,
                    SMOOTH=[1,3], swarmPlot_kw={}, bootPlot_kw={}, trendPlot_kw={},
                    lbl_rot=0, color_palette=None, FontScale=1, axs=None, figsize=None, stat=True):
    ''' INPUTS:
    - input_ = dict() containing the samples, indeces are labels
    - indeces = list of indeces used for multiple control analysis; 
            each list element contains the indeces of the samples to compare
            in each analysis -
            e.g. list(ind1,ind2) for 2 controls or list(ind) for just one control
            Default are all the keys/columns of input_
    - vertical = if true used a cumming's estimation layout, Gardner-Altman otherwise
    - EXC = if true display the distribution for the first class and set reference to 0
    - trend = if 0 plots no trend line
            if 1 plots the trend line bw mean of samples
            if >1 plots a trend line per sample (MAKE SURE DATA IS PAIRED)
    - spread = control spread of swarmplot and height of bootstrapped distribution
    - paired = set to True if data is paired
    - operation = specify which type of statistic to measure - e.g mean, median, ...
    - SWARM = set to 1 to plot a swarmplot, otherwise scatter uniformly
    - nsh = number of bootstrap samples
    - ci = confidence interval as ratio - e.g. .95
    - nbins = number of bins to estimate bootsrap distribution
    - BCA = use BCa correction 
    - SMOOTH = list of 2 elements, the first specifys whether to smooth the
            bootstrapped distribution, the second indicates the SD
    - swarmPlot_kw = keywords to modify the style of swarmPlot (to insert more)
            check individual function for more info
    - bootPlot_kw = keywords to modify the style of difference plot (to insert more)
            check individual function for more info
    - trendPlot_kw = keywords to modify the style of trend line plot
            check individual function for more info
    - lbl_rot = rotation param for x axis labels
    - color_palette = seaborn color_palette or list of colors to use
    - FontScale = seaborn font_scale parameter
    - figsize = size of the figure to plot as per plt figsize parameter
    - stat = set to False not to have mean and ci of the bootstrap distributions returned

    Swarmplot keyword args:
        try: swarmPlot_kw['label'] # swarmplot style
        except: swarmPlot_kw['label'] = 'Swarm plot'
        try: swarmPlot_kw['s']
        except: swarmPlot_kw['s'] = 7/np.log10(#samples)
        try: swarmPlot_kw['m']
        except: swarmPlot_kw['m'] = '.'
        try: swarmPlot_kw['mfc']
        except: swarmPlot_kw['mfc'] = color_palette
        try: swarmPlot_kw['err_width']
        except: swarmPlot_kw['err_width'] = nCols/2
        try: swarmPlot_kw['alpha']
        except: swarmPlot_kw['alpha'] = .5
        try: swarmPlot_kw['xticks']
        except: swarmPlot_kw['xticks'] = True
    
    Trendplot keyword arguments:
        try: trendPlot_kw['color']
        except: trendPlot_kw['color'] = [.5,.5,.5]
        try: trendPlot_kw['style']
        except: trendPlot_kw['style'] = '-'
        try: trendPlot_kw['alpha']
        except: trendPlot_kw['alpha'] = .5

    Bootplot keyword args:
        try: bootPlot_kw['ci_size']
        except: bootPlot_kw['ci_size'] = nCols*1.5  # size of black dot
        try: bootPlot_kw['ci_width']
        except: bootPlot_kw['ci_width'] = nCols/2 # width of ci line
        try: bootPlot_kw['ref_width']
        except: bootPlot_kw['ref_width'] = nCols/2 # width of ref line
        try: bootPlot_kw['ref_ls']
        except: bootPlot_kw['ref_style'] = '--' # style of ref line

    ------------
    
    OUTPUTS:
    fig, [m,ci], p-values = figure object and optionally mean, confidence interval
                            and p values of bootstrap estimation
    
    ------------
    
    EXAMPLES:
    - Unpaired example:

    import difference_estimation_plot as dpl
    input_ = {'sample 1': np.random.rand(100), 'sample 2': np.random.rand(90) + 0.4,
         'sample 3': np.random.rand(200) - 0.2}
    axs,m,p = dpl.estimation_plot(input_,)

    - No stats returned:
    
     import difference_estimation_plot as dpl
    input_ = {'sample 1': np.random.rand(100), 'sample 2': np.random.rand(90) + 0.4,
         'sample 3': np.random.rand(200) - 0.2}
    axs = dpl.estimation_plot(input_, stat=False)

    - Paired example:

    import difference_estimation_plot as dpl
    input_ = {'sample 1': np.random.rand(100), 'sample 2': np.random.rand(100) + 0.4,
         'sample 3': np.random.rand(100) - 0.2}
    axs,m,p = dpl.estimation_plot(input_, trend=1, paired=True) # links displayed
    axs,m,p = dpl.estimation_plot(input_, trend=0, paired=True) # no links displayed

    - Median difference example:

    import difference_estimation_plot as dpl
    input_ = {'sample 1': np.random.rand(100), 'sample 2': np.random.rand(100) + 0.4,
         'sample 3': np.random.rand(100) - 0.2}
    KEYS = list(input_.keys())
    axs,m,p = dpl.estimation_plot(input_, [KEYS], trend=1, operation=np.median)
    
    - Multiple controls
    input_ = {'sample 1': np.random.rand(100), 'sample 2': np.random.rand(100) + 0.4,
             'sample 3': np.random.rand(100) - 0.2, 'sample 4': np.random.rand(100) - 0.1}
    KEYS = list(input_.keys())
    axs,m,p = dpl.estimation_plot(input_, [KEYS[:2], KEYS[2:]], trend=1)


    - Nested subplots example:
    
     input_ = {'sample 1': np.random.rand(100), 'sample 2': np.random.rand(100) + 0.4,
             'sample 3': np.random.rand(100) - 0.2, 'sample 4': np.random.rand(100) - 0.1}
     KEYS = list(input_.keys())
     # obtain nested axes
     axs_nested = dpl.nested_subplots()
     # first estimation plot
     axs, m, p = dpl.estimation_plot(input_, [KEYS[:2],KEYS[2:]], axs=axs_nested[0])
     # second estimation plot
     axs, m, p = dpl.estimation_plot(input_, [KEYS[:2],KEYS[2:]], axs=axs_nested[1])

    '''

    # Process input -- convert to dataframe
    if not indeces: indeces = [list(input_.keys())] # use all entries if not specified otherwise
    df_ = []
    for i in input_.keys():
        df_.append(pd.DataFrame({i:input_[i]}))
    df = pd.concat(df_, axis=1)
    nCols = 0 # total number of groups and samples
    for i in indeces: nCols+=len(i)
    ns = len(df)
    
    # Set up the figure
    sns.set(style='ticks', font_scale=FontScale)
    if not axs:
        if vertical: # Cumming's estimation plot
            if figsize==None: figsize = (1*nCols,4)
            fig, axs = plt.subplots(2,1, sharex=False, sharey=False,
              gridspec_kw={'wspace': 0.1}, figsize=figsize)
        elif not vertical and not EXC: # G-A plot
            if figsize==None: figsize = (4,5)
            fig, axs = plt.subplots(1,2, sharex=False, sharey=False,
                  gridspec_kw={'wspace': 0.1, 'width_ratios': [nCols,nCols-1]},
                  figsize=figsize)
        else: # G-A plot w exception
            if figsize==None: figsize = (5,4)
            fig, axs = plt.subplots(1,2, sharex=False, sharey=False,
                  gridspec_kw={'wspace': 0.1}, figsize=figsize)
    
    # Swarmplot
    try: swarmPlot_kw['s'] # set the marker size here because it's not passed correctly
    except: swarmPlot_kw['s'] = 7/np.log10(ns) # set marker size depending on sample size
    axs[0] = swarmplot(df, indeces, axs[0], vertical, spread=spread, trend=trend, paired=paired,
              operation=operation, swarmPlot_kw=swarmPlot_kw, trendPlot_kw=trendPlot_kw, 
              color_palette=color_palette)
    # Distribution plot
    axs[1], m_b, ci_b, p = bootstrap_plot(df, indeces, axs[1], spread=spread, ci=ci, nbins=nbins,
                                       paired=paired, operation=operation, SMOOTH=SMOOTH,
                                       vertical=vertical, BCA=BCA, nsh=nsh, lbl_rot=lbl_rot,
                                       bootPlot_kw=bootPlot_kw, color_palette=color_palette)
    
    # set common x axis limits
    if nCols==2:
        xlim = (-.1/spread * 1.2*(nCols), nCols-1 + 1.1/spread * 1.2*(nCols))
    else:
        xlim = (-.1/spread * 1.2*(nCols/2), nCols-1 + 1.1/spread * 1.2*(nCols/2))
    axs[0].set_xlim(xlim); axs[1].set_xlim(xlim)
    if not vertical and not EXC:
        axs[1].set_xlim(xlim[0]+1, xlim[1])
    
    if stat: return axs, [m_b, ci_b], p
    else: return axs

