#!/usr/bin/env python3.6

import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import date2num,DateFormatter
import readTRbostock
import glob
import pytz
import obspy 

# Import internals
from CascadiaUtils import readdata
from .Cppostproc import detection  

# ----------------------------------------------------------------------------------------------
def simpleplot(dates,Cps,Cpc,tremors=None,axs=None,label=None):
    '''
    Make a simple plot of Cp
    Args:
        * dates
        * Cps
        * Cpc
        * tremors: None by default
        * axs: list of two existing subplot axes (None by default)
    '''

    
    if axs is None:
        fig = plt.figure(figsize=(18,5))
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(212,sharex=ax1)

    else:
        ax1,ax2 = axs
        fig = ax1.get_figure()

    p=ax1.plot_date(dates,Cpc,'-',lw=0.5,label=label)
    xlims = ax1.get_xlim() # Get xlim
    ax1.set_ylabel('Cp comp')
    if tremors is not None:
        [ax1.axvline(d,c='r',lw=0.5) for d in tremors]
        ax1.plot_date(dates,Cpc,'-',color=p[0].get_color(),label=None)
    plt.legend() # in case there are many lines

    ax2.plot_date(dates,Cps,'-',lw=0.5)
    ax2.set_ylabel('Cp station')
    if tremors is not None:
        [ax2.axvline(d,c='r',lw=0.5) for d in tremors]
        ax2.plot_date(dates,Cps,'-',lw=0.5,color=p[0].get_color())

    ax1.set_xlim(xlims)

    ax2.set_xlabel('Time')
    ax2.xaxis.set_major_formatter(DateFormatter('%d/%m/%y %H:%M'))
    ax2.xaxis.set_tick_params(rotation=30)
    plt.tight_layout()
    plt.setp(ax1.get_xticklabels(),visible=False)

    return [fig,ax1,ax2]

# ----------------------------------------------------------------------------------------------
def averagedplot(dates,Cps,Cpc,window,tremors=None,axs=None,label=None):
    '''
    '''

    # Start and end of current event
    tb = dates[0]
    te = dates[-1]
    
    # first time window
    t0 = tb
    t1 = tb + datetime.timedelta(seconds=window)

    # Mid where to plot
    tmid = []
    mCpc = []
    mCps = []

    while t1 <= te: # While not at the end of the event
        tmid.append(t0 + datetime.timedelta(seconds=window/2.))
        ix = np.where((dates>=t0)&(dates<=t1))[0]
        mCpc.append(np.nanmean(Cpc[ix]))
        mCps.append(np.nanmean(Cps[ix]))
        t0 = t1
        t1 += datetime.timedelta(seconds=window)

    mCps = np.array(mCps)
    mCpc = np.array(mCpc)
   
    #code.interact(local = locals())
    
    # Start proper plot
    if axs is None:
        fig = plt.figure(figsize=(18,5))
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(212,sharex=ax1)

    else:
        ax1,ax2 = axs
        fig = ax1.get_figure()

    # Make Cp comp plot
    p=ax1.plot_date(tmid,mCpc,'-',lw=0.5)
    xlims = ax1.get_xlim() # Get xlim
    ax1.set_ylabel('Cp comp')
    if tremors is not None:
        [ax1.axvline(d,c='r',lw=0.5) for d in tremors]
        ax1.plot_date(tmid,mCpc,'-',lw=0.5,color=p[0].get_color(),label=None)
    plt.legend() # in case there are many lines

    # Make Cp stat plot
    ax2 = fig.add_subplot(212,sharex=ax1)
    ax2.plot_date(tmid,mCps,'-',lw=0.5)
    ax2.set_ylabel('Cp station')
    if tremors is not None:
        [ax2.axvline(d,c='r',lw=0.5) for d in tremors]
        ax2.plot_date(tmid,mCps,'-',lw=0.5,color=p[0].get_color())

    ax1.set_xlim(xlims)

    # Cosmetic
    ax2.set_xlabel('Time')
    ax2.xaxis.set_major_formatter(DateFormatter('%d/%m/%y %H:%M'))
    ax2.xaxis.set_tick_params(rotation=30)
    plt.tight_layout()
    plt.setp(ax1.get_xticklabels(),visible=False)

    return [fig,ax1,ax2]

# ----------------------------------------------------------------------------------------------
def strain_vs_Cp(dates,Cp,S,t1=None,t2=None,tremors=None):
    '''
    Make a plot with data on top and Cp at bottom
    Args:
        * dates, Cp : matrics gron readresults() or readH5results()
        * S         : obspy trace or list ['station','channel']
        * t1        : starting date 
        * t2        : ending date 
        * tremors   : dates from bostock catalogue [OPT]
    '''

    # Check something
    assert(type(S) is obspy.Stream),'3rd arg must be a obspy Stream of data'
    
    # Copy strain data for just for
    S2 = S.copy()

    if t1 is not None:
        # Set timezone info
        utc = pytz.UTC
        t1d = t1.datetime
        t2d = t2.datetime
        t1d = t1d.replace(tzinfo=utc)
        t2d = t2d.replace(tzinfo=utc)
        ixcp = np.where((dates>=t1d)&(dates<=t2d))[0]
        dates = dates[ixcp]
        Cp = Cp[ixcp]
        S2.trim(t1,t2)

    # Get time of straindata
    t = np.array([ti.datetime for ti in S2[0].times(type='utcdatetime')])

    # Make figure
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    axs = []
    for i in range(len(S2)):
        axs.append(ax1.twinx())

    # Plot tremors if provided
    if tremors is not None:
        ix = np.where((tremors>=dates[0])&(tremors<=dates[-1]))[0]
        [ax1.axvline(d,c='r',lw=0.8) for d in tremors[ix]]

    # Plot strain data
    k = 0
    c = ['r','orange','m']
    for ax,tr in zip(axs,S2):
        ax.plot_date(t,tr.data,'-',c=c[k],label=tr.stats.channel)
        k += 1

    # Plot Cp
    ax1.plot_date(dates,Cp,'k-',lw=2) 

    # Some cosmetics
    ax1.set_ylabel('Cp')
    ax1.set_xlabel('Time')
    ax1.set_zorder(axs[-1].get_zorder()+1)
    ax1.patch.set_visible(False)
    fig.tight_layout()

    # All done
    return

# ----------------------------------------------------------------------------------------------
def data_vs_Cp(dates,Cp,data,t1,t2,tremors=None):
    '''
    Make a plot with data on top and Cp at bottom
    Args:
        * dates, Cp : matrics gron readresults() or readH5results()
        * data      : obspy trace or list ['station','channel']
        * t1        : starting date 
        * t2        : ending date 
        * tremors   : dates from bostock catalogue [OPT]
    '''

    # Set timezone info
    utc = pytz.UTC
    t1d = t1.datetime
    t2d = t2.datetime
    t1d = t1d.replace(tzinfo=utc)
    t2d = t2d.replace(tzinfo=utc)

    # Create figure
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212,sharex=ax1)


    # Plot tremors if provided
    if tremors is not None:
        ix = np.where((tremors>=t1)&(tremors<=t2))[0]
        [ax1.axvline(d,c='r',lw=0.8) for d in tremors[ix]]
        [ax2.axvline(d,c='r',lw=0.8) for d in tremors[ix]]


    # Plot results
    if type(data) is list:
        S = readdata(t1,t2+3600.)
        d = S.select(station=data[0],channel=data[1])
        data = d[0]
    data.trim(t1,t2)
    t = np.arange(data.stats.starttime.datetime,data.stats.endtime+data.stats.delta,datetime.timedelta(seconds=data.stats.delta))
    ax1.plot_date(t,data.data,'k-',lw=0.2)

    # plot Cp
    ixcp = np.where((dates>=t1d)&(dates<=t2d))[0]
    ax2.plot_date(dates[ixcp],Cp[ixcp],'k-',lw=1) 

    # Some cosmetics
    ax1.set_ylabel('Counts')
    ax2.set_ylabel('Cp')
    ax2.set_xlabel('Time')

    fig.tight_layout()
    
    return
    


# ----------------------------------------------------------------------------------------------
def averagedplot(dates,Cps,Cpc,window,tremors=None,axs=None,label=None):
    '''
    '''

    # Start and end of current event
    tb = dates[0]
    te = dates[-1]
    
    # first time window
    t0 = tb
    t1 = tb + datetime.timedelta(seconds=window)

    # Mid where to plot
    tmid = []
    mCpc = []
    mCps = []

    while t1 <= te: # While not at the end of the event
        tmid.append(t0 + datetime.timedelta(seconds=window/2.))
        ix = np.where((dates>=t0)&(dates<=t1))[0]
        mCpc.append(np.nanmean(Cpc[ix]))
        mCps.append(np.nanmean(Cps[ix]))
        t0 = t1
        t1 += datetime.timedelta(seconds=window)

    mCps = np.array(mCps)
    mCpc = np.array(mCpc)
   
    #code.interact(local = locals())
    
    # Start proper plot
    if axs is None:
        fig = plt.figure(figsize=(18,5))
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(212,sharex=ax1)

    else:
        ax1,ax2 = axs
        fig = ax1.get_figure()

    # Make Cp comp plot
    p=ax1.plot_date(tmid,mCpc,'-',lw=0.5)
    xlims = ax1.get_xlim() # Get xlim
    ax1.set_ylabel('Cp comp')
    if tremors is not None:
        [ax1.axvline(d,c='r',lw=0.5) for d in tremors]
        ax1.plot_date(tmid,mCpc,'-',lw=0.5,color=p[0].get_color(),label=None)
    plt.legend() # in case there are many lines

    # Make Cp stat plot
    ax2 = fig.add_subplot(212,sharex=ax1)
    ax2.plot_date(tmid,mCps,'-',lw=0.5)
    ax2.set_ylabel('Cp station')
    if tremors is not None:
        [ax2.axvline(d,c='r',lw=0.5) for d in tremors]
        ax2.plot_date(tmid,mCps,'-',lw=0.5,color=p[0].get_color())

    ax1.set_xlim(xlims)

    # Cosmetic
    ax2.set_xlabel('Time')
    ax2.xaxis.set_major_formatter(DateFormatter('%d/%m/%y %H:%M'))
    ax2.xaxis.set_tick_params(rotation=30)
    plt.tight_layout()
    plt.setp(ax1.get_xticklabels(),visible=False)

    return [fig,ax1,ax2]


# ----------------------------------------------------------------------------------------------
def slidingwin(dates,Cps,Cpc,Np,tremors=None,axs=None,label=None):
    '''
    Make a sliding window plot of Cp
    Args:
        * dates
        * Cps
        * Cpc
        * Np: Number of point used for sliding windoz
        * tremors: None by default
        * axs: list of two existing subplot axes (None by default)
    '''

    Np = int(Np)
    Np2 = int(Np/2)

    dates = dates[Np2:-(Np2-1)]
    Cps = running_mean(Cps,Np)
    Cpc = running_mean(Cpc,Np)

    if axs is None:
        fig = plt.figure(figsize=(18,5))
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(212,sharex=ax1)

    else:
        ax1,ax2 = axs
        fig = ax1.get_figure()

    p=ax1.plot_date(dates,Cpc,'-',lw=0.5)
    xlims = ax1.get_xlim() # Get xlim
    ax1.set_ylabel('Cp comp')
    if tremors is not None:
        [ax1.axvline(d,c='r',lw=0.5) for d in tremors]
        ax1.plot_date(dates,Cpc,'-',color=p[0].get_color(),label=None)
    plt.legend() # in case there are many lines

    ax2.plot_date(dates,Cps,'-',lw=0.5)
    ax2.set_ylabel('Cp station')
    if tremors is not None:
        [ax2.axvline(d,c='r',lw=0.5) for d in tremors]
        ax2.plot_date(dates,Cps,'-',lw=0.5,color=p[0].get_color())

    ax1.set_xlim(xlims)

    ax2.set_xlabel('Time')
    ax2.xaxis.set_major_formatter(DateFormatter('%d/%m/%y %H:%M'))
    ax2.xaxis.set_tick_params(rotation=30)
    plt.tight_layout()
    plt.setp(ax1.get_xticklabels(),visible=False)

    return [fig,ax1,ax2]

# ----------------------------------------------------------------------------------------------
def running_mean(x, N):
    cumsum = np.nancumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# ----------------------------------------------------------------------------------------------
def plotdetection(dates,Cps,Cpc,window,thres,tremors=None,nt=None,axs=None,label=None):
    ''' 
    Number of detections per time windows
    '''

    tmid,mCps,mCpc = detection(dates,Cps,Cpc,window,thres)

    # Start and end of current event
    tb = dates[0]
    te = dates[-1]
    
    # first time window
    t0 = tb
    t1 = tb + datetime.timedelta(seconds=window)

    # Mid where to plot
    tmid = []
    mCpc = []
    mCps = []

    stds = np.nanstd(Cps)
    stdc = np.nanstd(Cpc)

    while t1 <= te: # While not at the end of the event
        tmid.append(t0 + datetime.timedelta(seconds=window/2.))
        ix = np.where((dates>=t0)&(dates<=t1))[0]
        mCps.append(len(np.where(Cps[ix]>=thres*stds)[0]))
        mCpc.append(len(np.where(Cpc[ix]>=thres*stdc)[0]))
        t0 = t1
        t1 += datetime.timedelta(seconds=window)


    # Start proper plot
    if axs is None:
        fig = plt.figure(figsize=(18,5))
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(212,sharex=ax1)

    else:
        ax1,ax2 = axs
        fig = ax1.get_figure()


    ax1.set_title('TEMPLATE {} -- Threshold = {} x std'.format(nt,thres))
    p=ax1.plot_date(tmid,mCpc,'-')
    xlims = ax1.get_xlim() # Get xlim
    ax1.set_ylabel('Nd (inter-compo Cp)')
    if tremors is not None:
        [ax1.axvline(d,c='r',lw=0.5,alpha=0.1) for d in tremors]
        ax1.plot_date(tmid,mCpc,'-',color=p[0].get_color())

    ax2 = fig.add_subplot(212,sharex=ax1)
    ax2.plot_date(tmid,mCps,'-')
    ax2.set_ylabel('Nd (inter-station Cp)')
    if tremors is not None:
        [ax2.axvline(d,c='r',lw=0.5,alpha=0.1) for d in tremors]
        ax2.plot_date(tmid,mCps,'-',color=p[0].get_color())

    ax1.set_xlim(xlims)

    ax2.set_xlabel('Time')
    ax2.xaxis.set_major_formatter(DateFormatter('%d/%m/%y %H:%M'))
    ax2.xaxis.set_tick_params(rotation=30)
    plt.tight_layout()
    plt.setp(ax1.get_xticklabels(),visible=False)

    return [fig,ax1,ax2]



# ----------------------------------------------------------------------------------------------
def cumulativedetection(dates,Cps,Cpc,window,thres,tremors=None,nt=None,axs=None,label=None):
    ''' 
    Number of detections per time windows
    '''

    tmid,mCps,mCpc = detection(dates,Cps,Cpc,window,thres)



    # Start proper plot
    if axs is None:
        fig = plt.figure(figsize=(18,5))
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(212,sharex=ax1)

    else:
        ax1,ax2 = axs
        fig = ax1.get_figure()


    ax1.set_title('TEMPLATE {} -- Threshold = {} x std'.format(nt,thres))
    p=ax1.plot_date(tmid,mCpc,'-')
    xlims = ax1.get_xlim() # Get xlim
    ax1.set_ylabel('Nd (inter-compo Cp)')
    if tremors is not None:
        [ax1.axvline(d,c='r',lw=0.5,alpha=0.1) for d in tremors]
        ax1.plot_date(tmid,mCpc,'-',color=p[0].get_color())

    ax2 = fig.add_subplot(212,sharex=ax1)
    ax2.plot_date(tmid,mCps,'-')
    ax2.set_ylabel('Nd (inter-station Cp)')
    if tremors is not None:
        [ax2.axvline(d,c='r',lw=0.5,alpha=0.1) for d in tremors]
        ax2.plot_date(tmid,mCps,'-',color=p[0].get_color())

    ax1.set_xlim(xlims)

    ax2.set_xlabel('Time')
    ax2.xaxis.set_major_formatter(DateFormatter('%d/%m/%y %H:%M'))
    ax2.xaxis.set_tick_params(rotation=30)
    plt.tight_layout()
    plt.setp(ax1.get_xticklabels(),visible=False)

    return [fig,ax1,ax2]


