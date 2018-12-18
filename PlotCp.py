#!/usr/bin/env python3.6

import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import date2num,DateFormatter
import os
import sys
import readTRbostock
import code
import h5py
import glob

def readresults(NTEMPLATE,paf,NHOURS=550,buff=20,form=None):
    '''
    Read phase coherence results in paf
    '''

    dates = np.array(())
    Cps = np.array(())
    Cpc = np.array(())
    for i in range(NHOURS):
        try:
            if form is None:
                fid = os.path.join(paf,'PC_H{:03d}_T{:03d}.txt'.format(i,NTEMPLATE))
            else:
                fid = os.path.join(paf,form.format(i,NTEMPLATE))

            M = np.loadtxt(fid)[buff:,:]
            d = [datetime.datetime.fromtimestamp(int(M[i,0]-3600.)) for i in range(len(M))]
            Cps = np.append(Cps,M[:,2])
            Cpc = np.append(Cpc,M[:,1])
            dates = np.append(dates,d)
        except:
            continue

    return dates,Cps,Cpc

# ----------------------------------------------------------------------------------------------
def readH5results(NTEMPLATE,paf,days='all',dtime=True,prebuff=0,postbuff=None,mode='normal'):

    '''
    Read phase coherence results in paf stored in h5py files
    Args :
            * NTEMPLATE: Template to be read
            * paf      : Where files are
            * days     : Which days to read (default='all', provide list otherwise)
            * dtime    : If True (default), return an array of datetime objects. Else timestamp
            * buff     : Remove data at the beginning of each hour (default=0)
            * mode     : Type of h5 structure: 'normal' or 'interp'

    Returns :
            * dates,Cps,Cpc
    '''

    # get shortcut
    from datetime import datetime

    # Get day sof interests
    if days is 'all':
        days=range(6,25)
    elif type(days) is int:
        days=[days]
    else:
        assert(type(days) is list),'days must be "all", int, or list'

    dates = np.array(())
    if mode=='normal':
        Cps = []#np.array(())
        Cpc = []#np.array(())
    elif mode=='interp':
        Cps = np.array(())
        Cpc = np.array(())
    else:
        print('ERROR: mode arg must be interp or normal')
        sys.exit(0)
                
    # Get files
    files = []
    for d in days:
        fpaf = os.path.join(paf,'DAY.{:02d}*.h5'.format(d))
        files.append(glob.glob(fpaf))

    # Flatten list 
    files = [item for sub in files for item in sub]
    files.sort() # sort
    
    for fid in files:
        # open file
        f = h5py.File(fid,'r')
        
        # Get keys of template of interest
        keys = [s for s in list(f.keys()) if 'T{:03d}'.format(NTEMPLATE) in s]
        if len(keys[0].split('.'))==2: # If only one point per template and per hour
            for k in keys: # For each hour
                if postbuff is not None: 
                    # Get time
                    time = f[k]['Time'].value[prebuff:-postbuff] 
                else:
                    time = f[k]['Time'].value[prebuff:] 
        
                if dtime:
                    d = [datetime.fromtimestamp(t-3600.) for t in time]
                else:
                    d = np.array(time-3600.)
                dates = np.append(dates,d)
            
                # Get phase coheremce           
                if postbuff is not None: 
                    Cps.append(f[k]['CpS'].value[prebuff:-postbuff]) #= np.append(Cps,f[k]['CpS'].value[buff:])
                    Cpc.append(f[k]['CpC'].value[prebuff:-postbuff]) #= np.append(Cpc,f[k]['CpC'].value[buff:])
                else:
                    Cps.append(f[k]['CpS'].value[prebuff:]) 
                    Cpc.append(f[k]['CpC'].value[prebuff:]) 


        elif len(keys[0].split('.'))==3: # If on interp mode: H{}.T{}.P{}
            # Get hours 
            hours = np.unique([int(k.split('.')[0][1:]) for k in keys])
            for h in hours:
                # list of points for 1 hour and 1 template
                keys2 = [s for s in list(keys) if 'H{:02d}'.format(h) in s] 
                # Get phase coheremce          
                if postbuff is not None:                  
                    c1 =  np.array([f[k2]['CpS'].value[prebuff:-postbuff] for k2 in keys2]).T
                    c2 =  np.array([f[k2]['CpC'].value[prebuff:-postbuff] for k2 in keys2]).T
                else:
                    c1 =  np.array([f[k2]['CpS'].value[prebuff:] for k2 in keys2]).T
                    c2 =  np.array([f[k2]['CpC'].value[prebuff:] for k2 in keys2]).T

                # Get size
                snew=Cps.shape[0]+len(c1)
                Cps = np.append(Cps,c1).reshape(snew,c1.shape[1])
                Cpc = np.append(Cpc,c2).reshape(snew,c1.shape[1])
                if postbuff is not None:                  
                    time = f[keys2[0]]['Time'].value[prebuff:-postbuff]
                else:
                    time = f[keys2[0]]['Time'].value[prebuff:]
    
                if dtime:
                    d = np.array([datetime.fromtimestamp(t-3600.) for t in time])
                else:
                    d = np.array(time)

                dates = np.append(dates,d)

    # Make arrays
    dates = np.array(dates)
    Cps = np.array(Cps)
    Cpc = np.array(Cpc)

    # Sort dates
    ix = np.argsort(dates)
    dates = dates[ix]
    Cps   = Cps[ix]
    Cpc   = Cpc[ix]    

    # All done
    return dates,Cps,Cpc

# ----------------------------------------------------------------------------------------------
def mergeH5results(paf,outfile,templates='all', mode='normal',days='all',ind=None):
    '''
    Merge files from different times into a single bigass h5py file
    Args:

    '''

    # Which templates?
    if templates=='all':
        TEMPLATES = [1,2,3,5,6,7,10,12,15,17,19,20,21,22,23,26,30,31,32,34,36,37,38,40,41,43,45,47,49,52,53,55,58,61,62,63,65,66,68,70,74,76,78,99,101,102,113,121,125,127,132,141,142,144,145,147,152,154,156,158,159,162,176,181,191,230,231,232,233,234,240,241,242,243,244,245,246,247,248,250,251,252,253,254,255,256,257,258,259]+list(range(260,301))
        # Remove that weird template 
        TEMPLATES.remove(295)
        TEMPLATES = np.array(TEMPLATES)
    else:
        TEMPLATES = np.array(templates)


    # open out file and create datasets
    out = h5py.File(outfile,'w')

    for T in TEMPLATES: # Loop on templates
        print('Template {:03d}'.format(T))
        key='{:03d}'.format(T)
        # Read results
        d,Cps,Cpc = readH5results(T,paf,days=days,dtime=False,mode=mode)
        
        # If first template, create time vector dataset
        if T==TEMPLATES[0]:
            out.create_dataset('Time',data=np.array(d))
    
        out.create_group(key)
        if ind is None:
            out[key].create_dataset('CpS',data=Cps)
            out[key].create_dataset('CpC',data=Cpc)
        else:
            out[key].create_dataset('CpS',data=Cps[:,ind])
            out[key].create_dataset('CpC',data=Cpc[:,ind])

    out.close()
    
    # All done
    return


# ----------------------------------------------------------------------------------------------
def readBostock(NTEMPLATE):
    '''
    Read timing of Bostock templates
    '''

    # Read tremor catalogue
    iev,mags,nm,tms = readTRbostock.readTRbostock()
    ix = np.where(iev==NTEMPLATE)[0]
    Tdates = tms[ix]
    
    return Tdates

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
def getmaxcp(dates,Cps,Cpc,window,wlen=0.,mean=False):
    '''
    For a given time window, get the maximum phase coherence value
    Args:
        dates, Cps, Cpc : matrics gron readresults() or readH5results()
        window          : window on which to do computations (in sec)
        wlen            : length on the PC window computation (in sec)
                          If window is 120s and Phase coh. windows are 60s, then 
                          actual window is 180s long
    '''

    '''
    # Adapt time window if wlen is given
    if wlen is not None:
        assert(wlen<window),'wlen must be smaller than window you idiot'
    '''

    # Start and end of current event
    tb = dates[0]
    te = dates[-1]
    
    # first time window
    t0 = tb
    t1 = tb + datetime.timedelta(seconds=window)
    #t1 = tb + datetime.timedelta(seconds=(window+wlen/2.)
    ts = tb # Fo rmiddle window

    # Mid where to plot
    tmid = []
    mCpc = []
    mCps = []

    while t1 <= te: # While not at the end of the event

        tmid.append(t0 + datetime.timedelta(seconds=window/2.))
        #tm = ts + datetime.timedelta(seconds=(wlen+window)/2.)
        #ts += datetime.timedelta(seconds=(wlen+window))
        #tmid.append(tm)

        #ix = np.where((dates>=t0)&(dates<=t1-datetime.timedelta(seconds=wlen)))[0]
        ix = np.where((dates>=t0)&(dates<=t1-datetime.timedelta(seconds=wlen/2.)))[0]

        # Check if several Cp value for same time (i.e. interp mode)
        if len(Cps.shape)==1:        
            if mean:
                mCpc.append(np.nanmean(Cpc[ix]))
                mCps.append(np.nanmean(Cps[ix]))
            else:
                mCpc.append(np.nanmax(Cpc[ix]))
                mCps.append(np.nanmax(Cps[ix]))
        else:
            if mean:
                ds = [np.nanmean(Cps[ix,k]) for k in range(Cps.shape[1])]
                dc = [np.nanmean(Cpc[ix,k]) for k in range(Cpc.shape[1])]
            else:
                ds = [np.nanmax(Cps[ix,k]) for k in range(Cps.shape[1])]
                dc = [np.nanmax(Cpc[ix,k]) for k in range(Cpc.shape[1])]

            mCps.append(ds)
            mCpc.append(dc)

        t0 = t1
        t1 += datetime.timedelta(seconds=window)

    tmid = np.array(tmid)
    mCps = np.array(mCps)
    mCpc = np.array(mCpc)
   

    return tmid,mCps,mCpc

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
def detection(dates,Cps,Cpc,window,thres,std=None):
    ''' 
    Number of detections per time windows
    Args:
            * dates    : list or array of datetime
            * Cps,Cpc  : results of Phase Coh, same length of dates
            * window   : window length used to count detections
            * thres    : Threshold use to count detection (Cp>thres*STD)
            * [OPT] std: if None, std computed from Cps or Cpc, or can be 2d array
    
    Return:
            * tmid   : time at middle of windows        
            * mCps   : number of Cps detections in each windows
            * mCpc   : number of Cpc detections in each windows
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

    # Get standars deviation
    if std is None:
        stds = np.nanstd(Cps,axis=0)
        stdc = np.nanstd(Cpc,axis=0)
    else:
        if len(Cps.shape)==1:
            stds = std[0]
            stdc = std[1]
        else:
            stds = std[0]*np.ones(Cps.shape[1])
            stdc = std[1]*np.ones(Cpc.shape[1])

    # Compute detections
    while t1 <= te: # While not at the end of the event
        tmid.append(t0 + datetime.timedelta(seconds=window/2.))
        ix = np.where((dates>=t0)&(dates<=t1))[0]

        # Check if several Cp value for same time (i.e. interp mode)
        if len(Cps.shape)==1:
            mCps.append(len(np.where(Cps[ix,:]>=thres*stds)[0]))
            mCpc.append(len(np.where(Cpc[ix,:]>=thres*stdc)[0]))
        else:
            ds = [len(np.where(Cps[ix,k]>=thres*stds[k])[0]) for k in range(Cps.shape[1])]
            dc = [len(np.where(Cpc[ix,k]>=thres*stdc[k])[0]) for k in range(Cpc.shape[1])]
            mCps.append(ds)
            mCpc.append(dc)

        t0 = t1
        t1 += datetime.timedelta(seconds=window)
    
    # Make into arrays
    tmid = np.array(tmid)
    mCps = np.array(mCps)
    mCpc = np.array(mCpc)

    # All done
    return tmid,mCps,mCpc


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

# ----------------------------------------------------------------------------------------------
def getstd(paf,template='all'):
    '''
    get STD of PC results
    Args:
            * paf             : where to look for results
            * template ['all']: If not 'all', list of templates to read  
    
    Return:
            * STDs, STDc
    '''

    Cps = np.array(())
    Cpc = np.array(())
    
    if template is not 'all':
        for T in template:
            _,tCps,tCpc = readresults(int(T),'./',buff=0)          
            Cps = np.append(Cps,tCps)
            Cpc = np.append(Cpc,tCpc)

    else:
        # Which template
        TEMPLATES = [1,2,3,5,6,7,10,12,15,17,19,20,21,22,23,26,30,31,32,34,36,37,38,40,41,43,45,47,49,52,53,55,58,61,62,63,65,66,68,70,74,76,78,99,101,102,113,121,125,127,132,141,142,144,145,147,152,154,156,158,159,162,176,181,191,230,231,232,233,234,240,241,242,243,244,245,246,247,248,250,251,252,253,254,255,256,257,258,259,260]+list(range(260,301))
        # Remove that weird template 
        TEMPLATES.remove(295)
        TEMPLATES = np.array(TEMPLATES)

        for T in TEMPLATES:
            _,tCps,tCpc = readresults(int(T),'./',buff=0)          
            Cps = np.append(Cps,tCps)
            Cpc = np.append(Cpc,tCpc)

    # Get std
    stds = np.nanstd(Cps)
    stdc = np.nanstd(Cpc)

    # All done
    return stds,stdc
