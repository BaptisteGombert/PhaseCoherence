#!/usr/bin/env python3.6

import numpy as np
import datetime
import os
import sys
import code
import h5py
import glob
import pytz
import obspy
import pickle

# Import internals
import readTRbostock

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
def readH5results(NTEMPLATE,paf,days='all',dtime=True,prebuff=0,postbuff=None,mode='normal',\
                  fidformat='DAY.{:02d}*.h5'):

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
    utc=pytz.timezone('UTC')

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
        fpaf = os.path.join(paf,fidformat.format(d))
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
                    time = f[k]['Time'].value[()][prebuff:-postbuff] 
                else:
                    time = f[k]['Time'].value[()][prebuff:] 
        
                if dtime:
                    #d = [datetime.fromtimestamp(t-3600.) for t in time]
                    d = np.array([datetime.fromtimestamp(t).astimezone(utc) for t in time])
                else:
                    d = np.array(time)

                dates = np.append(dates,d)
            
                # Get phase coheremce           
                if postbuff is not None: 
                    Cps.append(f[k]['CpS'][()][prebuff:-postbuff]) #= np.append(Cps,f[k]['CpS'].value[buff:])
                    Cpc.append(f[k]['CpC'][()][prebuff:-postbuff]) #= np.append(Cpc,f[k]['CpC'].value[buff:])
                else:
                    Cps.append(f[k]['CpS'][()][prebuff:]) 
                    Cpc.append(f[k]['CpC'][()][prebuff:]) 
            f.close()


        elif len(keys[0].split('.'))==3: # If on interp mode: H{}.T{}.P{}
            # Get hours 
            hours = np.unique([int(k.split('.')[0][1:]) for k in keys])
            for h in hours:
                # list of points for 1 hour and 1 template
                keys2 = [s for s in list(keys) if 'H{:02d}'.format(h) in s] 
                # Get phase coheremce          
                if postbuff is not None:                  
                    c1 =  np.array([f[k2]['CpS'][()][prebuff:-postbuff] for k2 in keys2]).T
                    c2 =  np.array([f[k2]['CpC'][()][prebuff:-postbuff] for k2 in keys2]).T
                else:
                    c1 =  np.array([f[k2]['CpS'][()][prebuff:] for k2 in keys2]).T
                    c2 =  np.array([f[k2]['CpC'][()][prebuff:] for k2 in keys2]).T

                # Get size
                snew=Cps.shape[0]+len(c1)
                Cps = np.append(Cps,c1).reshape(snew,c1.shape[1])
                Cpc = np.append(Cpc,c2).reshape(snew,c1.shape[1])
                if postbuff is not None:                  
                    time = f[keys2[0]]['Time'][()][prebuff:-postbuff]
                else:
                    time = f[keys2[0]]['Time'][()][prebuff:]
    
                if dtime:
                    #d = np.array([datetime.fromtimestamp(t-3600.).astimezone(utc) for t in time])
                    d = np.array([datetime.fromtimestamp(t).astimezone(utc) for t in time])
                else:
                    d = np.array(time)

                dates = np.append(dates,d)
            f.close()
                    
    # Make arrays
    dates = np.array(dates)
    if len(keys[0].split('.'))==3:
        Cps = np.array(Cps)
        Cpc = np.array(Cpc)
    else:
        Cpc = np.array(Cpc).reshape(np.array(Cpc).size)
        Cps = np.array(Cps).reshape(np.array(Cps).size)

    # Sort dates
    ix = np.argsort(dates)
    dates = dates[ix]
    Cps   = Cps[ix]
    Cpc   = Cpc[ix]    

    # All done
    return dates,Cps,Cpc

# ----------------------------------------------------------------------------------------------
def readmergedH5(NTEMPLATE,resfile,dtime=True,prebuff=0,postbuff=0.,mode='normal'):
    '''
    Read results from a file which been constructed by mergeH5results()
    Args :
            * NTEMPLATE: Template to be read
            * resfile  : Where file is
            * dtime    : If True (default), return an array of datetime objects. Else timestamp
            * buff     : Remove data at the beginning of each hour (default=0)
            * mode     : Type of h5 structure: 'normal' or 'interp'

    Returns :
            * dates,Cps,Cpc
    '''

    # Get UTC timezone
    utc=pytz.timezone('UTC')

    key = '{:03d}'.format(NTEMPLATE)
    with h5py.File(resfile,'r') as fid:
        Cps = fid[key]['CpS'][()]
        Cpc = fid[key]['CpC'][()]
        ts  = fid['Time'][()]

    if dtime:
        dates = np.array([datetime.datetime.fromtimestamp(t).astimezone(utc) for t in ts])
    else:
        dates = np.array(ts)

    # Remove buffer
    t0 = dates[0]
    ixs = []
    while t0<dates[-1]:
        ix = [np.where(dates==t0)[0][0]+t for t in range(-postbuff,prebuff)]
        ixs.append(ix)
        t0 += datetime.timedelta(hours=1)

    ixs = np.array(ixs)
    ixs = ixs.reshape(ixs.size)
    val = np.ones(dates.shape,dtype='bool')
    val[ixs]=False

    if mode=='interp':
        Cps = Cps[val,:]
        Cpc = Cpc[val,:]
    else:
        Cps[~val] = np.nan
        Cpc[~val] = np.nan

    #dates = dates[val]
    
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
        elif type(ind)==int:
            out[key].create_dataset('CpS',data=Cps[:,ind])
            out[key].create_dataset('CpC',data=Cpc[:,ind])
        elif type(ind)==dict:
            out[key].create_dataset('CpS',data=Cps[:,ind[key]])
            out[key].create_dataset('CpC',data=Cpc[:,ind[key]])

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
            mCps.append(len(np.where(Cps[ix]>=thres*stds)[0]))
            mCpc.append(len(np.where(Cpc[ix]>=thres*stdc)[0]))
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
def getstd(template='all',paf='./',days='all',buff=0,mode='normal'):
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
            _,tCps,tCpc = readH5results(int(T),paf,days=days,dtime=True,prebuff=buff,postbuff=buff,mode=mode)
            Cps = np.append(Cps,tCps)
            Cpc = np.append(Cpc,tCpc)

    else:
        # Which template
        TEMPLATES = [1,2,3,5,6,7,10,12,15,17,19,20,21,22,23,26,30,31,32,34,36,37,38,40,41,43,45,47,49,52,53,55,58,61,62,63,65,66,68,70,74,76,78,99,101,102,113,121,125,127,132,141,142,144,145,147,152,154,156,158,159,162,176,181,191,230,231,232,233,234,240,241,242,243,244,245,246,247,248,250,251,252,253,254,255,256,257,258,259,260]+list(range(260,301))
        # Remove that weird template 
        TEMPLATES.remove(295)
        TEMPLATES = np.array(TEMPLATES)

        for T in TEMPLATES:
            _,tCps,tCpc = readH5results(int(T),paf,days=days,dtime=True,prebuff=buff,postbuff=buff,mode=mode)                      
            Cps = np.append(Cps,tCps)
            Cpc = np.append(Cpc,tCpc)

    # Get std
    stds = np.nanstd(Cps)
    stdc = np.nanstd(Cpc)

    # All done
    return stds,stdc

# ----------------------------------------------------------------------------------------------
def getpeaks(Cp,width=20,prominence=0.02,height=None,stdthres=2.5,win=11):
    '''
    Get peaks of Cp results using scipy.signal.find_peaks()
    Return indices of peaks and dictionnary of peaks properties
    
    Args :
            * Cp          : Phase-coherence time-serie
            * width       : Width of peak used for detection (in # of samples)
            * prominence  : prominence of peak used for detection
            * height      : Minimum threshold used to fin peaks (any peaks with 
                            max Cp value under this will be discarded. Def=None
            * stdthres    : if height is None, it will be set to stdthres*std (def=2.5)
            * win         : width of sliding window used to smooth Cp before peak detection (def=10)
   
   Return :
            * p      : indices of peaks
            * pmax   : indices of max value of peaks (can be diff to p due to smoothing effect)
            * prop   : dictionnary containing peak properties (i.e. width, prominence, bounds, etc...)
    '''
    
    # Import scipy functions that does all the heavy lifting
    from scipy.signal import find_peaks 
    
    # Import internal running_mean func
    from .PlotCp import running_mean

    # Get height value if not given
    if height is None:
        std   = np.nanstd(Cp[:20000]) # noise level before ETS
        mean  = np.nanmean(Cp[:20000]) # average Cp value
        height = mean + std*stdthres # get height value to feed to find_peaks

    # Smooth Cp for more accurate peak detection
    if win%2==0: # Make window odd number of simple for later simplicity (<== Baptiste doesn't know how to code)
        win+=1
    aCp = running_mean(Cp,win)
    # Make it the same length as original vector
    tmp = np.zeros(Cp.shape)
    tmp[int(win/2):int(win/2)+len(aCp)] = aCp
    aCp = tmp

    # Find peaks
    p,prop = find_peaks(aCp,width=width,prominence=prominence,height=height)

    # Find indices of the bounds of eahc peak and its max value
    bounds  = np.zeros((len(prop['right_ips']),2),dtype=int)
    peakval = np.zeros((len(p))) # Maximum of peaks
    pmax    = np.zeros((len(p)),dtype=int) # Argument of maximum value of each peak
    
    for k in range(len(bounds)): # loop on each peak
        l = prop['left_ips'][k] # left bound
        r = prop['right_ips'][k] # right bound
        bounds[k,0] = int(np.floor(l)) 
        bounds[k,1] = int(np.ceil(r)) 
        peakval[k] = np.nanmax(Cp[bounds[k,0]:bounds[k,1]])
        try:
            pmax[k] = np.nanargmax(Cp[bounds[k,0]:bounds[k,1]])+bounds[k,0]
        except ValueError:
            pmax[k] = p[k]

    prop['bounds'] = bounds # store it
    prop['peakvals'] = peakval # store it

    # All done
    return p,pmax,prop


# ----------------------------------------------------------------------------------------------
def getpeakstats(TEMPLATES,h5file,width=20,prominence=0.02,height=None,stdthres=2.5,win=11,save=False):
    '''
    Get and compile "detection" stats returned by find_peaks
    How many peaks? How long?
    Args:  
        [MANDATORY]
            * TEMPLATES    : list (or array) or int of LFE to use
            * h5file       : Result file
        [OPT]
            * width       : Width of peak used for detection (in # of samples)
            * prominence  : prominence of peak used for detection
            * height      : Minimum threshold used to fin peaks (any peaks with 
                            max Cp value under this will be discarded. Def=None
            * stdthres    : if height is None, it will be set to stdthres*std (def=2.5)
            * win         : width of sliding window used to smooth Cp before peak detection (def=10)
    '''
    
    import pdb

    # Transform into lists
    if type(TEMPLATES) is int:
        TEMPLATES = [TEMPLATES]
    if type(width) is not list:
        width = [width]
    l1 = len(width) 
    if type(prominence) is not list:
        prominence = [prominence]
    l2 = len(prominence)
    if type(height) is not list:
        height = [height] 
    l3 = len(height)
    if type(stdthres) is not list:
        stdthres = [stdthres]
    l4 = len(stdthres)
    if type(win) is not list:
        win = [win]       
    l5 = len(win)
   
    maxlen = np.max([l1,l2,l3,l4,l5])
    if (maxlen>1)&(not l1==l2==l3==l4==l5):
        if len(width)<maxlen:
            width = [width[0] for i in range(maxlen)]
        if len(prominence)<maxlen:
            prominence = [prominence[0] for i in range(maxlen)]
        if len(height)<maxlen:
            height = [height[0] for i in range(maxlen)]
        if len(stdthres)<maxlen:
            stdthres = [stdthres[0] for i in range(maxlen)]
        if len(win)<maxlen:
            win = [win[0] for i in range(maxlen)]

    # Get templates
    if TEMPLATES is 'all':
        TEMPLATES = [1,2,3,5,6,7,10,12,15,17,19,20,21,22,23,26,30,31,32,34,36,37,38,40,41,43,45,47,49,52,53,55,58,61,62,63,65,66,68,70,74,76,78,99,101,102,113,121,125,127,132,141,142,144,145,147,152,154,156,158,159,162,176,181,191,230,231,232,233,234,240,241,242,243,244,245,246,247,248,250,251,252,253,254,255,256,257,258,259,260]+list(range(260,301))
        # Remove that weird template 
        TEMPLATES.remove(295)
    TEMPLATES = np.array(TEMPLATES)

    # Make empty dictionnary
    properties = {}
    # Loop on templates
    for NT in TEMPLATES:
        # Start filling dictionnary
        key = '{:03d}'.format(NT)
        properties[key] = {}
        properties[key]['Cps'] = {}
        properties[key]['Cpc'] = {}
        properties[key]['widths'] = width
        properties[key]['prominences'] = prominence
        properties[key]['heights'] =  height
        properties[key]['stdthres'] = stdthres
        properties[key]['wins'] = win 
        

        for c in ['Cps','Cpc']:
            properties[key][c]['Npeaks'] = []
            properties[key][c]['durations'] = []
            properties[key][c]['Cpmax'] = []

        # Read data
        dates,Cps,Cpc = readmergedH5(NT,h5file,prebuff=20,postbuff=20,mode='normal') 
        dt = (dates[1]-dates[0]).seconds 

        # Get peak values
        for wid,pro,hei,std,wi in zip(width,prominence,height,stdthres,win):  
            pc,pmaxc,propc = getpeaks(Cpc,width=wid,prominence=pro,height=hei,stdthres=std,win=wi)
            ps,pmaxs,props = getpeaks(Cps,width=wid,prominence=pro,height=hei,stdthres=std,win=wi)

            # Store results in dico
            properties[key]['Cps']['Npeaks'].append(len(pmaxs))
            properties[key]['Cpc']['Npeaks'].append(len(pmaxc))
            properties[key]['Cps']['durations'].append(props['widths']*dt)
            properties[key]['Cpc']['durations'].append(propc['widths']*dt)
            properties[key]['Cps']['Cpmax'].append(Cps[pmaxs])
            properties[key]['Cpc']['Cpmax'].append(Cpc[pmaxc])

    Ntries = len(properties[key]['Cpc']['Npeaks']) # Hpw many parameters have been tried?

    # Concatenate results
    allevent = {}
    allevent['Cps'] = {}
    allevent['Cpc'] = {}
    allevent['widths'] = width
    allevent['prominences'] = prominence
    allevent['heights'] =  height
    allevent['stdthres'] = stdthres
    allevent['wins'] = win
    for c in ['Cps','Cpc']:
        allevent[c]['Npeaks'] = np.zeros((Ntries))
        allevent[c]['durations'] = [[] for p in range(Ntries)]#np.zeros((Ntries))
        allevent[c]['Cpmax'] = [[] for p in range(Ntries)]# np.zeros((Ntries))
    for NT in TEMPLATES:
        key = '{:03d}'.format(NT)
        for p in range(Ntries):
            #pdb.set_trace()
            allevent['Cps']['Npeaks'][p] += properties[key]['Cps']['Npeaks'][p]
            allevent['Cpc']['Npeaks'][p] += properties[key]['Cpc']['Npeaks'][p]
            allevent['Cps']['durations'][p] += list(properties[key]['Cps']['durations'][p])
            allevent['Cpc']['durations'][p] += list(properties[key]['Cpc']['durations'][p])
            allevent['Cps']['Cpmax'][p] += list(properties[key]['Cps']['Cpmax'][p])
            allevent['Cpc']['Cpmax'][p] += list(properties[key]['Cpc']['Cpmax'][p])

    # Save results
    '''
    if save:
        f1 = prefix+'prop.pkl'
        f2 = prefix+'allt.pkl'
        with open(f1,'wb') as pf1:
            pickle.dump(properties,pf1,protocol=pickle.HIGHEST_PROTOCOL)
        with open(f2,'wb') as pf2:
            pickle.dump(allevent,pf2,protocol=pickle.HIGHEST_PROTOCOL)
    '''

    return properties,allevent
