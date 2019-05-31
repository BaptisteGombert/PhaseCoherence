#!/usr/bin/env python3.6

# Import externales
import numpy as np
import datetime
import obspy
import os
import copy
import sys
import matplotlib.pyplot as plt


# Import internals
import PhaseCoherence.PhaseCoherence as PC


##########################################################################################################
def computePC(template,data,wintemp,buftemp,tlook,wlenlook,blim,reftemp,shifts,reflook):
    '''
    Compute one PC per hour
    '''

    # Initialise Phsae coherence object
    P = PC.PhaseCoherence('test',template,data=data)

    # Prepare data for computation
    P.PrepareData()

    # Define few extra parameters
    shtemp='t0'; shgrid=None 
    shtry=shifts;
    shlook=0.
    
    # Set parameters in objects
    P.setParams(reftemp, shtemp, wintemp,buftemp, reflook,shlook, wlenlook, tlook, blim, shtry, shgrid)
    
    # Make cross-correlation
    P.crosscorrDT()

    # Make tapering
    Mw  = int(np.round(P.params['wlenlook']/P.dtim)) # length of window
    tap = np.hamming(Mw).reshape(Mw,1)
    
    # Taper firtst cross-correlation 
    P.taperCrosscorr(taper=tap)

    # Compute phase coherence
    P.computeCp()

    time = P.params['tlook']


    return P,time


##########################################################################################################
if __name__ == '__main__':

    # Define dates of interests
    t1 = obspy.UTCDateTime(2010,8,20,1,0)
    t2 = obspy.UTCDateTime(2010,8,20,2,0)

    # frequency range
    blim=[1,10]

    # window for template, 
    wintemp = [-0.2,7.8]

    # we'll buffer by buftemp on either side of the template
    # the template tapers to zero with a cosine taper within the buffer
    buftemp = 1.

    # times of window centers: every 6 seconds
    # relative to reflook + shtemp + shtry
    wlenlook=60. # Windows where PC is computed are wlenlook seconds long
    dtl = 6. # Windows are separated by dtl seconds
    trange=[0,3600] # We look at 3600. seconds of data
    shift = []
    tlook = np.arange(trange[0],trange[1],dtl)

    # need to highpass to avoid aliasing
    hpfilt=np.minimum(np.diff(wintemp)[0],wlenlook)
    hpfilt=3/hpfilt
    
    # Get data waveform
    D = obspy.read('path/to/your/data') 
    D.merge()   
    D.trim(t1,t2) # Select time of interest
     
    # Get template 
    T = obspy.read('path/to/your/template')
        
    # Get time shifts based on arrival time differences
    shifts = {}
    lini = 99999999.
    for tr in T:
        nid = tr.stats.network+'.'+tr.stats.station+'.'+tr.stats.channel[-1]
        shifts[nid] = tr.stats.sac['t0']
        if tr.stats.sac['t0']<lini:
            lini=tr.stats.sac['t0']
        tr.stats.t0=tr.stats.sac['t0']
    # Make sure smallest time-shift is 0.
    for k in shifts.keys():
        shifts[k] -= lini

    # Filter that 
    T.filter('bandpass',freqmin=hpfilt,freqmax=19)
    D.filter('bandpass',freqmin=hpfilt,freqmax=19)

    # Resample data and template
    T.resample(sampling_rate=40,no_filter=True)
    D.resample(sampling_rate=40,no_filter=True)

    # Compute PC
    P,t   = computePC(T,D,wintemp,buftemp,tlook,wlenlook,blim,None,shifts,None)
   
    # Plot results
    Tdates = [datetime.timedelta(seconds=tt) + D[0].stats.starttime.datetime for tt in t] 
    Ddates = [datetime.timedelta(seconds=tt) + D[0].stats.starttime.datetime for tt in D[0].times()] 
    
    f,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(Ddates,D[0].data,'k')
    ax1.set_ylabel('Data (counts)')
    ax2.plot(Tdates,P.Cp['Cpstat'],'r')
    ax2.spines['right'].set_edgecolor('r')

    ax1.set_xlabel('Time')
    ax2.set_ylabel('Phase coherence')
    ax2.yaxis.label.set_color('r')
    ax2.tick_params(axis='y',colors='r')

    f.savefig('Results.png',bbox_inches='tight')
    '''
    Results are stored in a dictionnary
    P.Cp['Cpstat'] for inter-station phase coherence
    P.Cp['Cpcomp'] for inter-component phase coherence
    '''

