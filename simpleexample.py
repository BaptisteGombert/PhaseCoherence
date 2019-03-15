#!/usr/bin/env python3.6

# Import externales
import numpy as np
import datetime
import obspy
import os
import copy
import sys

# Import internals
import PhaseCoherence as PC
from CascadiaUtils import *


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
    blim=[1,6]

    # window for template, 
    wintemp = [-0.2,4.8]

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
    D1 = obspy.read('path/to/your/data') 
    D1.merge()   
    D1.trim(t1,t2) # Select time of interest
     
    # Get template 
    T = obspy.read('path/to/your/template')
        
    # Get time shifts based on arrival time differences
    shifts = {}
    lini = 99999999.
    for tr in T:
        nid = tr.stats.network+'.'+tr.stats.station+'.'+tr.stats.channel
        shifts[nid] = tr.stats.t0
        if tr.stats.t0<lini:
            lini=tr.stats.t0
    # Make sure smallest time-shift is 0.
    for k in shifts.keys():
        shifts1[k] -= lini

    # Filter that 
    T.filter('bandpass',freqmin=hpfilt,freqmax=19)
    D.filter('bandpass',freqmin=hpfilt,freqmax=19)

    # Resample data and template
    T.resample(sampling_rate=40,no_filter=True)
    D.resample(sampling_rate=40,no_filter=True)

    # Compute PC
    P,t   = computePC(T,D,wintemp,buftemp,tlook,wlenlook,blim,None,shifts,None)
    
    '''
    Results are stored in a dictionnary
    P.Cp['Cpstat'] for inter-station phase coherence
    P.Cp['Cpcomp'] for inter-component phase coherence
    '''

