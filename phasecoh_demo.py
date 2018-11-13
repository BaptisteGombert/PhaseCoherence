import numpy as np
import obspy
import os,glob,sys
import seisproc
import PhaseCoherence as PC
#import phasecoh
import time
import matplotlib.pyplot as plt
plt.ion()
plt.show()

# directory with data
# replace this with wherever the data is located now
ddir = os.path.join(os.environ['DATA'],'TREMOR','parkfieldtremor')

# read the files
fls = glob.glob(os.path.join(ddir,'*.SAC'))
st = obspy.Stream()
for fl in fls:
    st=st+obspy.read(fl)

# check for problematic intervals
for tr2 in st:
    if not isinstance(tr2.data,np.ma.masked_array):
        tr2.data=np.ma.masked_array(tr2.data,mask=False)
    tr2.data.mask=np.logical_or(tr2.data.mask,tr2.data==-12345)
seisproc.copyfromsacheader(st)

# frequency range
blim=[1,10]

# window for template, 
wintemp = [-0.2,3.8]

# we'll buffer by buftemp on either side of the template
# the template tapers to zero with a cosine taper within the buffer
buftemp = 1.

# times of window centers: every 2.5 seconds
# relative to reflook + shtemp + shtry
wlenlook=5.
dtl = 1.25
trange=[-500,500]
trange=[0,1000]
shift = []

tlook = np.arange(trange[0],trange[1],dtl)

# need to highpass to avoid aliasing
hpfilt=np.minimum(np.diff(wintemp)[0],wlenlook)
hpfilt=3/hpfilt

# filter and bandpass
msk=seisproc.prepfiltmask(st)
st.filter('bandpass',freqmin=hpfilt,freqmax=40)
st.resample(sampling_rate=100,no_filter=True)
msk.resample(sampling_rate=100,no_filter=True)
seisproc.addfiltmask(st,msk)

# relevant earthquake info
tref = obspy.UTCDateTime('2011-01-28T06:33:11.950000Z')

P = PC.PhaseCoherence('test',st)
P.ComputeAll(verbose=False,reftemp=None,shtemp='t3',wintemp=wintemp,
                    buftemp=buftemp,reflook=None,shlook=0.,
                    wlenlook=wlenlook,tlook=tlook,blim=blim)




sys.exit()





a = [0.1,0.15,0.2,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.]
for i in a: 
    tlook = np.arange(trange[0],trange[1],i)

    # need to highpass to avoid aliasing
    hpfilt=np.minimum(np.diff(wintemp)[0],wlenlook)
    hpfilt=3/hpfilt

    # filter and bandpass
    msk=seisproc.prepfiltmask(st)
    st.filter('bandpass',freqmin=hpfilt,freqmax=40)
    st.resample(sampling_rate=100,no_filter=True)
    msk.resample(sampling_rate=100,no_filter=True)
    seisproc.addfiltmask(st,msk)

    # relevant earthquake info
    tref = obspy.UTCDateTime('2011-01-28T06:33:11.950000Z')

    P = PC.PhaseCoherence('test',st)
    P.ComputeAll(verbose=False,reftemp=None,shtemp='t3',wintemp=wintemp,
                        buftemp=buftemp,reflook=None,shlook='t3',
                        wlenlook=wlenlook,tlook=tlook,blim=blim)

    plt.plot(tlook,P.Cp['Cpcomp'])
    shift.append(tlook[np.nanargmax(P.Cp['Cpcomp'])])
    print('{} : {}'.format(i,tlook[np.nanargmax(P.Cp['Cpcomp'])]))


'''
t0 = time.time()

P = PC.PhaseCoherence('test',st)
P.PrepareData()
for i in range(10):
    #P = PC.PhaseCoherence('test',st)
    P.ComputePhaseCoherence(verbose=False,reftemp=None,shtemp='t3',wintemp=wintemp,
                                   buftemp=buftemp,reflook=None,shlook='t3',
                                   wlenlook=wlenlook,tlook=tlook,blim=blim)
t1 = time.time()
print('new code --> {}'.format(t1-t0))

t0 = time.time()
for i in range(10):
    Cp=phasecoh.phasecoh(st,reftemp=None,shtemp='t3',wintemp=wintemp,
                     buftemp=buftemp,st2=st,reflook=None,shlook='t3',
                     wlenlook=wlenlook,tlook=tlook,blim=blim)
t2 = time.time()
print('old code --> {}'.format(t2-t0))

'''

