''' 
A base class to perform Phase Coherence analysis 

NOW USED FOR TEST-DEV PURPOSES ONLY - DO NOT USE FOR CP ANALYSIS

'''

# Import classic external stuff
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy
import datetime
import code
import itertools

# Import seismo external stuff
import obspy
import spectrum

# Import J. personnal modules
import general


class PhaseCoherence(object):

    '''
    Class implementing the phase coherence method

    Args required:
        * name      : Instance Name
        * template  : Obspy stream for templates

    Args optionnal:
        * data      : Obspy stream of data set to search (default: st1)
        * lat0      : Origin latitude (for plot, default=0)
        * lon0      : Origin longitude (for plot, default=0)
    '''
    
    # -------------------------------------------------------------------------------
    # Initialize class #
    def __init__(self,name, template, data=None, lon0=None, lat0=None):

        '''
        Initialize main variables of the class
        '''
        
        # Save name
        assert(type(name) is str), 'name argument must be a string'
        self.name = name
        
        # Lon/Lat of reference.
        if lon0 is None:
            self.lon0 = 0.
        else:
            self.lon0 = lon0

        if lat0 is None:
            self.lat0 = 0.
        else:
            self.lat0 = lat0

        # Copy data and template with copuy.deepcopy() to avoid changing original data
        # Save template
        assert(type(template) is obspy.core.stream.Stream), 'template argument must be an obspy stream'        
        self.template = template.copy()

        # default data
        if data is None:
            self.data = template.copy()
        else:
            assert(type(template) is obspy.core.stream.Stream), 'data argument must be an obspy stream'                    
            self.data = data.copy()

    # -------------------------------------------------------------------------------        
    def PrepareData(self,verbose=False):
        '''
        This function just check streams, remove unused traces, resample, etc...
        '''

        # 1/ Get infos on stations, compo, and common traces
        self._getCommonTraces()

        # 2/ Keep only traces which are in common in template and data
        self._removeUnusedTraces(verbose=verbose)

        # 3/ Resample data if necessary
        self._resampleTraces(verbose=verbose)

        # 4/ Merge and mask data
        self.merge()
        
        return

    # -------------------------------------------------------------------------------        
    def ComputePhaseCoherence(self,reftemp=None, shtemp='t0', wintemp=None,\
                 buftemp=None, reflook=None, shlook='t0', wlenlook=5., tlook=None,\
                 blim=None, shtry=None, shgrid=None, taper=True, cptype='both',\
                 verbose=True):
        '''
        This function assumes you already prepare the data, makes te rest
         * reftemp   : reference time for templates (default: often ignored, 
                      but first start time of template[0])
        * shtemp    : Time shift for templates (default: 't0'). Either
                        (1) a string for a marker (then reftemp is ignored)
                        (2) time shifts since reftemp in seconds
        * wintemp   : Window since reference for template (default: [0,3.])
        * buftemp   : A buffer on either side of the template window (default: diff(wintemp)/6)
        * reflook   : Reference times for search data (default: often ignored, 
                      but first start time of st2[0])
        * shlook    : Time shift for data (default: 't0'). Either
                        (1) a string for a marker---in which case reflook is ignored
                        (2) time shifts since reflook in seconds
        * wlenlook  : Window length for search (default: 5.)
        * tlook     : Times to search (default: all available)
        * blim      : Bandlimit to consider (default: [1,10.])
        * shtry     : A dictionary of time shifts to try for each component
        * shgrid    : Time spacing to use for gridding (default: wlen/4)
        * cptype    : Which phase coherence to compute?
                            - 'stat' for inter-station PC
                            - 'comp' for inter-component PC
                            - 'both' (default) for both

        * verbose   : Let the code whisper sweet words into your ears...                   
        '''

        # 1/ Set parameters
        self.setParams(reftemp, shtemp, wintemp,buftemp, reflook,\
                      shlook, wlenlook, tlook, blim, shtry, shgrid)

        # 2/ Make template-data cross-correlation
        self.crosscorrDT()

        # 3/ Taper crosscor if wanted
        if taper:
            self.taperCrosscorr()
        
        # 4/ Compute Cp
        self.computeCp(cptype=cptype)
             
        # All done
        return


    # -------------------------------------------------------------------------------        
    def ComputeAll(self,reftemp=None, shtemp='t0', wintemp=None,\
                 buftemp=None, reflook=None, shlook='t0', wlenlook=5., tlook=None,\
                 blim=None, shtry=None, shgrid=None, taper=True, cptype='both',\
                 verbose=True):

        '''
        Main function of this class. It will call a bunch of others to compute the
        phase coherence. Args to write

        * reftemp   : reference time for templates (default: often ignored, 
                      but first start time of template[0])
        * shtemp    : Time shift for templates (default: 't0'). Either
                        (1) a string for a marker (then reftemp is ignored)
                        (2) time shifts since reftemp in seconds
        * wintemp   : Window since reference for template (default: [0,3.])
        * buftemp   : A buffer on either side of the template window (default: diff(wintemp)/6)
        * reflook   : Reference times for search data (default: often ignored, 
                      but first start time of st2[0])
        * shlook    : Time shift for data (default: 't0'). Either
                        (1) a string for a marker---in which case reflook is ignored
                        (2) time shifts since reflook in seconds
        * wlenlook  : Window length for search (default: 5.)
        * tlook     : Times to search (default: all available)
        * blim      : Bandlimit to consider (default: [1,10.])
        * shtry     : A dictionary of time shifts to try for each component
        * shgrid    : Time spacing to use for gridding (default: wlen/4)
        * cptype    : Which phase coherence to compute?
                            - 'stat' for inter-station PC
                            - 'comp' for inter-component PC
                            - 'both' (default) for both

        * verbose   : Let the code whisper sweet words into your ears...                   
        '''

        # To be implemented
        if taper is False:
            taper=True
            print('taper was performed anyway, not implemented otherwise (yet)')       
       
        
        # 1/ Get infos on stations, compo, and common traces
        self._getCommonTraces()

        # 2/ Keep only traces which are in common in template and data
        self._removeUnusedTraces(verbose=verbose)

        # 3/ Resample data if necessary
        self._resampleTraces(verbose=verbose)

        # 4/ Merge and mask data
        self.merge()

        # 5/ Set parameters
        self.setParams(reftemp, shtemp, wintemp,buftemp, reflook,\
                      shlook, wlenlook, tlook, blim, shtry, shgrid)

        # 6/ Make template-data cross-correlation
        self.crosscorrDT()

        # 7/ Taper crosscor if wanted
        if taper:
            self.taperCrosscorr()
        
        # 8/ Compute Cp
        self.computeCp(cptype=cptype)
             
        # All done
        return

    # -------------------------------------------------------------------------------        
    def setParams(self,reftemp=None, shtemp='t0', wintemp=None,buftemp=None,\
                 reflook=None, shlook='t0', wlenlook=5., tlook=None,\
                 blim=None, shtry=None, shgrid=None):

        ''' 
        Set the different parameters used for the phase coherence analysis in a dictionnary
        Args:
        * reftemp   : reference time for templates (default: often ignored, 
                      but first start time of template[0])
        * shtemp    : Time shift for templates (default: 't0'). Either
                        (1) a string for a marker (then reftemp is ignored)
                        (2) time shifts since reftemp in seconds
        * wintemp   : Window since reference for template (default: [0,3.])
        * buftemp   : A buffer on either side of the template window (default: diff(wintemp)/6)
        * reflook   : Reference times for search data (default: often ignored, 
                      but first start time of st2[0])
        * shlook    : Time shift for data (default: 't0'). Either
                        (1) a string for a marker---in which case reflook is ignored
                        (2) time shifts since reflook in seconds
        * wlenlook  : Window length for search (default: 5.)
        * tlook     : Times to search (default: all available)
        * blim      : Bandlimit to consider (default: [1,10.])
        * shtry     : A dictionary of time shifts to try for each component
        * shgrid    : Time spacing to use for gridding (default: wlen/4)
        '''

        # Make empty dictionnary
        param = {}
  
        # default reference times
        if reftemp is None:
            param['reftemp'] = self.template[0].stats.starttime
        else:
            param['reftemp'] = reftemp

        if reflook is None:
            param['reflook'] = self.data[0].stats.starttime
        else:
            param['reflook'] = reflook

        # template info
        if wintemp is None:
            param['wintemp'] = np.array([0.,3.])
        else:
            assert(len(wintemp)==2), 'wintemp must have 2 values'            
            param['wintemp'] = np.array(wintemp)

        if buftemp is None:
            param['buftemp'] = np.diff(self.wintemp)[0]/6.
        else:
            param['buftemp'] = buftemp

        # Time shift for template and data
        param['shlook'] = shlook
        param['shtemp'] = shtemp
        
        # Length of looking window
        param['wlenlook'] = wlenlook            
        
        # frequency range
        if blim is None:
            param['blim'] = np.array([1.,10.]) 
        else:
            assert(len(blim)==2), 'dlim must have 2 values'
            param['blim'] = blim

        # default time shift gridding
        param['tlook'] = tlook
        
        if shgrid is None:
            shgrid = wlenlook/4.
            param['shgrid'] = np.maximum(shgrid,self.dtim)
        else:
            param['shgrid'] = shgrid

        if shtry is None:
            param['shtry'] = dict((nsci,np.array([0.])) for nsci in self.stations['CommonTr'])
        else:
            assert(type(shtry) is dict), 'shtry must be a dictionnary'
            param['shtry'] = shtry
        
        self.params = param

        # All done
        return

    # -------------------------------------------------------------------------------        
    def merge(self):
        '''
        Merge traces and add a mask if necessary
        '''
        self.template = self.template.merge()
        self.data     = self.data.merge()
        for tr in self.template+self.data:
            if not isinstance(tr.data,np.ma.masked_array):
                tr.data=np.ma.masked_array(tr.data,mask=np.isnan(tr.data))
        
        # All done
        return
    # -------------------------------------------------------------------------------        
    def resolvepick(self,tr,pk='t0',mkset=None,reftemp=None):
        '''
         INPUT
        
         tr          trace
         pk          reference time for tr
                        either a string for a marker, a time in seconds from the beginning,
                        or a time
         mkset       string of a marker to set (default: None, not set)
         reftemp     reference time (default: tr.stats.starttime)
        
         OUTPUT
          
         tsec        pick time in seconds relative to the start time
         tdat        pick time as a date
        '''


        if reftemp is None:
            reftemp=tr.stats.starttime

        if isinstance(pk,str):
            # time from marker
            tsec = tr.stats[pk]
            tdat = tr.stats.starttime+tsec
        elif isinstance(pk,float):
            # if it's a time since reference
            tdat = reftemp+pk
            tsec = reftemp-tr.stats.starttime
        elif isinstance(pk,datetime.datetime):
            # if it's just a time
            tdat = obspy.UTCDateTime(pk)
            tsec = tdat - tr.stats.starttime
        elif isinstance(pk,obspy.UTCDateTime):
            # if it's just a time
            tdat = pk
            tsec = tdat - tr.stats.starttime

        if mkset:
            # set marker
            tr.stats[mkset]=tsec

        return tsec,tdat

    # -------------------------------------------------------------------------------        
    def crosscorrDT(self):
        '''
        Make the cross-correlation between the data and the template 
        in every window
        '''

        assert(hasattr(self,'params')),'You need to set the parameters with setParams() first'

        # Get params
        wlenlook = self.params['wlenlook']
        wintemp  = self.params['wintemp']
        reftemp  = self.params['reftemp']
        reflook  = self.params['reflook']
        buftemp  = self.params['buftemp']
        tlook    = self.params['tlook']
        shlook   = self.params['shlook']
        shtemp   = self.params['shtemp']
        dtim     = self.dtim
        nsc      = self.stations['CommonTr']
        
        # Get time shift params
        shcalc, ishfs = self._makeGrid()
        Nst      = self.Nst

        # a time to allow buffering of the data
        buftime = (wlenlook+buftemp*2+np.diff(wintemp)[0])*0.05
        buftime = buftime + 3*dtim

        # length of template
        M=int(np.round((np.diff(wintemp)[0]+2*buftemp)/dtim))

        # length of window
        Mw=int(np.round(wlenlook/dtim))

        # number of indices
        Nt=len(tlook)

        # number of stations
        Ns=len(nsc)

        # cross-correlation
        xc = np.ndarray([Mw,Nt,Nst],dtype=float)
        self.tmp = np.ndarray([73401,Nst],dtype=float)
        dok = np.ndarray([Nt,Nst],dtype=bool)

        # relative time limits to extract
        tget = [wintemp[0]-buftemp+np.min(tlook)-wlenlook-buftime,
                wintemp[1]+buftemp+np.max(tlook)+wlenlook+buftime+4*dtim]

        # potential length of data to search
        N=int((tget[1]-tget[0])/dtim)

        # need to pick times
        ix=np.arange(0,N,1)*dtim
        ix=np.searchsorted(ix,tlook-wlenlook/2.+wintemp[0]-buftemp-tget[0])
        ix=np.minimum(ix,N-1)
        ix=np.maximum(ix,0)
                
        # grid to extract
        i1,i2=np.meshgrid(ix,np.arange(0,Mw,1))
        i1=i1+i2
        i1=np.minimum(i1,N-1)
        i1=np.maximum(i1,0)
        self.i1 = i1

        # and to extract for data percentage
        ix2=ix+Mw+M
        ix2=np.minimum(ix2,N-1)

        # computation starts
        d = obspy.Stream()
        t = obspy.Stream()

        # for each trace
        for nsci in nsc:
            for mm in range(0,len(ishfs[nsci])):
                # which time shifts to consider here
                tshf = shcalc[nsci][mm]
                m = ishfs[nsci][mm]

                # related data
                vls=nsc[m].split('.')
                tr1 = self.template.select(network=vls[0],station=vls[1],
                                           channel='*'+vls[2])[0].copy()
                tr2 = self.data.select(network=vls[0],station=vls[1],
                                       channel='*'+vls[2])[0].copy()
                
                # arrival time
                tsec,tdat1 = self.resolvepick(tr1,pk=shtemp,mkset=None,reftemp=reftemp)

                # grab the template (note this distorts the picks)
                tr1=tr1.trim(starttime=tdat1+wintemp[0]-buftemp,
                             endtime=tdat1+wintemp[1]+buftemp+3*dtim,pad=True)
                tr1.data=tr1.data[0:M]
                
                # taper
                tr1.taper(type='cosine',max_percentage=None,
                          max_length=buftemp,side='both')
                
                # arrival time
                tsec,tdat2 = self.resolvepick(tr2,pk=shlook,mkset=None,reftemp=reflook)

                # grab the limited data of interest 
                # (note this distorts the picks)
                tr2=tr2.trim(starttime=tdat2+tget[0]+tshf,
                             endtime=tdat2+tget[1]+6*dtim+tshf,
                             pad=True)
                tr2.data=tr2.data[0:N]

                t.append(tr1)
                d.append(tr2)

                # extract data and set missing data to zero
                data1,data2=tr1.data.data,tr2.data.data
                data1[tr1.data.mask]=0.
                data2[tr2.data.mask]=0.
                
                # cross-correlate
                xci=scipy.ndimage.filters.correlate1d(data2,data1)#/len(data2)

                # grab the correct portion
                ixc = int(data1.size/2)
                xci=xci[i1+ixc]
                
                # also figure out if the data is good enough
                data1=(~tr1.data.mask).astype(float)
                data2=(~tr2.data.mask).astype(float)
                data2=np.cumsum(data2)
                doki=(data2[ix2]-data2[ix])*(sum(data1)/M/(Mw+M))
                doki=doki>0.95
                
                # add to set
                dok[:,m]=doki
                xc[:,:,m]=xci


        # Save it
        self.dok = dok
        self.crosscorr  = xc
        self.tget = tget
        shift = ((tlook-wlenlook/2.+wintemp[0]-buftemp)[0]-tlook[0])/2.
        self.shift = shift

        self.d = d
        self.t = t
        del self.data
        del self.template

        # All done
        return


    # -------------------------------------------------------------------------------        
    def taperCrosscorr(self,taper=True):
        
        '''
        Taper the cross-correlation bewteen data and template using
        a discrete prolate spheroidal (Slepian) sequences.
        Changes self.xc
        '''

        # Get a couple of parameters
        Mw  = int(np.round(self.params['wlenlook']/self.dtim)) # length of window
        xc  = self.crosscorr.copy()
        Nt  = len(self.params['tlook']) # number of indices
        Ns  = len(self.stations['CommonTr']) # number of stations
        Nst = self.Nst
        wlenlook = self.params['wlenlook']
        blim     = self.params['blim']
        dtim     = self.dtim
        dok      = self.dok

        # tapers
        if type(taper) is bool:
            if taper is False:
                tap = np.ones([Mw,1])
                Ntap = 1
                NW = 2
            else:
                # compute tapers
                # comment out these three lines to exclude taper
                NW = 4
                [tap,V] = spectrum.mtm.dpss(Mw,NW)
                # Ditch crappy tapers
                ix = np.where(V>0.99)[0]
                V=V[ix]
                tap=tap[:,ix]
                Ntap = len(V)
        else:
            tap=taper
            Ntap=taper.shape[1]
            NW=2

        # repeat and multiply by tapers
        xc  = xc.reshape([Mw,Nt,Nst,1])
        tap = tap.reshape([Mw,1,1,Ntap])
        xc  = np.multiply(xc,tap)

        self.Ntap = Ntap

        #----------------------------------------------------------

        # fft
        Nft=Mw*2
        xc=np.fft.fft(xc,n=Nft,axis=0)

        # frequencies
        freq=np.fft.fftfreq(Nft,d=dtim)
        
        # just frequencies in range
        ixf=np.logical_and(freq>=blim[0],freq<=blim[1])
        ixf,=np.where(ixf)

        # frequency spacing
        dfreq=np.median(np.diff(freq))
        spc=2.*float(NW)/wlenlook
        spc=np.arange(0.,len(ixf)-1,spc/dfreq)
        spc=np.round(spc).astype(int)
        ixf = ixf[spc]

        freq=freq[ixf]
        Nf=len(ixf)
        xc=xc[ixf,:,:,:]
        
        if (type(taper) is bool)&(taper is True):
            xc=xc.reshape([Nf,Nt,Nst,Ntap])


        # average the cross-spectra over tapers
        #xc = np.mean(xc,axis=3)

        # Save unorm
        self.unorm_xc = xc.copy()
        # normalize
        xc = np.divide(xc,np.abs(xc))
        
        # set problematic intervals to zero
        #doki = dok.reshape([1,Nt,Nst]).astype(float)
        #xc = np.multiply(xc,doki).reshape([Nf,Nt,Nst])

        # Save tapered crosscorr
        self.xc = xc
        self.freq = freq

        # All done
        return

    #-------------------------------------------------------------
    def _computePt(self,cptype='both'):
        '''
        Compute the phase coherence from the template/data tappered cross correlation.
        Returns a dictionnary stored in self.Cp
        Args:   
            * comp  : Compute 
        
        '''
        # Check some stuff
        assert(cptype in ['comp','stat','both']),'cptype must be "stat","comp", or "both"'
        
        # Get some parameters
        tlook   = self.params['tlook']
        Nt      = len(tlook) # number of indices
        Nf      = self.xc.shape[0]
        Nc      = self.stations['Nc']
        icmp    = self.stations['icmp']
        ins     = self.stations['ins']
        nper    = self.stations['nper']
        nsc     = self.stations['CommonTr']
        xc      = self.unorm_xc
        dok     = self.dok
        Ntap    = self.Ntap
        

        if cptype in ['stat','both']:
            # sum, separated by station
            Ptstat=np.zeros((Nt,self.Ntap))

            # compute inter-station coherence for each component
            for t in range(self.Ntap):
                for ks in range(0,Nc):
                    
                    # at each component
                    ii=icmp==ks
                    inds = list(itertools.combinations(np.where(ii)[0],2))
                    for jj,kk in inds:
                        T = np.power(np.abs(np.conj(xc[:,:,jj,t])*xc[:,:,kk,t]),2)
                        
                        # average over frequencies
                        T=np.mean(T,axis=0).reshape([Nt])

                        # add to set
                        Ptstat[:,t]+=T
     
            # V2
            Ptstat2=np.zeros((Nt,self.Ntap))

            # compute inter-station coherence for each component
            for ks in range(0,Nc):    
                ii=icmp==ks
                inds = list(itertools.combinations(np.where(ii)[0],2))

                for jj,kk in inds:
                    for t in range(self.Ntap):
                        
                        T = np.power(np.abs(np.conj(xc[:,:,jj,t])*xc[:,:,kk,t]),2)
                        
                        # average over frequencies
                        T=np.mean(T,axis=0).reshape([Nt])

                        # add to set
                        Ptstat2[:,t]+=T  
        
        self.Ptstat = Ptstat
        self.norm   = np.sqrt((Ptstat**2).sum(1)) 
        return


    #-------------------------------------------------------------
    def computeCp(self,cptype='both'):
        '''
        Compute the phase coherence from the template/data tappered cross correlation.
        Returns a dictionnary stored in self.Cp
        Args:   
            * comp  : Compute 
        
        '''

        # Check some stuff
        assert(cptype in ['comp','stat','both']),'cptype must be "stat","comp", or "both"'
        
        # Get some parameters
        tlook   = self.params['tlook']
        Nt      = len(tlook) # number of indices
        Nf      = self.xc.shape[0]
        Nc      = self.stations['Nc']
        icmp    = self.stations['icmp']
        ins     = self.stations['ins']
        nper    = self.stations['nper']
        nsc     = self.stations['CommonTr']
        xc      = self.xc
        dok     = self.dok
        Ntap    = self.Ntap
        freq    = self.freq
        
        # Initialize Cp dict        
        Cp = {'tlook':tlook,'freq':freq,'Ntap':Ntap,'nsc':nsc}

        if cptype in ['stat','both']:
            # sum, separated by station
            Pstat=np.zeros((Nt,self.Ntap))
            Cpstat=np.zeros((Nt,self.Ntap))
            Nstat=np.zeros([Nt,Nc,self.Ntap])

            # compute inter-station coherence for each component
            for t in range(self.Ntap):
                for ks in range(0,Nc):
                    # compute and add for each frequency
                    #for ifreq in range(0,len(freq)):
                    
                    # at each component
                    ii=icmp==ks

                    if sum(ii)>1:
                        
                        Pt = np.abs(np.sum(xc[:,:,ii,t],axis=0))**2
                        # average without taking absolute value (Phase walkout???)
                        Rstati=np.sum(xc[:,:,ii,t],axis=2).reshape([Nf,Nt])

                        # and find amplitude of this vector
                        Rstati=np.abs(Rstati)

                        # count the number of stations per windows 
                        nn=np.sum(dok[:,ii],axis=1)
                        nn=nn.reshape([1,Nt]).astype(float)

                        # to phase coherence
                        Rstati=np.divide(np.divide(np.power(Rstati,2),nn)-1.,nn-1.)

                        # average over frequencies
                        Rstati=np.mean(Rstati,axis=0).reshape([Nt])

                        # check if there are enough stations
                        nn=nn.flatten()
                        nok=nn<2
                        nn[nok]=0
                        Rstati[nok]=0.

                        # add to set
                        Cpstat[:,t]=Cpstat[:,t]+Rstati
                        Nstat[:,ks,t]=nn
                

            # normalize inter-station coherence
            Nci = np.sum(Nstat>=2,axis=1).astype(float)
            Cpstat = np.divide(Cpstat,Nci)

            # expected standard deviation
            # why does Nstat not include component?
            stds = self._expstd(Nstat,Nf*Ntap)

            Cp['Pstat'] = Pstat
            Cp['Cpstat'] = Cpstat
            Cp['Nstat']  = Nstat
            Cp['stds']   = stds
            

        #------------------------------------------------------
        if cptype in ['comp','both']:
            # and consider components per station
            Cpcomp=np.zeros((Nt,self.Ntap))
            Ncomp=np.zeros([Nt,len(nper),self.Ntap])

            for t in range(self.Ntap):
                # for components
                for ks in range(0,len(nper)):
                    # at each station
                    ii=ins==ks
                    
                    if sum(ii)>1:
                        # average without taking absolute value
                        Rcompi=np.sum(xc[:,:,ii,t],axis=2).reshape([Nf,Nt])
                        Rcompi=np.abs(Rcompi)

                        # count the number of values
                        nn=np.sum(dok[:,ii],axis=1)
                        nn=nn.reshape([1,Nt])
                        
                        # to phase coherence
                        Rcompi=np.divide(np.divide(np.power(Rcompi,2),nn)-1,nn-1.)

                        # average over frequencies
                        Rcompi=np.mean(Rcompi,axis=0).reshape([Nt])

                        # only add if all the components are there
                        nn=nn.flatten()
                        nok=nn!=sum(ii)
                        nn[nok]=0
                        Rcompi[nok]=0.

                        Cpcomp[:,t]=Cpcomp[:,t]+Rcompi
                        Ncomp[:,ks,t]=nn

            # normalize inter-component coherence
            Nci = np.sum(Ncomp>=2,axis=1).astype(float)
            Cpcomp = np.divide(Cpcomp,Nci)

            # expected standard deviation
            stdc = self._expstd(Ncomp,Nf*Ntap)

            Cp['Cpcomp'] = Cpcomp
            Cp['Ncomp']  = Ncomp
            Cp['stdc']   = stdc

        #--------------------------------------------------------------

        # Save it
        self.Cp = Cp

        return 


    # -------------------------------------------------------------------------------        
    def _getCommonTraces(self):

        '''
        Get common traces between the template and station
        return:
                * self.stations : dictionnary containing some variables used after
                    -> Nc       : number of distinct components (1, 2, ou 3)
                    -> icmp     : For each trace, indice if which compo it is
                    -> ins      : For each trace, get station number
                    -> nper     : For each trace, get number of traces
                    -> nscT     : Traces in template
                    -> nscD     : Traces in data
                    -> CommonTr : List of traces in both template and data
        '''

        # Networks, stations, components to compare
        nsc1=np.array([tr.stats.network+'.'+tr.stats.station+'.'+
                       tr.stats.channel[-1] for tr in self.template])
        nsc2=np.array([tr.stats.network+'.'+tr.stats.station+'.'+
                       tr.stats.channel[-1] for tr in self.data])
        nsc=np.intersect1d(nsc1,nsc2)
        nsc=np.sort(nsc)

        # split by station
        ns=np.array([vl.split('.')[0]+'.'+vl.split('.')[1] for vl in nsc])
        nsa,ins,nper=np.unique(ns,return_inverse=True,return_counts=True)

        # split by component
        icmp,cmps = self._groupcomp(nsc)
        Nc=len(cmps)

        stations = {'Nc':Nc, 'icmp':icmp, 'ins':ins, 'nper':nper, \
                    'nscT':nsc1, 'nscD':nsc2, 'CommonTr':nsc} 

        self.stations = stations
        
        # all done
        return 
             
    # -------------------------------------------------------------------------------        
    def _groupcomp(self,nsc):
        """
        :param     nsc: list of network.station.component
        :return   icmp: index of components for each
        :return   cmps: components considered
        """

        # define groups
        cmps = np.array(['E1','N2','Z3'])

        # initialize
        icmp = np.ndarray(len(nsc),dtype=int)

        for k in range(0,len(nsc)):
            # split component
            vl = nsc[k].split('.')
            vl = vl[-1][-1]
            
            for m in range(0,len(cmps)):
                if vl in cmps[m]:
                    icmp[k]=m

        # only components that were used
        ii,icmp=np.unique(icmp,return_inverse=True)
        cmps=cmps[ii]

        # all done
        return icmp,cmps


    # -------------------------------------------------------------------------------        
    def _expstd(self,Nstat,Nave):
        """
        :param    Nstat:  number of stations used
                    could have a multiple columns if averaged over
                    components
        :param     Nave:  number of values averaged over after
        :return    stde:  expected std
        """

        # number of stations
        if Nstat.ndim>1:
            # number of components
            Nc = Nstat.shape[1]
            # average stations per component?
            # this really shouldn't change 
            Nstat = np.mean(Nstat,axis=1)
        else:
            # just one component
            Nc = 1

        # number of pairs
        Np = np.multiply(Nstat,Nstat-1)/2

        # total number of values
        Ntot = Np*Nc*Nave
        Ntot = np.atleast_1d(Ntot).astype(float)
        
        # std
        stde = np.divide(1.,2.*Ntot)
        stde = np.power(stde,0.5)

        return stde


    # -------------------------------------------------------------------------------        
    def _removeUnusedTraces(self,verbose=True):
        '''
        Remove traces which are neither in template or data
        '''
          
        # Check something
        assert(hasattr(self,'stations')), 'You need to compute stations info first'

        # Create empty obspy stream to fill with template
        st1i = obspy.Stream()

        # Select only used traces
        bo1 = np.isin(self.stations['nscT'],self.stations['CommonTr'])
        ix1 = np.where(bo1)[0]
        [st1i.append(self.template[i]) for i in ix1]

        if verbose:
            removed = self.stations['nscT'][~bo1]
            if removed.size==0:
                print('No traces removed in template')
            else:
                t=[print('Trace {} removed from template'.format(r)) for r in removed]


        # Create empty obspy stream to fill with data
        st2i = obspy.Stream()

        # Select only used traces
        bo2 = np.isin(self.stations['nscD'],self.stations['CommonTr'])
        ix2 = np.where(bo2)[0]
        [st2i.append(self.data[i]) for i in ix2]

        if verbose:
            removed = self.stations['nscD'][~bo2]
            if removed.size==0:
                print('No traces removed in data')
            else:
                t=[print('Trace {} removed from data'.format(r)) for r in removed]

        # Put them in current streams
        self.template = st1i
        self.data     = st2i

        # All done
        return

    # -------------------------------------------------------------------------------        
    def _resampleTraces(self,verbose=True):
        ''' 
        Resample data and template to the same time spacing of necessary
        '''

        # resample to the same time spacing if necessary
        dtim=[tr.stats.delta for tr in self.template]+[tr.stats.delta for tr in self.data]
        dtim=np.unique(np.array(dtim))

        if len(dtim)>1:
            if verbose:
                print('Resampling to common interval')
            self.template=self.template.resample(sampling_rate=1./min(dtim),no_filter=False)
            self.data=self.data.resample(sampling_rate=1./min(dtim),no_filter=False)
        
        # Save it 
        self.dtim = min(dtim)

        # All done
        return


    # -------------------------------------------------------------------------------        
    def _makeGrid(self,shgrid=None,shtry=None):
        '''
        Compute shcalc, ishfs, and Nst
        '''
        
        assert(hasattr(self,'params')),'You must set parameters with setParams() first'
        shgrid = self.params['shgrid']
        shtry  = self.params['shtry']
        nsc    = self.stations['CommonTr']
         
        # time shifts to calculate
        shcalc = dict((nsci,np.array([0.])) for nsci in nsc)
        for ky in shcalc.keys():
            shtry[ky] = np.atleast_1d(shtry[ky])
            vl = general.minmax(shtry[ky])
            nvl = int(np.ceil(np.diff(vl)[0]/shgrid))
            nvl = np.maximum(nvl,1)
            shcalc[ky] = np.linspace(vl[0],vl[1],nvl)
        
        # maximum number of shifts to calculate for each station
        # because we'll make a list with one x-c for each station/shift
        Nst=np.sum(np.array([len(vl) for vl in shcalc.values()]))
        i1=0
        ishfs = dict((nsci,np.array([0])) for nsci in nsc)
        for nsci in nsc:
            ishfs[nsci] = np.arange(0,len(shcalc[nsci]))+i1
            i1=i1+len(shcalc[nsci])

        # Save it
        self.Nst = Nst

        # All done
        return shcalc, ishfs



