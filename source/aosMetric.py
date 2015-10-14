#!/usr/bin/env python
##
# @authors: Bo Xin 
# @       Large Synoptic Survey Telescope

import sys

import numpy as np
import scipy.special as sp
from astropy.io import fits

from cwfsTools import padArray

from cwfsErrors import nonSquareImageError
from aosErrors import psfSamplingTooLowError

class aosMetric(object):

    def __init__(self,wfs, debugLevel):
        self.nArm=6
        armLen=[0.379,0.841, 1.237, 1.535, 1.708]
        armW = [0.2369,0.4786,0.5689,0.4786,0.2369]
        self.nRing=len(armLen)
        self.nField=self.nArm*self.nRing+1
        self.nFieldp4=self.nField+4
        self.fieldX = np.zeros(self.nFieldp4)
        self.fieldY = np.zeros(self.nFieldp4)
        self.fieldX[0] = 0
        self.fieldY[0] = 0
        self.w=[0]
        for i in range(self.nRing):
            self.w=np.concatenate((self.w,np.ones(self.nArm)*armW[i]))
            self.fieldX[i*self.nArm+1: (i+1)*self.nArm+1]=\
              armLen[i]*np.cos(np.arange(self.nArm)*(2*np.pi)/self.nArm)
            self.fieldY[i*self.nArm+1: (i+1)*self.nArm+1]=\
              armLen[i]*np.sin(np.arange(self.nArm)*(2*np.pi)/self.nArm)
        self.w = self.w/np.sum(self.w)
        self.fieldX[self.nField:]=[1.185, -1.185, -1.185, 1.185]
        self.fieldY[self.nField:]=[1.185, 1.185, -1.185, -1.185]

        #below, p is for PSF
        self.fieldXp = self.fieldX.copy()
        self.fieldYp = self.fieldY.copy()

        self.fieldXp[19] +=0.004
        self.fieldXp[22] -=0.004
        
        if debugLevel>=3:
            print(self.w.shape)
            print(self.w)

        aa =np.loadtxt('data/pssn_alpha.txt')
        self.pssnAlpha = aa[:, 0]
        #self.pssnRange = aa[: 1]
        
        self.znx2 = np.zeros((self.nFieldp4, wfs.znwcs3))
        
    def getFWHMfromZ(self):
        self.fwhm=np.zeros(self.nField)

    def getPSSNfromZ(self):
        pass

    def getPSSNandMore(self,state,wavelength,debugLevel):

        self.PSSN = np.zeros(self.nField)
        self.FWHMeff = np.zeros(self.nField)
        self.dm5 = np.zeros(self.nField)
        
        for i in range(self.nField):
            opdFile='%s/sim%d_iter%d_opd%d.fits'%(
                state.imageDir, state.iSim,state.iiter,i)
            IHDU = fits.open(opdFile)
            opd = IHDU[0].data*1e3 #from mm to um 
            IHDU.close()
            
            self.stampD = 2**np.ceil(np.log2(state.opdSize))
            if self.stampD>opd.shape[0]:
                a=opd
                opd=np.zeros((self.stampD,self.stampD))
                opd[:a.shape[0],:a.shape[1]] = a
            
            self.PSSN[i] = calc_pssn(opd,wavelength,debugLevel=debugLevel)
            self.FWHMeff[i] = np.sqrt(
                -1.2187*0.6040**2+0.8127*0.7386**2/self.PSSN[i])
            self.dm5[i] = -1.25*np.log10(self.PSSN[i])
            
            if debugLevel>=2:
                print('field#%d, PSSN=%7.4f'%(i,self.PSSN[i]))

        self.GQPSSN=np.sum(self.w*self.PSSN)
        if debugLevel>=2:
            print(self.GQPSSN)
    
    def getEllipticity(self,state, wavelength, debugLevel):
        self.elli=np.zeros(self.nField)
        for i in range(self.nField):
            opdFile='%s/sim%d_iter%d_opd%d.fits'%(
                state.imageDir, state.iSim,state.iiter,i)
            IHDU = fits.open(opdFile)
            opd = IHDU[0].data*1e3 #from mm to um 
            IHDU.close()
            
            self.stampD = 2**np.ceil(np.log2(state.opdSize))
            if self.stampD>opd.shape[0]:
                a=opd
                opd=np.zeros((self.stampD,self.stampD))
                opd[:a.shape[0],:a.shape[1]] = a
            
            self.elli[i] = psf2eAtmW(opd,wavelength,debugLevel=debugLevel)
            
            if debugLevel>=2:
                print('field#%d, elli=%7.4f'%(i,self.elli[i]))

        self.GQelli=np.sum(self.w*self.elli)
        if debugLevel>=2:
            print(self.GQelli)
    
def calc_pssn(array, wlum, type='opd', D=8.36,r0inmRef=0.1382, zen=0,
              pmask=0, imagedelta=0.2,fno=1.2335, debugLevel=0):
    """
    array: the array that contains eitehr opd or pdf
    opd need to be in microns
    wlum: wavelength in microns
    type: what is used to calculate pssn - either opd or psf
    psf doesn't matter, will be normalized anyway
    D: side length of OPD image in meter 
    r0inmRef: fidicial atmosphere r0@500nm in meter, Konstantinos uses 0.20
    Now that we use vonK atmosphere, r0in=0.1382 -> fwhm=0.6"
    earlier, we used Kol atmosphere, r0in=0.1679 -> fwhm=0.6"
    zen: telescope zenith angle

    The following are only needed when the input array is psf -    
    pmask: pupil mask. when opd is used, it can be generated using opd image,
    we can put 0 or -1 or whatever here.
    when psf is used, this needs to be provided separately with same
    size as array
    imagedelta and fno are only needed when psf is used. use 0,0 for opd

    THE INTERNAL RESOLUTION THAT FFTS OPERATE ON IS VERY IMPORTANT
    TO THE ACCUARCY OF PSSN.
    WHEN TYPE='OPD', NRESO=SIZE(ARRAY,1)
    WHEN TYPE='PSF', NRESO=SIZE(PMASK,1)
       for the psf option, we can not first convert psf back to opd then start over, 
       because psf=|exp(-2*OPD)|^2. information has been lost in the | |^2.
       we need to go forward with psf->mtf,
       and take care of the coordinates properly.
    """
    
    if array.ndim==3:
        array2D=array[0,:,:].squeeze()

    if type=='opd':
        try:
            m=max(array2D.shape)
        except NameError:
            m=max(array.shape)
    else:
        m = max(pmask.shape)
        #pupil needs to be padded k times larger to get imagedelta
        k=fno*wlum/imagedelta
        m=np.rint(m*k+1e-5)
        D=D*k

    mtfa = createMTFatm(D,m,wlum,zen,r0inmRef)
    
    if type=='opd':
        try:
            iad = (array2D!=0)
        except NameError:
            iad = (array!=0)
    elif type == 'psf':
        iad = padArray(pmask, m)

    #number of non-zero elements, used for normalization later        
    #miad2 = np.count_nonzero(iad)

    # Perfect telescope
    opdt = np.zeros((m,m))
    psft = opd2psf(opdt, iad,wlum,0,0,0,debugLevel)
    otft = psf2otf(psft) #OTF of perfect telescope
    otfa = otft *mtfa # add atmosphere to perfect telescope
    psfa = otf2psf(otfa)
    pssa = np.sum(psfa**2) # atmospheric PSS = 1/neff_atm
                                       
    # Error;
    if type == 'opd':
        if array.ndim == 2:
            ninst = 1
        else:
            ninst = array.shape[0]
        for i in range(ninst):
            if array.ndim == 2:
                array2D = array
            else:
                array2D=array[i,:,:].squeeze()
            psfei = opd2psf(array2D, iad, wlum, 0, 0, 0, debugLevel)
            if i==0:
                psfe = psfei
            else:
                psfe += psfei
        psfe=psfe/ninst
    else:
        psfe = padArray(array,m)
        psfe = psfe/np.sum(psfe)*np.sum(psft)

    otfe = psf2otf(psfe) #OTF of error
    otftot = otfe *mtfa # add atmosphere to error
    psftot = otf2psf(otftot)
    pss = np.sum(psftot**2) # atmospheric + error PSS

    pssn = pss/pssa # normalized PSS
    
    return pssn

def createMTFatm(D,m, wlum,zen,r0inmRef):
    
    wlm = wlum*1.e-6

    df = D/wlm/(m-1) # frequency resolution in 1/rad
    m0 = np.rint(0.5*(m+1)+1e-5)
    aa = np.arange(1,m+1)
    x, y=np.meshgrid(aa,aa)
    
    f = df*np.sqrt((x-m0)**2+(y-m0)**2)
    
    zen = zen*np.pi/180. # telescope zenith angle, change here
    r0aref = r0inmRef*np.cos(zen)**0.6 #atmosphere reference r0
    r0a = r0aref*(wlum/0.5)**1.2 #atmosphere r0, a function of wavelength
    L0=30  #outer scale in meter, only used when model=vonK

    #atmosphere structure function, in range [-D/2, D/2]    
    #sfa=atmSF('Kol',wlm*f,r0a,L0)
    sfa=atmSF('vonK',wlm*f,r0a,L0) 
    mtfa = np.exp(-0.5*sfa)

    return mtfa
        
def atmSF(model,r,r0a,L0):
    """
    create the atmosphere phase structure function
    model = 'Kol'
             = 'vonK'
    r is the input array, for calculating atmosphere OTF, r=lambda*f
    r0a is the atmosphere r0, a function of wavelength
    L0 is outer scale in meter, only meaningful for vonK model
    """

    if model == 'Kol':
        sfa = 6.88*(r/r0a)**(5/3)        
    elif model == 'vonK':
        sfa_c=2*sp.gamma(11/6)/2**(5/6)/np.pi**(8/3)*\
          (24/5*sp.gamma(6/5))**(5/6)*(r0a/L0)**(-5/3)
        sfa_k=sp.kv(5/6,(2*np.pi/L0*r)) #modified bessel of 2nd/3rd kind
        sfa=sfa_c*(2**(-1/6)*sp.gamma(5/6)-(2*np.pi/L0*r)**(5/6)*sfa_k)
        
        # if we don't do below, everything will be nan after ifft2
        # midp = r.shape[0]/2+1
        #1e-2 is to avoid x.49999 be rounded to x
        midp=np.rint(0.5*(r.shape[0]-1)+1e-2)
        sfa[midp,midp]=0 #at this single point, sfa_k=Inf, 0*Inf=Nan;

    return sfa

def psf2eAtmW(wfm, wlum, D=8.36,pmask=0,r0inmRef=0.1382,
              zen=0, imagedelta=0.2,fno=1.2335,debugLevel=0):

    psfe=opd2psf(wfm,0,wlum, imagedelta,1,fno, debugLevel)
    otfe = psf2otf(psfe) #OTF of error
    
    m = psfe.shape[0] #used to use wfm.shape[0]
    mtfa = createMTFatm(D, m, wlum, zen, r0inmRef)
            

def opd2psf(opd, pupil, wavelength, imagedelta, sensorFactor,fno,debugLevel):
    """
    wavefront OPD in micron
    imagedelta in micron, use 0 if pixel size is not specified
    wavelength in micron

    if pupil is a number, not an array, we will get pupil geometry from opd
    The following are not needed if imagedelta=0,
    sensorFactor, fno
    """

    opd[np.isnan(opd)]=0
    try:
        if (pupil.shape==opd.shape):
            pass
        else:
            raise AttributeError
    except AttributeError:
        pupil=(opd!=0)

    if imagedelta != 0:
        try:
            if opd.shape[0] != opd.shape[1]:
                raise(nonSquareImageError)
        except nonSquareImageError:
            print('Error (opd2psf): Only square images are accepted.')
            print('image size = (%d, %d)' % (
            self.image.shape[0], self.image.shape[1]))
            sys.exit()
    
        k=fno*wavelength/imagedelta
        padding=k/sensorFactor
        try:
            if padding<1:
                raise(psfSamplingTooLowError)
        except psfSamplingTooLowError:
            print('opd2psf: sampling too low, data inaccurate');
            print('imagedelta needs to be smaller than fno*wlum=%4.2f um'%(
                    fno*wavelength))
            print('         so that the padding factor > 1');
            print('         otherwise we have to cut pupil to be < D');
            sys.exit()
    
        sensorSamples=opd.shape[0]
        N=np.rint(padding*sensorSamples)
        pupil=padArray(pupil,N)
        opd=padArray(opd,N)
        if debugLevel>=3:
            print('padding=%8.6f'%padding)
        
    z = pupil*np.exp(-2j*np.pi*opd/wavelength)
    z = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(z),
                    s=z.shape)) #/sqrt(miad2/m^2)
    z = np.absolute(z**2)
    z = z/np.sum(z)
    
    if debugLevel>=3:
        print('imagedelta=%8.6f'%imagedelta)
        print(np.sum(z))
        
    return z

def psf2otf(psf):    
    otf = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psf),
                                       s=psf.shape))
    return otf

def otf2psf(otf):
    psf = np.absolute(np.fft.fftshift(np.fft.irfft2(np.fft.fftshift(otf),
                                       s=otf.shape)))
    return psf
