""" Generate a SMAP map from an input catalog"""

from __future__ import print_function

from .smap_struct import smap_map
from .smap_beam import get_gauss_beam
import numpy as np
from astropy.nddata import convolve

__all__ = ["cattomap_gauss"]

def cattomap_gauss(area, fluxes, wave=[250.0,350,500], 
		   pixsize=[6.0, 8.33333, 12.0], racen=25.0, deccen=0.0, 
		   fwhm=[17.6, 23.9, 35.2], nfwhm=5.0, bmoversamp=5, 
		   sigma_inst=None, verbose=False):
    """ Generates simulated maps as SMAP structures using a Gaussian beam
    from an input catalog of flux densities.
    
    Parameters
    ----------
    area: float
      Area of generated maps, in deg^2

    fluxes: ndarray
      Array of flux densities, of shape nsources by nbands, in Jy.

    wave: ndarray
      Wavelengths to generate maps at, in microns.  This must
      have the same number of elements as the second dimension
      of fluxes.

    pixsize: ndarray
      Pixel sizes of output maps, in arcsec.  The code may
      perform somewhat better if the finest pixel scale is first.

    racen: float
      Right ascension of generated maps
          
    deccen: float
      Declination of generated maps

    fwhm: ndarray
      Beam FWHM values, in arcsec.

    nfwhm: float
      How far out, in units of FWHM, to generate the beams

    sigma_inst: ndarray or None
      Map instrument noise, in Jy.  If None, no instrument
      noise is added.

    verbose: bool
      Print informational messages as it runs.

    Returns
    -------
      A tuple containing an array input maps and the x/y positions
      of the sources (in the first map)
    """

    import math	
    from numbers import Number

    # Check inputs
    if not isinstance(fluxes, np.ndarray):
        raise TypeError("Input fluxes not ndarray")
    
    # Get number of bands
    if len(fluxes.shape) == 1:
        # 1 band case
        nbands = 1
    elif len(fluxes.shape) == 2:
        # Usual case
        nbands = fluxes.shape[1]
    else:
        raise ValueError("Input fluxes of unexpected dimension")
    
    if len(wave) < nbands:
        raise ValueError("Number of wavelengths not the same as the"
                         " number of bands")
    if len(pixsize) < nbands:
        raise ValueError("Number of pixsize values not the same as the"
                         " number of bands")
    if len(fwhm) < nbands:
        raise ValueError("Number of FWHM values not the same as the"
                         " number of bands")

    nsrcs = fluxes.shape[0]

    # Figure out sigma situation
    if sigma_inst is None:
        has_sigma = False
    elif isinstance(sigma_inst, Number):
        # Single value -- replicate it out
        has_sigma = True
        int_sigma = sigma_inst * np.ones(nbands, dtype=np.float32)
    else:
        has_sigma = True
        int_sigma = np.asarray(sigma_inst, dtype=np.float32)
        if len(int_sigma) == 1 and nbands > 1:
            int_sigma = int_sigma[0] * np.ones(nbands, dtype=np.float32)
        elif len(int_sigma) < nbands:
            raise ValueError("Not enough instrument sigma values for #bands")

    if has_sigma and int_sigma.min() < 0:
        raise ValueError("Invalid (negative) instrument sigma")

    # Create the empty maps
    nextent = np.empty(nbands, dtype=np.int32)
    truearea = np.empty(nbands, dtype=np.float32)
    maps = []
    for i in range(nbands):
        pixarea = (pixsize[i] / 3600.0)**2
        nextent[i] = math.ceil(math.sqrt(area / pixarea))
        truearea[i] = nextent[i]**2 * pixarea
        s_map = smap_map()
        if has_sigma:
            s_map.create(np.zeros((nextent[i], nextent[i]), dtype=np.float32),
                         pixsize[i], racen, deccen, wave=wave[i],
                         error=int_sigma[i]*np.ones((nextent[i], nextent[i]), 
                                                    dtype=np.float32))
        if has_sigma:
            s_map.create(np.zeros((nextent[i], nextent[i]), dtype=np.float32),
                         pixsize[i], racen, deccen, wave=wave[i])
        maps.append(s_map)
 
    # Generate positions in the first band for all sources
    xpos = nextent[0] * np.random.rand(nsrcs)
    ypos = nextent[0] * np.random.rand(nsrcs)
    
    # Construct maps
    for i in range(nbands):
        if verbose:
            print("Preparing %0.1f map" % wave[i])

        # Add sources
        if verbose:
            print(" Inserting sources")
        relpix = pixsize[i] / pixsize[0]
        xf = np.floor(xpos * relpix)
        yf = np.floor(ypos * relpix)
        cmap = maps[i].image
        nx, ny = cmap.shape
        np.place(xf, xf > nx-1, nx-1)
        np.place(yf, yf > ny-1, ny-1)
        for cx, cy, cf in zip(xf, yf, fluxes[:,i]):
            cmap[cx, cy] += cf

        # Smooth
        if verbose:
            print(" Smoothing")
        bm = get_gauss_beam(fwhm[i], pixsize[i], nfwhm)
        convmap = convolve(cmap, bm, boundary='wrap')

        # Noise
        if verbose:
            print(" Adding noise")
        if has_sigma:
            convmap +=np.random.normal(scale=int_sigma[i], 
                                       size=convmap.shape)
            
        convmap -= convmap.mean()
        maps[i].image = convmap
        
    return (maps, xpos, ypos)
