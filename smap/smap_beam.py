
""" SMAP beam handling"""

import math
import numpy as np
from astropy.nddata.convolution.make_kernel import make_kernel

from .defaults import smap_pixsize, spire_fwhm

__all__ = ["get_gauss_beam", "get_spire_beam"]

def get_spire_beam(band, pixscale=None, nfwhm=5.0, oversamp=5):
    """ Get default SPIRE beam

    Parameters
    ----------
    band: string
      Name of band ('PSW', 'PMW', 'PLW')
    
    pixscale: float
      Pixel scale, in same units as FWHM.  If not provided, defaults
      based on band

    nfwhm: float
      Number of fwhm (approximately) of each dimension of the output beam.

    oversamp: int
      Odd integer giving the oversampling to use when constructing the
      beam.  The beam is generated in pixscale / oversamp size pixels,
      then rebinned to pixscale.
    """
    
    if not band in spire_fwhm:
        raise ValueError("Unknown band for FWHM: %s" % band)
    fwhm = spire_fwhm[band]

    if pixscale is None:
        if not band in smap_pixsize:
            raise ValueError("Unknown band for default pixel scale: %s" % band)
        pix = smap_pixsize[band]
    else:
        pix = float(pixscale)
        if pix <= 0:
            raise ValueError("Invalid (non-positive) pixel scale: %f" % pix)

    return get_gauss_beam(fwhm, pix, nfwhm=nfwhm, oversamp=oversamp)

def get_gauss_beam(fwhm, pixscale, nfwhm=5.0, oversamp=5):
    """ Generate Gaussian kernel

    Parameters
    ----------
    fwhm: float
      FWHM of the Gaussian beam.

    pixscale: float
      Pixel scale, in same units as FWHM.

    nfwhm: float
      Number of fwhm (approximately) of each dimension of the output beam.

    oversamp: int
      Odd integer giving the oversampling to use when constructing the
      beam.  The beam is generated in pixscale / oversamp size pixels,
      then rebinned to pixscale.
    
    Notes
    -----
      The beam is normalized by having a value of 1 in the center.
      If oversampling is used, the returned array will be the sum over
      the neighborhood of this maximum, so will not be one.
    """

    if fwhm <= 0:
        raise ValueError("Invalid (negative) FWHM")
    if pixscale <= 0:
        raise ValueError("Invalid (negative) pixel scale")
    if nfwhm <= 0.0:
        raise ValueError("Invalid (non-positive) nfwhm")
    if fwhm / pixscale < 2.5:
        raise ValueError("Insufficiently well sampled beam")
    if oversamp < 1:
        raise ValueError("Invalid (<1) oversampling")

    retext = round(fwhm * nfwhm / pixscale)
    if retext % 2 == 0:
        retext += 1

    bmsigma = fwhm / math.sqrt(8 * math.log(2))

    if oversamp == 1:
        # Easy case
        beam = make_kernel((retext, retext), bmsigma / pixscale, 
                           'gaussian').astype(np.float32)
        beam /= beam.max
    else:
        genext = retext * oversamp
        genpixscale = pixscale / oversamp
        gbeam = make_kernel((genext, genext), bmsigma / genpixscale, 
                           'gaussian').astype(np.float32)
        gbeam /= gbeam.max() # Normalize -before- rebinning

        # Rebinning -- tricky stuff!
        bmview = gbeam.reshape(retext, oversamp, retext, oversamp)
        beam = bmview.mean(axis=3).mean(axis=1)

    return beam
