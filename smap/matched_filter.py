""" Routines to apply matched filtering to SMAP maps

See Chapin et al. 2011, MNRAS, 411, 505, Appendix A.
"""

from __future__ import print_function

from .defaults import spire_fwhm, spire_confusion, smap_pixsize

import math
import numpy as np
import scipy.fftpack as fft

from astropy.nddata.convolution.make_kernel import make_kernel

__all__ = ["make_matched_filter", "make_red_matched_filter", 
           "matched_filter", "matched_filter_red"]

def make_beam(band, pixscale, nx, ny):
    """ Internal helper function to make full size, 0,0 centered beam.
    Note that the beam is [y, x] indexed, since smap maps are also done
    that way"""

    fwhm_pix = spire_fwhm[band] / pixscale
    sigma_pix = fwhm_pix / math.sqrt(8 * math.log(2))
    beam = make_kernel((ny, nx), kernelwidth=sigma_pix)
    beam /= beam.max()
    # The max is as nx // 2, ny // 2 -- we want it at 0, 0
    return np.roll(np.roll(beam, -(ny // 2), axis=0), -(nx // 2), axis=1)

def filt_norm_fac(filt, band, pixscale):
    """ Return the normalization factor for the filter"""

    from astropy.nddata import convolve

    # Do this empirically -- make a small image, smooth with this,
    # check the normalization.  This uses per-beam normalization.
    # The test image is just the un-smoothed beam for this band
    # with a maximum of 1.0
    sigma_pix = spire_fwhm[band] / (pixscale * math.sqrt(8 * math.log(2)))
    test_im = make_kernel((4 * filt.shape[0], 4 * filt.shape[1]),
                          kernelwidth=sigma_pix, force_odd=True)
    test_im /= test_im.max()

    # Convolve
    # Note that convolve modifies filt, so we have to use a copy
    test_im = convolve(test_im, filt.copy(), boundary='zeros')

    # Return the factor to multiply the filter by
    return 1.0 / test_im.max()

def make_matched_filter(band, nx, ny, inst_noise, conf_noise=None, 
                        pixscale=None):
    """ Builds the real space matched filters for a single SPIRE band.

    Parameters
    ----------
    band: string
      Name of the band ('PSW', 'PMW', or 'PLW').

    nx: int
      Size of the target maps along the x direction

    ny: int
      Size of the target maps along the y direction

    inst_noise: float
      Instrument (white) noise, in Jy

    conf_noise: float
      Confusion noise.  Has a band based default if not provided

    pixscale: float
      Pixel scale, in arcsec, of the maps.  Has a band based default
      if not provided

    inst_noise: float
      Instrument noise of combined maps, in Jy

    conf_noise: float
      Confusion noise of the combined maps, in Jy

    Returns
    -------
    filt: ndarray
      The real space matched filter.  Note that
      they are not applied to the data by this function.
    """
    
    if not band in spire_fwhm:
        raise ValueError("Unknown band: %s" % band)
    nx = int(nx)
    if nx <= 0:
        raise ValueError("Invalid (non-positive) nx")
    ny = int(ny)
    if ny <= 0:
        raise ValueError("Invalid (non-positive) ny")
    inst_noise = float(inst_noise)
    if inst_noise < 0:
        raise ValueError("Invalid (negative) instrument noise")
    if conf_noise is None:
        conf_noise = spire_confusion[band]
    else:
        conf_noise = float(conf_noise)
        if conf_noise < 0:
            raise ValueError("Invalid (negative) confusion "
                             "noise: %f" % conf_noise)
    if pixscale is None:
        pixscale = smap_pixsize[band]
    else:
        pixscale = float(pixscale)
        if pixscale <= 0.0:
            raise ValueError("Invalid (non-positive) pixel "
                             "scale: %f" % pixscale)

    if inst_noise == 0 and conf_noise == 0:
        raise ValueError("One of inst_noise or conf_noise must be non-zero")

    # First white noise
    if inst_noise > 0:
        p_noise = np.empty((ny, nx), dtype=np.float32)
        p_noise[:, :] = nx * ny * inst_noise**2
    else:
        p_noise = np.zeros((ny, nx), dtype=np.float32)


    # We build this ourselves to get the right size and
    # centered in the right place
    beam = make_beam(band, pixscale, nx, ny)
    beam_fft = fft.fft2(beam)

    # Add in confusion noise
    if conf_noise > 0:
        scale_confusion = conf_noise**2 / beam.var()
        p_noise += scale_confusion * np.abs(beam_fft)**2

    # Get real space filter, unshifting and taking only
    # the center. 
    filt = np.real(fft.ifft2(beam500_fft / p_noise))
    npix_take = math.ceil(6 * spire_fwhm[band] / pixscale)
    if npix_take % 2 == 0:
        npix_take += 1
    if npix_take > nx or npix_take > ny:
        raise ValueError("Map was too small to adequately apply filter")
    filt = np.roll(np.roll(filt, npix_take // 2, axis=0), 
                   npix_take // 2, axis=1)
    filt = filt[0:npix_take, 0:npix_take]

    # Normalize
    filt *= filt_norm_fac(filt, band, pixscale)

    return filt
    

def make_red_matched_filter(nx, ny, pixscale, inst_noise, conf_noise,
                            verbose=False):
    """ Builds the real space matched filters to match all SPIRE bands

    This builds the Ed Chapin style matched filter for
    the 500 micron band, and then finds the filters to match the
    other two bands to that beam.  

    Parameters
    ----------
    nx: int
      Size of the target maps along the x direction

    ny: int
      Size of the target maps along the y direction

    pixscale: float
      Pixel scale, in arcsec, of the maps

    inst_noise: float
      Instrument noise of combined maps, in Jy

    conf_noise: float
      Confusion noise of the combined maps, in Jy

    verbose: bool
      Print informational messages as the code runs

    Returns
    -------
    filts: tuple
      The real space matched filters at 250, 350, 500um.  Note that
      they are not applied to the data by this function.
    """

    nx = int(nx)
    if nx <= 0:
        raise ValueError("Invalid (non-positive) nx")
    ny = int(ny)
    if ny <= 0:
        raise ValueError("Invalid (non-positive) ny")
    pixscale = float(pixscale)
    if pixscale <= 0.0:
        raise ValueError("Invalid (non-positive) pixel scale")
    inst_noise = float(inst_noise)
    if inst_noise < 0:
        raise ValueError("Invalid (negative) instrument noise")
    conf_noise = float(conf_noise)
    if conf_noise < 0:
        raise ValueError("Invalid (negative) confusion noise")

    if inst_noise == 0 and conf_noise == 0:
        raise ValueError("One of inst_noise or conf_noise must be non-zero")

    # First white noise
    if inst_noise > 0:
        p_noise = np.empty((ny, nx), dtype=np.float32)
        p_noise[:, :] = nx * ny * inst_noise**2
    else:
        p_noise = np.zeros((ny, nx), dtype=np.float32)

    # Now, work out the matched filter for the 500um beam
    # We build this ourselves to get the right size and
    # centered in the right place
    if verbose:
        print("Preparing 500um filter")
    beam500 = make_beam('PLW', pixscale, nx, ny)
    beam500_fft = fft.fft2(beam500)

    # Add in confusion noise
    if conf_noise > 0:
        scale_confusion = conf_noise**2 / beam500.var()
        p_noise += scale_confusion * np.abs(beam500_fft)**2

    # Get real space 500um filter, unshifting and taking only
    # the center.  Note: we haven't normalized yet
    filt500 = np.real(fft.ifft2(beam500_fft / p_noise))
    npix_take = math.ceil(6 * spire_fwhm['PLW'] / pixscale)
    if npix_take % 2 == 0:
        npix_take += 1
    if npix_take > nx or npix_take > ny:
        raise ValueError("Map was too small to adequately apply filter")
    filt500 = np.roll(np.roll(filt500, npix_take // 2, axis=0), 
                      npix_take // 2, axis=1)
    filt500 = filt500[0:npix_take, 0:npix_take]

    # Build 250um filter next
    if verbose:
        print("Preparing 250um filter")
    beam250 = make_beam('PSW', pixscale, nx, ny)
    filt_fft = fft.fft2(beam250)
    filt_fft = beam500_fft**2 / (filt_fft * p_noise)
    filt250 = np.real(fft.ifft2(filt_fft))
    filt250 = np.roll(np.roll(filt250, npix_take // 2, axis=0), 
                      npix_take // 2, axis=1)
    filt250 = filt250[0:npix_take, 0:npix_take]

    # And finally 350
    if verbose:
        print("Preparing 350um filter")
    beam350 = make_beam('PMW', pixscale, nx, ny)
    filt_fft = fft.fft2(beam350)
    filt_fft = beam500_fft**2 / (filt_fft * p_noise)
    filt350 = np.real(fft.ifft2(filt_fft))
    filt350 = np.roll(np.roll(filt350, npix_take // 2, axis=0), 
                      npix_take // 2, axis=1)
    filt350 = filt350[0:npix_take, 0:npix_take]

    # Normalize
    if verbose:
        print("Normalizing filters")
    filt250 *= filt_norm_fac(filt250, 'PSW', pixscale)
    filt350 *= filt_norm_fac(filt350, 'PMW', pixscale)
    filt500 *= filt_norm_fac(filt500, 'PLW', pixscale)

    return (filt250, filt350, filt500)
    
    
def matched_filter(inmap, conf_noise=None, verbose=False):
    """ Apply matched filtering to the input SPIRE map

    Parameters
    ----------
    inmap: smap_map
      SPIRE map to filter

    conf_noise: float
      Confusion noise RMS level in combined pixel in Jy in the un-smoothed
      maps.  Has a default based on the band if not provided.

    verbose: bool
      Print informational messages as the code runs

    Returns
    -------
    filts: ndarray
      The real space filter that has been applied.  In addition,
      the input maps are smoothed.

    Notes
    -----
      Currently does not adjust the error maps.
    """
    
    # Make sure the input map is acceptable
    if not inmap.has_data:
        raise Exception("Input map has no data")
    if not inmap.has_error:
        raise Exception("Input map does not have error information")
    pixscale = inmap.pixscale
    nx = inmap.xsize
    ny = inmap.ysize

    # Make sure band name is known
    if not hasattr(inmap, 'names'):
        raise Exception("Don't know band of SPIRE map")
    if not inmap.names in spire_fwhm:
        raise Exception("Unknown band from input map: %s" % inmap.names)

    if conf_noise is None:
        conf_noise = spire_confusion[inmap.names]
    else:
        conf_noise = float(conf_noise)

    # Get the noise estimate
    inst_sigma = inmap.estimate_noise()
    if verbose:
        print("Estimated map noise: %0.5f" % inst_sigma)

    # Get filter
    filts = make_matched_filter(nx, ny, inst_sigma, conf_noise=conf_noise,
                                pixscale=inmap.pixscale)

    # Apply smoothing, taking into account bad values
    if verbose:
        print("Applying smoothing")
    inmap.convolve(filt)

    return filt


def matched_filter_red(map250, map350, map500, conf_noise, k1=-0.3919, k2=0.0,
                       verbose=False):
    """ Apply matched filtering to the triple image set.

    Parameters
    ----------
    map250: smap_map
      250um map.

    map350: smap_map
      350um map.  Must have the same pixel scale as map250 and be the same
      size.

    map500: smap_map
      500um map.  Must have the same pixel scale as map250 and be the same
      size.

    conf_noise: float
      Confusion noise RMS level in combined pixel in Jy in the un-smoothed
      maps.

    k1: float
      Scale for 250um map.  -1 <= k1 <= 1; used to compute instrument
      noise levels in the combined map.

    k2: float
      Scale for 350um map.  -1 + k1^2 <= k2 <= 1 - k1^2; used to compute 
      instrument noise levels in the combined map.

    verbose: bool
      Print informational messages as the code runs

    Returns
    -------
    filts: tuple
      The real space filters that have been applied.  In addition,
      the input maps are smoothed.

    Notes
    -----
      All three of the input maps should have error information.  The
      idea is that the maps will finally be combined using

        D = k1 * map250 + k2 * map350 + sqrt(1.0 - k1^2 - k2^2) * map500

      The matched combination is based on the propogated instrument noise
      (using the above coefficients) and the confusion noise
      estimate for the combination.  Since the latter is hard to determine,
      the user has to input it.

      Currently does not adjust the error maps.  Also, the code will
      always treat the first map as PSW, etc., even if it isn't.  So entering
      them in the wrong order will quietly do the wrong thing.
    """
    
    # Check k coefficients
    k1 = float(k1)
    k2 = float(k2)
    if k1**2 + k2**2 >= 1.0:
        raise ValueError("k1^2 + k2^2 must be less than 1.")
    k3 = math.sqrt(1.0 - k1**2 - k2**2)

    conf_noise = float(conf_noise)
    if conf_noise < 0.0:
        raise ValueError("Invalid (negative) confusion noise: %f" % conf_noise)
    
    # Make sure the maps have data
    if not map250.has_data:
        raise Exception("250um map has no data")
    if not map350.has_data:
        raise Exception("350um map has no data")
    if not map500.has_data:
        raise Exception("500um map has no data")

    # Make sure maps all have error information
    if not map250.has_error:
        raise Exception("250um map does not have error information")
    if not map350.has_error:
        raise Exception("350um map does not have error information")
    if not map500.has_error:
        raise Exception("500um map does not have error information")

    # Make sure pixel scales are in reasonable agreement
    pixscale = map250.pixscale
    if abs(map350.pixscale - pixscale) / pixscale > 1e-3:
        raise Exception("350um map has different pixel scale than 250um one")
    if abs(map500.pixscale - pixscale) / pixscale > 1e-3:
        raise Exception("500um map has different pixel scale than 250um one")

    # Make sure maps are the same size
    nx = map250.xsize
    ny = map250.ysize
    if map350.xsize != nx or map350.ysize != ny:
        raise Exception("350um map not the same size as 250um map")
    if map500.xsize != nx or map500.ysize != ny:
        raise Exception("500um map not the same size as 250um map")

    # Get the noise estimate
    white_var = 0.0
    if k1 > 0:
        white_var += (k1 * map250.estimate_noise())**2
    if k2 > 0:
        white_var += (k2 * map350.estimate_noise())**2
    if k3 > 0:
        white_var += (k3 * map500.estimate_noise())**2
    inst_sigma = math.sqrt(white_var)
    if verbose:
        print("Estimated map noise: %0.5f" % inst_sigma)

    # Get filters
    filts = make_red_matched_filter(nx, ny, pixscale, inst_sigma, conf_noise)

    # Apply smoothing, taking into account bad values
    if verbose:
        print("Applying smoothing")
    map250.convolve(filts[0])
    map350.convolve(filts[1])
    map500.convolve(filts[2])

    return filts
