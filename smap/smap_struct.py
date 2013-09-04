from __future__ import print_function

import math
import astropy.wcs as wcs
import astropy.io.fits as fits
import astropy.coordinates as coords
from astropy import units as u
import numpy as np

""" Reads and writes SMAP fits structures"""

__all__ = ["smap_map"]

class smap_map:
    """ Represents SMAP map

    Notes
    -----
     Following pyfits conventions, the indexing scheme for the image,
    error map, etc., is [y, x] rather than [x, y]
    """
    def __init__(self, filename=None):
        """ Basic constructor

        Parameters
        ----------
        filename: string or None
          If present, the file specificed by this filename is read in
        """

        self._has_data = False
        self._has_exposure = False
        self._has_error = False
        self._has_mask = False
        if not filename is None:
            self.read(filename)

    def read(self, filename, read_error=True, read_exposure=True,
             read_mask=True):
        """ Read from a FITS file.

        Parameters
        ----------
        filename: string
          FITS file to read in data from.

        read_error: boolean
          Read in error information (if present)

        read_exposure: boolean
          Read in exposure information (if present)

        read_mask: boolean
          Read in mask information (if present)
        """

        hdulist = fits.open(filename, uint=True)

        # Figure out what we have
        try:
            image_ext = hdulist.index_of('image')
        except KeyError:
            hdulist.close()
            raise IOError("File %s doesn't have primary image" % filename)
        
        self.image = hdulist[image_ext].data
        self.astrometry = wcs.WCS(hdulist[image_ext].header)
        if 'WAVELN' in hdulist[image_ext].header:
            self.wave = int(hdulist[image_ext].header['WAVELN'])
        elif hasattr(self, 'wave'):
            del self.wave
        if 'DESC' in hdulist[image_ext].header:
            self.bands = hdulist[image_ext].header['DESC']
        elif hasattr(self, 'bands'):
            del self.bands
        self.xsize = self.image.shape[1] # Note the transpose!
        self.ysize = self.image.shape[0]
        self._has_filter = False # Not supported yet
        # This is a non-FITS compliant keyword, so avoid it for now
        # self.todmask = hdulist[image_ext].header['TOD_EXCLUDEMASK']

        # Set name
        namedict = {250: 'PSW', 350: 'PMW', 500: 'PLW'}
        if hasattr(self, 'wave') and self.wave in namedict:
            self.names = namedict[self.wave]
        elif hasattr(self, 'names'):
            del self.names

        # Set up pixel scale
        crpix = self.astrometry.wcs.crpix
        tval = np.array([[crpix[0], crpix[1]], [crpix[0]+1, crpix[1]],
                         [crpix[0], crpix[1]+1]])
        world = self.astrometry.all_pix2world(tval, 1)
        p0 = coords.ICRSCoordinates(ra=world[0,0], dec=world[0,1], 
				    unit=(u.degree, u.degree))
        s1 = p0.separation(coords.ICRSCoordinates(ra=world[1,0],dec=world[1,1], 
						  unit=(u.degree, u.degree)))
        s2 = p0.separation(coords.ICRSCoordinates(ra=world[2,0],dec=world[2,1], 
						  unit=(u.degree, u.degree)))
        self.pixscale = math.sqrt(s1.arcsecs * s2.arcsecs)
        
        if read_error:
            try:
                error_ext = hdulist.index_of('error')
                self._has_error = True
                self.error = hdulist[error_ext].data
            except KeyError:
                self._has_error = False
                if hasattr(self, 'error'):
                    del self.error
        else:
            self._has_error = False
            if hasattr(self, 'error'):
                del self.error

        if read_exposure:
            try:
                exposure_ext = hdulist.index_of('exposure')
                self._has_exposure = True
                self.exposure = hdulist[exposure_ext].data
            except KeyError:
                self._has_exposure = False
                if hasattr(self, 'exposure'):
                    del self.exposure
        else:
            self._has_exposure = False
            if hasattr(self, 'exposure'):
                del self.exposure

        if read_mask:
            try:
                mask_ext = hdulist.index_of('mask')
                self._has_mask = True
                self.mask = hdulist[mask_ext].data
            except KeyError:
                self._has_mask = False
                if hasattr(self, 'mask'):
                    del self.mask
        else:
            self._has_mask = False
            if hasattr(self, 'mask'):
                del self.mask
            
        hdulist.close()
        self._has_data = True

    def write(self, filename):
        """ Write as FITS file.

        Parameters
        ----------
        filename: string
          File to write data to in FITS format.
        """
        
        if not self._has_data:
            raise IOError("Attempting to write empty smap map")

        # Set up the header
        head = self.astrometry.to_header()
        if hasattr(self, 'wave'):
            head['WAVELN'] = self.wave
        if hasattr(self, 'bands'):
            head['DESC'] = self.bands

        # Set up image
        hdulist = fits.HDUList(fits.PrimaryHDU())
        hdulist.append(fits.ImageHDU(data=self.image, 
                                     header=head, name='image'))
        if self._has_error:
            hdulist.append(fits.ImageHDU(data=self.error, 
                                         header=head, name='error'))
        if self._has_exposure:
            hdulist.append(fits.ImageHDU(data=self.exposure, 
                                         header=head, name='exposure'))
        if self._has_mask:
            hdulist.append(fits.ImageHDU(data=self.mask, uint=True,
                                         header=head, name='mask'))

        hdulist.writeto(filename, clobber=True, checksum=True)
        
    def create(self, image, pixscale, racen, deccen,  wave=None,
               bands=None, error=None, exposure=None, mask=None):
        """ Create an image from input data.

        Parameters
        ----------
        image: ndarray
          Map.  2D array.  Note that, as is the case for error,
          exposure, etc., this must use the indexing convention
          [y, x] following astropy.io.fits conventions.
        
        wave: float
          Wavelength in microns of map

        pixscale: float
          Pixel scale in arcsec

        racen: float
          Central ra value

        deccen: float
          Central dec value

        bands: string
          Optional band description

        error: ndarray
          Error array

        exposure: ndarray
          Exposure array

        mask: ndarray
          Mask array
        """
    

        if not type(image) == np.ndarray:
            raise Exception("Input image not ndarray")
        if len(image.shape) != 2:
            raise Exception("Non-2D image frame")

        self._has_data = True
        self.image = image.astype(np.float32)
        self.xsize = image.shape[1] # note transpose
        self.ysize = image.shape[0]

        self.pixscale = float(pixscale)
        self.astrometry = wcs.WCS(naxis=2)
        self.astrometry.wcs.crpix = [self.xsize/2, self.ysize/2]
        self.astrometry.wcs.cd = np.array([[-self.pixscale / 3600.0, 0],
                                           [0, self.pixscale / 3600.0]])
        self.astrometry.wcs.crval = np.array([racen, deccen],
                                             dtype=np.float64)
        self.astrometry.wcs.ctype = ['RA---TAN'.encode(), 
                                     'DEC--TAN'.encode()]

        if not wave is None:
            self.wave = int(wave)
            namedict = {250: 'PSW', 350: 'PMW', 500: 'PLW'}
            if self.wave in namedict:
                self.names = namedict[self.wave]
            elif hasattr(self, 'names'):
                del self.names
        else:
            if hasattr(self, 'wave'): del self.wave
            if hasattr(self, 'names'): del self.names
            
        if not bands is None:
            self.bands = str(bands)
        elif hasattr(self, 'bands'): 
            del self.bands

        if not error is None:
            if not type(error) == np.ndarray:
                raise ValueError("Input error array not ndarray")
            if len(error.shape) != 2:
                raise ValueError("Input error array not 2D")
            if error.shape[0] != self.image.shape[0] or \
                    error.shape[1] != self.image.shape[1]:
                raise ValueError("Input error array not same extent as image")
            self.error = error.astype(np.float32)
            self._has_error = True
        else:
            if hasattr(self, 'error'):
                del self.error
            self._has_error = False

        if not exposure is None:
            if not type(exposure) == np.ndarray:
                raise ValueError("Input exposure array not ndarray")
            if len(exposure.shape) != 2:
                raise ValueError("Input exposure array not 2D")
            if exposure.shape[0] != self.image.shape[0] or \
                    exposure.shape[1] != self.image.shape[1]:
                raise ValueError("Input exposure array not same "
                                 "extent as image")
            self.exposure = exposure.astype(np.float32)
            self._has_exposure = True
        else:
            if hasattr(self, 'exposure'):
                del self.exposure
            self._has_exposure = False

        if not mask is None:
            if not type(mask) == np.ndarray:
                raise ValueError("Input mask array not ndarray")
            if len(mask.shape) != 2:
                raise ValueError("Input mask array not 2D")
            if mask.shape[0] != self.image.shape[0] or \
                    mask.shape[1] != self.image.shape[1]:
                raise ValueError("Input mask array not same "
                                 "extent as image")
            self.mask = mask.astype(np.uint16)
            self._has_mask = True
        else:
            if hasattr(self, 'mask'):
                del self.mask
            self._has_mask = False
            
    @property
    def has_data(self):
        return self._has_data

    @property
    def has_error(self):
        return self._has_error

    @property
    def has_exposure(self):
        return self._has_exposure

    @property
    def has_mask(self):
        return self._has_mask

    def add_noise(self, sigma):
        """ Add Gaussian noise to map.

        Will update the error extension if present"""

        if not self._has_data:
            raise Exception("Trying to add noise to map with no data")

        sigval = float(sigma)
        if sigval < 0:
            raise ValueError("Invalid (negative) sigma: %f" % sigval)
        if sigval == 0:
            return #Nothing to do
        if self._has_error:
            self.error = np.sqrt(self.error**2 + sigval**2)

        self.map += np.random.normal(scale=sigval, size=self.map.shape)

    def where_bad(self):
        """ Returns a list of pixels that are bad (no exposure, masked,
        non-finite"""
        badpix = np.zeros(self.image.shape, dtype=np.bool)
        np.place(badpix, ~np.isfinite(self.image), True)
        if self._has_exposure:
            np.place(badpix, self.exposure <= 0.0, True)
        if self._has_mask:
            np.place(badpix, self.mask != 0, True)
        return np.nonzero(badpix)
        
    def estimate_noise(self):
        """ Attempts to estimate a representative noise level for the map

        The approach is to take the central 20% of the image (in each
        dimension), mask out all pixels with mask bits, zero
        exposure, or non-finite values, and return the median noise
        estimate.
        """
        
        if not self._has_data:
            raise Exception("Trying to estimate noise on empty map")
        if not self._has_error:
            raise Exception("Can't estimate noise in map with no error info")

        xcen = self.xsize // 2
        ycen = self.ysize // 2
        dx = math.ceil(self.xsize * 0.1)
        dy = math.ceil(self.ysize * 0.1)
        cen_im = self.image[ycen-dy:ycen+dy, xcen-dx:xcen+dx] # Note transpose!
        badpix = np.zeros(cen_im.shape, dtype=np.bool)
        np.place(badpix, ~np.isfinite(cen_im), True)
        if self._has_exposure:
            np.place(badpix, 
                     self.exposure[ycen-dy:ycen+dy, xcen-dx:xcen+dx] <= 0.0, 
                     True)
        if self._has_mask:
            np.place(badpix, 
                     self.mask[ycen-dy:ycen+dy, xcen-dx:xcen+dx] != 0,
                     True)
        nbad = np.count_nonzero(badpix)
        npix = cen_im.shape[0] * cen_im.shape[1]
        errmap = self.error[ycen-dy:ycen+dy, xcen-dx:xcen+dx]
        if nbad != 0:
            # There are some pixels to skip
            if nbad == npix:
                raise Exception("All pixels in central region masked")
            mederr = np.median(errmap[np.nonzero(~badpix)])
        else:
            mederr = np.median(errmap)
        return mederr

    def pix2world(self, xpos, ypos):
        """ Convert image positions to world coordinates (ra, dec)

        Parameters
        ----------
        xpos: ndarray
          X positions to convert, zero indexed.

        ypos: ndarray
          Y positions to convert, zero indexed.

        Returns
        -------
        radec: ndarray
          A npos by 2 array of ra/dec values in decimal degrees at
          whatever epoch the map coordinates are specified in.

        Notes
        -----
         xpos and ypos must be broadcastable to each other.
         Also remember that the data follows pyfits conventions,
         so the map values corresponding to xpos, ypos are
         map.image[ypos, xpos].
        """

        if not self._has_data:
            raise Exception("No data present")

        return self.astrometry.all_pix2world(xpos, ypos, 0)

    def world2pix(self, ra, dec):
        """ Convert image positions to world coordinates (ra, dec)

        Parameters
        ----------
        ra: ndarray
          RA positions to convert, in the epoch of the map.

        dec: ndarray
          DEC positions to convert, in the epoch of the map.

        Returns
        -------
        xypos
          A npos by 2 array of x, y positions, zero indexed.

        Notes
        -----
         ra, dec must be broadcastable to each other.
         Also remember that the data follows pyfits conventions,
         so the map values corresponding to xypos are
         map.image[xypos[:, 1], xypos[:, 0]].
        """

        if not self._has_data:
            raise Exception("No data present")

        return self.astrometry.all_world2pix(ra, dec, 0)

    def convolve(self, kernel):
        """ Convovles the map, dealing with bad pixels

        Parameters
        ----------
        kernel: ndarray
          Input smoothing filter in real space.  Should be much smaller 
          than the input map, or this will be quite slow.  Note that
          you want this to be transposed to follow the fits convention.

        Notes
        -----
          Currently does not alter the error information, just the
          actual map.  Uses brute-force (non-FFT) convolution, since
          this is usually faster for small kernels.
        """
    
        from astropy.nddata import convolve

        badpix = self.where_bad()
        nbad = len(badpix[0])

        # v0.2 of astropy.nddata.convolve modifies kernel, so make
        # a copy
        if nbad == 0:
            # Hey, no bad pixels.  probably a simulation, but easy
            self.image = convolve(self.image, kernel.copy(), boundary='wrap')
        else:
            self.image[badpix] = np.nan
            self.image = convolve(self.image, kernel.copy(), boundary='wrap')

    def __str__(self):
        """ String representation"""
        if not self._has_data:
            return "Empty SMAP map"
        outstr = "SMAP map\n"
        if hasattr(self, 'wave'):
            outstr += " wavelength: %d [um]" % self.wave
        if hasattr(self, 'names'):
            outstr += " desc: %s" % self.names
        outstr += "\n RA: %0.4f [deg]  DEC: %0.4f [deg]" %\
            (self.astrometry.wcs.crval[0], self.astrometry.wcs.crval[1])
        outstr += " Pixscale: %0.2f [arcsec]" % self.pixscale
        outstr += "\n Map [%d x %d]" % (self.xsize, self.ysize)
        if self._has_error:
            outstr += "\n Error map [%d x %d]" % (self.xsize, self.ysize)
        if self._has_exposure:
            outstr += "\n Exposure map [%d x %d]" % (self.xsize, self.ysize)
        if self._has_mask:
            outstr += "\n Mask map [%d x %d]" % (self.xsize, self.ysize)

        return outstr
