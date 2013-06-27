
""" Sets defaults for various SMAP parameters"""

""" Nominal wavelengths of SPIRE bands, in microns"""
spire_wavelength = {'PSW': 250, 'PMW': 350, 'PLW': 500}

""" FWHM of SPIRE beam, in arcsec"""
spire_fwhm = {'PSW': 17.6, 'PMW': 23.9, 'PLW': 35.2}

""" Confusion noise for SPIRE, in Jy (Nguyen et al. 2010)"""
spire_confusion = {'PSW': 0.0058, 'PMW': 0.0063, 'PLW': 0.0068}

""" Default SMAP pixel sizes, in arcsec, by band"""
smap_pixsize = {'PSW': 6.0, 'PMW': 25 / 3.0, 'PLW': 12.0}
