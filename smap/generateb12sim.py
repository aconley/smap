""" Generate a Bethermin et al. 2012 simulation and store in SMAP map"""

from .smap_struct import smap_map

def generate_gauss(area, wave=[250.0,350,500], pixsize=[6.0, 8.33333, 12.0],
                   racen=25.0, deccen=0.0, fwhm=[17.6, 23.9, 35.2], nfwhm=5.0, 
                   bmoversamp=5, gensize=150000, truthtable=False, log10Mb=11.2,
                   alpha=1.3, log10Mmin=8.5, log10Mmax=12.75, ninterpm=2000,
                   zmin=0.1, zmax=10.0, Om0=0.315, H0=67.7, phib0=-3.02,
                   gamma_sfmf=0.4, ninterpz=1000, rsb0=0.012, gammasb=1.0,
                   zsb=1.0, logsSFRM0=-10.2, betaMS=-0.2, zevo=2.5,
                   gammams=3.0, bsb=0.6, sigmams=0.15, sigmasb=0.2,
                   mnU_MS0=4.0, gammaU_MS0=1.3, z_UMS=2.0, mnU_SB0=35.0, 
                   gammaU_SB0=0.4, z_USB=3.1, scatU=0.2, ninterpdl=200,
                   sigma_inst=None, verbose=False):
        """ Generates simulated maps as SMAP structures using a Gaussian beam.

        Parameters
        ----------
        area: float
          Area of generated maps, in deg^2

        wave: ndarray
          Wavelengths to generate maps at, in microns.

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

        gensize: int
          In order to try to save memory, sources are added to the maps
          in chunks of this size.  If set to 0, all the sources are
          generated at once.
        
        truthtable: bool
          If set to true, then the truth table is also returned (giving
          the source fluxes and positions).  If this is set, then
          gensize is set to 0.

        log10Mb: float
          Log10 of Mb, in solar masses

        alpha: float
          Power-law slope of low mass distribution
 
        log10Mmin: float
          Log10 minimum mass to generate, in solar masses

        log10Mmax: float
          Log10 maximum mass to generate, in solar masses

        ninterpm: float
          Number of mass interpolation samples to use

        zmin: float
          Minimum z generated

        zmax: float
          Maximum z generated

        Om0: float
          Density parameter of matter

        H0: float
          Hubble constant in km / s / Mpc

        phib0: float
          log 10 number density at SFMF break in comoving Mpc^-3

        gamma_sfmf: float
          Evolution of the number density of the SFMF at z > 1

        ninterpz: int
          Number of z interpolation samples to use

        rsb0: float
          Relative amplitude of SB distribution to MS one.

        gammasb: float
          Redshift evolution in SB relative amplitude.

        zsb: float
          Redshift where SB relative amplitude stops evolving

        logsSFRM0: float
          Base Log 10 specific star formation rate.

        betaMs: float
          Slope of sFRM-M relation

        zevo: float
          Redshift where MS normalization stops evolving

        gammams: float
          Redshift evolution power law for sSFR

        bsb: float
          Boost in sSFR for starbursts, in dex

        sigmams: float
          Width of MS log-normal distribution

        sigmasb: float
          Width of SB log-normal distribution

        mnU_MS0: float
          Average ultraviolet intensity field in MS galaxies at z=0

        gammaU_MS0: float
          Evolution in <U> for MS

        z_UMS: float
          Redshift where <U> stops evolving for MS galaxies

        mnU_SB0: float
          Average ultraviolet intensity field in SB galaxies at z=0

        gammaU_SB0: float
          Evolution in <U> for SB

        z_USB: float
          Redshift where <U> stops evolving for SB galaxies

        scatU: float
          Scatter in <U>, in dex

        ninterpdl: float
          Number of interpolation points in the luminosity distance.

        sigma_inst: ndarray or None
          Map instrument noise, in Jy.  If None, no instrument
          noise is added.

        verbose: bool
          Print informational messages as it runs.

        Returns
        -------
          A tuple containing the input maps.  If truthtable is
        set on initialization, also includes the truth table of
        positions and fluxes, where the positions are relative
        to the first map.
        """

        try:
            from bethermin12_sim import genmap_gauss
        except ImportError:
            raise Exception("bethermin12_sim library not installed")

        # Generate maps
        gen = genmap_gauss(wave=wave, pixsize=pixsize, fwhm=fwhm,
                           nfwhm=nfwhm, bmoversamp=bmoversamp,
                           gensize=gensize, truthtable=False, log10Mb=log10Mb,
                           alpha=alpha, log10Mmin=log10Mmin, 
                           log10MMax=log10MMax, ninterpm=ninterpm, zmin=zmin,
                           zmax=zmax, Om0=Om0, H0=H0, phib0=phib0,
                           gamma_sfmf=gamma_sfmf, ninterpz=ninterpz,
                           rsb0=rsb0, gammasb=gammasb, zsb=zsb, 
                           logsSFRM0=logsSFRM0, betaMS=betaMS, zevo=zevo,
                           bsb=bsb, sigmams=sigmams, sigmasb=sigmasb,
                           mnU_MS0=mnU_MS0, gammaU_MS0=gammaU_MS0,
                           z_UMS=z_UMS, mnU_SB0=mnU_SB0, gammaU_SB0=gammaU_SB0,
                           z_USB=z_USB, scatU=scatU, ninterpdl=ninterpdl)
        maps = gen.generate(area, verbose=verbose)
        nmaps = len(wave)

        # Figure out sigma situation
        if sigma_inst is None:
            has_sigma = False
        elif type(sigma_inst) == list:
            if len(sigma_inst) != nmaps:
                if len(sigma_inst) == 1:
                    int_sigma = sigma_inst[0] * np.ones_like(gen.wave)
                else:
                    raise ValueError("Number of sigmas doesn't match number"
                                     " of wavelengths")
            else:
                int_sigma = np.asarray(sigma_inst, dtype=np.float32)
            has_sigma = True
        elif type(sigma_inst) == np.ndarray:
            if len(sigma_inst) != self._nbands:
                if len(sigma) == 1:
                    int_sigma = sigma_inst[0] * np.ones_like(gen.wave)
                else:
                    raise ValueError("Number of sigmas doesn't match number"
                                     " of wavelengths")
            else:
                int_sigma = sigma_inst.astype(np.float32, copy=False)
            has_sigma = True
        else:
            int_sigma=  float(sigma_inst) * np.ones_like(gen.wave)
            has_sigma = True

        for i in range(nmaps):
            cmap = maps[i]
            cmap -= cmap.mean()
            if has_sigma:
                error = int_sigma[i] * np.ones_like(cmap)
                cmap += np.random.normal(scale=int_sigma[i], size=cmap.shape)
            else:
                error = None
            mapstr = smap_map()
            mapstr.create(cmap, self._pixsize[i], racen, deccen,
                          wave=self._wave[i], error=error)
            maps[i] = mapstr
            del cmap

        return maps
