from distutils.core import setup

import sys
major, minor1, minor2, release, serial = sys.version_info

if (major < 3) and (minor1 < 7):
    raise SystemExit("smap requires at least python 2.7")

setup(
    name="smap_hermes",
    version="0.1.2",
    author="Alexander Conley",
    author_email="alexander.conley@colorado.edu",
    packages=["smap_hermes"],
    license="GPL",
    description="HerMES SMAP routines",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    requires=['numpy(>=1.7.0)', 'scipy (>=0.8.0)', 'astropy(>=0.4.0)']
)
