# Gaia_Distances
A python code to infer the distance to stars based upon Gaia parallaxes.  This code reproduces the inferred distances of Bailer-Jones et al. 2018.  In addition, this code offers the ability to include astrometric excess noise, change the zero point, and change the field star length scale.

Requirements:
Python 2.7
scipy
numpy

Use:
In your pthon code add these lines near the top.  Of course, change the /path/ to the location of Gaia_Distances

import sys
sys.path.append('/path/')
from Gaia_Distances import *

Then you may call the following useful functions

rmode = Calculate_Mode() #returns the mode of the distance
HDI = Calculate_HDI() # returns the highest density interval
stats = CalculateModeHDIForNstars() #returns mode and HDI for a set of stars

