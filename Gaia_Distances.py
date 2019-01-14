import numpy as np
from scipy.integrate import quadrature
import matplotlib.pyplot as pl
from sys import exit


#See
# Bailer-Jones, C.A.L. et al. 2018 https://arxiv.org/pdf/1804.10121.pdf
# Bailer-Jones, C.A.L. 2015 http://iopscience.iop.org/article/10.1086/683116

def Calculate_Mode(piobs,sigmapi,pioffset,ll):
	#piobs = observed parallax in mas
	#sigmapi = measured uncertainty
	#lengthscale = lengthscale in the exponential prior

	#To find the mode we use the cubic equation, eq. 19 of BJ2015
	#r^3/L - 2r^2 + \pi*r/sigma^2 - 1/sigma^2 = 0


	pitrue = piobs - pioffset
	a = 1./ll
	b = -2.
	c = pitrue/sigmapi**2
	d = -1./sigmapi**2
	coeffs = [a,b,c,d]
	
	roots = np.roots(coeffs)
	realroots = np.isreal(roots)
	#print "The roots are",roots

	reals = np.real(roots[realroots])
	if (np.size(reals) == 1):
		#only one real root, so return it
		rmode = reals[0]
	else:
		#multiple roots
		if (piobs >= 0.):
			rmode = np.amin(reals)
		else:
			rmode = np.amax(reals)

	return rmode


def posterior(x,piobs,sigma,pioffset,ll,norm=1.):
	pitrue = piobs-pioffset
	p = norm*x**2*np.exp(-x/ll - (pitrue-1./x)**2/(2.*sigma**2))
	return p

def CalculateBounds(rmode,piobs,sigma,pioffset,ll):
	pitrue = piobs - pioffset
	if (pitrue >= 0.):
		sigmar = sigma/pitrue*rmode
		xp = min(8.*sigmar,8.*ll)
		xupp = rmode+xp
		xlow = max(rmode-4.*sigmar,0.001)
	else:
		xupp = 8.*ll
		xlow = 0.001
	return xlow,xupp

def dydr(x,piobs,sigma,pioffset,ll):
	pitrue = piobs-pioffset
	return -1./ll - (pitrue-1./x)/(x*sigma)**2

def d2ydr2(x,piobs,sigma,pioffset,ll):
	pitrue = piobs-pioffset
	return (2.*pitrue - 3./x)/(x**3*sigma**2)

def dpdr(x,piobs,sigma,pioffset,ll,pnorm):
	dp = ( 2./x + dydr(x,piobs,sigma,pioffset,ll) ) * posterior(x,piobs,sigma,pioffset,ll,norm=pnorm)
	return dp

def d2pdr2(x,piobs,sigma,pioffset,ll,pnorm):
	dy = dydr(x,piobs,sigma,pioffset,ll)
	dy2 = d2ydr2(x,piobs,sigma,pioffset,ll)
	dp2 =  dy2 + dy**2 + 4.*dy/x + 2./x**2
	dp2 = dp2*posterior(x,piobs,sigma,pioffset,ll,norm=pnorm)
	return dp2

def Calculate_HDI(rmode,piobs,sigma,pioffset,ll,pnorm,ptarget=0.68):
	dp2 = d2pdr2(rmode,piobs,sigma,pioffset,ll,pnorm)
	pmode = posterior(rmode,piobs,sigma,pioffset,ll,norm=pnorm)
	deltap = -0.005*pmode
	deltar0 = np.sqrt(2.*deltap/dp2)
	rp = rmode + deltar0
	rm = rmode - deltar0
	pp = posterior(rp,piobs,sigma,pioffset,ll,norm=pnorm)
	pm = posterior(rm,piobs,sigma,pioffset,ll,norm=pnorm)
	areap = 0.5*(pmode + pp)*deltar0
	aream = 0.5*(pmode + pm)*deltar0
	area = areap + aream
	pm0 = pm
	pp0 = pp


	#Now start adding up linear taylor series expansion
	while area < ptarget:
		dpm = dpdr(rm,piobs,sigma,pioffset,ll,pnorm)
		dpp = dpdr(rp,piobs,sigma,pioffset,ll,pnorm)
		drm = deltap/dpm
		drp = deltap/dpp
		#drm = -deltar0
		#drp = deltar0
		rm = rm + drm
		rp = rp + drp
		pp = posterior(rp,piobs,sigma,pioffset,ll,norm=pnorm)
		pm = posterior(rm,piobs,sigma,pioffset,ll,norm=pnorm)
		areap = 0.5*(pp0 + pp)*drp
		aream = -0.5*(pm0 + pm)*drm
		area = area + areap + aream
		pm0 = pm
		pp0 = pp
		#print rm, rp, area
	return rm,rp

def CalculateModeHDIForNstars(omega,sig,offset,length,showpdf=True):
	nstars = np.shape(omega)[0]
	stats = np.empty((nstars,3))
	for i in range(nstars):
		rmode = Calculate_Mode(omega[i],sig[i],offset,length[i])
		print(rmode)
		xlow,xupp = CalculateBounds(rmode,omega[i],sig[i],offset,length[i])
		norm = quadrature(posterior,xlow,xupp,args=(omega[i],sig[i],offset,length[i]))[0]
		HDI = Calculate_HDI(rmode,omega[i],sig[i],offset,length[i],1./norm)
		stats[i,0] = rmode
		stats[i,1:] = HDI
		print("rmode, CI =",stats[i,:])
		if (showpdf):
			xtry = np.linspace(xlow,xupp,100)
			ps = posterior(xtry,omega[i],sig[i],offset,length[i],norm=1./norm)
			pmode = posterior(rmode,omega[i],sig[i],offset,length[i],norm=1./norm)
			pl.plot(xtry,ps)
			pl.plot([rmode,rmode],[0.,pmode])
			pl.plot([HDI[0],HDI[0]],[0.,pmode])
			pl.plot([HDI[1],HDI[1]],[0.,pmode])
			pl.show()
	return stats 
	
def main():
	data = np.loadtxt('yellowgiants.txt',delimiter=',')
	nstars =  np.shape(data)[0]
	for i in range(nstars):
		piobs = data[i,1]
		sigma = 1.081*data[i,2]
		length = data[i,3]
		offset = -0.08 #Stassun et al. 2018
		#offset = -0.046 #Riess et al. 2018
		#offset = -0.029 #BJ et al. 2018
		rmode = Calculate_Mode(piobs,sigma,offset,length)
		xlow,xupp = CalculateBounds(rmode,piobs,sigma,offset,length)
		norm = quadrature(posterior,xlow,xupp,args=(piobs,sigma,offset,length))[0]
		HDI = Calculate_HDI(rmode,piobs,sigma,offset,length,1./norm)
		xtry = np.linspace(xlow,xupp,100)
		ps = posterior(xtry,piobs,sigma,offset,length,norm=1./norm)
		pmode = posterior(rmode,piobs,sigma,offset,length,norm=1./norm)

		print "rmode, CI =",rmode, HDI
		pl.plot(xtry,ps)
		pl.plot([rmode,rmode],[0.,pmode])
		pl.plot([HDI[0],HDI[0]],[0.,pmode])
		pl.plot([HDI[1],HDI[1]],[0.,pmode])
		pl.show()

if (__name__ == "__main__"):
	main()
