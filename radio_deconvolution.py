import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib import rc
from scipy.ndimage import shift
from tqdm import tqdm

def logmin(a,minv):
    la=np.log10(a)
    la[(la<minv) | np.isnan(la)]=minv
    return la

def recentre(z):
    # centre an array where the bottom left pixel is 0,0
    rcz=np.zeros_like(z)
    hs=xm//2
    rcz[hs:,hs:]=z[:hs,:hs]
    rcz[hs:,:hs]=z[:hs,hs:]
    rcz[:hs,hs:]=z[hs:,:hs]
    rcz[:hs,:hs]=z[hs:,hs:]
    return rcz

# Main program -- do the computations and make the plots.
# Any FITS image of reasonable size should be OK here -- should be square.
# The example image comes from NED

# Open the FITS image. Check that it's square
hdu=fits.open('MESSIER_051 I R kfm2000.fits')
hdu[0].data-=np.min(hdu[0].data)
ym,xm=hdu[0].data.shape
assert(ym==xm)

# Set up the figure
plt.figure(figsize=(9.8,9))
fontsize=10
rc('font',**{'family':'serif','serif':['Times'],'size':fontsize})
rc('text', usetex=True)
plt.rcParams.update({'font.size':fontsize})
plt.tight_layout()

# FFT the FITS image and display first row:
z=np.fft.fft2(hdu[0].data)
hs=xm//2
rcz=recentre(z) # make 0,0 of u,v plane the centre of the image for plotting

# Original image in l-m co-ords
plt.subplot(331)
plt.imshow(logmin(hdu[0].data,0.1),origin='lower',extent=(-xm/2-0.5,xm/2-0.5,-xm/2-0.5,xm/2-0.5),vmin=2,vmax=4)
plt.xlabel('$\\ell$')
plt.ylabel('$m$')

# Amplitude of FT in u-v
plt.subplot(332)
plt.imshow(np.log10(np.absolute(rcz)),origin='lower',extent=(-xm/2-0.5,xm/2-0.5,-xm/2-0.5,xm/2-0.5))
plt.xlabel('$u$')
plt.ylabel('$v$')

# Phase angle of FT in u-v
plt.subplot(333)
plt.imshow(np.angle(rcz),origin='lower',cmap='bwr',extent=(-xm/2-0.5,xm/2-0.5,-xm/2-0.5,xm/2-0.5))
plt.xlabel('$u$')
plt.ylabel('$v$')

# generate a random set of 'antenna' positions

mask=np.zeros_like(hdu[0].data)

# Parameters of the 'array' 
nant=20
pixmin=10

# Log-uniform distribution of radii and random angles
r=10**np.random.uniform(np.log10(pixmin),np.log10(hs/2),size=nant)
theta=np.random.uniform(-np.pi,np.pi,size=nant)

time=np.linspace(0,np.pi/3,300)

x=r*np.cos(theta)
y=r*np.sin(theta)

# plot the antenna positions
plt.subplot(334)
plt.scatter(x,y,s=2)
plt.xlim((-hs,hs))
plt.ylim((-hs,hs))
plt.gca().set_aspect('equal')
plt.xlabel('$x$')
plt.ylabel('$y$')

# put the FT amplitude in as background
plt.subplot(335)
plt.imshow(np.log10(np.absolute(rcz)),origin='lower',extent=(-xm/2-0.5,xm/2-0.5,-xm/2-0.5,xm/2-0.5))
plt.xlabel('$u$')
plt.ylabel('$v$')

# Rotate the Earth
for t in time:
    # for each time, make a copy of the xs and ys and subtract -- this
    # gives all possible pairs in u and v
    x=r*np.cos(theta+t)
    y=r*np.sin(theta+t)
    xt=np.tile(x,(nant,1))
    yt=np.tile(y,(nant,1))
    u=(xt-xt.T).flatten()
    v=(yt-yt.T).flatten()
    plt.scatter(u,v,s=0.5,color='red')
    # fill out the mask in the uv plane. We make the mask slightly
    # bigger for display purposes.
    mask[np.round(v).astype(int),np.round(u).astype(int)]=1
    mask[(np.round(v)+1).astype(int),np.round(u).astype(int)]=1
    mask[(np.round(v)-1).astype(int),np.round(u).astype(int)]=1
    mask[np.round(v).astype(int),(np.round(u)-1).astype(int)]=1
    mask[np.round(v).astype(int),(np.round(u)+1).astype(int)]=1

# Plot the masked FT of the sky
mask[0,0]=0
plt.subplot(336)
maskz=z*mask
plt.imshow(np.log10(np.absolute(recentre(maskz))),origin='lower',extent=(-xm/2-0.5,xm/2-0.5,-xm/2-0.5,xm/2-0.5))
plt.xlabel('$u$')
plt.ylabel('$v$')

# Form the dirty image and beam by inverse-transforming the masked
# image and the mask itself.

dirty=np.real(np.fft.ifft2(maskz))
dbeam=np.real(np.fft.ifft2(mask))
dbeam/=np.max(dbeam) # normalize the dirty beam to 1

#here we could write out the dirty image
#hdu[0].data=dirty
#hdu.writeto('test.fits',overwrite=True)

# Plot the dirty image and beam
plt.subplot(337)
plt.imshow(logmin(dirty,0.1),origin='lower',extent=(-xm/2-0.5,xm/2-0.5,-xm/2-0.5,xm/2-0.5),vmin=2,vmax=4)
plt.xlabel('$\ell$')
plt.ylabel('$m$')
plt.subplot(338)
plt.imshow(recentre(dbeam)[hs-50:hs+50,hs-50:hs+50],origin='lower',extent=(-50.5,49.5,-50.5,49.5))
plt.xlabel('$\\ell$')
plt.ylabel('$m$')

# Now try an image-based CLEAN

niter=2000
gain=0.2 # the 'CLEAN gain'
model=np.zeros_like(dirty) # an image to store what we've CLEANEd

# Go round for niter iterations
for i in tqdm(range(niter)):
    # Find the peak of the dirty image
    index=np.argmax(dirty)
    y,x=np.unravel_index(index,dirty.shape)
    # Find the flux value at the peak
    value=dirty[y,x]
    # shift the dirty beam image to this position, scale it by the
    # CLEAN gain, and subtract the scaled version from the dirty image
    sdbeam=gain*value*np.roll(dbeam,(y,x),axis=(0,1))
    dirty-=sdbeam
    # Keep track of the values we've subtracted
    model[y,x]+=gain*value
    
# Plot the residual image
plt.subplot(339)
plt.imshow(logmin(dirty,0.1),origin='lower',extent=(-xm/2-0.5,xm/2-0.5,-xm/2-0.5,xm/2-0.5),vmin=2,vmax=4)

# Here we could use the model image to restore components to the residual.

plt.savefig('fourier.pdf')

