import numpy as np
from scipy.optimize import curve_fit

def three_gaussian_2d(x,y,amp1,cenx1,ceny1,widx1,widy1,amp2,cenx2,ceny2,widx2,widy2,amp3,cenx3,ceny3,widx3,widy3,bg):
    g1 = amp1*np.exp(-((x-cenx1)**2/(2*widx1)**2+(y-ceny1)**2/(2*widy1**2)))
    g2 = amp2*np.exp(-((x-cenx2)**2/(2*widx2)**2+(y-ceny2)**2/(2*widy2**2)))
    g3 = amp3*np.exp(-((x-cenx3)**2/(2*widx3)**2+(y-ceny3)**2/(2*widy3**2)))

    return g1 + g2 + g3 + bg

def three_gaussian_2d_new(x,y,amp1,widx1,widy1,amp2,cenx2,ceny2,widx2,widy2,bg):
    g1 = amp1*np.exp(-((x)**2/(2*widx1)**2+(y)**2/(2*widy1**2)))
    g2 = amp2*np.exp(-((x-cenx2)**2/(2*widx2)**2+(y-ceny2)**2/(2*widy2**2)))
    g3 = amp2*np.exp(-((x+cenx2)**2/(2*widx2)**2+(y+ceny2)**2/(2*widy2**2)))

    return g1 + g2 + g3 + bg

def one_gaussian_2d(x,y,amp1,cenx1,ceny1,widx1,widy1,bg):
    g1 = amp1*np.exp(-((x-cenx1)**2/(2*widx1**2)+(y-ceny1)**2/(2*widy1**2)))
    return g1 + bg

def _gaussian(M, *args):
    x,y = M
    return three_gaussian_2d(x,y,*args)

def _gaussian_new(M, *args):
    x,y = M
    return three_gaussian_2d_new(x,y,*args)

def fit_gaussian_new(X,Y,a,mask,p0=None,bounds=None):
    xdata = np.vstack((X[mask].ravel(),Y[mask].ravel()))
    #p0 = (1,0,0,3,1)
    if p0 is None:
        p0 = (1,0,0,1,1,
                1,0,-10,1,1,
                1,0,10,1,1,
                1)
    if bounds is None:
        bounds = ([0,-1,-1,0,0,
        0,-15,-30,0,0,
        0,-15,-30,0,0,
        0],
        [1,1,1,10,10,
        1,15,30,10,10,
        1,15,30,10,10,
        2])

    px, covx = curve_fit(_gaussian_new,xdata,a[mask].ravel(),p0,bounds=bounds)


    return px, covx

def fit_gaussian(X,Y,a,mask,p0=None,bounds=None):
    xdata = np.vstack((X[mask].ravel(),Y[mask].ravel()))
    #p0 = (1,0,0,3,1)
    if p0 is None:
        p0 = (1,0,0,1,1,
                1,0,-10,1,1,
                1,0,10,1,1,
                1)
    if bounds is None:
        bounds = ([0,-1,-1,0,0,
        0,-15,-30,0,0,
        0,-15,-30,0,0,
        0],
        [1,1,1,10,10,
        1,15,30,10,10,
        1,15,30,10,10,
        2])

    px, covx = curve_fit(_gaussian,xdata,a[mask].ravel(),p0,bounds=bounds)


    return px, covx