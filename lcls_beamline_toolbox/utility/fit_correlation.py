import numpy as np
from scipy.optimize import curve_fit

def three_gaussian_2d(x,y,amp1,cenx1,ceny1,wid1,amp2,cenx2,ceny2,wid2,amp3,cenx3,ceny3,wid3,bg):
    g1 = amp1*np.exp(-((x-cenx1)**2+(y-ceny1)**2)/(2*wid1**2))
    g2 = amp2*np.exp(-((x-cenx2)**2+(y-ceny2)**2)/(2*wid2**2))
    g3 = amp3*np.exp(-((x-cenx3)**2+(y-ceny3)**2)/(2*wid3**2))

    return g1 + g2 + g3 + bg

def one_gaussian_2d(x,y,amp1,cenx1,ceny1,wid1,bg):
    g1 = amp1*np.exp(-((x-cenx1)**2+(y-ceny1)**2)/(2*wid1**2))
    return g1 + bg

def _gaussian(M, *args):
    x,y = M
    return three_gaussian_2d(x,y,*args)

def fit_gaussian(X,Y,a,mask,bounds=None):
    xdata = np.vstack((X[mask].ravel(),Y[mask].ravel()))
    #p0 = (1,0,0,3,1)
    p0 = (1,0,0,3,
            1,0,-5,3,
            1,0,5,3,
            1)
    if bounds is None:
        bounds = ([0,-1,-1,0,
        0,-15,-20,0,
        0,-15,-20,0,
        0],
        [1,1,1,5,
        1,15,20,5,
        1,15,20,5,
        2])

    px, covx = curve_fit(_gaussian,xdata,a[mask].ravel(),p0,bounds=bounds)


    return px, covx