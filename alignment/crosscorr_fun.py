import scipy.fftpack as spf
import scipy as sp

def phase_corr(a, b):
    tmp = spf.fft2(a)*spf.fft2(b).conj()
    tmp /= abs(tmp)
    return spf.ifft2(tmp)

    
def norm_cross_corr(a,b):
    a-=sp.mean(a)
    b-=sp.mean(b)
    
    return sp.sum(cross_corr(a,b))/(l2_norm(a)*l2_norm(b))
    
    
def cross_corr(a, b):
    tmp = spf.fft2(a)*spf.fft2(b).conj()
    return spf.ifft2(tmp)    

