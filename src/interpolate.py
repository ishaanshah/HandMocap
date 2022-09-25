import numpy as np
from scipy import interpolate

def interpolate_nans(col, kind="cubic"):
    '''
    Interpolate missing points (of value nan)

    INPUTS:
    - col: pandas column of coordinates
    - kind: 'linear', 'slinear', 'quadratic', 'cubic'. Default: 'cubic'

    OUTPUT:
    - col_interp: interpolated pandas column
    '''

    idx = np.arange(len(col))
    idx_good = np.where(np.isfinite(col))[0] #index of non zeros
    # if len(idx_good) <= 10:
    #     return col

    f_interp = interpolate.interp1d(idx_good, col[idx_good], kind=kind, bounds_error=False)
    col_interp = np.where(np.isfinite(col), col, f_interp(idx)) #replace nans with interpolated values
    col_interp = np.where(np.isfinite(col_interp), col_interp, np.nanmean(col_interp)) #replace remaining nans

    return col_interp

