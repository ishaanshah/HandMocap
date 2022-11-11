import numpy as np

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def get_intr(param, undistort=False):
    intr = np.eye(3)
    intr[0,0] = param['fx_undist' if undistort else 'fx']
    intr[1,1] = param['fy_undist' if undistort else 'fy']
    intr[0,2] = param['cx_undist' if undistort else 'cx']
    intr[1,2] = param['cy_undist' if undistort else 'cy']

    # TODO: Make work for arbitrary dist params in opencv
    dist = np.asarray([param['k1'], param['k2'], param['p1'], param['p2']])

    return intr, dist

def get_extr(param):
    qvec = [param['qvecw'], param['qvecx'], param['qvecy'], param['qvecz']]
    tvec = np.asarray([param['tvecx'], param['tvecy'], param['tvecz']])
    r = qvec2rotmat(qvec)
    extr = np.vstack([np.hstack([r, tvec[:,None]]), np.zeros((1, 4))])
    extr[3,3] = 1
    extr = extr[:3]

    return extr

def read_params(params_path):
    params = np.loadtxt(params_path, dtype=[
                       ('cam_id', int), ('width', int), ('height', int),
                       ('fx', float), ('fy', float),
                       ('cx', float), ('cy', float),
                       ('k1', float), ('k2', float),
                       ('p1', float), ('p2', float),
                       ('cam_name', '<U10'),
                       ('qvecw', float), ('qvecx', float), ('qvecy', float), ('qvecz', float),
                       ('tvecx', float), ('tvecy', float), ('tvecz', float)
                    ]
                )
    params = np.sort(params, order='cam_name')

    return params
