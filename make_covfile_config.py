import magnificationbias_covariance
import gglens_x_ggclustering_likelihood as utils
import os,sys,json,copy
from numpy import deg2rad
import numpy as np
from collections import OrderedDict
import time

def make_covfile_config(config_filename):
    t0 = time.time()
    print('reference config is %s'%config_filename)
    mc = magnificationbias_covariance.magnificationbias_covariance()
    
    # change rmin and rmax to get data over full radial bins
    p = json.load(open(config_filename, 'r'), object_pairs_hook=OrderedDict)
    p['observables']['lensing']['rmin'] = [1e-3, 1e-3, 1e-3]
    p['observables']['lensing']['rmax'] = [100, 100, 100]
    temp_config_filename = os.path.join(os.path.dirname(__file__),'config_temp.json')
    json.dump(p, open(temp_config_filename, 'w'))
    
    # get config
    config = utils.config(temp_config_filename, 
                          use_interp_meas_corr=False) # option for faster measurement correction calc
    
    # prepare
    info = {}
    info['zl_list'] = config.parameters['observables']['lensing']['redshift']
    info['zs'] = 1.234
    info['galaxy_list'] = [1.78, 2.12, 2.28]

    Omega_s = (136.9*deg2rad(1.0)**2)
    info['Omega_s'] = Omega_s
    info['alpha_list'] = [2.259,3.563,3.729]
    info['shape_noise'] = 0.2207**2
    n_s = 7.9549 # arcmin^-2
    info['n_s'] = n_s / deg2rad(1.0/60.0)**2

    print('preparing for dcov..')
    mc.prepare_dcov(info)
    
    # load radial bins from config
    x,y,cov = config.load_data()
    n = config.probes.n_redshift['lensing']
    cov_lens = cov.get_covariance('lensing')
    R = x.get_radial_bin('lensing', 0)
    dlogR = np.diff(np.log10(R))[0]
    nr = len(R)
    print('radial bins = ',R)
    
    # compute
    print('start dcov computation. this takes several minutes')
    dR = np.array([10**(-dlogR/2.0),10**(dlogR/2.0)])
    dcov = np.zeros(cov_lens.shape)
    # compute off diagonal
    for i in range(n):
        for j in range(n):
            print('computing cov[z1=%d, z2=%d] : '%(i,j), end='')
            for ir in range(nr):
                for jr in range(nr):
                    cov_i = nr*i + ir
                    cov_j = nr*j + jr
                    if cov_i < cov_j:
                        R1 = dR*x.get_radial_bin('lensing',i)[ir]
                        R2 = dR*x.get_radial_bin('lensing',j)[jr]
                        dc = mc.get_dcov(i, j, R1, R2, show_integrand=False, dump_peak_ratio=1e2)
                        dcov[cov_i,cov_j] = dc
            print('done')
    dcov = dcov + dcov.T
    # compute diagonal
    for i in range(n):
        for ir in range(nr):
            cov_i = nr*i + ir
            R1 = dR*x.get_radial_bin('lensing',i)[ir]
            dc = mc.get_dcov(i, i, R1, R1, show_integrand=False, dump_peak_ratio=1e2)
            dcov[cov_i, cov_i] = dc
    cov_full = cov_lens + dcov
    
    # save new cov to file
    filename_list = config.parameters['observables']['lensing']['filename_covariance']
    new_config = copy.deepcopy(config.parameters)
    for i in range(n):
        for j in range(n):
            filename = filename_list[n*i+j].replace('cov','mag_cov')
            new_config['observables']['lensing']['filename_covariance'][n*i+j] = filename
            full_path = os.path.join(os.path.join(config.working_dir,filename))
            sub_mat = cov_full[i*nr:(i+1)*nr, j*nr:(j+1)*nr]
            print('saved (%d,%d) submatrix of lensing covariance to %s'%(i,j,full_path))
            np.savetxt(full_path, dcov[i*nr:(i+1)*nr, j*nr:(j+1)*nr])
            
    # save new config to working dir with suffix(_mag) appended
    new_config_filename = config_filename.replace('.json', '_mag.json')
    json.dump(new_config, open(os.path.join(new_config_filename), 'w'), ensure_ascii=False, indent=4)
    
    # show time
    t1 = time.time()
    print('time : %d sec'%(t1-t0))
    
    
    
if __name__ == '__main__':
    """
    Call this script with reference json config file from shell, like
        python make_covfile_config.py config_mock_fiducial.json
    """
    config_filename = sys.argv[1]
    make_covfile_config(config_filename)