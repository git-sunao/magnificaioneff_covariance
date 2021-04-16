import model_pt
from dark_emulator import model_hod
from scipy.integrate import simps
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import interp2d, interp1d
from scipy.special import jn
import numpy as np
import matplotlib.pyplot as plt

class magnificationbias_covariance:
    def __init__(self, galaxy_model='b1'):
        self.pt = model_pt.pt()
        self.galaxy_model = galaxy_model
        if self.galaxy_model == 'hod':
            self.hod= model_hod.darkemu_x_hod()
        self.set_cosmology()
        self.k = np.logspace(-5.0, 2, 200)
        self.l = np.logspace(-1, 4.0, 10000)
        
        self.Cls = []
        self.Cll = []
        self.Css = {}
        self.Cgs = []
        
    def set_cosmology(self):
        #takahashi paper: https://arxiv.org/pdf/1706.01472.pdf
        Omega_cdm = 0.233
        Omega_b = 0.046
        Omega_m = Omega_cdm+Omega_b # 0.279
        Omega_de = 1.0-Omega_m
        h = 0.7
        ns = 0.97
        sigma8 = 0.82
        
        lnAs_temp = 3.094
        cparam = np.array([[ Omega_b*h**2, Omega_cdm*h**2, Omega_de, lnAs_temp, ns, -1.]])
        self.pt.set_cosmology(cparam)
        sigma8_temp = self.pt.get_sigma8()
        print(2.0*np.log(sigma8/sigma8_temp))
        cparam[0][3] = lnAs_temp + 2.0*np.log(sigma8/sigma8_temp)
        self.pt.set_cosmology(cparam)
        if self.galaxy_model == 'hod':
            self.hod.set_cosmology(cparam)
        
        self.H0 = self.pt.halofit.cosmo.H0.value*1e3/model_pt.constants.c.value / self.pt.halofit.cosmo.h
        self.h = self.pt.halofit.cosmo.h
        self.Om = 1.0 - self.pt.cosmo.cparam[0][2]
        
    def set_galaxy(self, galaxy):
        if self.galaxy_model == 'hod':
            self.set_hod(galaxy)
        elif self.galaxy_model == 'b1':
            self.set_b1(galaxy)
    
    def set_b1(self, b1=2.0):
        self.pt.set_bias({'b1':b1})
    
    def set_hod(self, hod='default'):
        # temp
        if isinstance(hod, str):
            if hod == 'default':
                hod = {"logMmin":13.13, "sigma_sq":0.22, "logM1": 14.21, "alpha": 1.13, "kappa": 1.25, # hod
                       "poff": 0.2, "Roff": 0.1, # off-centering parameters p_off is the fraction of off-centered galaxies. Roff is the typical off-centered scale with respect to R200m.
                       "sat_dist_type": "emulator", # satellite distribution. Chosse emulator of NFW. In the case of NFW, the c-M relation by Diemer & Kravtsov (2015) is assumed.
                       "alpha_inc": 0.44, "logM_inc": 13.57} # incompleteness parameters. For details, see More et al. (2015)
            else:
                print('hod?')
        self.hod.set_galaxy(hod)
        
    def set_obs_params(self, Omega_s, alpha_list, sigma_s2, n_s):
        self.Omega_s = Omega_s
        self.alpha_list = alpha_list
        self.shape_noise = sigma_s2/n_s
        
    def lensing_kernel(self, zl, zs):
        comoving = self.pt.halofit.cosmo.comoving_distance
        xl = comoving(zl).value * self.h
        xs= comoving(zs).value * self.h
        ans = 3.0/2.0*self.H0**2*self.Om*(1+zl) * xl*(xs-xl)/xs
        sel = ans < 0
        ans[sel] = 0.0
        return ans
    
    def get_x(self, z):
        return self.pt.halofit.cosmo.comoving_distance(z).value * self.h
    
    def compute_Pmm(self, zs):
        z = np.linspace(0.0, zs, 100)
        pkhalo = []
        for _z in z:
            pklin = self.pt.get_pklin_from_z(self.k, _z)
            self.pt.halofit.set_pklin(self.k, pklin, _z)
            _pkhalo = self.pt.halofit.get_pkhalo()
            pkhalo.append(_pkhalo)
        pkhalo = np.array(pkhalo)
        self.Pmm_interp = interp2d(self.k, z, pkhalo, bounds_error=False, fill_value=0)
        
    def compute_Pgm(self, z):
        if self.galaxy_model == 'hod':
            self.hod.get_ds(np.linspace(1.0, 10.0, 4), z)
            p_tot = self.hod.p_cen + self.hod.p_cen_off + self.hod.p_sat
            self.Pgm_interp = interp1d(self.hod.fftlog_1h.k, p_tot, bounds_error=False, fill_value=0)
        elif self.galaxy_model == 'b1':
            pklin = self.pt.get_pklin_from_z(self.k, z)
            self.pt.halofit.set_pklin(self.k, pklin, z)
            pkhalo = self.pt.halofit.get_pkhalo()
            self.Pgm_interp = interp1d(self.k, pkhalo, bounds_error=False, fill_value=0)
            
    def compute_Cl(self, zl_list, zs, galaxy_list):
        self.compute_Pmm(zs)
        
        if not isinstance(zl_list, list):
            zl_list = [zl_list]
            
        l_sparse = np.logspace(np.log10(min(self.l)), np.log10(max(self.l)), 150)
        
        # C_gs
        self.C_gs = []
        for zl, gal in zip(zl_list, galaxy_list):
            xl = self.get_x(zl)
            self.set_galaxy(gal)
            self.compute_Pgm(zl)
            C_gs = self.lensing_kernel(np.array([zl]), zs) / xl**2 * self.Pgm_interp(self.l/xl)
            self.C_gs.append(C_gs)
        # C_ls
        self.C_ls = []
        for zl in zl_list:
            z = np.linspace(1e-3, zl, 100)
            x = self.get_x(z)
            C_ls = []
            for l in l_sparse:
                integrand = self.lensing_kernel(z, zl)*self.lensing_kernel(z, zs) / x**2 * np.diag(self.Pmm_interp(l/x, z))
                C_ls.append(simps(integrand, x))
            self.C_ls.append(ius(l_sparse,np.array(C_ls))(self.l))
        # C_ss
        C_ss = []
        z = np.linspace(1e-3, zs, 100)
        x = self.get_x(z)
        for l in l_sparse:
            integrand = self.lensing_kernel(z, zs)**2 / x**2 * np.diag(self.Pmm_interp(l/x, z)) # interp2d ok ?
            C_ss.append(simps(integrand, x))
        self.C_ss = ius(l_sparse, np.array(C_ss))(self.l)
        n = len(zl_list)
        # C_gl
        self.C_gl = [[0 for j in range(n)] for i in range(n)]
        for i in range(n):
            zl1 = zl_list[i]
            for j in range(n):
                zl2 = zl_list[j]
                if zl1 == zl2:
                    self.C_gl[i][j] = 0.0*self.l
                elif zl1 < zl2:
                    gal = galaxy_list[i]
                    xl = self.get_x(zl1)
                    self.set_galaxy(gal)
                    self.compute_Pgm(zl1)
                    self.C_gl[i][j] = self.lensing_kernel(zl1,np.array([zl2])) / xl**2 * self.Pgm_interp(self.l/xl)
                elif zl1 > zl2:
                    self.C_gl[i][j] = 0.0*self.l
        # C_ll 
        self.C_ll = [[0 for j in range(n)] for i in range(n)]
        for i in range(n):
            zl1 = zl_list[i]
            for j in range(i+1):
                zl2 = zl_list[j]
                zlmin = min([zl1,zl2])
                z = np.linspace(1e-3, zlmin, 100)
                x = self.get_x(z)
                C_ll = []
                for l in l_sparse:
                    integrand = self.lensing_kernel(z, zl1) * self.lensing_kernel(z, zl2) / x**2 * np.diag(self.Pmm_interp(l/x, z)) # interp2d ok ?
                    C_ll.append(simps(integrand, x))
                self.C_ll[i][j] = ius(l_sparse,np.array(C_ll))(self.l)
        for i in range(n):
            for j in range(i):
                self.C_ll[j][i] = self.C_ll[i][j]
                
    def get_j2l(self, rmin, rmax, zl):
        xl = self.get_x(zl)
        tmin, tmax = rmin/xl, rmax/xl
        return self.j2_ave(self.l, tmin, tmax)
                
    def j2_ave(self, k, Rmin, Rmax):
        ak = Rmin*k
        bk = Rmax*k
        ans = (2*jn(0, ak)+ak*jn(1,ak) - 2.0*jn(0,bk) - bk*jn(1,bk)) * 2.0/(bk**2-ak**2)
        return ans
    
    def Sigma_crit(self, zl, zs):
        comoving = self.pt.halofit.cosmo.comoving_distance
        xl = comoving(zl).value * self.h
        xs= comoving(zs).value * self.h
        ans = 2.0/3.0 * model_pt.rho_cr * xs/xl/(xs-xl) / (1.0+zl) / self.H0**2 / 1e12
        return ans
    
    def prepare_dcov(self, info):
        self.zl_list = info['zl_list']
        self.zs = info['zs']
        self.galaxy_list = info['galaxy_list']
        
        Omega_s = info['Omega_s']
        shape_noise = info['shape_noise']
        alpha_list = info['alpha_list']
        n_s = info['n_s']
        
        self.compute_Cl(self.zl_list, self.zs, self.galaxy_list)
        self.set_obs_params(Omega_s, alpha_list, shape_noise, n_s)
        
    def get_dcov(self, i, j, r_range1, r_range2, show_integrand=False, dump_peak_ratio=1e2,show_order=False, show_summand_order=False):
        j2l_i = self.get_j2l(r_range1[0], r_range1[1], self.zl_list[i])
        j2l_j = self.get_j2l(r_range2[0], r_range2[1], self.zl_list[j])
        alpha_i = self.alpha_list[i]
        alpha_j = self.alpha_list[j]
        Sigma_crit_i = self.Sigma_crit(self.zl_list[i], self.zs)
        Sigma_crit_j = self.Sigma_crit(self.zl_list[j], self.zs)
        
        C_sum = self.C_gs[i]*2.0*(alpha_j-1.0)*self.C_ls[j] + 2.0*(alpha_i-1.0)*self.C_ls[i]*self.C_gs[j] \
                + (2.0*(alpha_j-1.0)*self.C_gl[i][j] + 2.0*(alpha_i-1.0)*self.C_gl[j][i]) * (self.C_ss+self.shape_noise) \
                + 4.0*(alpha_i-1.0)*(alpha_j-1.0) * (self.C_ll[i][j]*(self.C_ss+self.shape_noise) + self.C_ls[i]*self.C_ls[j])
        
        l_J2 = min([self.get_x(self.zl_list[i]), self.get_x(self.zl_list[j])])\
                /max([np.exp(np.mean(np.log(r_range1))), np.exp(np.mean(np.log(r_range2)))])
        l_dump = l_J2 * dump_peak_ratio
        
        integrand_nowave = Sigma_crit_i * Sigma_crit_j * C_sum * self.l / (2.0*np.pi*self.Omega_s)
        wave = j2l_i * j2l_j
        dump = np.exp(-(self.l/l_dump)**2)
        integrand = wave * integrand_nowave * dump
        
        ans = simps(integrand, self.l)
        if show_integrand:
            fig, ax = plt.subplots(1,1)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel(r'$l$')
            ax.set_ylabel(r"$l J_2(lR/\chi_l)J_2(lR'/\chi_l')\Sigma_{\rm cr}(z_{\rm l},z_{\rm s})\Sigma_{\rm cr}(z_{\rm l'},z_{\rm s}) \sum C / 2\pi$", fontsize=15)
            ax.plot(self.l, integrand, color='k')
            ax.plot(self.l,-integrand, color='k', linestyle='--')
            ylim = (max(integrand)/1e7, max(integrand)*5)
            ax.set_ylim(ylim)
            xlim = ax.get_xlim()
            ax.set_xlim(xlim)
            ax.fill_between(xlim, ylim[0], max(integrand)/1e4, color='gray', alpha=0.2)
            ax2 = ax.twinx()
            ax2.set_yscale('log')
            ax2.set_yscale('log')
            ax2.plot(self.l, integrand_nowave/max(integrand_nowave),alpha=0.5, label=r'no $J_2$')
            ax2.plot(self.l, wave/max(wave),alpha=0.5, label=r'$J_2$')
            ax2.plot(self.l, dump/max(dump),alpha=0.5, label=r'dump')
            ax2.set_ylim((1e-7,2.0))
            ax2.set_ylabel('relative contribution \n (for colored lines)')
            ax2.legend(fontsize=15, bbox_to_anchor=(0.5, 0.0), loc='lower center')
            plt.show()
            
        if show_order:
            C = []
            C.append(max(self.C_gs[i]))
            C.append(max(2.0*(alpha_j-1.0)*self.C_ls[j]))
            
            C.append(max(2.0*(alpha_i-1.0)*self.C_ls[i]))
            C.append(max(self.C_gs[j]))
            
            C.append(max(2.0*(alpha_j-1.0)*self.C_gl[i][j]+2.0*(alpha_i-1.0)*self.C_gl[j][i]))
            C.append(max(self.C_ss+self.shape_noise))
            
            C.append(max(4.0*(alpha_i-1.0)*(alpha_j-1.0)*self.C_ll[i][j]))
            C.append(max(self.C_ss+self.shape_noise))
            
            C.append(max(2.0*(alpha_i-1.0)*self.C_ls[i]))
            C.append(max(2.0*(alpha_j-1.0)*self.C_ls[j]))
            
            label = ['C_gs[1]', 'C_ls[2]', 'C_ls[1]', 'C_gs[2]', 'C_gl', 'C_ss', 'C_ll', 'C_ss', 'C_ls[1]', 'C_ls[2]']
            
            print('max = %s = %e = O(10^{%d})'%(label[np.where(np.array(C)==max(C))[0][0]], max(C), int(np.log10(max(C))) ))
        
        if show_summand_order:
            s = []
            s.append(max(self.C_gs[i]*2.0*(alpha_j-1.0)*self.C_ls[j]))
            s.append(max(2.0*(alpha_i-1.0)*self.C_ls[i]*self.C_gs[j]))
            s.append(max((2.0*(alpha_j-1.0)*self.C_gl[i][j]+2.0*(alpha_i-1.0)*self.C_gl[j][i]) * (self.C_ss+self.shape_noise)))
            s.append(max(4.0*(alpha_i-1.0)*(alpha_j-1.0) * self.C_ll[i][j]*(self.C_ss+self.shape_noise)))
            s.append(max(4.0*(alpha_i-1.0)*(alpha_j-1.0) * self.C_ls[i]*self.C_ls[j]))
            label = ['C_gs[1] * C_ls[2]', 'C_gs[2] * C_ls[1]', 'C_gl * C_ss', 'C_ll * C_ss', 'C_ls[1] * C_ls[2]']
            print('max summand = %s = %e'%(label[np.where(np.array(s)==max(s))[0][0]], max(s)))
            
        return ans
    
    ###########################################################################