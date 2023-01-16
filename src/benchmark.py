import numpy as np
import xarray as xr
import pyinterp
from scipy import signal
from itertools import chain
import pandas as pd
import warnings
import matplotlib.pylab as plt
warnings.filterwarnings("ignore")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from .swot import *


class Benchmark(object):

    def __init__(self, gridstep=1):
        self._gridstep = gridstep
        self._stats = ['ssh','grad_ssh_across','grad_ssh_along','ssh_rmse', 'ug_rmse', 'ksi_rmse', 'grad_ssh_across_rmse','grad_ssh_along_rmse']
        self._d_stats = dict()
        self._init_accumulators()

    def _init_accumulators(self):
        """ creation des accumulateurs """
        self._xaxis = pyinterp.Axis(np.arange(-180,180,self.gridstep), is_circle=True)
        self._yaxis = pyinterp.Axis(np.arange(-90, 90,self.gridstep), is_circle=False)

        for k in self._stats:
            self._d_stats[k] = pyinterp.Binning2D(self._xaxis, self._yaxis)
            self._d_stats[k].clear()
        
    def raz(self):
        """ remise a zero des accumulateurs """
        for k in self._stats:
            self._d_stats[k].clear()
    
    
    def _coriolis_parameter(self, lat):  
        """Compute the Coriolis parameter for the given latitude:
        ``f = 2*omega*sin(lat)``, where omega is the angular velocity 
        of the Earth.
    
        Parameters
        ----------
        lat : array
          Latitude [degrees].
        """
        omega = 7.2921159e-05  # angular velocity of the Earth [rad/s]
        fc = 2*omega*np.sin(lat*np.pi/180.)
        # avoid zero near equator, bound fc by min val as 1.e-8
        fc = np.sign(fc)*np.maximum(np.abs(fc), 1.e-8)
    
        return fc


    def compute_stats(self, l_files, etuvar, l_files_input):
        """ 
        calcul des stats 
        INPUT:
            l_files: list of swot track files
            refavr: uncalibrated (true) SSH variable name
            etuvar: calibrated SSH variable name
        """

        self.mean_ssh_rmse = 0
        self.mean_ug_rmse = 0
        self.mean_ksi_rmse = 0
        
        self.mean_ssh_nocalib_rmse = 0
        self.mean_ug_nocalib_rmse = 0
        self.mean_ksi_nocalib_rmse = 0
        
        for i, fname in enumerate(l_files):
            
            # Calibrated field
            swt = SwotTrack(fname) #._dset
            swt.compute_geos_current(etuvar, 'calib_geos_current')
            swt.compute_relative_vorticity('calib_geos_current_x', 'calib_geos_current_y', 'calib_ksi')
            
            # Truth
            swt_input = SwotTrack(l_files_input[i])#._dset
            swt_input.compute_geos_current('ssh_true', 'true_geos_current')
            swt_input.compute_relative_vorticity('true_geos_current_x', 'true_geos_current_y', 'true_ksi')
            swt_input.compute_geos_current('ssh_err', 'err_geos_current')
            swt_input.compute_relative_vorticity('err_geos_current_x', 'err_geos_current_y', 'err_ksi')
            # NEED TO CHEK CONSISTENCY BETWEEN Fileterd and thruth if file not sorted
            
            # SSH RMSE
            self.stats_dict['ssh_rmse'].push(
                swt._dset.lon.values.flatten(),
                swt._dset.lat.values.flatten(),
                ((swt._dset[etuvar].values - swt_input._dset['ssh_true'].values)**2).flatten(),
                False
            )
            ssh_rmse = ((swt._dset[etuvar].values - swt_input._dset['ssh_true'].values)**2).flatten()
            ssh_rmse = ssh_rmse[~np.isnan(ssh_rmse)]
            self.mean_ssh_rmse += np.sqrt(np.mean(ssh_rmse))
             
            ssh_nocalib_rmse = ((swt_input._dset['ssh_err'].values - swt_input._dset['ssh_true'].values)**2).flatten()
            ssh_nocalib_rmse = ssh_nocalib_rmse[~np.isnan(ssh_nocalib_rmse)]
            self.mean_ssh_nocalib_rmse += np.sqrt(np.mean(ssh_nocalib_rmse))
            
            # GEOST CURRENT RMSE
            self.stats_dict['ug_rmse'].push(
                swt._dset.lon.values.flatten(),
                swt._dset.lat.values.flatten(),
                ((swt._dset['calib_geos_current'].values - swt_input._dset['true_geos_current'].values)**2).flatten(),
                False
            )
            ug_rmse = ((swt._dset['calib_geos_current'].values - swt_input._dset['true_geos_current'].values)**2).flatten()
            ug_rmse = ug_rmse[~np.isnan(ug_rmse)] 
            self.mean_ug_rmse += np.sqrt(np.mean(ug_rmse))
            
            ug_nocalib_rmse = ((swt_input._dset['err_geos_current'].values - swt_input._dset['true_geos_current'].values)**2).flatten()
            ug_nocalib_rmse = ug_nocalib_rmse[~np.isnan(ug_nocalib_rmse)] 
            self.mean_ug_nocalib_rmse += np.sqrt(np.mean(ug_nocalib_rmse))
            
            # VORTICITY RMSE
            self.stats_dict['ksi_rmse'].push(
                swt._dset.lon.values.flatten(),
                swt._dset.lat.values.flatten(),
                ((swt._dset['calib_ksi'].values - swt_input._dset['true_ksi'].values)**2).flatten(),
                False
            )
            ksi_rmse = ((swt._dset['calib_ksi'].values - swt_input._dset['true_ksi'].values)**2).flatten()
            ksi_rmse = ksi_rmse[~np.isnan(ksi_rmse)] 
            self.mean_ksi_rmse += np.sqrt(np.mean(ksi_rmse))
            
            ksi_nocalib_rmse = ((swt_input._dset['err_ksi'].values - swt_input._dset['true_ksi'].values)**2).flatten()
            ksi_nocalib_rmse = ksi_nocalib_rmse[~np.isnan(ksi_nocalib_rmse)] 
            self.mean_ksi_nocalib_rmse += np.sqrt(np.mean(ksi_nocalib_rmse))
            
        self.mean_ssh_rmse = self.mean_ssh_rmse/np.size(l_files)
        self.mean_ug_rmse = self.mean_ug_rmse/np.size(l_files)
        self.mean_ksi_rmse = self.mean_ksi_rmse/np.size(l_files)
        self.mean_ssh_nocalib_rmse = self.mean_ssh_nocalib_rmse/np.size(l_files)
        self.mean_ug_nocalib_rmse = self.mean_ug_nocalib_rmse/np.size(l_files)
        self.mean_ksi_nocalib_rmse = self.mean_ksi_nocalib_rmse/np.size(l_files)
            
    def _compute_grad_diff(self, etu, ref, f_coriolis):
        """ compute differences of gradients """
        mask = np.isnan(etu)
        ref[mask] = np.nan
        
        # swath resolution is 2kmx2km
        dx = 2000 # m
        dy = 2000 # m
        gravity = 9.81
        
    
        ref_gx, ref_gy = gravity/f_coriolis*np.gradient(ref, dx, edge_order=2)
        etu_gx, etu_gy = gravity/f_coriolis*np.gradient(etu, dx, edge_order=2)
    
        delta_x = etu_gx - ref_gx
        delta_y = etu_gy - ref_gy
        return delta_x, delta_y

    def write_stats(self, fname, **kwargs):
        """ export des résultats vers un fichier NetCDF """
        to_write = xr.Dataset(
            data_vars=dict(
                ssh_rmse=(["lat", "lon"], np.sqrt(self.stats_dict['ssh_rmse'].variable('mean').T)),
                ug_rmse=(["lat", "lon"], np.sqrt(self.stats_dict['ug_rmse'].variable('mean').T)),
                ksi_rmse=(["lat", "lon"], np.sqrt(self.stats_dict['ksi_rmse'].variable('mean').T)),
            ),
            coords=dict(
                lon=(["lon"], self.stats_dict['ssh'].x),
                lat=(["lat"], self.stats_dict['ssh'].y),
            ),
            attrs=dict(
                description=kwargs['description'] if 'description' in kwargs else 'statistics_of_residuals',
                calib_type=kwargs['calib'] if 'calib' in kwargs else 'None',
            ),
        )
        to_write.to_netcdf(fname)
        
        #del self.stats_dict['ssh_rmse']
        #del self.stats_dict['ug_rmse']
        #del self.stats_dict['ksi_rmse']
        
        if 'calib' in kwargs:
            self.calib_name = kwargs['calib']
        else :
            self.calib_name = 'None'
            
            
        
    def display_stats(self, fname, **kwargs):
        
        ds = xr.open_dataset(fname)
        
        plt.figure(figsize=(18, 15))
        
        ax = plt.subplot(311, projection=ccrs.PlateCarree())
        vmin = np.nanpercentile(ds.ssh_rmse, 5)
        vmax = np.nanpercentile(ds.ssh_rmse, 95)
        if 'vmin' in kwargs:
            ds.ssh_rmse.plot(x='lon', y='lat', cmap='Reds', cbar_kwargs={'label': '[m]'}, **kwargs)
            del kwargs["vmin"]  
            del kwargs["vmax"] 
        else:
            vmin = np.nanpercentile(ds.ssh_rmse, 5)
            vmax = np.nanpercentile(ds.ssh_rmse, 95)
            ds.ssh_rmse.plot(x='lon', y='lat', vmin=vmin, vmax=vmax, cmap='Reds', cbar_kwargs={'label': '[m]'}, **kwargs)
        plt.title('RMSE SSH field', fontweight='bold')
        ax.add_feature(cfeature.LAND, zorder=2)
        ax.coastlines(zorder=2)
        print('SSH:',np.mean(ds.ssh_rmse))

        ax = plt.subplot(312, projection=ccrs.PlateCarree())
        vmin = np.nanpercentile(ds.ug_rmse, 5)
        vmax = np.nanpercentile(ds.ug_rmse, 95)
        ds.ug_rmse.plot(x='lon', y='lat', vmin=vmin, vmax=vmax, cmap='Reds', cbar_kwargs={'label': '[m.s$^{-1}$]'}, **kwargs)
        plt.title('RMSE GEOSTROPHIC CURRENT field', fontweight='bold')
        ax.add_feature(cfeature.LAND, zorder=2)
        ax.coastlines(zorder=2)
        print('Ug:',np.mean(ds.ug_rmse))
        
        ax = plt.subplot(313, projection=ccrs.PlateCarree())
        vmin = np.nanpercentile(ds.ksi_rmse, 5)
        vmax = np.nanpercentile(ds.ksi_rmse, 95)
        ds.ksi_rmse.plot(x='lon', y='lat', vmin=vmin, vmax=vmax, cmap='Reds', cbar_kwargs={'label': '[]'}, **kwargs)
        plt.title('RMSE RELATIVE VORTICITY field', fontweight='bold')
        ax.add_feature(cfeature.LAND, zorder=2)
        ax.coastlines(zorder=2)
        print('Ksi:',np.mean(ds.ksi_rmse))

        plt.show()
        
        
    def compute_stats_by_regime(self, l_files, etuvar, l_files_inputs, regime_type ='all'):
        """ 
        calcul des stats 
        INPUT:
            l_files: list of swot track files
            refavr: unfiltered (true) SSH variable name
            etuvar: filtered SSH variable name
        """
        ds = xr.open_mfdataset(l_files, combine='nested', concat_dim='num_lines', parallel=True)
        ds_input = xr.open_mfdataset(l_files_inputs, combine='nested', concat_dim='num_lines', parallel=True)
        
        ds['residual_noise'] = ds[etuvar] - ds_input['ssh_true']
        ds['karin_noise'] = ds_input['ssh_karin'] - ds_input['ssh_true']
        ds['ssh_true'] = ds_input['ssh_true']
        ds['ssh_karin'] = ds_input['ssh_karin']
        swt = SwotTrack(dset=ds)
        swt.compute_geos_current(etuvar, 'filtered_geos_current')
        swt.compute_relative_vorticity('filtered_geos_current_x', 'filtered_geos_current_y', 'filtered_ksi')
        swt.compute_geos_current('ssh_true', 'true_geos_current')
        swt.compute_relative_vorticity('true_geos_current_x', 'true_geos_current_y', 'true_ksi')
        swt.compute_geos_current('ssh_karin', 'karin_geos_current')
        swt.compute_relative_vorticity('karin_geos_current_x', 'karin_geos_current_y', 'karin_ksi')
        self.num_pixels = ds.num_pixels.values
        del ds
        
        def compute_mean_rmse(ds, msk):
            
            tmp = ((ds*msk)**2).load()
            ds_stat = pyinterp.DescriptiveStatistics(tmp, axis=0)
            rmse_ac = np.sqrt(ds_stat.mean())
            
            del ds_stat
            
            ds_stat = pyinterp.DescriptiveStatistics(tmp)
            rmse = np.sqrt(ds_stat.mean())
            
            del ds_stat
            del tmp
            
            return rmse_ac, rmse
        
        
        # GLOBAL STAT
        print('processing global')
        msk = np.ones(ds_input.mask_coastline_200km.shape)        
        self.rmse_ac_residual_noise_global, self.rmse_residual_noise_global = compute_mean_rmse(swt._dset['residual_noise'], msk)
        self.rmse_ac_karin_noise_global, self.rmse_karin_noise_global = compute_mean_rmse(swt._dset['karin_noise'], msk)
        self.rmse_ac_residual_noise_global_ug, self.rmse_residual_noise_global_ug = compute_mean_rmse(swt._dset['filtered_geos_current'] - swt._dset['true_geos_current'], msk)
        self.rmse_ac_karin_noise_global_ug, self.rmse_karin_noise_global_ug = compute_mean_rmse(swt._dset['karin_geos_current'] - swt._dset['true_geos_current'], msk)
        self.rmse_ac_residual_noise_global_ksi, self.rmse_residual_noise_global_ksi = compute_mean_rmse(swt._dset['filtered_ksi'] - swt._dset['true_ksi'], msk)
        self.rmse_ac_karin_noise_global_ksi, self.rmse_karin_noise_global_ksi = compute_mean_rmse(swt._dset['karin_ksi'] - swt._dset['true_ksi'], msk)
        del msk
        
        # COASTAL (< 200 km) STAT
        print('processing coastal')
        msk = np.ones(ds_input.mask_coastline_200km.shape) 
        msk[ds_input['mask_coastline_200km']==1] = np.nan        
        self.rmse_ac_residual_noise_coastal, self.rmse_residual_noise_coastal = compute_mean_rmse(swt._dset['residual_noise'], msk)
        self.rmse_ac_karin_noise_coastal, self.rmse_karin_noise_coastal = compute_mean_rmse(swt._dset['karin_noise'], msk)
        self.rmse_ac_residual_noise_coastal_ug, self.rmse_residual_noise_coastal_ug = compute_mean_rmse(swt._dset['filtered_geos_current'] - swt._dset['true_geos_current'], msk)
        self.rmse_ac_karin_noise_coastal_ug, self.rmse_karin_noise_coastal_ug = compute_mean_rmse(swt._dset['karin_geos_current'] - swt._dset['true_geos_current'], msk)
        self.rmse_ac_residual_noise_coastal_ksi, self.rmse_residual_noise_coastal_ksi = compute_mean_rmse(swt._dset['filtered_ksi'] - swt._dset['true_ksi'], msk)
        self.rmse_ac_karin_noise_coastal_ksi, self.rmse_karin_noise_coastal_ksi = compute_mean_rmse(swt._dset['karin_ksi'] - swt._dset['true_ksi'], msk)
        del msk
        
        # OFFSHORE (> 200 km) + LOW VARIABILITY ( < 200cm2) STAT 
        print('processing low var')
        msk = ds_input['mask_coastline_200km']*ds_input['mask_ssh_var_over200cm2']
        self.rmse_ac_residual_noise_offshore_lowvar, self.rmse_residual_noise_offshore_lowvar = compute_mean_rmse(swt._dset['residual_noise'], msk)
        self.rmse_ac_karin_noise_offshore_lowvar, self.rmse_karin_noise_offshore_lowvar = compute_mean_rmse(swt._dset['karin_noise'], msk)
        self.rmse_ac_residual_noise_offshore_lowvar_ug, self.rmse_residual_noise_offshore_lowvar_ug = compute_mean_rmse(swt._dset['filtered_geos_current'] - swt._dset['true_geos_current'], msk)
        self.rmse_ac_karin_noise_offshore_lowvar_ug, self.rmse_karin_noise_offshore_lowvar_ug = compute_mean_rmse(swt._dset['karin_geos_current'] - swt._dset['true_geos_current'], msk)
        self.rmse_ac_residual_noise_offshore_lowvar_ksi, self.rmse_residual_noise_offshore_lowvar_ksi = compute_mean_rmse(swt._dset['filtered_ksi'] - swt._dset['true_ksi'], msk)
        self.rmse_ac_karin_noise_offshore_lowvar_ksi, self.rmse_karin_noise_offshore_lowvar_ksi = compute_mean_rmse(swt._dset['karin_ksi'] - swt._dset['true_ksi'], msk)
        del msk
        
        # OFFSHORE (> 200 km) + HIGH VARIABILITY ( < 200cm2) STAT
        print('processing high var')
        msk = np.ones(ds_input.mask_ssh_var_over200cm2.shape) 
        msk[ds_input['mask_ssh_var_over200cm2']==1] = np.nan
        msk = ds_input['mask_coastline_200km']*msk
        self.rmse_ac_residual_noise_offshore_highvar, self.rmse_residual_noise_offshore_highvar = compute_mean_rmse(swt._dset['residual_noise'], msk)
        self.rmse_ac_karin_noise_offshore_highvar, self.rmse_karin_noise_offshore_highvar = compute_mean_rmse(swt._dset['karin_noise'], msk)
        self.rmse_ac_residual_noise_offshore_highvar_ug, self.rmse_residual_noise_offshore_highvar_ug = compute_mean_rmse(swt._dset['filtered_geos_current'] - swt._dset['true_geos_current'], msk)
        self.rmse_ac_karin_noise_offshore_highvar_ug, self.rmse_karin_noise_offshore_highvar_ug = compute_mean_rmse(swt._dset['karin_geos_current'] - swt._dset['true_geos_current'], msk)
        self.rmse_ac_residual_noise_offshore_highvar_ksi, self.rmse_residual_noise_offshore_highvar_ksi = compute_mean_rmse(swt._dset['filtered_ksi'] - swt._dset['true_ksi'], msk)
        self.rmse_ac_karin_noise_offshore_highvar_ksi, self.rmse_karin_noise_offshore_highvar_ksi = compute_mean_rmse(swt._dset['karin_ksi'] - swt._dset['true_ksi'], msk)
        del msk
        
        
        
    def write_stats_by_regime(self, fname, **kwargs):
        """ export des résultats vers un fichier NetCDF """
        
        to_write = xr.Dataset(
            data_vars=dict(
                rmse_ac_residual_noise_global = (["num_pixels"], self.rmse_ac_residual_noise_global),
                rmse_residual_noise_global = (["x"], self.rmse_residual_noise_global),
                rmse_ac_karin_noise_global = (["num_pixels"], self.rmse_ac_karin_noise_global),
                rmse_karin_noise_global = (["x"], self.rmse_karin_noise_global),
                rmse_ac_residual_noise_global_ug = (["num_pixels"], self.rmse_ac_residual_noise_global_ug),
                rmse_residual_noise_global_ug = (["x"], self.rmse_residual_noise_global_ug),
                rmse_ac_karin_noise_global_ug = (["num_pixels"], self.rmse_ac_karin_noise_global_ug),
                rmse_karin_noise_global_ug = (["x"], self.rmse_karin_noise_global_ug),
                rmse_ac_residual_noise_global_ksi = (["num_pixels"], self.rmse_ac_residual_noise_global_ksi),
                rmse_residual_noise_global_ksi = (["x"], self.rmse_residual_noise_global_ksi),
                rmse_ac_karin_noise_global_ksi = (["num_pixels"], self.rmse_ac_karin_noise_global_ksi),
                rmse_karin_noise_global_ksi = (["x"], self.rmse_karin_noise_global_ksi),
                
                rmse_ac_residual_noise_coastal = (["num_pixels"], self.rmse_ac_residual_noise_coastal),
                rmse_residual_noise_coastal = (["x"], self.rmse_residual_noise_coastal),
                rmse_ac_karin_noise_coastal = (["num_pixels"], self.rmse_ac_karin_noise_coastal),
                rmse_karin_noise_coastal = (["x"], self.rmse_karin_noise_coastal),
                rmse_ac_residual_noise_coastal_ug = (["num_pixels"], self.rmse_ac_residual_noise_coastal_ug),
                rmse_residual_noise_coastal_ug = (["x"], self.rmse_residual_noise_coastal_ug),
                rmse_ac_karin_noise_coastal_ug = (["num_pixels"], self.rmse_ac_karin_noise_coastal_ug),
                rmse_karin_noise_coastal_ug = (["x"], self.rmse_karin_noise_coastal_ug),
                rmse_ac_residual_noise_coastal_ksi = (["num_pixels"], self.rmse_ac_residual_noise_coastal_ksi),
                rmse_residual_noise_coastal_ksi = (["x"], self.rmse_residual_noise_coastal_ksi),
                rmse_ac_karin_noise_coastal_ksi = (["num_pixels"], self.rmse_ac_karin_noise_coastal_ksi),
                rmse_karin_noise_coastal_ksi = (["x"], self.rmse_karin_noise_coastal_ksi),
                
                rmse_ac_residual_noise_offshore_highvar = (["num_pixels"], self.rmse_ac_residual_noise_offshore_highvar),
                rmse_residual_noise_offshore_highvar = (["x"], self.rmse_residual_noise_offshore_highvar),
                rmse_ac_karin_noise_offshore_highvar = (["num_pixels"], self.rmse_ac_karin_noise_offshore_highvar),
                rmse_karin_noise_offshore_highvar = (["x"], self.rmse_karin_noise_offshore_highvar),
                rmse_ac_residual_noise_offshore_highvar_ug = (["num_pixels"], self.rmse_ac_residual_noise_offshore_highvar_ug),
                rmse_residual_noise_offshore_highvar_ug = (["x"], self.rmse_residual_noise_offshore_highvar_ug),
                rmse_ac_karin_noise_offshore_highvar_ug = (["num_pixels"], self.rmse_ac_karin_noise_offshore_highvar_ug),
                rmse_karin_noise_offshore_highvar_ug = (["x"], self.rmse_karin_noise_offshore_highvar_ug),
                rmse_ac_residual_noise_offshore_highvar_ksi = (["num_pixels"], self.rmse_ac_residual_noise_offshore_highvar_ksi),
                rmse_residual_noise_offshore_highvar_ksi = (["x"], self.rmse_residual_noise_offshore_highvar_ksi),
                rmse_ac_karin_noise_offshore_highvar_ksi = (["num_pixels"], self.rmse_ac_karin_noise_offshore_highvar_ksi),
                rmse_karin_noise_offshore_highvar_ksi = (["x"], self.rmse_karin_noise_offshore_highvar_ksi),
                
                rmse_ac_residual_noise_offshore_lowvar = (["num_pixels"], self.rmse_ac_residual_noise_offshore_lowvar),
                rmse_residual_noise_offshore_lowvar = (["x"], self.rmse_residual_noise_offshore_lowvar),
                rmse_ac_karin_noise_offshore_lowvar = (["num_pixels"], self.rmse_ac_karin_noise_offshore_lowvar),
                rmse_karin_noise_offshore_lowvar = (["x"], self.rmse_karin_noise_offshore_lowvar),
                rmse_ac_residual_noise_offshore_lowvar_ug = (["num_pixels"], self.rmse_ac_residual_noise_offshore_lowvar_ug),
                rmse_residual_noise_offshore_lowvar_ug = (["x"], self.rmse_residual_noise_offshore_lowvar_ug),
                rmse_ac_karin_noise_offshore_lowvar_ug = (["num_pixels"], self.rmse_ac_karin_noise_offshore_lowvar_ug),
                rmse_karin_noise_offshore_lowvar_ug = (["x"], self.rmse_karin_noise_offshore_lowvar_ug),
                rmse_ac_residual_noise_offshore_lowvar_ksi = (["num_pixels"], self.rmse_ac_residual_noise_offshore_lowvar_ksi),
                rmse_residual_noise_offshore_lowvar_ksi = (["x"], self.rmse_residual_noise_offshore_lowvar_ksi),
                rmse_ac_karin_noise_offshore_lowvar_ksi = (["num_pixels"], self.rmse_ac_karin_noise_offshore_lowvar_ksi),
                rmse_karin_noise_offshore_lowvar_ksi = (["x"], self.rmse_karin_noise_offshore_lowvar_ksi),
            
            ),
            coords=dict(
                num_pixels=(["num_pixels"], self.num_pixels),
                x=(["x"], [1]),
            ),
            attrs=dict(
                description=kwargs['description'] if 'description' in kwargs else 'statistics_of_residuals_by_regime',
                filter_type=kwargs['filter'] if 'filter' in kwargs else 'None',
            ),
        )
        to_write.to_netcdf(fname)


    def plot_stats_by_regime(self, fname): 

        ds = xr.open_dataset(fname).drop('x')  

        fig = plt.figure(figsize=(12, 12))
        
        plt.subplot(221)
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_global'].values,'b', label='residual_noise_global')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_coastal'].values,'r', label='residual_noise_coastal')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_offshore_lowvar'].values,'g', label='residual_noise_offshore_lowvar')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_offshore_highvar'].values,'k', label='residual_noise_offshore_highvar')

        plt.plot(2*ds.num_pixels - 70,ds['rmse_ac_karin_noise_global'].values,'b--', lw=2, label='karin_noise_global')
        plt.plot(2*ds.num_pixels - 70,ds['rmse_ac_karin_noise_coastal'].values,'r--', lw=2, label='karin_noise_coastal')
        plt.plot(2*ds.num_pixels - 70,ds['rmse_ac_karin_noise_offshore_lowvar'].values,'g--', lw=2, label='karin_noise_offshore_lowvar')
        plt.plot(2*ds.num_pixels - 70,ds['rmse_ac_karin_noise_offshore_highvar'].values,'k--', lw=2, label='karin_noise_offshore_highvar')
        
        plt.ylim(0, 0.03)
        plt.xlabel('Across-track distance [km]', fontweight='bold')
        plt.ylabel('RMSE [m]', fontweight='bold')
        plt.title('SEA SURFACE HEIGHT', fontweight='bold')
        plt.grid()
        
        plt.subplot(222)
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_global_ug'].values,'b', label='residual_noise_global')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_coastal_ug'].values,'r', label='residual_noise_coastal')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_offshore_lowvar_ug'].values,'g', label='residual_noise_offshore_lowvar')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_offshore_highvar_ug'].values,'k', label='residual_noise_offshore_highvar')

        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_global_ug'].values,'b--', label='karin_noise_global')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_coastal_ug'].values,'r--', label='karin_noise_coastal')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_offshore_lowvar_ug'].values,'g--', label='karin_noise_offshore_lowvar')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_offshore_highvar_ug'].values,'k--', label='karin_noise_offshore_highvar')
        #plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="upper left", ncol=2)
        plt.title('GEOSTROPHIC CURRENTS', fontweight='bold')
        plt.xlabel('Across-track distance [km]', fontweight='bold')
        plt.ylabel('RMSE [m.s$^{-1}$]', fontweight='bold')
        plt.grid()
        
        plt.subplot(223)
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_global_ksi'].values,'b', label='residual_noise_global')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_coastal_ksi'].values,'r', label='residual_noise_coastal')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_offshore_lowvar_ksi'].values,'g', label='residual_noise_offshore_lowvar')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_offshore_highvar_ksi'].values,'k', label='residual_noise_offshore_highvar')

        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_global_ksi'].values,'b--', label='karin_noise_global')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_coastal_ksi'].values,'r--', label='karin_noise_coastal')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_offshore_lowvar_ksi'].values,'g--', label='karin_noise_offshore_lowvar')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_offshore_highvar_ksi'].values,'k--', label='karin_noise_offshore_highvar')
        plt.title('RELATIVE VORTIVITY', fontweight='bold')
        plt.xlabel('Across-track distance [km]', fontweight='bold')
        plt.ylabel('RMSE []', fontweight='bold')
        plt.grid()
        plt.legend(bbox_to_anchor=(1.02, 0.5, 1, 0.2), loc="upper left", ncol=2)
          
    
    def compute_along_track_psd(self, l_files, etuvar, l_files_inputs, lengh_scale=512, overlay=128., details=False, psd_type='welch'):
        """ compute along track psd """
            
        
        def create_segments_from_1d(lon, lat, ssh_true, ssh_noisy, ssh_calib, npt=512, n_overlay=128, center=True):
            """
            decoupage en segments d'une serie lon,lat,ssh 1D
            on suppose que les lon/lat sont toutes définies, mais les ssh peuvent avoir des trous
            """ 
            
            l_segments_ssh_true = []
            l_segments_ssh_noisy = []
            l_segments_ssh_calib = []
    
            # parcours des données
            n_obs = len(lon)
            ii=0 
            while ii+npt < n_obs: 
                seg_ssh_true = ssh_true[ii:ii+npt]
                seg_ssh_noisy = ssh_noisy[ii:ii+npt]
                seg_ssh_calib = ssh_calib[ii:ii+npt]
               
                if ((not np.any(np.isnan(seg_ssh_true))) and (not np.any(np.isnan(seg_ssh_noisy))) and (not np.any(np.isnan(seg_ssh_calib)))):
                #if not ( np.any(np.isnan(seg_ssh_true)) + np.any(np.isnan(seg_ssh_noisy)) + np.any(np.isnan(seg_ssh_filtered)) ):
                
                    l_segments_ssh_true.append(seg_ssh_true)
                    l_segments_ssh_noisy.append(seg_ssh_noisy)
                    l_segments_ssh_calib.append(seg_ssh_calib)
                
                # l_segments.append(
                #     Segment(
                #         lon[ii:ii+l],
                #         lat[ii:ii+l],
                #         seg_ssh-np.mean(seg_ssh)
                #     )
                # )
                ii+=npt-n_overlay
            return l_segments_ssh_true, l_segments_ssh_noisy, l_segments_ssh_calib
        
        l_segment_ssh_true = []
        l_segment_ssh_noisy = []
        l_segment_ssh_calib = []
        
        l_segment_ug_true = []
        l_segment_ug_noisy = []
        l_segment_ug_calib = []
        
        l_segment_ksi_true = []
        l_segment_ksi_noisy = []
        l_segment_ksi_calib = []
        
        resolution = 2. # along-track resolution in km
        npt = int(lengh_scale/resolution)
        n_overlap = int(npt*overlay)
        
        
        for i, fname in enumerate(l_files):
            
            # calib field
            swt = SwotTrack(fname)
            swt.compute_geos_current(etuvar, 'calib_geos_current')
            swt.compute_relative_vorticity('calib_geos_current_x', 'calib_geos_current_y', 'calib_ksi')
            
            # Truth
            swt_input = SwotTrack(l_files_inputs[i])
            swt_input.compute_geos_current('ssh_true', 'true_geos_current')
            swt_input.compute_relative_vorticity('true_geos_current_x', 'true_geos_current_y', 'true_ksi')
            
            # 
            swt_input.compute_geos_current('ssh_err', 'simulated_noise_geos_current')
            swt_input.compute_relative_vorticity('simulated_noise_geos_current_x', 'simulated_noise_geos_current_y', 'simulated_noise_ksi')
            # NEED TO CHEK CONSISTENCY BETWEEN Fileterd and thruth if file not sorted

        
            # parcours des différentes lignes along-track
            for ac_index in swt._dset.nC.values:

                # extraction des lon/lat/ssh
                lon = swt._dset.lon.values[:,ac_index]
                lat = swt._dset.lat.values[:,ac_index]
                
                ssh_true = swt_input._dset['ssh_true'].values[:, ac_index]
                ssh_noisy = swt_input._dset['ssh_err'].values[:, ac_index]
                ssh_calib = swt._dset[etuvar].values[:, ac_index]
                
                ug_true = swt_input._dset['true_geos_current'].values[:,ac_index]
                ug_noisy = swt_input._dset['simulated_noise_geos_current'].values[:,ac_index]
                ug_calib = swt._dset['calib_geos_current'].values[:,ac_index]
                
                ksi_true = swt_input._dset['true_ksi'].values[:,ac_index]
                ksi_noisy = swt_input._dset['simulated_noise_ksi'].values[:,ac_index]
                ksi_calib = swt._dset['calib_ksi'].values[:,ac_index]
            
                # construction de la liste des segments
                al_seg_ssh_true, al_seg_ssh_noisy, al_seg_ssh_calib = create_segments_from_1d(lon, 
                                                                                                 lat, 
                                                                                                 ssh_true, 
                                                                                                 ssh_noisy, 
                                                                                                 ssh_calib, 
                                                                                                 npt=npt,  
                                                                                                 n_overlay=n_overlap)
                
                al_seg_ug_true, al_seg_ug_noisy, al_seg_ug_calib = create_segments_from_1d(lon, 
                                                                                                 lat, 
                                                                                                 ug_true, 
                                                                                                 ug_noisy, 
                                                                                                 ug_calib, 
                                                                                                 npt=npt,  
                                                                                                 n_overlay=n_overlap)
                    
                    
                al_seg_ksi_true, al_seg_ksi_noisy, al_seg_ksi_calib = create_segments_from_1d(lon, 
                                                                                                 lat, 
                                                                                                 ksi_true, 
                                                                                                 ksi_noisy, 
                                                                                                 ksi_calib, 
                                                                                                 npt=npt,  
                                                                                                 n_overlay=n_overlap)
                l_segment_ssh_true.append(al_seg_ssh_true)
                l_segment_ssh_noisy.append(al_seg_ssh_noisy)
                l_segment_ssh_calib.append(al_seg_ssh_calib)
                
                l_segment_ug_true.append(al_seg_ug_true)
                l_segment_ug_noisy.append(al_seg_ug_noisy)
                l_segment_ug_calib.append(al_seg_ug_calib)
                
                l_segment_ksi_true.append(al_seg_ksi_true)
                l_segment_ksi_noisy.append(al_seg_ksi_noisy)
                l_segment_ksi_calib.append(al_seg_ksi_calib)
        
        # on met la liste à plat
        l_flat_ssh_true = np.asarray(list(chain.from_iterable(l_segment_ssh_true))).flatten()
        l_flat_ssh_noisy = np.asarray(list(chain.from_iterable(l_segment_ssh_noisy))).flatten()
        l_flat_ssh_calib = np.asarray(list(chain.from_iterable(l_segment_ssh_calib))).flatten()
        
        l_flat_ug_true = np.asarray(list(chain.from_iterable(l_segment_ug_true))).flatten()
        l_flat_ug_noisy = np.asarray(list(chain.from_iterable(l_segment_ug_noisy))).flatten()
        l_flat_ug_calib = np.asarray(list(chain.from_iterable(l_segment_ug_calib))).flatten()
        
        l_flat_ksi_true = np.asarray(list(chain.from_iterable(l_segment_ksi_true))).flatten()
        l_flat_ksi_noisy = np.asarray(list(chain.from_iterable(l_segment_ksi_noisy))).flatten()
        l_flat_ksi_calib = np.asarray(list(chain.from_iterable(l_segment_ksi_calib))).flatten()
        
        # PSD 
        freq, cross_spectrum = signal.csd(l_flat_ssh_noisy,l_flat_ssh_calib, fs=1./resolution, nperseg=npt, noverlap=0)
        if psd_type == 'welch':
            freq, psd_ssh_true  = signal.welch(l_flat_ssh_true, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
            freq, psd_ssh_noisy = signal.welch(l_flat_ssh_noisy, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
            freq, psd_ssh_calib = signal.welch(l_flat_ssh_calib, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
            freq, psd_err = signal.welch(l_flat_ssh_calib - l_flat_ssh_true, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
            freq, psd_err_err = signal.welch(l_flat_ssh_noisy - l_flat_ssh_true, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')

            freq, psd_ug_true  = signal.welch(l_flat_ug_true, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
            freq, psd_ug_noisy = signal.welch(l_flat_ug_noisy, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
            freq, psd_ug_calib = signal.welch(l_flat_ug_calib, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
            freq, psd_err_ug = signal.welch(l_flat_ug_calib - l_flat_ug_true, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
            freq, psd_err_err_ug = signal.welch(l_flat_ug_noisy - l_flat_ug_true, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')

            freq, psd_ksi_true  = signal.welch(l_flat_ksi_true, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
            freq, psd_ksi_noisy = signal.welch(l_flat_ksi_noisy, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
            freq, psd_ksi_calib = signal.welch(l_flat_ksi_calib, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
            freq, psd_err_ksi = signal.welch(l_flat_ksi_calib - l_flat_ksi_true, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
            freq, psd_err_err_ksi = signal.welch(l_flat_ksi_noisy - l_flat_ksi_true, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
        
        
        elif psd_type == 'powerspec': 
            from src import powerspec
            
            lon = swt._dset.lon.values 
            lat = swt._dset.lat.values 
            
            swt_input.fill_nadir_gap('ssh_true')
            swt_input.fill_nadir_gap('ssh_err')
            swt.fill_nadir_gap(etuvar)
                
            ssh_true = np.ma.array(swt_input._dset['ssh_true'].values )
            ssh_noisy = np.ma.array(swt_input._dset['ssh_err'].values )
            ssh_calib = np.ma.array(swt._dset[etuvar].values )

            swt_input.fill_nadir_gap('true_geos_current')
            swt_input.fill_nadir_gap('simulated_noise_geos_current')
            swt.fill_nadir_gap('calib_geos_current')
            
            ug_true = np.ma.array(swt_input._dset['true_geos_current'].values )
            ug_noisy = np.ma.array(swt_input._dset['simulated_noise_geos_current'].values )
            ug_calib = np.ma.array(swt._dset['calib_geos_current'].values )

            swt_input.fill_nadir_gap('true_ksi')
            swt_input.fill_nadir_gap('simulated_noise_ksi')
            swt.fill_nadir_gap('calib_ksi')
            
            ksi_true = np.ma.array(swt_input._dset['true_ksi'].values )
            ksi_noisy = np.ma.array(swt_input._dset['simulated_noise_ksi'].values )
            ksi_calib = np.ma.array(swt._dset['calib_ksi'].values )
            
            freq, psd_ssh_true  = powerspec.wavenumber_spectra(ssh_true,lon,lat,'Tukey',None)
            freq, psd_ssh_noisy = powerspec.wavenumber_spectra(ssh_noisy,lon,lat,'Tukey',None)
            freq, psd_ssh_calib = powerspec.wavenumber_spectra(ssh_calib,lon,lat,'Tukey',None)
            freq, psd_err       = powerspec.wavenumber_spectra(ssh_calib-ssh_true,lon,lat,'Tukey',None)
            freq, psd_err_err   = powerspec.wavenumber_spectra(ssh_noisy-ssh_true,lon,lat,'Tukey',None)

            freq, psd_ug_true  = powerspec.wavenumber_spectra(ug_true,lon,lat,'Tukey',None)
            freq, psd_ug_noisy = powerspec.wavenumber_spectra(ug_noisy,lon,lat,'Tukey',None)
            freq, psd_ug_calib = powerspec.wavenumber_spectra(ug_calib,lon,lat,'Tukey',None)
            freq, psd_err_ug   = powerspec.wavenumber_spectra(ug_calib-ug_true,lon,lat,'Tukey',None)
            freq, psd_err_err_ug = powerspec.wavenumber_spectra(ug_noisy-ug_true,lon,lat,'Tukey',None)

            freq, psd_ksi_true  = powerspec.wavenumber_spectra(ksi_true,lon,lat,'Tukey',None)
            freq, psd_ksi_noisy = powerspec.wavenumber_spectra(ksi_noisy,lon,lat,'Tukey',None)
            freq, psd_ksi_calib = powerspec.wavenumber_spectra(ksi_calib,lon,lat,'Tukey',None)
            freq, psd_err_ksi = powerspec.wavenumber_spectra(ksi_calib-ksi_true,lon,lat,'Tukey',None)
            freq, psd_err_err_ksi = powerspec.wavenumber_spectra(ksi_noisy-ksi_true,lon,lat,'Tukey',None)
            
            freq = freq*1e3

            
        
        self.freq = freq
        self.cross_spectrum = cross_spectrum
        self.psd_ssh_true = psd_ssh_true
        self.psd_ssh_noisy = psd_ssh_noisy
        self.psd_ssh_calib = psd_ssh_calib
        self.psd_err = psd_err
        self.psd_err_err = psd_err_err
        
        self.psd_ug_true = psd_ug_true
        self.psd_ug_noisy = psd_ug_noisy
        self.psd_ug_calib = psd_ug_calib
        self.psd_err_ug = psd_err_ug
        self.psd_err_err_ug = psd_err_err_ug
        
        self.psd_ksi_true = psd_ksi_true
        self.psd_ksi_noisy = psd_ksi_noisy
        self.psd_ksi_calib = psd_ksi_calib
        self.psd_err_ksi = psd_err_ksi
        self.psd_err_err_ksi = psd_err_err_ksi
    
    
    def write_along_track_psd(self, fname, psd_type = 'welch', **kwargs):
        """ export des résultats vers un fichier NetCDF """
        
        def compute_snr1(array, wavenumber, threshold=0.5):
            """
            :param array:
            :param wavenumber:
            :param threshold:
            :return:
            """

            flag_multiple_crossing = False

            zero_crossings =  np.where(np.diff(np.sign(array - threshold)) != 0.)[0]
            if len(zero_crossings) > 1:
                #print('Multiple crossing', len(zero_crossings))
                flag_multiple_crossing = True
                
            list_of_res = []
            if len(zero_crossings) > 0:    
                for index in range(zero_crossings.size):
            
                    if zero_crossings[index] + 1 < array.size:

                        array1 = array[zero_crossings[index]] - threshold
                        array2 = array[zero_crossings[index] + 1] - threshold
                        dist1 = np.log(wavenumber[zero_crossings[index]])
                        dist2 = np.log(wavenumber[zero_crossings[index] + 1])
                        log_wavenumber_crossing = dist1 - array1 * (dist1 - dist2) / (array1 - array2)
                        
                        resolution_scale = 1. / np.exp(log_wavenumber_crossing)

                    else:
                        resolution_scale = 0.
            
                    list_of_res.append(resolution_scale)
                
                if len(list_of_res) > 0:
                    resolution_scale = np.nanmax(np.asarray(list_of_res))
                else: 
                    resolution_scale = np.nanmin(1./wavenumber[wavenumber!=0])
 
            else:  
                if np.all( array - threshold>0 )>0:
                    resolution_scale = np.nan
                else : 
                    resolution_scale = np.nanmin(1./wavenumber[wavenumber!=0])
            #print(list_of_res) 
        
            return resolution_scale#, flag_multiple_crossing
        
        self.wavelength_snr1_calib = compute_snr1(self.psd_err/self.psd_ssh_true, self.freq)
        self.wavelength_snr1_nocalib = compute_snr1(self.psd_err_err/self.psd_ssh_true, self.freq)
        
        self.wavelength_snr1_calib_ug = compute_snr1(self.psd_err_ug/self.psd_ug_true, self.freq)
        self.wavelength_snr1_nocalib_ug = compute_snr1(self.psd_err_err_ug/self.psd_ug_true, self.freq)
        
        self.wavelength_snr1_calib_ksi = compute_snr1(self.psd_err_ksi/self.psd_ksi_true, self.freq)
        self.wavelength_snr1_nocalib_ksi = compute_snr1(self.psd_err_err_ksi/self.psd_ksi_true, self.freq)
        
        
        if psd_type == 'welch':
            data_vars0 = dict(
                    psd_ssh_true=(["wavenumber"], self.psd_ssh_true),
                    cross_spectrum_r=(["wavenumber"], np.real(self.cross_spectrum)),
                    cross_spectrum_i=(["wavenumber"], np.imag(self.cross_spectrum)),
                    psd_ssh_noisy=(["wavenumber"], self.psd_ssh_noisy),
                    psd_ssh_calib=(["wavenumber"], self.psd_ssh_calib),
                    psd_err=(["wavenumber"], self.psd_err),
                    psd_err_err=(["wavenumber"], self.psd_err_err),
                    snr1_calib=(["wavelength_snr1"], [1]),

                    psd_ug_true=(["wavenumber"], self.psd_ug_true),
                    psd_ug_noisy=(["wavenumber"], self.psd_ug_noisy),
                    psd_ug_calib=(["wavenumber"], self.psd_ug_calib),
                    psd_err_ug=(["wavenumber"], self.psd_err_ug),
                    psd_err_err_ug=(["wavenumber"], self.psd_err_err_ug),

                    psd_ksi_true=(["wavenumber"], self.psd_ksi_true),
                    psd_ksi_noisy=(["wavenumber"], self.psd_ksi_noisy),
                    psd_ksi_calib=(["wavenumber"], self.psd_ksi_calib),
                    psd_err_ksi=(["wavenumber"], self.psd_err_ksi),
                    psd_err_err_ksi=(["wavenumber"], self.psd_err_err_ksi),

                )
        elif psd_type == 'powerspec':
            data_vars0 = dict(
                    psd_ssh_true=(["wavenumber"], self.psd_ssh_true), 
                    psd_ssh_noisy=(["wavenumber"], self.psd_ssh_noisy),
                    psd_ssh_calib=(["wavenumber"], self.psd_ssh_calib),
                    psd_err=(["wavenumber"], self.psd_err),
                    psd_err_err=(["wavenumber"], self.psd_err_err),
                    snr1_calib=(["wavelength_snr1"], [1]),

                    psd_ug_true=(["wavenumber"], self.psd_ug_true),
                    psd_ug_noisy=(["wavenumber"], self.psd_ug_noisy),
                    psd_ug_calib=(["wavenumber"], self.psd_ug_calib),
                    psd_err_ug=(["wavenumber"], self.psd_err_ug),
                    psd_err_err_ug=(["wavenumber"], self.psd_err_err_ug),

                    psd_ksi_true=(["wavenumber"], self.psd_ksi_true),
                    psd_ksi_noisy=(["wavenumber"], self.psd_ksi_noisy),
                    psd_ksi_calib=(["wavenumber"], self.psd_ksi_calib),
                    psd_err_ksi=(["wavenumber"], self.psd_err_ksi),
                    psd_err_err_ksi=(["wavenumber"], self.psd_err_err_ksi),

                )
        
        
        to_write = xr.Dataset(
            data_vars=data_vars0,
            coords=dict(
                wavenumber=(["wavenumber"], self.freq),
                wavelength_snr1_calib=(["wavelength_snr1"], [self.wavelength_snr1_calib]),
                wavelength_snr1_calib_ug=(["wavelength_snr1_ug"], [self.wavelength_snr1_calib_ug]),
                wavelength_snr1_calib_ksi=(["wavelength_snr1_ksi"], [self.wavelength_snr1_calib_ksi]),
                wavelength_snr1_nocalib=(["wavelength_snr1"], [self.wavelength_snr1_nocalib]),
                wavelength_snr1_nocalib_ug=(["wavelength_snr1_ug"], [self.wavelength_snr1_nocalib_ug]),
                wavelength_snr1_nocalib_ksi=(["wavelength_snr1_ksi"], [self.wavelength_snr1_nocalib_ksi]),
            ),
            attrs=dict(
                description=kwargs['description'] if 'description' in kwargs else 'PSD analysis',
                calib_type=kwargs['calib'] if 'calib' in kwargs else 'None',
            ),
        )
        
        to_write.to_netcdf(fname)
        
    def display_psd(self, fname):
        
        ds = xr.open_dataset(fname)

        ds = ds.assign_coords({'wavelength': 1./ds['wavenumber']})

        fig = plt.figure(figsize=(15, 18)) 

        ax = plt.subplot(321)
        ds['psd_ssh_true'].plot(x='wavelength', label='PSD(SSH$_{true}$)', color='k', xscale='log', yscale='log', lw=3)
        ds['psd_ssh_noisy'].plot(x='wavelength', label='PSD(SSH$_{noisy}$)', color='r', lw=2)
        ds['psd_ssh_calib'].plot(x='wavelength', label='PSD(SSH$_{calib}$)', color='b', lw=2)
        ds['psd_err'].plot(x='wavelength', label='PSD(SSH$_{err}$)', color='grey', lw=2)
        plt.grid(which='both')
        plt.loglog(ds['wavelength'],ds['wavelength']**(5)/(ds['psd_ssh_noisy'][0]),'k--' ,label='k$^{-5}$') 
        plt.legend()
        plt.xlabel('wavelenght [km]')
        plt.ylabel('PSD [m$^2$.cy$^{-1}$.km$^{-1}$]')
        ax.invert_xaxis()
        plt.title('PSD Sea Surface Height')

        ds['SNR_calib'] = ds['psd_err']/ds['psd_ssh_true']
        ds['SNR_nocalib'] = ds['psd_err_err']/ds['psd_ssh_true']
        ax = plt.subplot(322)
        ds['SNR_calib'].plot(x='wavelength', label='PSD(SSH$_{err}$)/PSD(SSH$_{true}$)', color='b', xscale='log', lw=3)
        ds['SNR_nocalib'].plot(x='wavelength', label='PSD(Err$_{noise}$)/PSD(SSH$_{true}$)', color='r', lw=2)
        (ds['SNR_calib']/ds['SNR_calib']*0.5).plot(x='wavelength', label='SNR=1', color='grey', lw=2)
        plt.scatter(ds.wavelength_snr1_calib, 0.5, color='b', zorder=4, label="SNR1 AFTER calib")
        plt.scatter(ds.wavelength_snr1_nocalib, 0.5, color='r', zorder=4, label="SNR1 BEFORE calib")
        plt.grid(which='both')
        plt.legend()
        plt.xlabel('wavelenght [km]')
        plt.ylabel('SNR')

        plt.ylim(0, 2)
        ax.invert_xaxis()
        plt.title('SNR Sea Surface Height')


        ax = plt.subplot(323)
        ds['psd_ug_true'].plot(x='wavelength', label='PSD(Ug$_{true}$)', color='k', xscale='log', yscale='log', lw=3)
        ds['psd_ug_noisy'].plot(x='wavelength', label='PSD(Ug$_{noisy}$)', color='r', lw=2)
        ds['psd_ug_calib'].plot(x='wavelength', label='PSD(Ug$_{calib}$)', color='b', lw=2)
        ds['psd_err_ug'].plot(x='wavelength', label='PSD(err)', color='grey', lw=2)
        plt.grid(which='both')
        plt.loglog(ds['wavelength'],ds['wavelength']**(3),'k--' ,label='k$^{-3}$') 
        plt.legend()
        plt.xlabel('wavelenght [km]')
        plt.ylabel('PSD [m$^2$.s$^{-2}$.cy$^{-1}$.km$^{-1}$]')
        ax.invert_xaxis()
        plt.title('PSD Geostrophic current')

        ds['SNR_calib_ug'] = ds['psd_err_ug']/ds['psd_ug_true']
        ds['SNR_nocalib_ug'] = ds['psd_err_err_ug']/ds['psd_ug_true']
        ax = plt.subplot(324)
        ds['SNR_calib_ug'].plot(x='wavelength', label='PSD(Ug$_{err}$)/PSD(Ug$_{true}$)', color='b', xscale='log', lw=3)
        ds['SNR_nocalib_ug'].plot(x='wavelength', label='PSD(Ug$_{noise}$)/PSD(Ug$_{true}$)', color='r', lw=2)
        (ds['SNR_calib_ug']/ds['SNR_calib_ug']*0.5).plot(x='wavelength', label='SNR=1', color='grey', lw=2)
        plt.scatter(ds.wavelength_snr1_calib_ug, 0.5, color='b', zorder=4, label="SNR1 AFTER calib")
        plt.scatter(ds.wavelength_snr1_nocalib_ug, 0.5, color='r', zorder=4, label="SNR1 BEFORE calib")
        plt.grid(which='both')
        plt.legend()
        plt.ylim(0, 2)
        ax.invert_xaxis()
        plt.title('SNR Geostrophic current')
        plt.xlabel('wavelenght [km]')
        plt.ylabel('SNR')


        ax = plt.subplot(325)
        ds['psd_ksi_true'].plot(x='wavelength', label='PSD($\zeta_{true}$)', color='k', xscale='log', yscale='log', lw=3)
        ds['psd_ksi_noisy'].plot(x='wavelength', label='PSD($\zeta_{noisy}$)', color='r', lw=2)
        ds['psd_ksi_calib'].plot(x='wavelength', label='PSD($\zeta_{calib}$)', color='b', lw=2)
        ds['psd_err_ksi'].plot(x='wavelength', label='psd_err', color='grey', lw=2)
        plt.grid(which='both')
        plt.legend()
        plt.xlabel('wavelenght [km]')
        plt.ylabel('PSD [cy$^{-1}$.km$^{-1}$]')
        ax.invert_xaxis()
        plt.title('PSD Relative vorticity')

        ds['SNR_calib_ksi'] = ds['psd_err_ksi']/ds['psd_ksi_true']
        ds['SNR_nocalib_ksi'] = ds['psd_err_err_ksi']/ds['psd_ksi_true']
        ax = plt.subplot(326)
        ds['SNR_calib_ksi'].plot(x='wavelength', label='PSD($\zeta_{err}$)/PSD($\zeta_{true}$)', color='b', xscale='log', lw=3)
        ds['SNR_nocalib_ksi'].plot(x='wavelength', label='PSD($\zeta_{noise}$)/PSD($\zeta_{true}$)', color='r', lw=2)
        (ds['SNR_calib_ksi']/ds['SNR_calib_ksi']*0.5).plot(x='wavelength', label='SNR=1', color='grey', lw=2)
        plt.scatter(ds.wavelength_snr1_calib_ksi, 0.5, color='b', zorder=4, label="SNR1 AFTER calib")
        plt.scatter(ds.wavelength_snr1_nocalib_ksi, 0.5, color='r', zorder=4, label="SNR1 BEFORE calib")
        plt.grid(which='both')
        plt.legend()
        plt.ylim(0, 2)
        ax.invert_xaxis()
        plt.title('SNR Relative vorticity')
        plt.xlabel('wavelenght [km]')
        plt.ylabel('SNR')
        
        plt.show()
    
    
    def summary(self, notebook_name):
        
        wavelength_snr1_calib = self.wavelength_snr1_calib
        wavelength_snr1_nocalib = self.wavelength_snr1_nocalib
         
            
        data = [['no calib', 
                 'SSH [m]',
                 self.mean_ssh_nocalib_rmse,  
                 np.round(wavelength_snr1_nocalib, 1),
                 ''], 
                ['no calib', 
                 'Geostrophic current [m.s$^-1$]',
                 self.mean_ug_nocalib_rmse,  
                 np.round(self.wavelength_snr1_nocalib_ug, 1),
                 ''],
                ['no calib', 
                 'Relative vorticity []',
                 self.mean_ksi_nocalib_rmse,  
                 np.round(self.wavelength_snr1_nocalib_ksi, 1),
                 '']
               
               ]
        
        Leaderboard_nocalib = pd.DataFrame(data, 
                           columns=['Method',
                                    'Field',
                                    "µ(RMSE)",    
                                    'λ(SNR1) [km]', 
                                    'Reference'])
            
            
        data = [[self.calib_name, 
                 'SSH [m]',
                 self.mean_ssh_rmse,  
                 np.round(wavelength_snr1_calib, 1),
                 notebook_name], 
                [self.calib_name, 
                 'Geostrophic current [m.s$^-1$]',
                 self.mean_ug_rmse,  
                 np.round(self.wavelength_snr1_calib_ug, 1),
                 notebook_name],
                [self.calib_name, 
                 'Relative vorticity []',
                 self.mean_ksi_rmse,  
                 np.round(self.wavelength_snr1_calib_ksi, 1),
                 notebook_name]
               
               ]
        
        Leaderboard_calib = pd.DataFrame(data, 
                           columns=['Method',
                                    'Field',
                                    "µ(RMSE)",    
                                    'λ(SNR1) [km]', 
                                    'Reference'])
        
        
        print("Summary of the leaderboard metrics:")
        print(Leaderboard_nocalib.to_markdown())
        print(Leaderboard_calib.to_markdown())
            
    
    

    

    @property
    def longitude(self):
        return self.stats_dict['ssh'].x

    @property
    def latitude(self):
        return self.stats_dict['ssh'].y
        
    @property
    def gridstep(self):
        return self._gridstep

    @property
    def stats_dict(self):
        return self._d_stats
    