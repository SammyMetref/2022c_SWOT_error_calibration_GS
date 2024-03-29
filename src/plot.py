import numpy as np
import xarray as xr
import matplotlib.pylab as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from .swot import *


def compare_stat(filename_ref, filename_etu, **kwargs):
    
    ds_ref = xr.open_dataset(filename_ref)
    ref_filter = ds_ref.calib_type
    ds_etu = xr.open_dataset(filename_etu)
    etu_filter = ds_etu.calib_type
    
    ds = 100*(ds_etu - ds_ref)/ds_ref
    
    plt.figure(figsize=(18, 15))

        
    ax = plt.subplot(311, projection=ccrs.PlateCarree())
    vmin = np.nanpercentile(ds.ssh_rmse, 5)
    vmax = np.nanpercentile(ds.ssh_rmse, 95)
    vmin = np.maximum(np.abs(vmin), np.abs(vmax))
    ds.ssh_rmse.plot(x='lon', y='lat', vmin=-vmin, vmax=vmin, cmap='bwr', cbar_kwargs={'label': '[%]'}, **kwargs)
    plt.title('$\Delta$ RMSE SSH field ' + f'{etu_filter} vs {ref_filter}', fontweight='bold')
    ax.add_feature(cfeature.LAND, zorder=2)
    ax.coastlines(zorder=2)
    ax.axis([-65,-55,33,43])

    ax = plt.subplot(312, projection=ccrs.PlateCarree())
    vmin = np.nanpercentile(ds.ug_rmse, 5)
    vmax = np.nanpercentile(ds.ug_rmse, 95)
    vmin = np.maximum(np.abs(vmin), np.abs(vmax))
    ds.ug_rmse.plot(x='lon', y='lat', vmin=-vmin, vmax=vmin, cmap='bwr', cbar_kwargs={'label': '[%]'}, **kwargs)
    plt.title('$\Delta$ RMSE GEOSTROPPHIC CURRENT field ' + f'{etu_filter} vs {ref_filter}', fontweight='bold')
    ax.add_feature(cfeature.LAND, zorder=2)
    ax.coastlines(zorder=2)
    ax.axis([-65,-55,33,43])
        
    ax = plt.subplot(313, projection=ccrs.PlateCarree())
    vmin = np.nanpercentile(ds.ksi_rmse, 5)
    vmax = np.nanpercentile(ds.ksi_rmse, 95)
    vmin = np.maximum(np.abs(vmin), np.abs(vmax))
    ds.ksi_rmse.plot(x='lon', y='lat', vmin=-vmin, vmax=vmin, cmap='bwr', cbar_kwargs={'label': '[%]'}, **kwargs)
    plt.title('$\Delta$ RMSE Relative vorticity '+ f'{etu_filter} vs {ref_filter}', fontweight='bold')
    ax.add_feature(cfeature.LAND, zorder=2)
    ax.coastlines(zorder=2)
    ax.axis([-65,-55,33,43])

    plt.show()
    

def compare_stats_by_regime(list_of_filename, list_of_label): 
    
    ds = xr.concat([xr.open_dataset(filename) for filename in list_of_filename], dim='experiment')
    ds['experiment'] = list_of_label
    
    fig = plt.figure(figsize=(12, 13))
    
    plt.subplot(221)
    
    plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_global'][0, :].values, c='k', lw=2, label='karin_noise')
    for exp in ds['experiment'].values:
        ds_sel = ds.where(ds.experiment == exp, drop=True)
        plt.plot(2*ds_sel.num_pixels - 70, ds_sel['rmse_ac_residual_noise_global'].squeeze(), label=f'residual_noise ({exp})')
    plt.title('GLOBAL', fontweight='bold')
    plt.ylabel('Height Error [m]')  
    plt.xlabel('Ground Range [km]')
    plt.legend(loc='best')
    plt.grid()
    plt.ylim(0, 0.03)
    
    plt.subplot(222)
    plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_coastal'][0, :].values,c='k', lw=2, label='karin_noise')
    for exp in ds['experiment'].values:
        ds_sel = ds.where(ds.experiment == exp, drop=True)
        plt.plot(2*ds_sel.num_pixels - 70, ds_sel['rmse_ac_residual_noise_coastal'].squeeze(), label=f'residual_noise ({exp})')
    plt.title('COASTAL', fontweight='bold')
    plt.ylabel('Height Error [m]')  
    plt.xlabel('Ground Range [km]')
    plt.legend(loc='best')
    plt.grid()
    plt.ylim(0, 0.03)
    
    plt.subplot(223)
    plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_offshore_lowvar'][0, :].values,c='k', lw=2, label='karin_noise_offshore_lowvar')
    for exp in ds['experiment'].values:
        ds_sel = ds.where(ds.experiment == exp, drop=True)
        plt.plot(2*ds_sel.num_pixels - 70, ds_sel['rmse_ac_residual_noise_offshore_lowvar'].squeeze(), label=f'residual_noise ({exp})')
    plt.title('OFFSHORE (> 200km),\n LOW VARIBILITY (< 200cm$^2$)', fontweight='bold')
    plt.ylabel('Height Error [m]')  
    plt.xlabel('Ground Range [km]')
    plt.legend(loc='best')
    plt.grid()
    plt.ylim(0, 0.03)
    
    plt.subplot(224)
    plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_offshore_highvar'][0, :].values,c='k', lw=2, label='karin_noise')
    for exp in ds['experiment'].values:
        ds_sel = ds.where(ds.experiment == exp, drop=True)
        plt.plot(2*ds_sel.num_pixels - 70, ds_sel['rmse_ac_residual_noise_offshore_highvar'].squeeze(), label=f'residual_noise ({exp})')
    plt.title('OFFSHORE (> 200km),\n HIGH VARIBILITY (> 200cm$^2$)', fontweight='bold')
    plt.ylabel('Height Error [m]')  
    plt.xlabel('Ground Range [km]')
    plt.legend(loc='best')
    plt.grid()
    plt.ylim(0, 0.03)
    fig.show()   


    fig = plt.figure(figsize=(12, 13))
    
    plt.subplot(221)
    
    plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_global_ug'][0, :].values, c='k', lw=2, label='karin_noise')
    for exp in ds['experiment'].values:
        ds_sel = ds.where(ds.experiment == exp, drop=True)
        plt.plot(2*ds_sel.num_pixels - 70, ds_sel['rmse_ac_residual_noise_global_ug'].squeeze(), label=f'residual_noise ({exp})')
    plt.title('GLOBAL', fontweight='bold')
    plt.ylabel('Geost. current Error [m.s$^{-1}$]')  
    plt.xlabel('Ground Range [km]')
    plt.legend(loc='best')
    plt.grid()
    plt.ylim(0, 2.)
    
    plt.subplot(222)
    plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_coastal_ug'][0, :].values,c='k', lw=2, label='karin_noise')
    for exp in ds['experiment'].values:
        ds_sel = ds.where(ds.experiment == exp, drop=True)
        plt.plot(2*ds_sel.num_pixels - 70, ds_sel['rmse_ac_residual_noise_coastal_ug'].squeeze(), label=f'residual_noise ({exp})')
    plt.title('COASTAL', fontweight='bold')
    plt.ylabel('Geost. current Error [m.s$^{-1}$]')  
    plt.xlabel('Ground Range [km]')
    plt.legend(loc='best')
    plt.grid()
    plt.ylim(0, 2.)
    
    plt.subplot(223)
    plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_offshore_lowvar_ug'][0, :].values,c='k', lw=2, label='karin_noise_offshore_lowvar')
    for exp in ds['experiment'].values:
        ds_sel = ds.where(ds.experiment == exp, drop=True)
        plt.plot(2*ds_sel.num_pixels - 70, ds_sel['rmse_ac_residual_noise_offshore_lowvar_ug'].squeeze(), label=f'residual_noise ({exp})')
    plt.title('OFFSHORE (> 200km),\n LOW VARIBILITY (< 200cm$^2$)', fontweight='bold')
    plt.ylabel('Geost. current Error [m.s$^{-1}$]')  
    plt.xlabel('Ground Range [km]')
    plt.legend(loc='best')
    plt.grid()
    plt.ylim(0, 2.)
    
    plt.subplot(224)
    plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_offshore_highvar_ug'][0, :].values,c='k', lw=2, label='karin_noise')
    for exp in ds['experiment'].values:
        ds_sel = ds.where(ds.experiment == exp, drop=True)
        plt.plot(2*ds_sel.num_pixels - 70, ds_sel['rmse_ac_residual_noise_offshore_highvar_ug'].squeeze(), label=f'residual_noise ({exp})')
    plt.title('OFFSHORE (> 200km),\n HIGH VARIBILITY (> 200cm$^2$)', fontweight='bold')
    plt.ylabel('Geost. current Error [m.s$^{-1}$]')  
    plt.xlabel('Ground Range [km]')
    plt.legend(loc='best')
    plt.grid()
    plt.ylim(0, 2.)
    fig.show()


    fig = plt.figure(figsize=(12, 13))
    
    plt.subplot(221)
    
    plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_global_ksi'][0, :].values, c='k', lw=2, label='karin_noise')
    for exp in ds['experiment'].values:
        ds_sel = ds.where(ds.experiment == exp, drop=True)
        plt.plot(2*ds_sel.num_pixels - 70, ds_sel['rmse_ac_residual_noise_global_ksi'].squeeze(), label=f'residual_noise ({exp})')
    plt.title('GLOBAL', fontweight='bold')
    plt.ylabel('Relative vorticity Error []')  
    plt.xlabel('Ground Range [km]')
    plt.legend(loc='best')
    plt.grid()
    plt.ylim(0, 35)
    
    plt.subplot(222)
    plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_coastal_ksi'][0, :].values,c='k', lw=2, label='karin_noise')
    for exp in ds['experiment'].values:
        ds_sel = ds.where(ds.experiment == exp, drop=True)
        plt.plot(2*ds_sel.num_pixels - 70, ds_sel['rmse_ac_residual_noise_coastal_ksi'].squeeze(), label=f'residual_noise ({exp})')
    plt.title('COASTAL', fontweight='bold')
    plt.ylabel('Relative vorticity Error []')  
    plt.xlabel('Ground Range [km]')
    plt.legend(loc='best')
    plt.grid()
    plt.ylim(0, 35)
    
    plt.subplot(223)
    plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_offshore_lowvar_ksi'][0, :].values,c='k', lw=2, label='karin_noise_offshore_lowvar')
    for exp in ds['experiment'].values:
        ds_sel = ds.where(ds.experiment == exp, drop=True)
        plt.plot(2*ds_sel.num_pixels - 70, ds_sel['rmse_ac_residual_noise_offshore_lowvar_ksi'].squeeze(), label=f'residual_noise ({exp})')
    plt.title('OFFSHORE (> 200km),\n LOW VARIBILITY (< 200cm$^2$)', fontweight='bold')
    plt.ylabel('Relative vorticity Error []')  
    plt.xlabel('Ground Range [km]')
    plt.legend(loc='best')
    plt.grid()
    plt.ylim(0, 35)
    
    plt.subplot(224)
    plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_offshore_highvar_ksi'][0, :].values,c='k', lw=2, label='karin_noise')
    for exp in ds['experiment'].values:
        ds_sel = ds.where(ds.experiment == exp, drop=True)
        plt.plot(2*ds_sel.num_pixels - 70, ds_sel['rmse_ac_residual_noise_offshore_highvar_ug'].squeeze(), label=f'residual_noise ({exp})')
    plt.title('OFFSHORE (> 200km),\n HIGH VARIBILITY (> 200cm$^2$)', fontweight='bold')
    plt.ylabel('Relative vorticity Error []')  
    plt.xlabel('Ground Range [km]')
    plt.legend(loc='best')
    plt.grid()
    plt.ylim(0, 35)
    fig.show()   

    

def compare_psd(list_of_filename, list_of_label):
        
    ds = xr.concat([xr.open_dataset(filename) for filename in list_of_filename], dim='experiment')
    ds['experiment'] = list_of_label
    ds = ds.assign_coords({'wavelength': 1./ds['wavenumber']})

    fig = plt.figure(figsize=(15, 18))

    ax = plt.subplot(321)
    ds['psd_ssh_true'][0, :].plot(x='wavelength', label='PSD(SSH$_{true}$)', color='k', xscale='log', yscale='log', lw=3)
    ds['psd_ssh_noisy'][0, :].plot(x='wavelength', label='PSD(SSH$_{noisy}$)', color='r', lw=2)
    for exp in ds['experiment'].values:
        (ds['psd_ssh_calib'].where(ds['experiment']==exp, drop=True)).plot(x='wavelength', label='PSD(SSH$_{calib}$)' + f'({exp})', lw=2)
        #(ds['psd_err'].where(ds['experiment']==exp, drop=True)).plot(x='wavelength', label='PSD(SSH$_{err}$)' + f'({exp})', lw=2)
    plt.grid(which='both')
    plt.legend()
    plt.xlabel('wavelenght [km]')
    plt.ylabel('PSD [m$^2$.cy$^{-1}$.km$^{-1}$]')
    ax.invert_xaxis()
    plt.title('PSD Sea Surface Height')

    ds['SNR_calib'] = ds['psd_err']/ds['psd_ssh_true']
    ds['SNR_nocalib'] = ds['psd_err_err']/ds['psd_ssh_true']
    ax = plt.subplot(322)
    for exp in ds['experiment'].values:
        (ds['SNR_calib'].where(ds['experiment']==exp, drop=True)).plot(x='wavelength', label='PSD(SSH$_{err}$)/PSD(SSH$_{true}$)' + f'({exp})', xscale='log', lw=3)
        plt.scatter(ds.wavelength_snr1_calib.where(ds['experiment']==exp, drop=True), 0.5, zorder=4)
    ds['SNR_nocalib'][0, :].plot(x='wavelength', label='PSD(Err$_{noise}$)/PSD(SSH$_{true}$)', color='r', lw=2)
    (ds['SNR_calib'][0, :]/ds['SNR_calib'][0, :]*0.5).plot(x='wavelength', label='SNR=1', color='grey', lw=2)
    plt.scatter(ds.wavelength_snr1_nocalib[0], 0.5, color='r', zorder=4)
    plt.grid(which='both')
    plt.legend()
    plt.xlabel('wavelenght [km]')
    plt.ylabel('SNR')

    plt.ylim(0, 2)
    ax.invert_xaxis()
    plt.title('SNR Sea Surface Height')


    ax = plt.subplot(323)
    ds['psd_ug_true'][0, :].plot(x='wavelength', label='PSD(Ug$_{true}$)', color='k', xscale='log', yscale='log', lw=3)
    ds['psd_ug_noisy'][0, :].plot(x='wavelength', label='PSD(Ug$_{noisy}$)', color='r', lw=2)
    for exp in ds['experiment'].values:
        (ds['psd_ug_calib'].where(ds['experiment']==exp, drop=True)).plot(x='wavelength', label='PSD(Ug$_{calib}$)' + f'({exp})', lw=2)
        #(ds['psd_err_ug'].where(ds['experiment']==exp, drop=True)).plot(x='wavelength', label=f'PSD(err)' + f'({exp})', lw=2)
    plt.grid(which='both')
    plt.legend()
    plt.xlabel('wavelenght [km]')
    plt.ylabel('PSD [m$^2$.s$^{-2}$.cy$^{-1}$.km$^{-1}$]')
    ax.invert_xaxis()
    plt.title('PSD Geostrophic current')

    ds['SNR_calib_ug'] = ds['psd_err_ug']/ds['psd_ug_true']
    ds['SNR_nocalib_ug'] = ds['psd_err_err_ug']/ds['psd_ug_true']
    ax = plt.subplot(324)
    for exp in ds['experiment'].values:
        (ds['SNR_calib_ug'].where(ds['experiment']==exp, drop=True)).plot(x='wavelength', label='PSD(Ug$_{err}$)/PSD(Ug$_{true}$)' + f'({exp})', xscale='log', lw=3)
        plt.scatter(ds.wavelength_snr1_calib_ug.where(ds['experiment']==exp, drop=True), 0.5, zorder=4)
    ds['SNR_nocalib_ug'][0, :].plot(x='wavelength', label='PSD(Ug$_{noise}$)/PSD(Ug$_{true}$)', color='r', lw=2)
    (ds['SNR_calib_ug'][0, :]/ds['SNR_calib_ug'][0, :]*0.5).plot(x='wavelength', label='SNR=1', color='grey', lw=2)
    plt.scatter(ds.wavelength_snr1_nocalib_ug[0], 0.5, color='r', zorder=4)
    plt.grid(which='both')
    plt.legend()
    plt.ylim(0, 2)
    ax.invert_xaxis()
    plt.title('SNR Geostrophic current')
    plt.xlabel('wavelenght [km]')
    plt.ylabel('SNR')


    ax = plt.subplot(325)
    ds['psd_ksi_true'][0, :].plot(x='wavelength', label='PSD($\zeta_{true}$)', color='k', xscale='log', yscale='log', lw=3)
    ds['psd_ksi_noisy'][0, :].plot(x='wavelength', label='PSD($\zeta_{noisy}$)', color='r', lw=2)
    for exp in ds['experiment'].values:
        (ds['psd_ksi_calib'].where(ds['experiment']==exp, drop=True)).plot(x='wavelength', label='PSD($\zeta_{calib}$)' + f'({exp})', lw=2)
        #(ds['psd_err_ksi'].where(ds['experiment']==exp, drop=True)).plot(x='wavelength', label=f'PSD(err)' + f'({exp})', lw=2)
    plt.grid(which='both')
    plt.legend()
    plt.xlabel('wavelenght [km]')
    plt.ylabel('PSD [cy$^{-1}$.km$^{-1}$]')
    ax.invert_xaxis()
    plt.title('PSD Relative vorticity')

    ds['SNR_calib_ksi'] = ds['psd_err_ksi']/ds['psd_ksi_true']
    ds['SNR_nocalib_ksi'] = ds['psd_err_err_ksi']/ds['psd_ksi_true']
    ax = plt.subplot(326)
    for exp in ds['experiment'].values:
        (ds['SNR_calib_ksi'].where(ds['experiment']==exp, drop=True)).plot(x='wavelength', label='PSD($\zeta_{err}$)/PSD($\zeta_{true}$)' + f'({exp})', xscale='log', lw=3)
        plt.scatter(ds.wavelength_snr1_calib_ksi.where(ds['experiment']==exp, drop=True), 0.5, zorder=4)
    ds['SNR_nocalib_ksi'][0, :].plot(x='wavelength', label='PSD($\zeta_{noise}$)/PSD($\zeta_{true}$)', color='r', lw=2)
    (ds['SNR_calib_ksi'][0, :]/ds['SNR_calib_ksi'][0, :]*0.5).plot(x='wavelength', label='SNR=1', color='grey', lw=2)
    plt.scatter(ds.wavelength_snr1_nocalib_ksi[0], 0.5, color='r', zorder=4)
    plt.grid(which='both')
    plt.legend()
    plt.ylim(0, 2)
    ax.invert_xaxis()
    plt.title('SNR Relative vorticity')
    plt.xlabel('wavelenght [km]')
    plt.ylabel('SNR')
        
    plt.show()
    
        
    
def compare_leaderboard(path_leaderboards):
        
    print("Summary of the leaderboard comparison:")
    
    ds_ldb = xr.open_dataset('../results/no_calib/ldb_nocalib.nc')
    df_ldb = ds_ldb.to_dataframe() 
    print(df_ldb.to_markdown())
        
    for path_leaderboard in path_leaderboards:  
         
        ds_ldb = xr.open_dataset(path_leaderboard)  
        df_ldb = ds_ldb.to_dataframe() 
        print(df_ldb.to_markdown())
         
            
    
    
    
    
    
def plot_demo_pass(file_ref_input, file_calib):
    swt_input = SwotTrack(file_ref_input)
    swt_input.compute_geos_current('ssh_true', 'true_geos_current')
    swt_input.compute_relative_vorticity('true_geos_current_x', 'true_geos_current_y', 'true_ksi')
    swt_input.compute_geos_current('ssh_err', 'err_geos_current')
    swt_input.compute_relative_vorticity('err_geos_current_x', 'err_geos_current_y', 'err_ksi')
    
    
    swt_calib = SwotTrack(file_calib)
    swt_calib.compute_geos_current('ssh_err_calib', 'calib_geos_current')
    swt_calib.compute_relative_vorticity('calib_geos_current_x', 'calib_geos_current_y', 'calib_ksi')
     
    x_al0 = 2*(swt_input._dset.x_al-swt_input._dset.x_al[0]) 
    n_p, n_l = np.meshgrid(swt_input._dset.nC.values,x_al0)
    
    fig, axs = plt.subplots(2, 3,figsize=(10,16))
    row,col = 0,0
    ax = axs[row,col]
    ssh_true = swt_input._dset.ssh_true.values
    msk = np.isnan(swt_input._dset.ssh_err)
    ssh_true[msk] = np.nan
    vmin = np.nanpercentile(ssh_true, 5)
    vmax = np.nanpercentile(ssh_true, 95)
    ax.pcolormesh(n_p,n_l,ssh_true, vmin=vmin, vmax=vmax,cmap='Spectral_r') 
    ax.title.set_text('True SSH') 

    row,col = 1,0
    ax = axs[row,col] 
    pcm =ax.pcolormesh(n_p,n_l,ssh_true, vmin=vmin, vmax=vmax,cmap='Spectral_r') 
    ax.title.set_text('True SSH') 

    fig.colorbar(pcm, ax=axs[:,col], shrink=0.6,location='left', label='[m]') 


    row,col = 0,1
    ax = axs[row,col]
    ssh_err = swt_input._dset.ssh_err.values
    ax.pcolormesh(n_p,n_l,ssh_err,vmin=vmin, vmax=vmax, cmap='Spectral_r') 
    ax.title.set_text('SSH with errors') 

    row,col = 1,1
    ax = axs[row,col]
    ssh_err_calib = swt_calib._dset.ssh_err_calib.values
    pcm =ax.pcolormesh(n_p,n_l, ssh_err_calib, vmin=vmin, vmax=vmax,cmap='Spectral_r') 
    ax.title.set_text('Calibrated SSH') 

    cb=fig.colorbar(pcm, ax=axs[:,col], shrink=0.6, label='[m]') 
    cb.remove()

    row,col = 0,2
    ax = axs[row,col]
    ssh_diff = ssh_true - ssh_err
    vmin = np.nanpercentile(ssh_diff, 5)
    vmax = np.nanpercentile(ssh_diff, 95)
    vdata = np.maximum(np.abs(vmin), np.abs(vmax))
    ax.pcolormesh(n_p,n_l,ssh_diff,vmin=-vdata,vmax=vdata,cmap='bwr') 
    ax.title.set_text('SWOT errors on SSH') 

    row,col = 1,2
    ax = axs[row,col]
    ssh_diff_calib = ssh_true-ssh_err_calib
    pcm =ax.pcolormesh(n_p,n_l,ssh_diff_calib,vmin=-vdata,vmax=vdata,cmap='bwr') 
    ax.title.set_text('Residual errors') 

    fig.colorbar(pcm, ax=axs[:,col], shrink=0.6, label='[m]') 


    plt.subplots_adjust(left=0.25,wspace=0.5,right=0.8) 

    fig.show()
    
    
    
    
    
    fig, axs = plt.subplots(2, 3,figsize=(10,16))
    row,col = 0,0
    ax = axs[row,col]
    ssh_true = swt_input._dset.true_geos_current.values
    msk = np.isnan(swt_input._dset.ssh_err)
    ssh_true[msk] = np.nan
    vmin = np.nanpercentile(ssh_true, 5)
    vmax = np.nanpercentile(ssh_true, 95)
    ax.pcolormesh(n_p,n_l,ssh_true, vmin=vmin, vmax=vmax,cmap='Spectral_r') 
    ax.title.set_text('True Ug') 

    row,col = 1,0
    ax = axs[row,col] 
    pcm =ax.pcolormesh(n_p,n_l,ssh_true, vmin=vmin, vmax=vmax,cmap='Spectral_r') 
    ax.title.set_text('True Ug') 

    fig.colorbar(pcm, ax=axs[:,col], shrink=0.6,location='left', label='[m.s$^{-1}$]') 


    row,col = 0,1
    ax = axs[row,col]
    ssh_err = swt_input._dset.err_geos_current.values
    ax.pcolormesh(n_p,n_l,ssh_err,vmin=vmin, vmax=vmax, cmap='Spectral_r') 
    ax.title.set_text('Ug with errors') 

    row,col = 1,1
    ax = axs[row,col]
    ssh_err_calib = swt_calib._dset.calib_geos_current.values
    pcm =ax.pcolormesh(n_p,n_l, ssh_err_calib, vmin=vmin, vmax=vmax,cmap='Spectral_r') 
    ax.title.set_text('Calibrated Ug') 

    cb=fig.colorbar(pcm, ax=axs[:,col], shrink=0.6, label='[m.s$^{-1}$]') 
    cb.remove()

    row,col = 0,2
    ax = axs[row,col]
    ssh_diff = ssh_true - ssh_err
    vmin = np.nanpercentile(ssh_diff, 5)
    vmax = np.nanpercentile(ssh_diff, 95)
    vdata = np.maximum(np.abs(vmin), np.abs(vmax))
    ax.pcolormesh(n_p,n_l,ssh_diff,vmin=-vdata,vmax=vdata,cmap='bwr') 
    ax.title.set_text('SWOT errors on Ug') 

    row,col = 1,2
    ax = axs[row,col]
    ssh_diff_calib = ssh_true-ssh_err_calib
    pcm =ax.pcolormesh(n_p,n_l,ssh_diff_calib,vmin=-vdata,vmax=vdata,cmap='bwr') 
    ax.title.set_text('Residual errors Ug') 
    fig.colorbar(pcm, ax=axs[:,col], shrink=0.6, label='[m.s$^{-1}$]') 


    plt.subplots_adjust(left=0.25,wspace=0.5,right=0.8) 

    fig.show()
    
    
    
    
    
    
    
    
    
    
    fig, axs = plt.subplots(2, 3,figsize=(10,16))
    row,col = 0,0
    ax = axs[row,col]
    ssh_true = swt_input._dset.true_ksi.values
    msk = np.isnan(swt_input._dset.ssh_err)
    ssh_true[msk] = np.nan
    vmin = np.nanpercentile(ssh_true, 5)
    vmax = np.nanpercentile(ssh_true, 95)
    ax.pcolormesh(n_p,n_l,ssh_true, vmin=vmin, vmax=vmax,cmap='Spectral_r') 
    ax.title.set_text('True vorticity') 

    row,col = 1,0
    ax = axs[row,col] 
    pcm =ax.pcolormesh(n_p,n_l,ssh_true, vmin=vmin, vmax=vmax,cmap='Spectral_r') 
    ax.title.set_text('True vorticity') 

    fig.colorbar(pcm, ax=axs[:,col], shrink=0.6,location='left', label='[]') 


    row,col = 0,1
    ax = axs[row,col]
    ssh_err = swt_input._dset.err_ksi.values
    ax.pcolormesh(n_p,n_l,ssh_err,vmin=vmin, vmax=vmax, cmap='Spectral_r') 
    ax.title.set_text('Vorticity with errors') 

    row,col = 1,1
    ax = axs[row,col]
    ssh_err_calib = swt_calib._dset.calib_ksi.values
    pcm =ax.pcolormesh(n_p,n_l, ssh_err_calib, vmin=vmin, vmax=vmax,cmap='Spectral_r') 
    ax.title.set_text('Calibrated vorticity') 
    
    cb=fig.colorbar(pcm, ax=axs[:,col], shrink=0.6, label='[]') 
    cb.remove()

    row,col = 0,2
    ax = axs[row,col]
    ssh_diff = ssh_true - ssh_err
    vmin = np.nanpercentile(ssh_diff, 5)
    vmax = np.nanpercentile(ssh_diff, 95)
    vdata = np.maximum(np.abs(vmin), np.abs(vmax))
    ax.pcolormesh(n_p,n_l,ssh_diff,vmin=-vdata,vmax=vdata,cmap='bwr') 
    ax.title.set_text('SWOT errors on vorticity') 
    
    row,col = 1,2
    ax = axs[row,col]
    ssh_diff_calib = ssh_true-ssh_err_calib
    pcm =ax.pcolormesh(n_p,n_l,ssh_diff_calib,vmin=-vdata,vmax=vdata,cmap='bwr') 
    ax.title.set_text('Residual errors vorticity') 
    fig.colorbar(pcm, ax=axs[:,col], shrink=0.6, label='[]') 


    plt.subplots_adjust(left=0.25,wspace=0.5,right=0.8) 
    
    fig.show()
