import xarray as xr
import numpy as np
import sys
import os
import pyinterp
import pyinterp.fill
import matplotlib.pylab as plt
from scipy import interpolate
sys.path.append('..')  


def compute_segment_alongtrack(time_alongtrack, 
                               lat_alongtrack, 
                               lon_alongtrack, 
                               ssh_alongtrack, 
                               ssh_map_interp, 
                               lenght_scale,
                               delta_x,
                               delta_t):

    segment_overlapping = 0.25
    max_delta_t_gap = 4 * np.timedelta64(1, 's')  # max delta t of 4 seconds to cut tracks

    list_lat_segment = []
    list_lon_segment = []
    list_ssh_alongtrack_segment = []
    list_ssh_map_interp_segment = []

    # Get number of point to consider for resolution = lenghtscale in km
    delta_t_jd = delta_t / (3600 * 24)
    npt = int(lenght_scale / delta_x)

    # cut track when diff time longer than 4*delta_t
    indi = np.where((np.diff(time_alongtrack) > max_delta_t_gap))[0]
    track_segment_lenght = np.insert(np.diff(indi), [0], indi[0])

    # Long track >= npt
    selected_track_segment = np.where(track_segment_lenght >= npt)[0]

    if selected_track_segment.size > 0: 

        for track in selected_track_segment:

            if track-1 >= 0:
                index_start_selected_track = indi[track-1]
                index_end_selected_track = indi[track]
            else:
                index_start_selected_track = 0
                index_end_selected_track = indi[track]

            start_point = index_start_selected_track
            end_point = index_end_selected_track

            for sub_segment_point in range(start_point, end_point - npt, int(npt*segment_overlapping)):

                # Near Greenwhich case
                if ((lon_alongtrack[sub_segment_point + npt - 1] < 50.)
                    and (lon_alongtrack[sub_segment_point] > 320.)) \
                        or ((lon_alongtrack[sub_segment_point + npt - 1] > 320.)
                            and (lon_alongtrack[sub_segment_point] < 50.)):

                    tmp_lon = np.where(lon_alongtrack[sub_segment_point:sub_segment_point + npt] > 180,
                                       lon_alongtrack[sub_segment_point:sub_segment_point + npt] - 360,
                                       lon_alongtrack[sub_segment_point:sub_segment_point + npt])
                    mean_lon_sub_segment = np.median(tmp_lon)

                    if mean_lon_sub_segment < 0:
                        mean_lon_sub_segment = mean_lon_sub_segment + 360.
                else:

                    mean_lon_sub_segment = np.median(lon_alongtrack[sub_segment_point:sub_segment_point + npt])

                mean_lat_sub_segment = np.median(lat_alongtrack[sub_segment_point:sub_segment_point + npt])

                ssh_alongtrack_segment = np.ma.masked_invalid(ssh_alongtrack[sub_segment_point:sub_segment_point + npt])

                ssh_map_interp_segment = []
                ssh_map_interp_segment = np.ma.masked_invalid(ssh_map_interp[sub_segment_point:sub_segment_point + npt])
                if np.ma.is_masked(ssh_map_interp_segment):
                    ssh_alongtrack_segment = np.ma.compressed(np.ma.masked_where(np.ma.is_masked(ssh_map_interp_segment), ssh_alongtrack_segment))
                    ssh_map_interp_segment = np.ma.compressed(ssh_map_interp_segment)

                if ssh_alongtrack_segment.size > 0:
                    list_ssh_alongtrack_segment.append(ssh_alongtrack_segment)
                    list_lon_segment.append(mean_lon_sub_segment)
                    list_lat_segment.append(mean_lat_sub_segment)
                    list_ssh_map_interp_segment.append(ssh_map_interp_segment)


    return list_lon_segment, list_lat_segment, list_ssh_alongtrack_segment, list_ssh_map_interp_segment, npt 



def interp_swot2nadir(sats, dir_of_swottracks, swottracks_type='inputs', nremoval=6, ref_nadir='model'):
    
    import copy

    if swottracks_type == 'inputs':
        errorquad_true = []
        errorquad_err = []
    elif swottracks_type == 'calib':
        errorquad_calib = []

    else: 
        raise KeyError('Wrong swottracks_type')

    lon = []
    lat = []
    ref_rm = []
    x_ac = []

    list_of_swottracks = sorted(os.listdir(dir_of_swottracks))

    for iswot in list_of_swottracks:
        if iswot[:2] != 'dc':
            print('Warning: Some files in directory do not start by dc_*:'+iswot)
            continue
        ds_swot = xr.open_mfdataset(dir_of_swottracks+iswot)
        print(iswot)
        for isat in sats: 
            ds_sat = xr.open_dataset(isat)

            timedelay = np.timedelta64(12,'h') 
            mintime,maxtime = np.array(np.mean(ds_swot.time), dtype='datetime64[h]')-timedelay,np.array(np.mean(ds_swot.time), dtype='datetime64[h]')+timedelay

            ds_sat_cut = ds_sat.sel({'time':slice(mintime,maxtime)})

            if ds_sat_cut.time.size != 0:  

                if list_of_swottracks[0] == iswot:
                    time = np.array(ds_sat_cut.time)
                else:
                    time = np.hstack((time,np.array(ds_sat_cut.time)))

                lon = np.hstack((lon,np.array(ds_sat_cut.lon)))
                lat = np.hstack((lat,np.array(ds_sat_cut.lat)))
                
                x_ac_out = interpolate.griddata((np.array(ds_swot.lon).ravel(),np.array(ds_swot.lat).ravel()),
                           np.array(ds_swot.x_ac).ravel(),
                           (ds_sat_cut.lon,ds_sat_cut.lat))

                x_ac = np.hstack((x_ac,x_ac_out)) 

                if swottracks_type == 'inputs':
                    var_out = interpolate.griddata((np.array(ds_swot.lon).ravel(),np.array(ds_swot.lat).ravel()),
                               np.array(ds_swot.ssh_true).ravel(),
                               (ds_sat_cut.lon,ds_sat_cut.lat)) 
                    errorquad_true = np.hstack((errorquad_true,var_out)) 
  

                    var_out = interpolate.griddata((np.array(ds_swot.lon).ravel(),np.array(ds_swot.lat).ravel()),
                               np.array(ds_swot.ssh_err).ravel(),
                               (ds_sat_cut.lon,ds_sat_cut.lat))
                    
                    errorquad_err = np.hstack((errorquad_err,var_out))

                    eq_true_rm = copy.deepcopy(errorquad_true)
                    eq_err_rm = copy.deepcopy(errorquad_err) 
                              

                else:  
                    var_out = interpolate.griddata((np.array(ds_swot.lon).ravel(),np.array(ds_swot.lat).ravel()),
                               np.array(ds_swot.ssh_err_calib).ravel(),
                               (ds_sat_cut.lon,ds_sat_cut.lat))  
                    errorquad_calib = np.hstack((errorquad_calib,var_out)) 
  
                    eq_calib_rm = copy.deepcopy(errorquad_calib)
                    
                if ref_nadir == 'model':
                    ref_rm = np.hstack((ref_rm,np.array(ds_sat_cut.ssh_model*var_out/var_out)))
                if ref_nadir == 'obs':
                    ref_rm = np.hstack((ref_rm,np.array(ds_sat_cut.ssh_obs*var_out/var_out)))
                    
    
    if nremoval !=0:
        if swottracks_type == 'inputs':
            for i in range(np.shape(x_ac)[0]): 
                if x_ac[i]<-60+nremoval: 
                    eq_true_rm[i] = np.nan 
                    eq_err_rm[i] = np.nan  
                if x_ac[i]>60-nremoval: 
                    eq_true_rm[i] = np.nan  
                    eq_err_rm[i] = np.nan 
                if x_ac[i]>-10-nremoval and x_ac[i]<10+nremoval: 
                    eq_true_rm[i] = np.nan  
                    eq_err_rm[i] = np.nan 
        else:
            for i in range(np.shape(x_ac)[0]): 
                if x_ac[i]<-60+nremoval: 
                    eq_calib_rm[i] = np.nan  
                if x_ac[i]>60-nremoval: 
                    eq_calib_rm[i] = np.nan   
                if x_ac[i]>-10-nremoval and x_ac[i]<10+nremoval: 
                    eq_calib_rm[i] = np.nan  
        
            

    if swottracks_type == 'inputs':
        return lon, lat, time, ref_rm, x_ac, eq_true_rm, eq_err_rm
    else: 
        return lon, lat, time, ref_rm, x_ac, eq_calib_rm