# This script is for plotting barplot
# 7/16/20

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
import numpy as np
import pickle
import numpy.ma as ma

# load data
a_file = open("urban_frac.pkl", "rb")
urban_frac = pickle.load(a_file)
a_file.close()
a_file = open("time.pkl", "rb")
time = pickle.load(a_file)
a_file.close()
a_file = open("lon_lat.pkl", "rb")
lon_lat = pickle.load(a_file)
a_file.close()
a_file = open("mask_universal_clear_daytime.pkl", "rb")
mask_universal_clear_daytime = pickle.load(a_file)
a_file.close()
a_file = open("attribution_clear_daytime_all.pkl", "rb")
attribution_clear_daytime_all = pickle.load(a_file)
a_file.close()

mask_clear_daytime_universal = mask_universal_clear_daytime['mask_universal']
(nrow, ncol, N) = mask_clear_daytime_universal.shape  # N is number of months in the whole time series
yr_End = time['yr_End'][0, 0]
yr_Num = time['yr_Num'][0, 0]
yr_Start = time['yr_Start'][0, 0]
lat = lon_lat['lat']
lon = lon_lat['lon']
lon[lon > 180] = lon[lon > 180] - 360

# create mask_urbn
mask_urban_temp = np.empty(shape=(nrow, ncol))
mask_urban_temp[:] = np.nan
mask_urban_temp[urban_frac['urban_frac'] > 0.001] = 1
mask_urban = np.repeat(mask_urban_temp[:, :, np.newaxis], N, axis=2)

# create mask_season
average_period_value = 2  # average_period_value = 1 (annual), 2(summer), 3(winter)
mask_season = np.empty(shape=(nrow, ncol, N))
mask_season[:] = np.nan
ind_NH = np.where(lat >= 0)
ind_SH = np.where(lat < 0)
if average_period_value == 1:
    mask_season[:] = 1
elif average_period_value == 2:
    for i in range(yr_Num):
        mask_season[:, ind_NH, (i * 12 + 5):(i * 12 + 8)] = 1
        mask_season[:, ind_SH, (i * 12):(i * 12 + 2)] = 1
        mask_season[:, ind_SH, i * 12 + 11] = 1
elif average_period_value == 3:
    for i in range(yr_Num):
        mask_season[:, ind_NH, (i * 12):(i * 12 + 2)] = 1
        mask_season[:, ind_NH, i * 12 + 11] = 1
        mask_season[:, ind_SH, (i * 12 + 5):(i * 12 + 8)] = 1
else:
    print("average_period_value needs to be 1 or 2 or 3")

mask_only_urban_season = mask_season * mask_urban
mask_to_use = mask_clear_daytime_universal * mask_season * mask_urban

# select regions of interest
region_names = ['Global', 'US', 'China', 'Europe', 'CentralAmerica', 'NorthAfrica', 'India']
region_Num = len(region_names)
global_index = 0
US_index = 1
China_index = 2
EU_index = 3
CentralAmerica_index = 4
NorthAfrica_index = 5
India_index = 6

lat_index = []
lon_index = []
# global
lat_index = [np.arange(len(lat))]
lon_index = [np.arange(len(lon))]
# US
lon_left_EA = -90
lon_right_EA = -70
lat_low_EA = 30
lat_up_EA = 50
lat_index.append(np.where((lat > lat_low_EA) & (lat < lat_up_EA))[0])
lon_index.append(np.where((lon > lon_left_EA) & (lon < lon_right_EA))[0])
# China
lon_left_EC = 100
lon_right_EC = 120
lat_low_EC = 20
lat_up_EC = 40
lat_index.append(np.where((lat > lat_low_EC) & (lat < lat_up_EC))[0])
lon_index.append(np.where((lon > lon_left_EC) & (lon < lon_right_EC))[0])
# Europe
lon_left_EU = 15
lon_right_EU = 30
lat_low_EU = 35
lat_up_EU = 55
lat_index.append(np.where((lat > lat_low_EU) & (lat < lat_up_EU))[0])
lon_index.append(np.where((lon > lon_left_EU) & (lon < lon_right_EU))[0])
# CentralAmerica
lon_left_CA = -92
lon_right_CA = -78
lat_low_CA = 8
lat_up_CA = 20
lat_index.append(np.where((lat > lat_low_CA) & (lat < lat_up_CA))[0])
lon_index.append(np.where((lon > lon_left_CA) & (lon < lon_right_CA))[0])
# NorthAfrica
lon_left_NA = -20
lon_right_NA = 50
lat_low_NA = 10
lat_up_NA = 35
lat_index.append(np.where((lat > lat_low_NA) & (lat < lat_up_NA))[0])
lon_index.append(np.where(((lon > lon_left_NA) & (lon < 0)) | ((lon > 0) & (lon < lon_right_NA)))[0])
# India
lon_left_IN = 70
lon_right_IN = 87
lat_low_IN = 8
lat_up_IN = 28
lat_index.append(np.where((lat > lat_low_IN) & (lat < lat_up_IN))[0])
lon_index.append(np.where((lon > lon_left_IN) & (lon < lon_right_IN))[0])


# calculate spatial average and variability

def temporal_mean_std(var_input):
    """calculate spatial average and variability"""
    n = var_input.shape[2]
    var_temporal = np.zeros(shape=(1, n))
    for k in range(n):
        var_selected = var_input[:, :, k]
        var_temporal[0, k] = np.nanmean(var_selected)  # monthly average
    var_temporal_average = np.nanmean(var_temporal)
    var_temporal_variability = np.nanstd(var_temporal)
    return [var_temporal, var_temporal_average, var_temporal_variability]


variable_all = ['Diff_Ts_all', 'Ts_sum_TRM_all', 'Ts_term_alpha_TRM_all', 'Ts_term_ra_sum_TRM_all', 'Ts_term_rs_TRM_all',
                'Ts_term_Grnd_TRM_all']
var_Num = len(variable_all)
exper = ['all_Ts_rural_urbn_clear_daytime', 'all_WGT_rural_urbn_clear_daytime',
         'all_T2_rural_urbn_clear_daytime', 'all_WGT_2m_rural_urbn_clear_daytime']
exper_Num = len(exper)

dic = {}
for ivar in range(var_Num):
    var1 = np.zeros((region_Num, exper_Num, N))
    var1[:] = np.nan
    var2 = np.zeros((region_Num, exper_Num))
    var2[:] = np.nan
    var3 = np.zeros((region_Num, exper_Num))
    var3[:] = np.nan
    for iexper in range(exper_Num):
        for region_index in range(region_Num):
            var_name = variable_all[ivar]
            variable = attribution_clear_daytime_all[var_name][0, iexper]
            variable_lon = variable[lon_index[region_index], :, :]
            variable_lon_lat = variable_lon[:, lat_index[region_index], :]
            mask2use_lon = mask_to_use[lon_index[region_index], :, :]
            mask2use_lon_lat = mask2use_lon[:, lat_index[region_index], :]
            [var1[region_index, iexper, :], var2[region_index, iexper], var3[region_index, iexper]] \
                = temporal_mean_std(variable_lon_lat * mask2use_lon_lat)

    dic[variable_all[ivar] + '_global'] = var1
    dic[variable_all[ivar] + '_mean'] = var2
    dic[variable_all[ivar] + '_std'] = var3

# plot spatial average and variability for each attribution term
# first plot
fig, axes = plt.subplots(3, 1)
fig.set_size_inches(12, 10)
font = 14
# set width of bar
barWidth = 0.225
title = ['US', 'China', 'Europe']
text = [r'$\mathrm{a)}$', r'$\mathrm{b)}$', r'$\mathrm{c)}$']
xticks = ['GFDL', 'TRM', r'$\alpha$', r'$r_{a}$', r'$r_{s}$', 'G']
for i in range(3):
    axes[i].set_title(title[i], fontsize=font, fontweight='bold')
    axes[i].set_xticks([r + 1.5*barWidth for r in range(var_Num)])
    axes[i].set_xticklabels(xticks)
    axes[i].set_ylabel('Urban - Rural')
    axes[i].text(-0.05, 1.1, text[i], fontsize=font-2, verticalalignment='top', transform=axes[i].transAxes)
experiment_names = ['Canopy air temperature' + r'$\/ \mathrm{(^oC)}$', 'Canopy air SWBGT',
                     'Reference temperature' + r'$\/ \mathrm{(^oC)}$', 'Reference SWBGT']

# set height of bar
bar = np.zeros((exper_Num, var_Num))
err = np.zeros((exper_Num, var_Num))
r = np.zeros((exper_Num, var_Num))
color = ['black', 'red', 'green', 'blue']
for region_index in range(1, 4):
    for iexper in range(exper_Num):
        bar[iexper, :] = [dic[variable_all[ivar] + '_mean'][region_index, iexper] for ivar in range(var_Num)]
        err[iexper, :] = [dic[variable_all[ivar] + '_std'][region_index, iexper] for ivar in range(var_Num)]
        # Set position of bar on X axis
        if iexper == 0:
            r[iexper, :] = np.arange(len(bar[iexper, :]))
        else:
            r[iexper, :] = [x + barWidth for x in r[iexper - 1]]
        # make the plot
        p1 = axes[region_index - 1].bar(r[iexper, :], bar[iexper, :], color=color[iexper], width=barWidth,
                                        edgecolor='white', label=experiment_names[iexper], yerr=err[iexper, :],
                                        align='center', alpha=0.8, ecolor='black', capsize=5)
    axes[region_index - 1].axhline(linewidth=0.5, color='grey')
axes[0].legend()
# axes[0].set_axis_off()
plt.show()
fig.savefig('figures/figure_daytime_summer_TRM1_ra_all.png', dpi=200)


# Second plot
fig1, axes1 = plt.subplots(3, 1)
fig1.set_size_inches(12, 10)
font = 14
# set width of bar
barWidth = 0.225
title = ['Central America', 'North Africa', 'India']
text = [r'$\mathrm{a)}$', r'$\mathrm{b)}$', r'$\mathrm{c)}$']
xticks = ['GFDL', 'TRM', r'$\alpha$', r'$r_{a}$', r'$r_{s}$', 'G']
for i in range(3):
    axes1[i].set_title(title[i], fontsize=font, fontweight='bold')
    axes1[i].set_xticks([r + 1.5*barWidth for r in range(var_Num)])
    axes1[i].set_xticklabels(xticks)
    axes1[i].set_ylabel('Urban - Rural')
    axes1[i].text(-0.05, 1.1, text[i], fontsize=font-2, verticalalignment='top', transform=axes1[i].transAxes)
experiment_names = ['Canopy air temperature' + r'$\/ \mathrm{(^oC)}$', 'Canopy air SWBGT',
                     'Reference temperature' + r'$\/ \mathrm{(^oC)}$', 'Reference SWBGT']

# set height of bar
bar = np.zeros((exper_Num, var_Num))
err = np.zeros((exper_Num, var_Num))
r = np.zeros((exper_Num, var_Num))
color = ['black', 'red', 'green', 'blue']
for region_index in range(4, 7):
    for iexper in range(exper_Num):
        bar[iexper, :] = [dic[variable_all[ivar] + '_mean'][region_index, iexper] for ivar in range(var_Num)]
        err[iexper, :] = [dic[variable_all[ivar] + '_std'][region_index, iexper] for ivar in range(var_Num)]
        # Set position of bar on X axis
        if iexper == 0:
            r[iexper, :] = np.arange(len(bar[iexper, :]))
        else:
            r[iexper, :] = [x + barWidth for x in r[iexper - 1]]
        # make the plot
        p1 = axes1[region_index - 4].bar(r[iexper, :], bar[iexper, :], color=color[iexper], width=barWidth,
                                        edgecolor='white', label=experiment_names[iexper], yerr=err[iexper, :],
                                        align='center', alpha=0.8, ecolor='black', capsize=5)
    axes1[region_index - 4].axhline(linewidth=0.5, color='grey')
axes1[0].legend()
# axes[0].set_axis_off()
plt.show()
fig1.savefig('figures/figure_daytime_summer_TRM2_ra_all.png', dpi=200)

# for ra only
variable_all2 = ['Diff_Ts_all', 'Ts_sum_TRM_all', 'Ts_term_alpha_TRM_all', 'Ts_term_ra_TRM_all', 'Ts_term_rs_TRM_all',
                'Ts_term_Grnd_TRM_all']
var_Num = len(variable_all2)
exper = ['all_Ts_rural_urbn_clear_daytime', 'all_WGT_rural_urbn_clear_daytime',
         'all_T2_rural_urbn_clear_daytime', 'all_WGT_2m_rural_urbn_clear_daytime']
exper_Num = len(exper)

dic = {}
for ivar in range(var_Num):
    var1 = np.zeros((region_Num, exper_Num, N))
    var1[:] = np.nan
    var2 = np.zeros((region_Num, exper_Num))
    var2[:] = np.nan
    var3 = np.zeros((region_Num, exper_Num))
    var3[:] = np.nan
    for iexper in range(exper_Num):
        for region_index in range(region_Num):
            var_name = variable_all2[ivar]
            variable = attribution_clear_daytime_all[var_name][0, iexper]
            variable_lon = variable[lon_index[region_index], :, :]
            variable_lon_lat = variable_lon[:, lat_index[region_index], :]
            mask2use_lon = mask_to_use[lon_index[region_index], :, :]
            mask2use_lon_lat = mask2use_lon[:, lat_index[region_index], :]
            [var1[region_index, iexper, :], var2[region_index, iexper], var3[region_index, iexper]] \
                = temporal_mean_std(variable_lon_lat * mask2use_lon_lat)

    dic[variable_all2[ivar] + '_global'] = var1
    dic[variable_all2[ivar] + '_mean'] = var2
    dic[variable_all2[ivar] + '_std'] = var3

# plot3
fig3, axes = plt.subplots(3, 1)
fig3.set_size_inches(12, 10)
font = 14
# set width of bar
barWidth = 0.225
title = ['US', 'China', 'Europe']
text = [r'$\mathrm{a)}$', r'$\mathrm{b)}$', r'$\mathrm{c)}$']
xticks = ['GFDL', 'TRM', r'$\alpha$', r'$r_{a}$', r'$r_{s}$', 'G']
for i in range(3):
    axes[i].set_title(title[i], fontsize=font, fontweight='bold')
    axes[i].set_xticks([r + 1.5*barWidth for r in range(var_Num)])
    axes[i].set_xticklabels(xticks)
    axes[i].set_ylabel('Urban - Rural')
    axes[i].text(-0.05, 1.1, text[i], fontsize=font-2, verticalalignment='top', transform=axes[i].transAxes)
experiment_names = ['Canopy air temperature' + r'$\/ \mathrm{(^oC)}$', 'Canopy air SWBGT',
                     'Reference temperature' + r'$\/ \mathrm{(^oC)}$', 'Reference SWBGT']

# set height of bar
bar = np.zeros((exper_Num, var_Num))
err = np.zeros((exper_Num, var_Num))
r = np.zeros((exper_Num, var_Num))
color = ['black', 'red', 'green', 'blue']
for region_index in range(1, 4):
    for iexper in range(exper_Num):
        bar[iexper, :] = [dic[variable_all2[ivar] + '_mean'][region_index, iexper] for ivar in range(var_Num)]
        err[iexper, :] = [dic[variable_all2[ivar] + '_std'][region_index, iexper] for ivar in range(var_Num)]
        # Set position of bar on X axis
        if iexper == 0:
            r[iexper, :] = np.arange(len(bar[iexper, :]))
        else:
            r[iexper, :] = [x + barWidth for x in r[iexper - 1]]
        # make the plot
        p1 = axes[region_index - 1].bar(r[iexper, :], bar[iexper, :], color=color[iexper], width=barWidth,
                                        edgecolor='white', label=experiment_names[iexper], yerr=err[iexper, :],
                                        align='center', alpha=0.8, ecolor='black', capsize=5)
    axes[region_index - 1].axhline(linewidth=0.5, color='grey')
axes[0].legend()
# axes[0].set_axis_off()
plt.show()
fig3.savefig('figures/figure_daytime_summer_TRM1_ra.png', dpi=200)


# plot4
fig4, axes1 = plt.subplots(3, 1)
fig4.set_size_inches(12, 10)
font = 14
# set width of bar
barWidth = 0.225
title = ['Central America', 'North Africa', 'India']
text = [r'$\mathrm{a)}$', r'$\mathrm{b)}$', r'$\mathrm{c)}$']
xticks = ['GFDL', 'TRM', r'$\alpha$', r'$r_{a}$', r'$r_{s}$', 'G']
for i in range(3):
    axes1[i].set_title(title[i], fontsize=font, fontweight='bold')
    axes1[i].set_xticks([r + 1.5*barWidth for r in range(var_Num)])
    axes1[i].set_xticklabels(xticks)
    axes1[i].set_ylabel('Urban - Rural')
    axes1[i].text(-0.05, 1.1, text[i], fontsize=font-2, verticalalignment='top', transform=axes1[i].transAxes)
experiment_names = ['Canopy air temperature' + r'$\/ \mathrm{(^oC)}$', 'Canopy air SWBGT',
                     'Reference temperature' + r'$\/ \mathrm{(^oC)}$', 'Reference SWBGT']

# set height of bar
bar = np.zeros((exper_Num, var_Num))
err = np.zeros((exper_Num, var_Num))
r = np.zeros((exper_Num, var_Num))
color = ['black', 'red', 'green', 'blue']
for region_index in range(4, 7):
    for iexper in range(exper_Num):
        bar[iexper, :] = [dic[variable_all2[ivar] + '_mean'][region_index, iexper] for ivar in range(var_Num)]
        err[iexper, :] = [dic[variable_all2[ivar] + '_std'][region_index, iexper] for ivar in range(var_Num)]
        # Set position of bar on X axis
        if iexper == 0:
            r[iexper, :] = np.arange(len(bar[iexper, :]))
        else:
            r[iexper, :] = [x + barWidth for x in r[iexper - 1]]
        # make the plot
        p1 = axes1[region_index - 4].bar(r[iexper, :], bar[iexper, :], color=color[iexper], width=barWidth,
                                        edgecolor='white', label=experiment_names[iexper], yerr=err[iexper, :],
                                        align='center', alpha=0.8, ecolor='black', capsize=5)
    axes1[region_index - 4].axhline(linewidth=0.5, color='grey')
axes1[0].legend()
# axes[0].set_axis_off()
plt.show()
fig4.savefig('figures/figure_daytime_summer_TRM2_ra.png', dpi=200)

# reference layer only
variable_all3 = ['Diff_Ts_all', 'Ts_sum_TRM_all', 'Ts_term_alpha_TRM_all', 'Ts_term_ra_TRM_all',
                 'Ts_term_ra_prime_TRM_all', 'Ts_term_rs_TRM_all', 'Ts_term_Grnd_TRM_all']
var_Num = len(variable_all3)
# exper = ['all_Ts_rural_urbn_clear_daytime', 'all_WGT_rural_urbn_clear_daytime',
#          'all_T2_rural_urbn_clear_daytime', 'all_WGT_2m_rural_urbn_clear_daytime']
exper_Num = 2

dic = {}
for ivar in range(var_Num):
    var1 = np.zeros((region_Num, exper_Num, N))
    var1[:] = np.nan
    var2 = np.zeros((region_Num, exper_Num))
    var2[:] = np.nan
    var3 = np.zeros((region_Num, exper_Num))
    var3[:] = np.nan
    for iexper in (2, 3):  # T2 and SWBGT2 only
        for region_index in range(region_Num):
            var_name = variable_all3[ivar]
            variable = attribution_clear_daytime_all[var_name][0, iexper]
            variable_lon = variable[lon_index[region_index], :, :]
            variable_lon_lat = variable_lon[:, lat_index[region_index], :]
            mask2use_lon = mask_to_use[lon_index[region_index], :, :]
            mask2use_lon_lat = mask2use_lon[:, lat_index[region_index], :]
            [var1[region_index, iexper-2, :], var2[region_index, iexper-2], var3[region_index, iexper-2]] \
                = temporal_mean_std(variable_lon_lat * mask2use_lon_lat)

    dic[variable_all3[ivar] + '_global'] = var1
    dic[variable_all3[ivar] + '_mean'] = var2
    dic[variable_all3[ivar] + '_std'] = var3

# plot5
fig5, axes = plt.subplots(3, 1)
fig5.set_size_inches(12, 10)
font = 14
# set width of bar
barWidth = 0.3
title = ['US', 'China', 'Europe']
text = [r'$\mathrm{a)}$', r'$\mathrm{b)}$', r'$\mathrm{c)}$']
xticks = ['GFDL', 'TRM', r'$\alpha$', r'$r_{a}$', r'$r^\prime_{a}$', r'$r_{s}$', 'G']
for i in range(3):
    axes[i].set_title(title[i], fontsize=font, fontweight='bold')
    axes[i].set_xticks([r + 0.5*barWidth for r in range(var_Num)])
    axes[i].set_xticklabels(xticks)
    axes[i].set_ylabel('Urban - Rural')
    axes[i].text(-0.05, 1.1, text[i], fontsize=font-2, verticalalignment='top', transform=axes[i].transAxes)
experiment_names = ['Reference temperature' + r'$\/ \mathrm{(^oC)}$', 'Reference SWBGT']

# set height of bar
bar = np.zeros((exper_Num, var_Num))
err = np.zeros((exper_Num, var_Num))
r = np.zeros((exper_Num, var_Num))
color = ['black', 'red', 'green', 'blue']
for region_index in range(1, 4):
    for iexper in range(exper_Num):
        bar[iexper, :] = [dic[variable_all3[ivar] + '_mean'][region_index, iexper] for ivar in range(var_Num)]
        err[iexper, :] = [dic[variable_all3[ivar] + '_std'][region_index, iexper] for ivar in range(var_Num)]
        # Set position of bar on X axis
        if iexper == 0:
            r[iexper, :] = np.arange(len(bar[iexper, :]))
        else:
            r[iexper, :] = [x + barWidth for x in r[iexper - 1]]
        # make the plot
        p1 = axes[region_index - 1].bar(r[iexper, :], bar[iexper, :], color=color[iexper], width=barWidth,
                                        edgecolor='white', label=experiment_names[iexper], yerr=err[iexper, :],
                                        align='center', alpha=0.8, ecolor='black', capsize=5)
    axes[region_index - 1].axhline(linewidth=0.5, color='grey')
axes[0].legend()
# axes[0].set_axis_off()
plt.show()
fig5.savefig('figures/figure_daytime_summer_TRM1_reference.png', dpi=200)


# plot6
fig6, axes1 = plt.subplots(3, 1)
fig6.set_size_inches(12, 10)
font = 14
# set width of bar
barWidth = 0.225
title = ['Central America', 'North Africa', 'India']
text = [r'$\mathrm{a)}$', r'$\mathrm{b)}$', r'$\mathrm{c)}$']
xticks = ['GFDL', 'TRM', r'$\alpha$', r'$r_{a}$', r'$r^\prime_{a}$', r'$r_{s}$', 'G']
for i in range(3):
    axes1[i].set_title(title[i], fontsize=font, fontweight='bold')
    axes1[i].set_xticks([r + 0.5*barWidth for r in range(var_Num)])
    axes1[i].set_xticklabels(xticks)
    axes1[i].set_ylabel('Urban - Rural')
    axes1[i].text(-0.05, 1.1, text[i], fontsize=font-2, verticalalignment='top', transform=axes1[i].transAxes)
experiment_names = ['Reference temperature' + r'$\/ \mathrm{(^oC)}$', 'Reference SWBGT']

# set height of bar
bar = np.zeros((exper_Num, var_Num))
err = np.zeros((exper_Num, var_Num))
r = np.zeros((exper_Num, var_Num))
color = ['black', 'red', 'green', 'blue']
for region_index in range(4, 7):
    for iexper in range(exper_Num):
        bar[iexper, :] = [dic[variable_all3[ivar] + '_mean'][region_index, iexper] for ivar in range(var_Num)]
        err[iexper, :] = [dic[variable_all3[ivar] + '_std'][region_index, iexper] for ivar in range(var_Num)]
        # Set position of bar on X axis
        if iexper == 0:
            r[iexper, :] = np.arange(len(bar[iexper, :]))
        else:
            r[iexper, :] = [x + barWidth for x in r[iexper - 1]]
        # make the plot
        p1 = axes1[region_index - 4].bar(r[iexper, :], bar[iexper, :], color=color[iexper], width=barWidth,
                                        edgecolor='white', label=experiment_names[iexper], yerr=err[iexper, :],
                                        align='center', alpha=0.8, ecolor='black', capsize=5)
    axes1[region_index - 4].axhline(linewidth=0.5, color='grey')
axes1[0].legend()
# axes[0].set_axis_off()
plt.show()
fig6.savefig('figures/figure_daytime_summer_TRM2_reference.png', dpi=200)
