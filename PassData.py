# This script is for passing data from Matlab to Python
# 7/2/20

from scipy.io import loadmat
import numpy as np
import pickle
from plot_map import world_map
from plot_diff import fun
# from plot_map import world_map
# from plot_map import diff_map
import matplotlib.pyplot as plt

urban_frac = loadmat('Input/urban_frac.mat')
time = loadmat('Input/time.mat')
lon_lat = loadmat('Input/lon_lat.mat')

a_file = open("urban_frac.pkl", "wb")
pickle.dump(urban_frac, a_file)
a_file.close()
a_file = open("time.pkl", "wb")
pickle.dump(time, a_file)
a_file.close()
a_file = open("lon_lat.pkl", "wb")
pickle.dump(lon_lat, a_file)
a_file.close()

all_T2_rural_urbn_clear_daytime = loadmat('Input/all_T2_rural_urbn_clear_daytime.mat')
all_Ts_rural_urbn_clear_daytime = loadmat('Input/all_Ts_rural_urbn_clear_daytime.mat')
all_WGT_2m_rural_urbn_clear_daytime = loadmat('Input/all_WGT_2m_rural_urbn_clear_daytime.mat')
all_WGT_rural_urbn_clear_daytime = loadmat('Input/all_WGT_rural_urbn_clear_daytime.mat')
a_file = open("all_T2_rural_urbn_clear_daytime.pkl", "wb")
pickle.dump(all_T2_rural_urbn_clear_daytime, a_file)
a_file.close()
a_file = open("all_Ts_rural_urbn_clear_daytime.pkl", "wb")
pickle.dump(all_Ts_rural_urbn_clear_daytime, a_file)
a_file.close()
a_file = open("all_WGT_2m_rural_urbn_clear_daytime.pkl", "wb")
pickle.dump(all_WGT_2m_rural_urbn_clear_daytime, a_file)
a_file.close()
a_file = open("all_WGT_rural_urbn_clear_daytime.pkl", "wb")
pickle.dump(all_WGT_rural_urbn_clear_daytime, a_file)
a_file.close()

all_T2_rural_urbn_clear_nighttime = loadmat('Input/all_T2_rural_urbn_clear_nighttime.mat')
all_Ts_rural_urbn_clear_nighttime = loadmat('Input/all_Ts_rural_urbn_clear_nighttime.mat')
all_WGT_2m_rural_urbn_clear_nighttime = loadmat('Input/all_WGT_2m_rural_urbn_clear_nighttime.mat')
all_WGT_rural_urbn_clear_nighttime = loadmat('Input/all_WGT_rural_urbn_clear_nighttime.mat')


a_file = open("all_T2_rural_urbn_clear_nighttime.pkl", "wb")
pickle.dump(all_T2_rural_urbn_clear_nighttime, a_file)
a_file.close()
a_file = open("all_Ts_rural_urbn_clear_nighttime.pkl", "wb")
pickle.dump(all_Ts_rural_urbn_clear_nighttime, a_file)
a_file.close()
a_file = open("all_WGT_2m_rural_urbn_clear_nighttime.pkl", "wb")
pickle.dump(all_WGT_2m_rural_urbn_clear_nighttime, a_file)
a_file.close()
a_file = open("all_WGT_rural_urbn_clear_nighttime.pkl", "wb")
pickle.dump(all_WGT_rural_urbn_clear_nighttime, a_file)
a_file.close()

attribution_clear_daytime_all = loadmat('Input/attribution_clear_daytime_all.mat')
attribution_clear_nighttime_all = loadmat('Input/attribution_clear_nighttime_all.mat')
a_file = open("attribution_clear_daytime_all.pkl", "wb")
pickle.dump(attribution_clear_daytime_all, a_file)
a_file.close()
a_file = open("attribution_clear_nighttime_all.pkl", "wb")
pickle.dump(attribution_clear_nighttime_all, a_file)
a_file.close()

# load data
# a_file = open("all_T2_rural_urbn_clear_daytime.pkl", "rb")
# all_T2_rural_urbn_clear_daytime = pickle.load(a_file)
# a_file.close()
# a_file = open("all_Ts_rural_urbn_clear_daytime.pkl", "rb")
# all_Ts_rural_urbn_clear_daytime = pickle.load(a_file)
# a_file.close()
# a_file = open("all_WGT_2m_rural_urbn_clear_daytime.pkl", "rb")
# all_WGT_2m_rural_urbn_clear_daytime = pickle.load(a_file)
# a_file.close()
# a_file = open("all_WGT_rural_urbn_clear_daytime.pkl", "rb")
# all_WGT_rural_urbn_clear_daytime = pickle.load(a_file)
# a_file.close()
# a_file = open("all_T2_rural_urbn_clear_nighttime.pkl", "rb")
# all_T2_rural_urbn_clear_nighttime = pickle.load(a_file)
# a_file.close()
# a_file = open("all_Ts_rural_urbn_clear_nighttime.pkl", "rb")
# all_Ts_rural_urbn_clear_nighttime = pickle.load(a_file)
# a_file.close()
# a_file = open("all_WGT_2m_rural_urbn_clear_nighttime.pkl", "rb")
# all_WGT_2m_rural_urbn_clear_nighttime = pickle.load(a_file)
# a_file.close()
# a_file = open("all_WGT_rural_urbn_clear_nighttime.pkl", "rb")
# all_WGT_rural_urbn_clear_nighttime = pickle.load(a_file)
# a_file.close()

# need modified
# experiment_names = ['all_T2_rural_urbn_clear_nighttime', 'all_Ts_rural_urbn_clear_nighttime',
                    'all_WGT_2m_rural_urbn_clear_nighttime', 'all_WGT_rural_urbn_clear_nighttime']

# urban-rural contrast for Ts
# Diff_alpha = eval(experiment_names[1] + '[\'alpha_sel\']') - eval(
#     experiment_names[1] + '[\'alpha_ref\']')  # specifically in Ts
# Diff_ra_prime = eval(experiment_names[0] + '[\'ra_prime_sel\']') - eval(
#     experiment_names[0] + '[\'ra_prime_ref\']')  # specifically in T2
# Diff_ra = eval(experiment_names[1] + '[\'ra_sel\']') - eval(experiment_names[1] + '[\'ra_ref\']')
# Diff_rs = eval(experiment_names[1] + '[\'rs_sel\']') - eval(experiment_names[1] + '[\'rs_ref\']')
# Diff_Grnd = eval(experiment_names[1] + '[\'Grnd_sel\']') - eval(experiment_names[1] + '[\'Grnd_ref\']')
# Diff_Rn_str = eval(experiment_names[1] + '[\'Rn_str_sel\']') - eval(experiment_names[1] + '[\'Rn_str_ref\']')
# Diff_Qh = eval(experiment_names[1] + '[\'Qh_sel\']') - eval(experiment_names[1] + '[\'Qh_ref\']')
# Diff_Qle = eval(experiment_names[1] + '[\'Qle_sel\']') - eval(experiment_names[1] + '[\'Qle_ref\']')
# Diff_Ts = all_Ts_rural_urbn_clear_nighttime['Diff_Ts']
# Diff_T2 = all_T2_rural_urbn_clear_nighttime['Diff_T2']
# Diff_WGT = all_WGT_rural_urbn_clear_nighttime['Diff_WGT']
# Diff_WGT2 = all_WGT_2m_rural_urbn_clear_nighttime['Diff_WGT']
#
# (nrow, ncol, N) = Diff_alpha.shape  # N is number of months in the whole time series
# yr_End = time['yr_End'][0, 0]
# yr_Num = time['yr_Num'][0, 0]
# yr_Start = time['yr_Start'][0, 0]
# lat = lon_lat['lat']
# lats = np.repeat(lat, nrow, axis=1)
# lats = np.transpose(lats)
# lon = lon_lat['lon']
# lon[lon > 180] = lon[lon > 180] - 360
# lons = np.repeat(lon, ncol, axis=1)
# with open('lat_lon.npy', 'wb') as f:
#     np.save(f, lats)
#     np.save(f, lons)

# create mask_urbn
# mask_urban_temp = np.empty(shape=(nrow, ncol))
# mask_urban_temp[:] = np.nan
# mask_urban_temp[urban_frac['urban_frac'] > 0.001] = 1
# mask_urban = np.repeat(mask_urban_temp[:, :, np.newaxis], N, axis=2)


# def annual_average(data_monthly, period, yr_num):
#     """calculate annual mean value"""
#     index_day = 0
#     data_annual_average = np.empty((nrow, ncol, yr_num))
#     data_annual_average[:] = np.nan
#     for iYrInd in range(yr_num):
#         data_selected = data_monthly[:, :, index_day:(index_day + 12)]
#         tem = data_selected[:, :, period]
#         data_annual_average[:, :, iYrInd] = np.nanmean(tem, 2)
#         # tem2float = tem.astype(np.float)
#         # data_annual_average[:, :, iYrInd] = np.nanmean(tem2float, 2)
#         index_day = index_day + 12
#
#         # print('%d' % iYrInd)
#     return data_annual_average
#
#
# def multiannual_average(values_input, mask, average_period_value, yr_num):
#     """calculate multi-annual mean value"""
#     if average_period_value == 1:
#         print('1')
#         average_period = np.arange(12)
#         # annual average
#         values_input_average = annual_average(np.multiply(values_input, mask), average_period, yr_num)
#         # multi-annual average
#         values_input_annual = np.nanmean(values_input_average, 2)  # 2*2 array
#         return values_input_annual
#
#     elif average_period_value == 2:
#         print('2')
#         average_period_north = np.arange(5, 8)
#         average_period_south = np.append(np.arange(2), 11)
#         values_input_average_north = annual_average(np.multiply(values_input, mask), average_period_north, yr_num)
#         values_input_annual_north = np.nanmean(values_input_average_north, 2)
#         values_input_average_south = annual_average(np.multiply(values_input, mask), average_period_south, yr_num)
#         values_input_annual_south = np.nanmean(values_input_average_south, 2)
#         values_input_summer = np.empty((nrow,ncol))
#         values_input_summer[lats >= 0] = values_input_annual_north[lats >= 0]
#         values_input_summer[lats <= 0] = values_input_annual_south[lats <= 0]
#         return values_input_summer
#
#     else:
#         print('3')
#         average_period_north = np.append(np.arange(2), 11)
#         average_period_south = np.arange(5, 8)
#         values_input_average_north = annual_average(np.multiply(values_input, mask), average_period_north, yr_num)
#         values_input_annual_north = np.nanmean(values_input_average_north, 2)
#         values_input_average_south = annual_average(np.multiply(values_input, mask), average_period_south, yr_num)
#         values_input_annual_south = np.nanmean(values_input_average_south, 2)
#         values_input_winter = np.empty((nrow, ncol))
#         values_input_winter[lats >= 0] = values_input_annual_north[lats >= 0]
#         values_input_winter[lats <= 0] = values_input_annual_south[lats <= 0]
#         return values_input_winter


# average_period_value = 1 (annual), 2(summer), 3(winter)
# mask_season = np.empty(shape=(nrow, ncol, N))
# mask_season[:] = np.nan
# Average_period_value = 2
# if Average_period_value == 1:
#     mask_season[:] = 1
# elif Average_period_value == 2:
#     for i in range(yr_Num):
#         mask_season[:, :, i * 12 + 5: i * 12 + 7] = 1
# else:
#     for i in range(yr_Num):
#         mask_season[:, :, i * 12: i * 12 + 1] = 1
#         mask_season[:, :, i * 12 + 11] = 1
#
# # with open('Values_input_summer.npy', 'wb') as f:
# #     np.save(f, mask_urban)
# #     np.save(f, Average_period_value)
# #     np.save(f, yr_Num)
#
# # world_map
# Values_input_summer_name = ['Diff_Ts', 'Diff_T2', 'Diff_WGT', 'Diff_WGT2']
# for iname in range(len(Values_input_summer_name)-2):
#     plt.subplots(221 + iname)
#     input_value = eval(Values_input_summer_name[iname])
#     Values_input_summer = multiannual_average(input_value, mask_urban, Average_period_value, yr_Num)
#     # fig = world_map(Values_input_summer, lons, lats, iname)
#     world_map(Values_input_summer, lons, lats, iname)
# plt.tight_layout()
# plt.show()
#
# # diff_map
# diff_name = ['Diff_alpha', 'Diff_rs', 'Diff_ra', 'Diff_ra_prime',
#              'Diff_Grnd', 'Diff_Rn_str', 'Diff_Qh', 'Diff_Qle']
# for iname in range(len(diff_name)):
#     plt.subplots(221+i)
#     input_value = eval(diff_name[iname])
#     Values_input_summer = multiannual_average(input_value, mask_urban, Average_period_value, yr_Num)
#     fun(Values_input_summer, lons, lats, iname)
#
# plt.show()
#
# plt.tight_layout()
# plt.show()
#
# fig, axes = plt.subplots(nrows=4, ncols=2)
#
# for i, ax in enumerate(axes.flat):
#     ax.set_title('Test Axes {}'.format(i))
#     fun(Values_input_summer, lons, lats, i)
#
# fig.tight_layout()
#
# plt.show()
