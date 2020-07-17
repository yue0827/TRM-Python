# This script is for plotting global map
# 7/16/20

from scipy.io import loadmat
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
import numpy as np
import pickle
import numpy.ma as ma
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


def annual_average(data_monthly, period, yr_num):
    """calculate annual mean value"""
    index_day = 0
    data_annual_average = np.empty((nrow, ncol, yr_num))
    data_annual_average[:] = np.nan
    for iYrInd in range(yr_num):
        data_selected = data_monthly[:, :, index_day:(index_day + 12)]
        tem = data_selected[:, :, period]
        data_annual_average[:, :, iYrInd] = np.nanmean(tem, 2)
        # tem2float = tem.astype(np.float)
        # data_annual_average[:, :, iYrInd] = np.nanmean(tem2float, 2)
        index_day = index_day + 12

        # print('%d' % iYrInd)
    return data_annual_average


def multiannual_average(values_input, mask, average_period_value, yr_num, lats):
    """calculate multi-annual mean value"""
    if average_period_value == 1:
        print('1')
        average_period = np.arange(12)
        # annual average
        values_input_average = annual_average(np.multiply(values_input, mask), average_period, yr_num)
        # multi-annual average
        values_input_annual = np.nanmean(values_input_average, 2)  # 2*2 array
        return values_input_annual

    elif average_period_value == 2:
        print('2')
        average_period_north = np.arange(5, 8)
        average_period_south = np.append(np.arange(2), 11)
        values_input_average_north = annual_average(np.multiply(values_input, mask), average_period_north, yr_num)
        values_input_annual_north = np.nanmean(values_input_average_north, 2)
        values_input_average_south = annual_average(np.multiply(values_input, mask), average_period_south, yr_num)
        values_input_annual_south = np.nanmean(values_input_average_south, 2)
        values_input_summer = np.empty((nrow, ncol))
        values_input_summer[lats >= 0] = values_input_annual_north[lats >= 0]
        values_input_summer[lats <= 0] = values_input_annual_south[lats <= 0]
        return values_input_summer

    else:
        print('3')
        average_period_north = np.append(np.arange(2), 11)
        average_period_south = np.arange(5, 8)
        values_input_average_north = annual_average(np.multiply(values_input, mask), average_period_north, yr_num)
        values_input_annual_north = np.nanmean(values_input_average_north, 2)
        values_input_average_south = annual_average(np.multiply(values_input, mask), average_period_south, yr_num)
        values_input_annual_south = np.nanmean(values_input_average_south, 2)
        values_input_winter = np.empty((nrow, ncol))
        values_input_winter[lats >= 0] = values_input_annual_north[lats >= 0]
        values_input_winter[lats <= 0] = values_input_annual_south[lats <= 0]
        return values_input_winter


class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x2, y2 = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x2, y2), np.isnan(value))



a_file = open("urban_frac.pkl", "rb")
urban_frac = pickle.load(a_file)
a_file.close()
a_file = open("time.pkl", "rb")
time = pickle.load(a_file)
a_file.close()
a_file = open("lon_lat.pkl", "rb")
lon_lat = pickle.load(a_file)
a_file.close()

a_file = open("all_T2_rural_urbn_clear_daytime.pkl", "rb")
all_T2_rural_urbn_clear_daytime = pickle.load(a_file)
a_file.close()
a_file = open("all_Ts_rural_urbn_clear_daytime.pkl", "rb")
all_Ts_rural_urbn_clear_daytime = pickle.load(a_file)
a_file.close()
a_file = open("all_WGT_2m_rural_urbn_clear_daytime.pkl", "rb")
all_WGT_2m_rural_urbn_clear_daytime = pickle.load(a_file)
a_file.close()
a_file = open("all_WGT_rural_urbn_clear_daytime.pkl", "rb")
all_WGT_rural_urbn_clear_daytime = pickle.load(a_file)
a_file.close()

# need modified
exper = ['all_T2_rural_urbn_clear_daytime', 'all_Ts_rural_urbn_clear_daytime',
         'all_WGT_2m_rural_urbn_clear_daytime', 'all_WGT_rural_urbn_clear_daytime']

# urban-rural contrast
Diff_alpha = eval(exper[1] + '[\'alpha_sel\']') - eval(
    exper[1] + '[\'alpha_ref\']')  # specifically in Ts
Diff_ra_prime = eval(exper[0] + '[\'ra_prime_sel\']') - eval(
    exper[0] + '[\'ra_prime_ref\']')  # specifically in T2
Diff_ra = eval(exper[1] + '[\'ra_sel\']') - eval(exper[1] + '[\'ra_ref\']')
Diff_rs = eval(exper[1] + '[\'rs_sel\']') - eval(exper[1] + '[\'rs_ref\']')
Diff_Grnd = eval(exper[1] + '[\'Grnd_sel\']') - eval(exper[1] + '[\'Grnd_ref\']')
Diff_Rn_str = eval(exper[1] + '[\'Rn_str_sel\']') - eval(exper[1] + '[\'Rn_str_ref\']')
Diff_Qh = eval(exper[1] + '[\'Qh_sel\']') - eval(exper[1] + '[\'Qh_ref\']')
Diff_Qle = eval(exper[1] + '[\'Qle_sel\']') - eval(exper[1] + '[\'Qle_ref\']')
Diff_Ts = all_Ts_rural_urbn_clear_daytime['Diff_Ts']
Diff_T2 = all_T2_rural_urbn_clear_daytime['Diff_T2']
Diff_WGT = all_WGT_rural_urbn_clear_daytime['Diff_WGT']
Diff_WGT2 = all_WGT_2m_rural_urbn_clear_daytime['Diff_WGT']

dTs_dalpha = eval(exper[1] + '[\'dTs_dalpha_TRM\']')
dTs_drs = eval(exper[1] + '[\'dTs_drs_TRM\']')
dTs_dra = eval(exper[1] + '[\'dTs_dra_TRM\']')
dTs_dGrnd = eval(exper[1] + '[\'dTs_dGrnd_TRM\']')

dWGT_dalpha = eval(exper[3] + '[\'dWGT_dalpha_TRM\']')
dWGT_drs = eval(exper[3] + '[\'dWGT_drs_TRM\']')
dWGT_dra = eval(exper[3] + '[\'dWGT_dra_TRM\']')
dWGT_dGrnd = eval(exper[3] + '[\'dWGT_dGrnd_TRM\']')

dT2_dalpha = eval(exper[0] + '[\'dT2_dalpha_TRM\']')
dT2_drs = eval(exper[0] + '[\'dT2_drs_TRM\']')
dT2_dra = eval(exper[0] + '[\'dT2_dra_TRM\']')
dT2_dra_prime = eval(exper[0] + '[\'dT2_dra_prime_TRM\']')
dT2_dGrnd = eval(exper[0] + '[\'dT2_dGrnd_TRM\']')

dWGT2_dalpha = eval(exper[2] + '[\'dWGT_dalpha_TRM\']')
dWGT2_drs = eval(exper[2] + '[\'dWGT_drs_TRM\']')
dWGT2_dra = eval(exper[2] + '[\'dWGT_dra_TRM\']')
dWGT2_dra_prime = eval(exper[2] + '[\'dWGT_dra_prime_TRM\']')
dWGT2_dGrnd = eval(exper[2] + '[\'dWGT_dGrnd_TRM\']')

(nrow, ncol, N) = Diff_alpha.shape  # N is number of months in the whole time series
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

Average_period_value = 2  # summer
x, y = np.meshgrid(lon, lat)
xx = np.ndarray.flatten(x.T)
yy = np.ndarray.flatten(y.T)

# First plot (temperature difference)
Values_input_summer_name = ['Diff_Ts', 'Diff_T2', 'Diff_WGT', 'Diff_WGT2']
ind = [[0, 0], [0, 1], [1, 0], [1, 1]]
font = 14

#  making plot
fig, axes = plt.subplots(2, 2)
fig.set_size_inches(12.5, 7.5)
axes[0, 0].set_title(r'$\Delta T_{ca} \/ \mathrm{(^oC)}$', fontsize=font)
axes[0, 0].text(-182, 95, r'$\mathrm{a)}$', fontsize=font)
axes[0, 1].set_title(r'$\Delta T_{ref}\/ \mathrm{(^oC)}$', fontsize=font)
axes[0, 1].text(-182, 95, r'$\mathrm{b)}$', fontsize=font)
axes[1, 0].set_title(r'$\Delta SWGBT$', fontsize=font)
axes[1, 0].text(-182, 95, r'$\mathrm{c)}$', fontsize=font)
axes[1, 1].set_title(r'$\Delta SWGBT_{ref}$', fontsize=font)
axes[1, 1].text(-182, 95, r'$\mathrm{d)}$', fontsize=font)

for ix in range(2):
    for iy in range(2):
        axes[ix, iy].xaxis.set_ticks(np.arange(-120, 180, 60))
        axes[ix, iy].yaxis.set_ticks(np.arange(-30, 90, 30))
        axes[ix, iy].tick_params(direction='in', length=6, width=2)
        axes[ix, iy].xaxis.set_ticklabels([])
        axes[ix, iy].yaxis.set_ticklabels([])

for i in range(len(Values_input_summer_name)):
    # plt.subplots(221 + i)
    input_value = eval(Values_input_summer_name[i])
    Values_input_summer = multiannual_average(input_value, mask_urban, Average_period_value, yr_Num, y.T)

    mp = Basemap(llcrnrlon=-180., llcrnrlat=-60., urcrnrlon=180., urcrnrlat=90.,
                 projection='cyl', ax=axes[ind[i][0], ind[i][1]], suppress_ticks=False)
    mp.drawmapboundary(fill_color='w')
    mp.drawcoastlines()

    Z = Values_input_summer.T
    Zm = ma.masked_where(np.isnan(Z), Z)
    Zm2 = ma.masked_where(Zm > np.nanquantile(Values_input_summer, 0.995), Zm)
    Zm2 = ma.masked_where(Zm2 < np.nanquantile(Values_input_summer, 0.005), Zm2)

    cmin = np.nanmin(Zm2)
    cmax = np.nanmax(Zm2)
    cc = np.ndarray.flatten(Values_input_summer)
    if abs(cmax) > abs(cmin):
        cmin = - cmax
        # print('cmax {} >cmin {}' .format(i,i))
        colorhex = mp.hexbin(xx, yy, C=cc, vmin=cmin, vmax=cmax, gridsize=(134, 80), cmap='RdBu_r',
                             extent=(-180, 180, -60, 90),
                             # norm=MidpointNormalize(midpoint=0, vmin=cmin, vmax=cmax)
                             )

        cb = mp.colorbar(colorhex, location='bottom', pad=0.4, ax=axes[ind[i][0], ind[i][1]])
        # cb.set_label("0.995-quantile")
    else:
        cmax = - cmin
        # print("cmin {} >cmax {}" .format(i,i))
        colorhex = mp.hexbin(xx, yy, C=cc, vmin=cmin, vmax=cmax, gridsize=(134, 80), cmap='RdBu_r',
                             extent=(-180, 180, -60, 90),
                             # norm=MidpointNormalize(midpoint=0, vmin=cmin, vmax=cmax)
                             )

        cb = mp.colorbar(colorhex, location='bottom', pad=0.4, ax=axes[ind[i][0], ind[i][1]])
        # cb.set_label("0.005-quantile")

    mp.drawparallels(np.arange(-60, 90, 30), labels=[1, 0, 0, 0], linewidth=0, ax=axes[ind[i][0], ind[i][1]])
    mp.drawmeridians(np.arange(-180, 180, 60), labels=[0, 0, 0, 1], linewidth=0, ax=axes[ind[i][0], ind[i][1]])
    plt.text(-175, 10, 'JJA', fontname="Arial", fontsize=10, fontweight='bold', ha='left', va='center', color='k')
    plt.text(-175, -10, 'DJF', fontname="Arial", fontsize=10, fontweight='bold', ha='left', va='center', color='k')
plt.show()
fig.savefig('figures/figure_daytime_summer_Tdiff.png', dpi=200)

# Second plot (difference in properties and fluxes)
diff_name1 = ['Diff_alpha', 'Diff_rs', 'Diff_ra', 'Diff_ra_prime']
diff_name2 = ['Diff_Rn_str', 'Diff_Qh', 'Diff_Qle', 'Diff_Grnd']
ind2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
font2 = 14
# Plot2.1
fig2_1, axes2_1 = plt.subplots(2, 2)
# fig2.set_size_inches(9, 11)
fig2_1.set_size_inches(12.5, 7.5)
axes2_1[0, 0].set_title(r'$\Delta \alpha$', fontsize=font2)
axes2_1[0, 0].text(-182, 95, r'$\mathrm{a)}$', fontsize=font2)
axes2_1[0, 1].set_title(r'$\Delta r_{s} \/ \mathrm{(s/m)}$', fontsize=font2)
axes2_1[0, 1].text(-182, 95, r'$\mathrm{b)}$', fontsize=font2)
axes2_1[1, 0].set_title(r'$\Delta r_{a} \/ \mathrm{(s/m)}$', fontsize=font2)
axes2_1[1, 0].text(-182, 95, r'$\mathrm{c)}$', fontsize=font2)
axes2_1[1, 1].set_title(r'$\Delta r^\prime_{a} \/ \mathrm{(s/m)}$', fontsize=font2)
axes2_1[1, 1].text(-182, 95, r'$\mathrm{d)}$', fontsize=font2)

for ix in range(2):
    for iy in range(2):
        axes2_1[ix, iy].xaxis.set_ticks(np.arange(-120, 180, 60))
        axes2_1[ix, iy].yaxis.set_ticks(np.arange(-30, 90, 30))
        axes2_1[ix, iy].tick_params(direction='in', length=6, width=2)
        axes2_1[ix, iy].xaxis.set_ticklabels([])
        axes2_1[ix, iy].yaxis.set_ticklabels([])

for i in range(len(diff_name1)):
    input_value = eval(diff_name1[i])
    Values_input_summer = multiannual_average(input_value, mask_urban, Average_period_value, yr_Num, y.T)

    mp = Basemap(llcrnrlon=-180., llcrnrlat=-60., urcrnrlon=180., urcrnrlat=90.,
                 projection='cyl', ax=axes2_1[ind2[i][0], ind2[i][1]], suppress_ticks=False)

    mp.drawmapboundary(fill_color='w')
    mp.drawcoastlines()
    Z = Values_input_summer.T
    Zm = ma.masked_where(np.isnan(Z), Z)
    cc = np.ndarray.flatten(Values_input_summer)
    if i == 0:  # alpha
        cmin = np.nanmin(Zm)
        cmax = np.nanmax(Zm)
        if abs(cmax) > abs(cmin):
            cmin = - cmax
            colorhex = mp.hexbin(xx, yy, C=cc, gridsize=(134, 80), vmin=cmin, vmax=cmax, cmap='RdBu_r',
                                 extent=(-180, 180, -60, 90),
                                 # norm=MidpointNormalize(midpoint=0, vmin=cmin, vmax=cmax)
                                 )
            cb = mp.colorbar(colorhex, location='bottom', pad=0.4, ax=axes2_1[ind2[i][0], ind2[i][1]])
            # cb.set_label("max")
        else:
            cmax = - cmin
            colorhex = mp.hexbin(xx, yy, C=cc, gridsize=(134, 80), vmin=cmin, vmax=cmax, cmap='RdBu_r',
                                 extent=(-180, 180, -60, 90),
                                 # norm=MidpointNormalize(midpoint=0, vmin=cmin, vmax=cmax)
                                 )
            cb = mp.colorbar(colorhex, location='bottom', pad=0.4, ax=axes2_1[ind2[i][0], ind2[i][1]])
            # cb.set_label("min")
    elif i == 1:  # rs
        Zm2 = ma.masked_where(Zm > np.nanquantile(Values_input_summer, 0.85), Zm)
        Zm2 = ma.masked_where(Zm2 < np.nanquantile(Values_input_summer, 0.15), Zm2)
        cmin = np.nanmin(Zm2)
        cmax = np.nanmax(Zm2)
        if abs(cmax) > abs(cmin):
            cmin = - cmax
            colorhex = mp.hexbin(xx, yy, C=cc, gridsize=(134, 80), vmin=cmin, vmax=cmax, cmap='RdBu_r',
                                 extent=(-180, 180, -60, 90),
                                 norm=MidpointNormalize(midpoint=0, vmin=cmin, vmax=cmax))
            cb = mp.colorbar(colorhex, location='bottom', pad=0.4, ax=axes2_1[ind2[i][0], ind2[i][1]])
            # cb.set_label("0.85-quantile")
        else:
            cmax = - cmin
            colorhex = mp.hexbin(xx, yy, C=cc, gridsize=(134, 80), vmin=cmin, vmax=cmax, cmap='RdBu_r',
                                 extent=(-180, 180, -60, 90),
                                 norm=MidpointNormalize(midpoint=0, vmin=cmin, vmax=cmax))
            cb = mp.colorbar(colorhex, location='bottom', pad=0.4, ax=axes2_1[ind2[i][0], ind2[i][1]])
            # cb.set_label("0.15-quantile")
    elif i == 2:  # ra
        Zm2 = ma.masked_where(Zm > np.nanquantile(Values_input_summer, 0.9), Zm)
        Zm2 = ma.masked_where(Zm2 < np.nanquantile(Values_input_summer, 0.1), Zm2)
        cmin = np.nanmin(Zm2)
        cmax = np.nanmax(Zm2)
        if abs(cmax) > abs(cmin):
            cmin = - cmax
            colorhex = mp.hexbin(xx, yy, C=cc, gridsize=(134, 80), vmin=cmin, vmax=cmax, cmap='RdBu_r',
                                 extent=(-180, 180, -60, 90),
                                 norm=MidpointNormalize(midpoint=0, vmin=cmin, vmax=cmax))
            cb = mp.colorbar(colorhex, location='bottom', pad=0.4, ax=axes2_1[ind2[i][0], ind2[i][1]])
            # cb.set_label("0.9-quantile")
        else:
            cmax = - cmin
            colorhex = mp.hexbin(xx, yy, C=cc, gridsize=(134, 80), vmin=cmin, vmax=cmax, cmap='RdBu_r',
                                 extent=(-180, 180, -60, 90),
                                 norm=MidpointNormalize(midpoint=0, vmin=cmin, vmax=cmax))
            cb = mp.colorbar(colorhex, location='bottom', pad=0.4, ax=axes2_1[ind2[i][0], ind2[i][1]])
            # cb.set_label("0.1-quantile")
    elif i == 3:  # ra'
        Zm2 = ma.masked_where(Zm > np.nanquantile(Values_input_summer, 0.95), Zm)
        Zm2 = ma.masked_where(Zm2 < np.nanquantile(Values_input_summer, 0.05), Zm2)
        cmin = np.nanmin(Zm2)
        cmax = np.nanmax(Zm2)
        if abs(cmax) > abs(cmin):
            cmin = - cmax
            colorhex = mp.hexbin(xx, yy, C=cc, gridsize=(134, 80), vmin=cmin, vmax=cmax, cmap='RdBu_r',
                                 extent=(-180, 180, -60, 90),
                                 norm=MidpointNormalize(midpoint=0, vmin=cmin, vmax=cmax))
            cb = mp.colorbar(colorhex, location='bottom', pad=0.4, ax=axes2_1[ind2[i][0], ind2[i][1]])
            # cb.set_label("0.95-quantile")
        else:
            cmax = - cmin
            colorhex = mp.hexbin(xx, yy, C=cc, gridsize=(134, 80), vmin=cmin, vmax=cmax, cmap='RdBu_r',
                                 extent=(-180, 180, -60, 90),
                                 norm=MidpointNormalize(midpoint=0, vmin=cmin, vmax=cmax))
            cb = mp.colorbar(colorhex, location='bottom', pad=0.4, ax=axes2_1[ind2[i][0], ind2[i][1]])
            # cb.set_label("0.05-quantile")

    # colormesh = mp.pcolormesh(x, y, Zm, vmin=cmin, vmax=cmax, cmap='RdBu_r')

    mp.drawparallels(np.arange(-60, 90, 30), labels=[1, 0, 0, 0], linewidth=0)
    mp.drawmeridians(np.arange(-180, 180, 60), labels=[0, 0, 0, 1], linewidth=0)
    plt.text(-175, 10, 'JJA', fontname="Arial", fontsize=8, fontweight='bold', ha='left', va='center', color='k')
    plt.text(-175, -10, 'DJF', fontname="Arial", fontsize=8, fontweight='bold', ha='left', va='center', color='k')
plt.show()
fig2_1.savefig('figures/figure_daytime_summer_delta2_1.png', dpi=200)

# Plot2.2
fig2_2, axes2_2 = plt.subplots(2, 2)
fig2_2.set_size_inches(12.5, 7.5)
axes2_2[0, 0].set_title(r'$\Delta R^*_n \/ \mathrm{(W/m^2)}$', fontsize=font2)
axes2_2[0, 0].text(-182, 95, r'$\mathrm{a)}$', fontsize=font2)
axes2_2[0, 1].set_title(r'$\Delta H \/ \mathrm{(W/m^2)}$', fontsize=font2)
axes2_2[0, 1].text(-182, 95, r'$\mathrm{b)}$', fontsize=font2)
axes2_2[1, 0].set_title(r'$\Delta LE \/ \mathrm{(W/m^2)}$', fontsize=font2)
axes2_2[1, 0].text(-182, 95, r'$\mathrm{c)}$', fontsize=font2)
axes2_2[1, 1].set_title(r'$\Delta G \/ \mathrm{(W/m^2)}$', fontsize=font2)
axes2_2[1, 1].text(-182, 95, r'$\mathrm{d)}$', fontsize=font2)

for ix in range(2):
    for iy in range(2):
        axes2_2[ix, iy].xaxis.set_ticks(np.arange(-120, 180, 60))
        axes2_2[ix, iy].yaxis.set_ticks(np.arange(-30, 90, 30))
        axes2_2[ix, iy].tick_params(direction='in', length=6, width=2)
        axes2_2[ix, iy].xaxis.set_ticklabels([])
        axes2_2[ix, iy].yaxis.set_ticklabels([])

for i in range(len(diff_name1)):
    input_value = eval(diff_name2[i])
    Values_input_summer = multiannual_average(input_value, mask_urban, Average_period_value, yr_Num, y.T)

    mp = Basemap(llcrnrlon=-180., llcrnrlat=-60., urcrnrlon=180., urcrnrlat=90.,
                 projection='cyl', ax=axes2_2[ind2[i][0], ind2[i][1]], suppress_ticks=False)

    mp.drawmapboundary(fill_color='w')
    mp.drawcoastlines()
    Z = Values_input_summer.T
    Zm = ma.masked_where(np.isnan(Z), Z)
    cc = np.ndarray.flatten(Values_input_summer)
    Zm2 = ma.masked_where(Zm > np.nanquantile(Values_input_summer, 0.99), Zm)
    Zm2 = ma.masked_where(Zm2 < np.nanquantile(Values_input_summer, 0.01), Zm2)
    cmin = np.nanmin(Zm2)
    cmax = np.nanmax(Zm2)
    if abs(cmax) > abs(cmin):
        cmin = - cmax
        colorhex = mp.hexbin(xx, yy, C=cc, gridsize=(134, 80), vmin=cmin, vmax=cmax, cmap='RdBu_r',
                             extent=(-180, 180, -60, 90),
                             norm=MidpointNormalize(midpoint=0, vmin=cmin, vmax=cmax))
        cb = mp.colorbar(colorhex, location='bottom', pad=0.4, ax=axes2_2[ind2[i][0], ind2[i][1]])
        # cb.set_label("0.99-quantile")
    else:
        cmax = - cmin
        colorhex = mp.hexbin(xx, yy, C=cc, gridsize=(134, 80), vmin=cmin, vmax=cmax, cmap='RdBu_r',
                             extent=(-180, 180, -60, 90),
                             norm=MidpointNormalize(midpoint=0, vmin=cmin, vmax=cmax))
        cb = mp.colorbar(colorhex, location='bottom', pad=0.4, ax=axes2_2[ind2[i][0], ind2[i][1]])
        # cb.set_label("0.01-quantile")

    mp.drawparallels(np.arange(-60, 90, 30), labels=[1, 0, 0, 0], linewidth=0)
    mp.drawmeridians(np.arange(-180, 180, 60), labels=[0, 0, 0, 1], linewidth=0)
    plt.text(-175, 10, 'JJA', fontname="Arial", fontsize=8, fontweight='bold', ha='left', va='center', color='k')
    plt.text(-175, -10, 'DJF', fontname="Arial", fontsize=8, fontweight='bold', ha='left', va='center', color='k')

plt.show()
fig2_2.savefig('figures/figure_daytime_summer_delta2_2.png', dpi=200)

# Plot3 (sensitivities of canopy air temperature)
name3 = ['dTs_dalpha', 'dTs_drs', 'dTs_dra', 'dTs_dGrnd']
ind = [[0, 0], [0, 1], [1, 0], [1, 1]]
font3 = 14
fig3, axes3 = plt.subplots(2, 2)
fig3.set_size_inches(12.5, 7.5)
axes3[0, 0].set_title(r'${\partial T_{ca}}/{\partial \alpha} \/ \mathrm{(^oC)}$', fontsize=font3)
axes3[0, 0].text(-182, 95, r'$\mathrm{a)}$', fontsize=font3)
axes3[0, 1].set_title(r'${\partial T_{ca}}/{\partial r_{s}} \/ \mathrm{(^oC \/ s \/ m^{-1})}$', fontsize=font3)
axes3[0, 1].text(-182, 95, r'$\mathrm{b)}$', fontsize=font3)
axes3[1, 0].set_title(r'${\partial T_{ca}}/{\partial r_{a}} \/ \mathrm{(^oC \/ s \/ m^{-1})}$', fontsize=font3)
axes3[1, 0].text(-182, 95, r'$\mathrm{c)}$', fontsize=font3)
axes3[1, 1].set_title(r'${\partial T_{ca}}/{\partial G} \/ \mathrm{(^oC \/ m^{2} \/ W^{-1})}$', fontsize=font3)
axes3[1, 1].text(-182, 95, r'$\mathrm{d)}$', fontsize=font3)

for ix in range(2):
    for iy in range(2):
        axes3[ix, iy].xaxis.set_ticks(np.arange(-120, 180, 60))
        axes3[ix, iy].yaxis.set_ticks(np.arange(-30, 90, 30))
        axes3[ix, iy].tick_params(direction='in', length=6, width=2)
        axes3[ix, iy].xaxis.set_ticklabels([])
        axes3[ix, iy].yaxis.set_ticklabels([])

for i in range(len(name3)):
    input_value = eval(name3[i])
    Values_input_summer = multiannual_average(input_value, mask_urban, Average_period_value, yr_Num, y.T)

    mp = Basemap(llcrnrlon=-180., llcrnrlat=-60., urcrnrlon=180., urcrnrlat=90.,
                 projection='cyl', ax=axes3[ind[i][0], ind[i][1]], suppress_ticks=False)
    mp.drawmapboundary(fill_color='w')
    mp.drawcoastlines()
    uq = 0.99
    lq = 0.01
    Z = Values_input_summer.T
    Zm = ma.masked_where(np.isnan(Z), Z)
    Zm2 = ma.masked_where(Zm > np.nanquantile(Values_input_summer, uq), Zm)
    Zm2 = ma.masked_where(Zm2 < np.nanquantile(Values_input_summer, lq), Zm2)

    cmin = np.nanmin(Zm2)
    cmax = np.nanmax(Zm2)
    cc = np.ndarray.flatten(Values_input_summer)
    if abs(cmax) > abs(cmin):
        cmin = - cmax
        # print('cmax {} >cmin {}' .format(i,i))
        colorhex = mp.hexbin(xx, yy, C=cc, vmin=cmin, vmax=cmax, gridsize=(134, 80), cmap='RdBu_r',
                             extent=(-180, 180, -60, 90))

        cb = mp.colorbar(colorhex, location='bottom', pad=0.4, ax=axes3[ind[i][0], ind[i][1]])
        # cb.set_label("{}-quantile".format(uq))
    else:
        cmax = - cmin
        # print("cmin {} >cmax {}" .format(i,i))
        colorhex = mp.hexbin(xx, yy, C=cc, vmin=cmin, vmax=cmax, gridsize=(134, 80), cmap='RdBu_r',
                             extent=(-180, 180, -60, 90))

        cb = mp.colorbar(colorhex, location='bottom', pad=0.4, ax=axes3[ind[i][0], ind[i][1]])
        # cb.set_label("{}-quantile" .format(lq))

    mp.drawparallels(np.arange(-60, 90, 30), labels=[1, 0, 0, 0], linewidth=0, ax=axes3[ind[i][0], ind[i][1]])
    mp.drawmeridians(np.arange(-180, 180, 60), labels=[0, 0, 0, 1], linewidth=0, ax=axes3[ind[i][0], ind[i][1]])
    plt.text(-175, 10, 'JJA', fontname="Arial", fontsize=10, fontweight='bold', ha='left', va='center', color='k')
    plt.text(-175, -10, 'DJF', fontname="Arial", fontsize=10, fontweight='bold', ha='left', va='center', color='k')
plt.show()
fig3.savefig('figures/figure_daytime_summer_Ts_partial.png', dpi=200)

# Plot4 (sensitivities of reference layer air temperature)
name4 = ['dT2_dalpha', 'dT2_drs', 'dT2_dra', 'dT2_dra_prime', 'dT2_dGrnd']
ind = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]]
font4 = 14
title = [[r'${\partial T_{ref}}/{\partial \alpha} \/ \mathrm{(^oC)}$',
         r'${\partial T_{ref}}/{\partial r_{s}} \/ \mathrm{(^oC \/ s \/ m^{-1})}$'],
         [r'${\partial T_{ref}}/{\partial r_{a}} \/ \mathrm{(^oC \/ s \/ m^{-1})}$',
         r'${\partial T_{ref}}/{\partial r^\prime_{a}} \/ \mathrm{(^oC \/ s \/ m^{-1})}$'],
         [r'${\partial T_{ref}}/{\partial G} \/ \mathrm{(^oC \/ m^{2} \/ W^{-1})}$']]
text = [[r'$\mathrm{a)}$', r'$\mathrm{b)}$'], [r'$\mathrm{c)}$', r'$\mathrm{d)}$'], [r'$\mathrm{e)}$']]

fig4, axes4 = plt.subplots(3, 2)
axes4[2, 1].axis('off')
fig4.set_size_inches(14, 12)
shouldbreak = False
for i in range(3):
    for j in range(2):
        if i == 2 and j == 1:
            shouldbreak = True
            break
        else:
            axes4[i, j].set_title(title[i][j], fontsize=font4)
            axes4[i, j].text(-182, 95, text[i][j], fontsize=font4)
    if shouldbreak:
        break

shouldbreak = False
for ix in range(3):
    for iy in range(2):
        if ix == 2 and iy == 1:
            shouldbreak = True
            break
        else:
            axes4[ix, iy].xaxis.set_ticks(np.arange(-120, 180, 60))
            axes4[ix, iy].yaxis.set_ticks(np.arange(-30, 90, 30))
            axes4[ix, iy].tick_params(direction='in', length=6, width=2)
            axes4[ix, iy].xaxis.set_ticklabels([])
            axes4[ix, iy].yaxis.set_ticklabels([])
    if shouldbreak:
        break
# plt.show()
# making maps
for i in range(len(name4)):
    input_value = eval(name4[i])
    Values_input_summer = multiannual_average(input_value, mask_urban, Average_period_value, yr_Num, y.T)
    mp = Basemap(llcrnrlon=-180., llcrnrlat=-60., urcrnrlon=180., urcrnrlat=90.,
                 projection='cyl', ax=axes4[ind[i][0], ind[i][1]], suppress_ticks=False)
    mp.drawmapboundary(fill_color='w')
    mp.drawcoastlines()
    uq = 0.99
    lq = 0.01
    Z = Values_input_summer.T
    Zm = ma.masked_where(np.isnan(Z), Z)
    Zm2 = ma.masked_where(Zm > np.nanquantile(Values_input_summer, uq), Zm)
    Zm2 = ma.masked_where(Zm2 < np.nanquantile(Values_input_summer, lq), Zm2)

    cmin = np.nanmin(Zm2)
    cmax = np.nanmax(Zm2)
    cc = np.ndarray.flatten(Values_input_summer)
    if abs(cmax) > abs(cmin):
        cmin = - cmax
        # print('cmax {} >cmin {}' .format(i,i))
        colorhex = mp.hexbin(xx, yy, C=cc, vmin=cmin, vmax=cmax, gridsize=(134, 80), cmap='RdBu_r',
                             extent=(-180, 180, -60, 90))

        cb = mp.colorbar(colorhex, location='bottom', pad=0.4, ax=axes4[ind[i][0], ind[i][1]])
        # cb.set_label("{}-quantile" .format(uq))
    else:
        cmax = - cmin
        # print("cmin {} >cmax {}" .format(i,i))
        colorhex = mp.hexbin(xx, yy, C=cc, vmin=cmin, vmax=cmax, gridsize=(134, 80), cmap='RdBu_r',
                             extent=(-180, 180, -60, 90))

        cb = mp.colorbar(colorhex, location='bottom', pad=0.4, ax=axes4[ind[i][0], ind[i][1]])
        # cb.set_label("{}-quantile" .format(lq))

    mp.drawparallels(np.arange(-60, 90, 30), labels=[1, 0, 0, 0], linewidth=0, ax=axes4[ind[i][0], ind[i][1]])
    mp.drawmeridians(np.arange(-180, 180, 60), labels=[0, 0, 0, 1], linewidth=0, ax=axes4[ind[i][0], ind[i][1]])
    plt.text(-175, 10, 'JJA', fontname="Arial", fontsize=10, fontweight='bold', ha='left', va='center', color='k')
    plt.text(-175, -10, 'DJF', fontname="Arial", fontsize=10, fontweight='bold', ha='left', va='center', color='k')
plt.show()
fig4.savefig('figures/figure_daytime_summer_T2_partial.png', dpi=200)

# Plot5 (sensitivities of SWGT)
name5 = ['dWGT_dalpha', 'dWGT_drs', 'dWGT_dra', 'dWGT_dGrnd']
ind = [[0, 0], [0, 1], [1, 0], [1, 1]]
font5 = 14

fig5, axes5 = plt.subplots(2, 2)
fig5.set_size_inches(12.5, 7.5)
axes5[0, 0].set_title(r'${\partial SWBGT}/{\partial \alpha} \/ \mathrm{(^oC)}$', fontsize=font5)
axes5[0, 0].text(-182, 95, r'$\mathrm{a)}$', fontsize=font5)
axes5[0, 1].set_title(r'${\partial SWBGT}/{\partial r_{s}} \/ \mathrm{(^oC \/ s \/ m^{-1})}$', fontsize=font5)
axes5[0, 1].text(-182, 95, r'$\mathrm{b)}$', fontsize=font5)
axes5[1, 0].set_title(r'${\partial SWBGT}/{\partial r_{a}} \/ \mathrm{(^oC \/ s \/ m^{-1})}$', fontsize=font5)
axes5[1, 0].text(-182, 95, r'$\mathrm{c)}$', fontsize=font5)
axes5[1, 1].set_title(r'${\partial SWBGT}/{\partial G} \/ \mathrm{(^oC \/ m^{2} \/ W^{-1})}$', fontsize=font5)
axes5[1, 1].text(-182, 95, r'$\mathrm{d)}$', fontsize=font5)
for ix in range(2):
    for iy in range(2):
        axes5[ix, iy].xaxis.set_ticks(np.arange(-120, 180, 60))
        axes5[ix, iy].yaxis.set_ticks(np.arange(-30, 90, 30))
        axes5[ix, iy].tick_params(direction='in', length=6, width=2)
        axes5[ix, iy].xaxis.set_ticklabels([])
        axes5[ix, iy].yaxis.set_ticklabels([])
for i in range(len(name5)):
    input_value = eval(name5[i])
    Values_input_summer = multiannual_average(input_value, mask_urban, Average_period_value, yr_Num, y.T)

    mp = Basemap(llcrnrlon=-180., llcrnrlat=-60., urcrnrlon=180., urcrnrlat=90.,
                 projection='cyl', ax=axes5[ind[i][0], ind[i][1]], suppress_ticks=False)
    mp.drawmapboundary(fill_color='w')
    mp.drawcoastlines()
    uq = 0.99
    lq = 0.01
    Z = Values_input_summer.T
    Zm = ma.masked_where(np.isnan(Z), Z)
    Zm2 = ma.masked_where(Zm > np.nanquantile(Values_input_summer, uq), Zm)
    Zm2 = ma.masked_where(Zm2 < np.nanquantile(Values_input_summer, lq), Zm2)

    cmin = np.nanmin(Zm2)
    cmax = np.nanmax(Zm2)
    cc = np.ndarray.flatten(Values_input_summer)
    if abs(cmax) > abs(cmin):
        cmin = - cmax
        # print('cmax {} >cmin {}' .format(i,i))
        colorhex = mp.hexbin(xx, yy, C=cc, vmin=cmin, vmax=cmax, gridsize=(134, 80), cmap='RdBu_r',
                             extent=(-180, 180, -60, 90))

        cb = mp.colorbar(colorhex, location='bottom', pad=0.4, ax=axes5[ind[i][0], ind[i][1]])
        # cb.set_label("{}-quantile" .format(uq))
    else:
        cmax = - cmin
        # print("cmin {} >cmax {}" .format(i,i))
        colorhex = mp.hexbin(xx, yy, C=cc, vmin=cmin, vmax=cmax, gridsize=(134, 80), cmap='RdBu_r',
                             extent=(-180, 180, -60, 90))

        cb = mp.colorbar(colorhex, location='bottom', pad=0.4, ax=axes5[ind[i][0], ind[i][1]])
        # cb.set_label("{}-quantile" .format(lq))

    mp.drawparallels(np.arange(-60, 90, 30), labels=[1, 0, 0, 0], linewidth=0, ax=axes5[ind[i][0], ind[i][1]])
    mp.drawmeridians(np.arange(-180, 180, 60), labels=[0, 0, 0, 1], linewidth=0, ax=axes5[ind[i][0], ind[i][1]])
    plt.text(-175, 10, 'JJA', fontname="Arial", fontsize=10, fontweight='bold', ha='left', va='center', color='k')
    plt.text(-175, -10, 'DJF', fontname="Arial", fontsize=10, fontweight='bold', ha='left', va='center', color='k')
plt.show()
fig5.savefig('figures/figure_daytime_summer_SWGT_partial.png', dpi=200)

# Plot6 (sensitivities of SWGT2)
name6 = ['dWGT2_dalpha', 'dWGT2_drs', 'dWGT2_dra', 'dWGT2_dra_prime', 'dWGT2_dGrnd']
ind = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]]
font6 = 14
title = [[r'${\partial SWBGT_{ref}}/{\partial \alpha} \/ \mathrm{(^oC)}$',
         r'${\partial SWBGT_{ref}}/{\partial r_{s}} \/ \mathrm{(^oC \/ s \/ m^{-1})}$'],
         [r'${\partial SWBGT_{ref}}/{\partial r_{a}} \/ \mathrm{(^oC \/ s \/ m^{-1})}$',
         r'${\partial SWBGT_{ref}}/{\partial r^\prime_{a}} \/ \mathrm{(^oC \/ s \/ m^{-1})}$'],
         [r'${\partial SWBGT_{ref}}/{\partial G} \/ \mathrm{(^oC \/ m^{2} \/ W^{-1})}$']]
text = [[r'$\mathrm{a)}$', r'$\mathrm{b)}$'], [r'$\mathrm{c)}$', r'$\mathrm{d)}$'], [r'$\mathrm{e)}$']]

fig6, axes6 = plt.subplots(3, 2)
axes6[2, 1].axis('off')
fig6.set_size_inches(14, 12)
shouldbreak = False
for i in range(3):
    for j in range(2):
        if i == 2 and j == 1:
            shouldbreak = True
            break
        else:
            axes6[i, j].set_title(title[i][j], fontsize=font6)
            axes6[i, j].text(-182, 95, text[i][j], fontsize=font6)
    if shouldbreak:
        break

shouldbreak = False
for ix in range(3):
    for iy in range(2):
        if ix == 2 and iy == 1:
            shouldbreak = True
            break
        else:
            axes6[ix, iy].xaxis.set_ticks(np.arange(-120, 180, 60))
            axes6[ix, iy].yaxis.set_ticks(np.arange(-30, 90, 30))
            axes6[ix, iy].tick_params(direction='in', length=6, width=2)
            axes6[ix, iy].xaxis.set_ticklabels([])
            axes6[ix, iy].yaxis.set_ticklabels([])
    if shouldbreak:
        break

for i in range(len(name6)):
    input_value = eval(name6[i])
    Values_input_summer = multiannual_average(input_value, mask_urban, Average_period_value, yr_Num, y.T)

    mp = Basemap(llcrnrlon=-180., llcrnrlat=-60., urcrnrlon=180., urcrnrlat=90.,
                 projection='cyl', ax=axes6[ind[i][0], ind[i][1]], suppress_ticks=False)
    mp.drawmapboundary(fill_color='w')
    mp.drawcoastlines()
    uq = 0.99
    lq = 0.01
    Z = Values_input_summer.T
    Zm = ma.masked_where(np.isnan(Z), Z)
    Zm2 = ma.masked_where(Zm > np.nanquantile(Values_input_summer, uq), Zm)
    Zm2 = ma.masked_where(Zm2 < np.nanquantile(Values_input_summer, lq), Zm2)

    cmin = np.nanmin(Zm2)
    cmax = np.nanmax(Zm2)
    cc = np.ndarray.flatten(Values_input_summer)
    if abs(cmax) > abs(cmin):
        cmin = - cmax
        # print('cmax {} >cmin {}' .format(i,i))
        colorhex = mp.hexbin(xx, yy, C=cc, vmin=cmin, vmax=cmax, gridsize=(134, 80), cmap='RdBu_r',
                             extent=(-180, 180, -60, 90))

        cb = mp.colorbar(colorhex, location='bottom', pad=0.4, ax=axes6[ind[i][0], ind[i][1]])
        # cb.set_label("{}-quantile" .format(uq))
    else:
        cmax = - cmin
        # print("cmin {} >cmax {}" .format(i,i))
        colorhex = mp.hexbin(xx, yy, C=cc, vmin=cmin, vmax=cmax, gridsize=(134, 80), cmap='RdBu_r',
                             extent=(-180, 180, -60, 90))

        cb = mp.colorbar(colorhex, location='bottom', pad=0.4, ax=axes6[ind[i][0], ind[i][1]])
        # cb.set_label("{}-quantile" .format(lq))

    mp.drawparallels(np.arange(-60, 90, 30), labels=[1, 0, 0, 0], linewidth=0, ax=axes6[ind[i][0], ind[i][1]])
    mp.drawmeridians(np.arange(-180, 180, 60), labels=[0, 0, 0, 1], linewidth=0, ax=axes6[ind[i][0], ind[i][1]])
    plt.text(-175, 10, 'JJA', fontname="Arial", fontsize=10, fontweight='bold', ha='left', va='center', color='k')
    plt.text(-175, -10, 'DJF', fontname="Arial", fontsize=10, fontweight='bold', ha='left', va='center', color='k')
plt.show()
fig6.savefig('figures/figure_daytime_summer_SWGT2_partial.png', dpi=200)

# plot temperature difference




####
# fig = plt.figure(figsize=(9, 3), dpi=80)
# # Divide figure to grid.
# # gs1 is a grid to draw u, v with one colorbar.
# # gs2 is another grid to draw w with another colorbar.
# gs1 = GridSpec(nrows=2, ncols=1, left=0.06, right=0.46, height_ratios=[0.2, 0.02])
# gs2 = GridSpec(nrows=2, ncols=1, left=0.55, right=0.95, height_ratios=[0.2, 0.02])
#
# # Assign axes from gs1 and gs2.
# ax11 = fig.add_subplot(gs1[0, 0])
# ax12 = fig.add_subplot(gs1[1, 0], sharex=ax11)
# ax21 = fig.add_subplot(gs2[0, 0])
# ax22 = fig.add_subplot(gs2[1, 0], sharex=ax21)
#
# # Adjust grid
# gs1.update(wspace=0.15, hspace=0.15)  # set the spacing between axes.
# gs2.update(wspace=0.15, hspace=0.15)  # set the spacing between axes.
# # ax21.set_ylabel("y")
# # ax22.set_xlabel("x")
# # ax11.set_ylabel("y")
# # ax12.set_xlabel("x")
# # ax11.yaxis.set_label_coords(-0.17, 0.4)
# plt.setp(ax11.get_xticklabels(), visible=False)
# plt.setp(ax21.get_xticklabels(), visible=False)
#
# # Plot map
# mp1 = Basemap(llcrnrlon=-180., llcrnrlat=-60., urcrnrlon=180., urcrnrlat=90.,
#               projection='cyl', ax=ax11)
#
# mp1.drawmapboundary(fill_color='w')
# mp1.drawcoastlines()
#
# cc = np.ndarray.flatten(Values_input_summer)
# cmin = np.nanmin(Values_input_summer)
# cmax = np.nanmax(Values_input_summer)
# colormesh = mp1.pcolormesh(lons, lats, Values_input_summer, vmin=cmin, vmax=cmax, cmap='RdBu_r')
# # mp1.hexbin(xx, yy, C=cc, gridsize=(134, 80), cmap='RdBu_r', extent=(-180, 180, -60, 90))
# mp1.drawparallels(np.arange(-60, 90, 30), labels=[1, 1, 0, 0])
# mp1.drawmeridians(np.arange(-180, 180, 60), labels=[0, 0, 0, 1])
# cb1 = mp1.colorbar(colormesh, location='bottom', pad=0.3, ax=ax11)
# # cb1 = plt.colorbar(mp1, cax=ax12)
#
# plt.show()
# set the colormap and centre the colorbar
