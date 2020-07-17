from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def fun(values_input, lons2, lats2, ind2):
    """Urban rural difference"""
    m = Basemap(llcrnrlon=-180., llcrnrlat=-60., urcrnrlon=180., urcrnrlat=90., projection='cyl')

    # map.fillcontinents(color='coral',lake_color='w')
    m.drawcoastlines()

    # x, y = map(lons[0:50,50], lats[0:50,50])
    # c = Values_input_annual[0:50,50]
    # map.scatter(x, y, marker='s', color='b')
    x, y = lons2, lats2
    xx = np.ndarray.flatten(x)
    yy = np.ndarray.flatten(y)
    c = values_input
    cc = np.ndarray.flatten(c)
    m.drawmapboundary(fill_color='w')
    # map.fillcontinents(color='#cc9966',lake_color='#99ffff')
    m.drawparallels(np.arange(-60, 90, 30), labels=[1, 1, 0, 0])
    m.drawmeridians(np.arange(-180, 180, 60), labels=[0, 0, 0, 1])

    # map.hexbin(xx, yy, C=cc,  cmap='YlOrBr')
    # map.hexbin(xx, yy, C=cc,  cmap=mpl.cm.cool)
    m.hexbin(xx, yy, C=cc, gridsize=(134, 80), cmap='RdBu_r', extent=(-180, 180, -60, 90))
    m.colorbar(location='bottom', pad=0.3)
    if ind2 == 0:
        m.title = plt.title(r'$\Delta \alpha$', fontsize=20)
        fig = plt.gcf()
        plt.show()
        fig.savefig('figures/figure_nighttime_summer_alpha.png')

    elif ind2 == 1:
        m.title = plt.title(r'$\Delta r_{s}(s/m)$', fontsize=20)
        fig = plt.gcf()
        plt.show()
        fig.savefig('figures/figure_nighttime_summer_rs.png')

    elif ind2 == 2:
        m.title = plt.title(r'$\Delta r_{a}(s/m)$', fontsize=20)
        fig = plt.gcf()
        plt.show()
        fig.savefig('figures/figure_nighttime_summer_ra.png')

    elif ind2 == 3:
        m.title = plt.title(r'$\Delta r_{a}^\'(s/m)$', fontsize=20)
        fig = plt.gcf()
        plt.show()
        fig.savefig('figures/figure_nighttime_summer_ra_prime.png')

    elif ind2 == 4:
        m.title = plt.title(r'$\Delta G(W/m^2)$', fontsize=20)
        fig = plt.gcf()
        plt.show()
        fig.savefig('figures/figure_nighttime_summer_G.png')

    elif ind2 == 5:
        m.title = plt.title(r'$\Delta R_n^*(W/m^2)$', fontsize=20)
        fig = plt.gcf()
        plt.show()
        fig.savefig('figures/figure_nighttime_summer_Rn_str.png')

    elif ind2 == 6:
        m.title = plt.title(r'$\Delta H(W/m^2)$', fontsize=20)
        fig = plt.gcf()
        plt.show()
        fig.savefig('figures/figure_nighttime_summer_Qh.png')

    else:
        m.title = plt.title(r'$\Delta L_e(W/m^2)$', fontsize=20)
        fig = plt.gcf()
        plt.show()
        fig.savefig('figures/figure_nighttime_summer_Qle.png')

    return


# im1 = Image.open(r'figures/figure_nighttime_summer_alpha.png')
# rgb_im1 = im1.convert('RGB')
# rgb_im1.save(r'figures/figure_nighttime_summer_alpha.jpg')

fig1, axes1 = plt.subplots(1, 2)
axes1[0].set_title(r'${\partial T_{ca}}/{\partial \alpha}$')
axes1[1].set_title(r'${\partial T_{ca}}/{\partial r_{s}}$')

mp = Basemap(llcrnrlon=-180., llcrnrlat=-60., urcrnrlon=180., urcrnrlat=90.,
             projection='cyl',ax=axes1[0])
# mp.drawmapboundary(fill_color='w')
# mp.drawcoastlines()
# colorhex = mp.hexbin(xx, yy, C=cc, gridsize=(134, 80), vmin=cmin, vmax=cmax, cmap='RdBu_r',
#                      extent=(-180, 180, -60, 90))
# cb = mp.colorbar(colorhex, location='bottom', pad=0.4)
# mp.drawparallels(np.arange(-60, 90, 30), labels=[1, 1, 0, 0], linewidth=0)
# mp.drawmeridians(np.arange(-180, 180, 60), labels=[0, 0, 0, 1], linewidth=0)
# ax.set_xticks([0,1,2])
# plt.show()

mp1 = Basemap(llcrnrlon=-180., llcrnrlat=-60., urcrnrlon=180., urcrnrlat=90.,
             projection='cyl', ax=axes1[0])
plt.show()


