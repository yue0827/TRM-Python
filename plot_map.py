from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors


def world_map(values_input_summer, lons, lats, ind):
    if __name__ == '__main__':
        """Plot temperature difference"""
        mp = Basemap(llcrnrlon=-180., llcrnrlat=-60., urcrnrlon=180., urcrnrlat=90., projection='cyl')

        # map.fillcontinents(color='coral',lake_color='w')
        mp.drawcoastlines()

        # x, y = map(lons[0:50,50], lats[0:50,50])
        # c = Values_input_annual[0:50,50]
        # map.scatter(x, y, marker='s', color='b')
        x, y = lons, lats
        xx = np.ndarray.flatten(x)
        yy = np.ndarray.flatten(y)
        c = values_input_summer
        cc = np.ndarray.flatten(c)
        mp.drawmapboundary(fill_color='w')
        # map.fillcontinents(color='#cc9966',lake_color='#99ffff')
        mp.drawparallels(np.arange(-60,90,30),labels=[1,1,0,0])
        mp.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1])

        # map.hexbin(xx, yy, C=cc,  cmap='YlOrBr')
        # map.hexbin(xx, yy, C=cc,  cmap=mpl.cm.cool)
        mp.hexbin(xx, yy, C=cc, gridsize=(134, 80), cmap='RdBu_r', extent=(-180, 180, -60, 90))
        mp.colorbar(location='bottom', pad=0.3)
        if ind == 0:
            mp.title = plt.title(r'$\Delta T_{ca}(^oC)$', fontsize=20)
            fig = plt.gcf()
            plt.show()
            fig.savefig('figures/figure_nighttime_summer_Tca.png')

        elif ind == 1:
            mp.title = plt.title(r'$\Delta T_{ref}(^oC)$', fontsize=20)
            fig = plt.gcf()
            plt.show()
            fig.savefig('figures/figure_nighttime_summer_Tref.png')

        elif ind == 2:
            mp.title = plt.title(r'$\Delta SWGBT$', fontsize=20)
            fig = plt.gcf()
            plt.show()
            fig.savefig('figures/figure_nighttime_summer_SWGBT.png')

        else:
            mp.title = plt.title(r'$\Delta SWGBT_{ref}$', fontsize=20)
            fig = plt.gcf()
            plt.show()
            fig.savefig('figures/figure_nighttime_summer_SWGBT2.png')
    return fig


# def fun(values_input, lons2, lats2, ind2):
#     """Urban rural difference"""
#     m = Basemap(llcrnrlon=-180., llcrnrlat=-60., urcrnrlon=180., urcrnrlat=90., projection='cyl')
#
#     # map.fillcontinents(color='coral',lake_color='w')
#     m.drawcoastlines()
#
#     # x, y = map(lons[0:50,50], lats[0:50,50])
#     # c = Values_input_annual[0:50,50]
#     # map.scatter(x, y, marker='s', color='b')
#     x, y = m(lons2, lats2)
#     xx = np.ndarray.flatten(x)
#     yy = np.ndarray.flatten(y)
#     c = values_input
#     cc = np.ndarray.flatten(c)
#     m.drawmapboundary(fill_color='w')
#     # map.fillcontinents(color='#cc9966',lake_color='#99ffff')
#     m.drawparallels(np.arange(-60, 90, 30), labels=[1, 1, 0, 0])
#     m.drawmeridians(np.arange(-180, 180, 60), labels=[0, 0, 0, 1])
#
#     # map.hexbin(xx, yy, C=cc,  cmap='YlOrBr')
#     # map.hexbin(xx, yy, C=cc,  cmap=mpl.cm.cool)
#     m.hexbin(xx, yy, C=cc, gridsize=(134, 80), cmap='RdBu_r', extent=(-180, 180, -60, 90))
#     m.colorbar(location='bottom', pad=0.3)
#     if ind2 == 0:
#         map.title = plt.title(r'$\Delta \alpha$', fontsize=20)
#         fig = plt.gcf()
#         # plt.show()
#         # fig.savefig('figures/figure_daytime_summer_Tca.png')
#
#     elif ind2 == 1:
#         map.title = plt.title(r'$\Delta r_{s}(s/m)$', fontsize=20)
#         fig = plt.gcf()
#         # plt.show()
#         # fig.savefig('figures/figure_daytime_summer_Tref.png')
#
#     elif ind2 == 2:
#         map.title = plt.title(r'$\Delta r_{a}(s/m)$', fontsize=20)
#         fig = plt.gcf()
#         # plt.show()
#         # fig.savefig('figures/figure_daytime_summer_SWGBT.png')
#
#     elif ind2 == 3:
#         map.title = plt.title(r'$\Delta r_{a}\'(s/m)$', fontsize=20)
#         fig = plt.gcf()
#
#     elif ind2 == 4:
#         map.title = plt.title(r'$\Delta G\'(W/m^2)$', fontsize=20)
#         fig = plt.gcf()
#
#     elif ind2 == 5:
#         map.title = plt.title(r'$\Delta R_n^*(W/m^2)$', fontsize=20)
#         fig = plt.gcf()
#
#     elif ind2 == 6:
#         map.title = plt.title(r'$\Delta H(W/m^2)$', fontsize=20)
#         fig = plt.gcf()
#
#     else:
#         map.title = plt.title(r'$\Delta l_e(W/m^2)$', fontsize=20)
#         fig = plt.gcf()
#
#     return fig



# if __name__ == '__main__':
#     world_map()
#     urbn_rural_diff()
