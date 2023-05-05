##WEATHER ROUTING

#https://fastseas.com

import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib import dates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
import maps as mp
import matplotlib.dates as mdates
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
import xarray as xr
import datetime as dt
import matplotlib.ticker as mticker
import re
from netCDF4 import Dataset
import matplotlib
def cor_uv_to_dir(U,V):
    """
    Calculates the wind direction from the u and v component of wind.
    Takes into account the wind direction coordinates is different than the 
    trig unit circle coordinate. If the wind directin is 360 then returns zero
    (by %360)
    Inputs:
      U = west/east direction (wind from the west is positive, from the east is negative)
      V = south/noth direction (wind from the south is positive, from the north is negative)
    """
    WDIR= (270-np.rad2deg(np.arctan2(V,U)))%360
    if ((WDIR>=0) &  (WDIR<180)):
      WDIR=WDIR+180
    else:
      WDIR=WDIR-180
    return WDIR

 
def angle_correction(deltaangle, dirship_initial, cor_fac):
  if (deltaangle < -100):
    deltaangle=deltaangle+360
  if (deltaangle > 100):
    deltaangle=deltaangle-360
  if (deltaangle > 0):
    a=dirship_initial + deltaangle*cor_fac
    if a >= 360:
      return a - 360
    else:
      return a
  elif (deltaangle < 0):
    a=dirship_initial + deltaangle*cor_fac
    if a < 0:
      return a + 360
    else:
      return a
  else:
    return dirship_initial

def vel_conv(vel,dir):
  if dir <= 90:
    u = vel*np.sin(np.radians(dir))
    v = vel*np.cos(np.radians(dir))
  if dir > 90 and dir <=180:
    dir=dir-90
    u = vel*np.cos(np.radians(dir))
    v = -vel*np.sin(np.radians(dir))
  if dir > 180 and dir <=270:
    dir=dir-180
    u = -vel*np.sin(np.radians(dir))
    v = -vel*np.cos(np.radians(dir))
  if dir > 270 and dir <=360:
    dir=dir-270
    u = -vel*np.cos(np.radians(dir))
    v = vel*np.sin(np.radians(dir))
  return(u,v)  

dirship=140
dirship_initial=dirship
ub,vb=vel_conv(1,dirship)



woplim=20   #wind operational lim
coplim=2    #current operational lim
wavelim=3    #wave operational lim

###############adcp
########WIND


sp=1


today = dt.datetime.today()
year = today.strftime("%Y")
month = today.strftime("%m")
day = today.strftime("%d")

tomorrow = today + dt.timedelta(days=1)
tomorrow = today - dt.timedelta(days=1)
tomorrow = today 
year2 = tomorrow.strftime("%Y")
month2 = tomorrow.strftime("%m")
day2 = tomorrow.strftime("%d")

tomorrow = today + dt.timedelta(days=4)
year3 = tomorrow.strftime("%Y")
month3 = tomorrow.strftime("%m")
day3 = tomorrow.strftime("%d")


pathgfsu = f'/mnt/c/Users/fernando.barreto/Desktop/boletins/wind_gfs/GFS_F_UWIND_{year}{month}{day}.nc'

pathgfsv = f'/mnt/c/Users/fernando.barreto/Desktop/boletins/wind_gfs/GFS_F_VWIND_{year}{month}{day}.nc'

pathgfsu = f'/mnt/c/Users/fernando.barreto/Desktop/boletins/wind_gfs/download_gfs_python/GFS_F_UWIND_20220418.nc'

pathgfsv = f'/mnt/c/Users/fernando.barreto/Desktop/boletins/wind_gfs/download_gfs_python/GFS_F_VWIND_20220418.nc'


res=0.08
latmin=-45
latmax=-10
lonmin=-55
lonmax=-20
lat=np.arange(latmin ,latmax , res)
lon=np.arange(lonmin ,lonmax,  res)


lon2, lat2 = np.meshgrid(lon, lat)



dsu = xr.open_dataset(pathgfsu)
dsv = xr.open_dataset(pathgfsv)


dtt=1

times=np.arange(dt.datetime(int(year2), int(month2), int(day2), 3), dt.datetime(int(year3), int(month3), int(day3), dtt*2), dt.timedelta(hours=dtt))

dsu = dsu.interp(time = times, lon=lon+360, lat=lat)
dsv = dsv.interp(time = times,  lon=lon+360, lat=lat)

u = dsu.ugrd10m
v = dsv.vgrd10m

#velocidade = (((u**2)+(v**2))**0.5)/0.5144

#dir=np.zeros(u.shape)
#dir= (270-np.rad2deg(np.arctan2(v,u)))%360
#dir = np.array(dir)
#r1=np.where((dir>=0) &  (dir<180))
#r2=np.where((dir>=180) &  (dir<360))
#dir[r1]=dir[r1] +180
#dir[r2]=dir[r2] -180
#dir=np.round(dir)
#dir[np.where(dir)==360]=0

velcom=np.zeros(u.shape)


#np.dot([u[0,100,100],v[0,100,100]], [ub,vb])/np.linalg.norm([ub,vb])
#https://stackoverflow.com/questions/55226011/using-python-to-calculate-vector-projection
# for i in range(u.shape[0]):
  # velcom[i,:]=np.array(list(map(lambda x,y:np.dot([x,y], [ub,vb]), np.array(u[i]).ravel(), np.array(v[i]).ravel()))).reshape(u.shape[1], u.shape[2])
  # velcom[i][np.where(velcom[i]>0)]=velcom[i][np.where(velcom[i]>0)]/velcom[i].max()
  # velcom[i][np.where(velcom[i]<0)]=velcom[i][np.where(velcom[i]<0)]/np.abs(velcom[i].min())

# wind_index = xr.Dataset(data_vars={'wind_index':(['time','lat', 'lon'], velcom)},
                 # coords={'time':times,
                         # 'lat': lat,
                          # 'lon': lon})
# # i=0
# # sp=3
# # fig = plt.figure(figsize=(8,10), dpi=200)
# # ax = fig.add_subplot(projection=ccrs.PlateCarree())
# # im=ax.pcolor(lon2,lat2, velcom[i], cmap = 'rainbow')								
# # u_arrow = u[i]
# # v_arrow = v[i]
# # U = (u_arrow / np.sqrt(u_arrow**2 + v_arrow**2)); #esse passo é necessário pra setas ficarem do mesmo tamanho
# # V = (v_arrow / np.sqrt(u_arrow**2 + v_arrow**2));   
# # plt.quiver(lon2[0:-1:sp, 0:-1:sp],lat2[0:-1:sp, 0:-1:sp],U[0:-1:sp, 0:-1:sp],V[0:-1:sp, 0:-1:sp],color='k', width = 0.0025, headwidth=7, headlength=4, headaxislength=3, scale=50)	   
# # gl = ax.gridlines(draw_labels=True, alpha=0.9, color='1', linestyle='dotted')
# # gl.xlabels_top = False
# # gl.ylabels_left = True
# # gl.ylabels_right = False
# # gl.xlabel_style = {'size': 12}
# # gl.ylabel_style = {'size': 12}
# # gl.xlocator = mticker.FixedLocator([-43,-42, -41,-40,-39])
# # gl.ylocator = mticker.FixedLocator([-22, -23, -24])
# # ax.add_feature(cfeature.LAND, zorder=1)
# # ax.coastlines(zorder=1)
# # #ax.set_xlim(-44, -38)
# # #ax.set_ylim(-24.5,-20)
# # cbaxes = fig.add_axes([0.8,0.15, 0.02, 0.7]) #[x,y,dx,dy]
# # cb=plt.colorbar(im, format='%.1f', cax = cbaxes, orientation='vertical')
# # cb.ax.tick_params(labelsize=12)
# # cb.set_label('Velocidade\nda corrente\n(nós)', labelpad=-40, y=5.05, x=-0.1,rotation='horizontal', fontsize = 10)
# # plt.show()


#LAT_INI=-24


GRAVITY  = 9.8
R_TERRA  = 6371000.
PI=np.pi


veln=9   #knots
veln=[veln*0.51]  #m/s

#YPOS=-1000
#XPOS=-1000


import numpy
import math

def get_bearing(lat1, long1, lat2, long2):
    dLon = (long2 - long1)
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
    brng = numpy.arctan2(x,y)
    brng = numpy.degrees(brng)
    if brng < 0:
      brng=brng+360
    return brng


def coordangle(XPOS, YPOS, LON_REF, LAT_REF):
  LATOUT  = LAT_REF + YPOS / ( (PI/180.)* R_TERRA)
  LONOUT  = LON_REF + XPOS / ( (PI/180.)* R_TERRA*np.cos(LATOUT*PI/180.) )   
  return LONOUT,LATOUT



def distancex_y(distance, dirship):
  if (dirship <=90):
    XPOS=np.sin(np.radians(dirship))*distance
    YPOS=np.cos(np.radians(dirship))*distance
  elif ((dirship <=180) & (dirship >90)):
    dirship=dirship-90
    XPOS=np.cos(np.radians(dirship))*distance
    YPOS=-np.sin(np.radians(dirship))*distance
  elif dirship > 180 and dirship <=270:
    dirship=dirship-180
    XPOS=-np.sin(np.radians(dirship))*distance
    YPOS=-np.cos(np.radians(dirship))*distance
  elif dirship > 270 and dirship <=360:
    dirship=dirship-270
    XPOS=-np.cos(np.radians(dirship))*distance
    YPOS=np.sin(np.radians(dirship))*distance
  return(XPOS, YPOS)

from math import sin, cos, sqrt, atan2, radians
R = 6373.0
def distance_loc(lat1, lon1, lat2, lon2):
  lat1=radians(lat1)
  lat2=radians(lat2)
  lon1=radians(lon1)
  lon2=radians(lon2)  
  dlon = lon2 - lon1
  dlat = lat2 - lat1
  a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
  c = 2 * atan2(sqrt(a), sqrt(1 - a))
  distance = R * c   #km
  return (distance)

#distancex_y(distance,dirship)
###############adcp
LAT_INI=-23.02
LON_INI=-43.109

LAT_F=-25.81
LON_F=-41.04

numdays=6

dirship=140
dirship=get_bearing(LAT_INI, LON_INI, LAT_F, LON_F)

cor_fac=1
          #correction_factor
timen=1800  #s
distance=veln[-1]*timen

timeinitial=dt.datetime(int(year2), int(month2), int(day2), 3)

timl=np.arange(0, numdays*24*60*60, timen)
cor_fac=np.linspace(0,cor_fac, timl.shape[0])

lonship=[LON_INI]
latship=[LAT_INI]
timeship=[timeinitial]

#x = [[] for i in range(3)]

disini=distance_loc(LAT_INI, LON_INI, LAT_F, LON_F)*1000
timedist=disini/veln[-1]
timl2=np.arange(0, timedist, timen)

df=50   #degree
df=np.linspace(df,0,timl2.shape[0]+50)
dd=1   #d degree

wtrand=0.5 #weight for transversal velocity


endo=0
# angles=np.arange(dirship_initial-df, dirship_initial+df+dd, dd)    #degree of freedom
# angles[np.where(angles>=360)]=angles[np.where(angles>=360)] -360
# angles[np.where(angles<0)]=angles[np.where(angles<0)] +360

# indexes=np.zeros(len(angles))
# latg=np.zeros(len(angles))
# long=np.zeros(len(angles))
#df[:]=0
sp=10
j=0
#for j in range(1,20):
timeh=0
for loop in timl:
  timeh=timeh+(timen/(60*60))
  j=j+1
  ntime=timeship[j-1] + dt.timedelta(seconds=timen)
  timeship.append(ntime)
  angles=np.arange(dirship-df[j-1], dirship+df[j-1] + dd, dd)
  angles[np.where(angles>=360)]=angles[np.where(angles>=360)] -360
  angles[np.where(angles<0)]=angles[np.where(angles<0)] +360
  latg=np.zeros(len(angles))
  long=np.zeros(len(angles))
  indexes=np.zeros(len(angles))
  ub,vb=vel_conv(1,dirship)
  print('dirship', dirship, cor_uv_to_dir(ub, vb))
#  timeship[j-1]
  velcom=np.array(list(map(lambda x,y:np.dot([x,y], [ub,vb]), np.array(u.interp(time=ntime)).ravel(), np.array(v.interp(time=ntime)).ravel()))).reshape(u.shape[1], u.shape[2])
  mag=np.array(list(map(lambda x,y:np.sqrt(x**2 + y**2), np.array(u.interp(time=ntime)).ravel(), np.array(v.interp(time=ntime)).ravel()))).reshape(u.shape[1], u.shape[2])
  tcom=np.array(list(map(lambda x,y:np.sqrt(x**2 - y**2), np.array(mag).ravel(), np.array(velcom).ravel()))).reshape(u.shape[1], u.shape[2])
  velcom=velcom - (np.abs(tcom))*wtrand
  velcomp=velcom.copy()  
#  velcom[np.where(velcom>woplim)]=woplim - velcom[np.where(velcom>woplim)]
  velcom[np.where((mag>woplim) &(velcom>=0))]=(woplim - mag[np.where((mag>woplim) &(velcom>=0))])*3
  velcom[np.where(velcom>0)]=velcom[np.where(velcom>0)]/np.nanmax(velcom)
  velcom[np.where(velcom<0)]=velcom[np.where(velcom<0)]/np.abs(np.nanmin(velcom))
  wind_index = xr.Dataset(data_vars={'wind_index':(['lat', 'lon'], velcom)},
                 coords={ 'lat': lat,
                          'lon': lon})
  velcomp2 = xr.Dataset(data_vars={'velcomp':(['lat', 'lon'], velcomp)},
                 coords={ 'lat': lat,
                          'lon': lon})
  for i in range(len(angles)):
   distance=veln[-1]*timen
   xn, yn=distancex_y(distance, angles[i])
   long[i],latg[i]=coordangle(xn, yn, lonship[j-1], latship[j-1])
#   indexes[i]=wind_index.interp(time=ntime, lat=latg[i], lon=long[i]).wind_index.values
   indexes[i]=wind_index.interp(lat=latg[i], lon=long[i]).wind_index.values
  lonship.append(long[np.nanargmax(indexes)])
  latship.append(latg[np.nanargmax(indexes)])
  lastangle=angles[np.argmax(indexes)]
#  deltaangle=dirship_initial - lastangle
#  dirship=angle_correction(deltaangle, dirship_initial, cor_fac[j-1])
#  print(latg[i], i, latship[-1])
#  stop
  dirship = get_bearing(latship[-1], lonship[-1], LAT_F, LON_F)
  distanceb2=distance_loc(latship[-1], lonship[-1], LAT_F, LON_F)
  print('distance2', distanceb2)
  print('velcomp', velcomp2.interp(lat=latship[-1], lon=lonship[-1]).velcomp.values)
  veln.append(veln[0] + velcomp2.interp(lat=latship[-1], lon=lonship[-1]).velcomp.values*0.1)
  if distanceb2<100:
    df[:]=0
  if distanceb2<10:
    endo=1
  if timeh >= 5 or endo ==1:
    fig = plt.figure(figsize=(10,7), dpi=200)
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
    im=ax.pcolor(lon2, lat2, wind_index.wind_index, vmin=-1, vmax=1)
    u_arrow = u.interp(time=ntime)
    v_arrow = v.interp(time=ntime)
#  U = (u_arrow / np.sqrt(u_arrow**2 + v_arrow**2)); #esse passo é necessário pra setas ficarem do mesmo tamanho
#  V = (v_arrow / np.sqrt(u_arrow**2 + v_arrow**2));
    U = (u_arrow); #esse passo é necessário pra setas ficarem do mesmo tamanho
    V = (v_arrow);    
#  ax.quiver(lon2[0:-1:sp, 0:-1:sp],lat2[0:-1:sp, 0:-1:sp],U[0:-1:sp, 0:-1:sp],V[0:-1:sp, 0:-1:sp],color='k', width = 0.0025, headwidth=7, headlength=4, headaxislength=3, scale=50, alpha=0.5)	   
    aa2=ax.quiver(lon2[0:-1:sp, 0:-1:sp],lat2[0:-1:sp, 0:-1:sp],U[0:-1:sp, 0:-1:sp],V[0:-1:sp, 0:-1:sp],color='k', width = 0.0030, headwidth=7, headlength=3, headaxislength=3, scale=80, alpha=0.5)	   
    ax.quiverkey(aa2, 0.25, 0.86, 5, '5 m/s', coordinates='axes')
    ax.plot(lonship[:j+1], latship[:j+1], '-', color='black', linewidth=2)
    ax.plot(lonship[j], latship[j],marker= "^", color='black',  markersize=12)
    ax.plot(lonship[0], latship[0],marker= ".", color='black',  markersize=13)
    ax.plot(LON_F, LAT_F,marker= ".", color='red',  markersize=13)
    ax.add_feature(cfeature.LAND, zorder=1)
    ax.coastlines(zorder=1)
    gl = ax.gridlines(draw_labels=True, alpha=0.8, color='0.4', linestyle='dotted')
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    parallels = np.arange(-40,-10,1)
    meridians = np.arange(-60,-15,1)
    gl.xlocator = mticker.FixedLocator(meridians)
    gl.ylocator = mticker.FixedLocator(parallels)
    ax.set_xlim(-45, -38)
    ax.set_ylim(-27,-21)
    cax = fig.add_axes([0.85, 0.3, 0.02, 0.38])
    cbar=fig.colorbar(im, shrink=0.8, extend='both', ax=ax, cax=cax)
    cbar.ax.set_ylabel('Wind index', rotation=270)
    cbar.ax.get_yaxis().labelpad = 11
    text = cbar.ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(family='times new roman', style='normal', size=11)
    text.set_font_properties(font)
    mp.scale_bar(ax,location=[0.07, 0.0825], length=200,unit_name='km', linewidth=2, metres_per_unit=1000)
    left, bottom, width, height = [0.85, 0.75, 0.09, 0.29]
    axins4 = fig.add_axes([left, bottom, width, height])
    axins4.set_xticks([]) 
    axins4.set_yticks([])
    axins4.axis("off")
    figdateslocal=timeship[j] - dt.timedelta(hours=3)
    axins4.text(0,0,figdateslocal.strftime("%Y/%m/%d - %H:%M") + ' Local Time', fontsize=10, fontweight='bold', bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 5})
    plt.savefig(f'/mnt/c/Users/fernando.barreto/Desktop/weather_routing/wind/apr/{j}_{round(cor_fac[j-1])}_{round(dirship_initial)}.png', transparent=False, bbox_inches="tight")
    plt.close()
    timeh=0
  if endo==1:
    print('END OF JOURNAY')
    break




######################33 CURRENT


pathroms = f'/mnt/c/Users/fernando.barreto/Desktop/boletins/current_mercator/previsao_corrente_{year}{month}{day}.nc'
pathroms = f'/mnt/c/Users/fernando.barreto/Desktop/boletins/current_mercator/previsao_corrente_20220416.nc'


ds = xr.open_dataset(pathroms)


ds = ds.interp(time = times, longitude=lon, latitude=lat)

urs = ds.uo[:,0,:]
vrs = ds.vo[:,0,:]


velcom=np.zeros(urs.shape)


#np.dot([u[0,100,100],v[0,100,100]], [ub,vb])/np.linalg.norm([ub,vb])
#https://stackoverflow.com/questions/55226011/using-python-to-calculate-vector-projection

# for i in range(urs.shape[0]):
  # velcom[i,:]=np.array(list(map(lambda x,y:np.dot([x,y], [ub,vb]), np.array(urs[i]).ravel(), np.array(vrs[i]).ravel()))).reshape(urs.shape[1], urs.shape[2])
  # velcom[i][np.where(velcom[i]>0)]=velcom[i][np.where(velcom[i]>0)]/np.nanmax(velcom[i])
  # velcom[i][np.where(velcom[i]<0)]=velcom[i][np.where(velcom[i]<0)]/np.abs(np.nanmin(velcom[i]))

# cur_index = xr.Dataset(data_vars={'cur_index':(['time','lat', 'lon'], velcom)},
                 # coords={'time':times,
                         # 'lat': lat,
                          # 'lon': lon})


veln=9   #knots
veln=[veln*0.51]  #m/s

LAT_INI=-26
LON_INI=-26

LAT_F=-22
LON_F=-38

numdays=6

dirship=140
dirship=get_bearing(LAT_INI, LON_INI, LAT_F, LON_F)

cor_fac=1
          #correction_factor
timen=1800  #s
distance=veln[-1]*timen

timeinitial=dt.datetime(int(year2), int(month2), int(day2), 3)

timl=np.arange(0, numdays*24*60*60, timen)
cor_fac=np.linspace(0,cor_fac, timl.shape[0])

lonship=[LON_INI]
latship=[LAT_INI]
timeship=[timeinitial]

disini=distance_loc(LAT_INI, LON_INI, LAT_F, LON_F)*1000
timedist=disini/veln[-1]
timl2=np.arange(0, timedist, timen)

df=50   #degree
df=np.linspace(df,0,timl2.shape[0]+50)
dd=1   #d degree

wtrand=0.5 #weight for transversal velocity


endo=0
# angles=np.arange(dirship_initial-df, dirship_initial+df+dd, dd)    #degree of freedom
# angles[np.where(angles>=360)]=angles[np.where(angles>=360)] -360
# angles[np.where(angles<0)]=angles[np.where(angles<0)] +360

# indexes=np.zeros(len(angles))
# latg=np.zeros(len(angles))
# long=np.zeros(len(angles))
#df[:]=0
sp=10
j=0
#for j in range(1,20):
timeh=0
for loop in timl:
  timeh=timeh+(timen/(60*60))
  j=j+1
  ntime=timeship[j-1] + dt.timedelta(seconds=timen)
  #timeship[j-1]
  timeship.append(ntime)
  angles=np.arange(dirship-df[j-1], dirship+df[j-1] + dd, dd)
  angles[np.where(angles>=360)]=angles[np.where(angles>=360)] -360
  angles[np.where(angles<0)]=angles[np.where(angles<0)] +360
  latg=np.zeros(len(angles))
  long=np.zeros(len(angles))
  indexes=np.zeros(len(angles))
  ub,vb=vel_conv(1,dirship)
#  print('dirship', dirship, cor_uv_to_dir(ub, vb))
  velcom=np.array(list(map(lambda x,y:np.dot([x,y], [ub,vb]), np.array(urs.interp(time=ntime)).ravel(), np.array(vrs.interp(time=ntime)).ravel()))).reshape(urs.shape[1], urs.shape[2])
  mag=np.array(list(map(lambda x,y:np.sqrt(x**2 + y**2), np.array(urs.interp(time=ntime)).ravel(), np.array(vrs.interp(time=ntime)).ravel()))).reshape(urs.shape[1], urs.shape[2])
  tcom=np.array(list(map(lambda x,y:np.sqrt(x**2 - y**2), np.array(mag).ravel(), np.array(velcom).ravel()))).reshape(urs.shape[1], urs.shape[2])
  velcom=velcom - (np.abs(tcom))*wtrand
  velcomp=velcom.copy()    
#  velcom[np.where(velcom>coplim)]=coplim - velcom[np.where(velcom>coplim)]
  velcom[np.where((mag>coplim) &(velcom>=0))]=coplim - mag[np.where((mag>coplim) &(velcom>=0))]
  velcom[np.where(velcom>0)]=velcom[np.where(velcom>0)]/np.nanmax(velcom)
  velcom[np.where(velcom<0)]=velcom[np.where(velcom<0)]/np.abs(np.nanmin(velcom))
  cur_index = xr.Dataset(data_vars={'cur_index':(['lat', 'lon'], velcom)},
                 coords={ 'lat': lat,
                          'lon': lon})
  velcomp2 = xr.Dataset(data_vars={'velcomp':(['lat', 'lon'], velcomp)},
                 coords={ 'lat': lat,
                          'lon': lon})
  for i in range(len(angles)):
   distance=veln[-1]*timen
   xn, yn=distancex_y(distance, angles[i])
   long[i],latg[i]=coordangle(xn, yn, lonship[j-1], latship[j-1])
#   indexes[i]=cur_index.interp(time=ntime, lat=latg[i], lon=long[i]).cur_index.values
   indexes[i]=cur_index.interp(lat=latg[i], lon=long[i]).cur_index.values
  lonship.append(long[np.nanargmax(indexes)])
  latship.append(latg[np.nanargmax(indexes)])
  lastangle=angles[np.argmax(indexes)]
#  deltaangle=dirship_initial - lastangle
#  dirship=angle_correction(deltaangle, dirship_initial, cor_fac[j-1])
  dirship = get_bearing(latship[-1], lonship[-1], LAT_F, LON_F)
  distanceb2=distance_loc(latship[-1], lonship[-1], LAT_F, LON_F)
  veln.append(veln[0] + velcomp2.interp(lat=latship[-1], lon=lonship[-1]).velcomp.values)
  print('distance2', distanceb2)
  print('vel', velcomp2.interp(lat=latship[-1], lon=lonship[-1]).velcomp.values, veln[0] + velcomp2.interp(lat=latship[-1], lon=lonship[-1]).velcomp.values,veln[0])
  if distanceb2<100:
    df[:]=0
  if distanceb2<10:
    endo=1
  if timeh >= 5 or endo ==1:
    fig = plt.figure(figsize=(10,7), dpi=200)
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
#    im=ax.pcolor(lon2, lat2, cur_index.interp(time=ntime).cur_index, vmin=-1, vmax=1)
    im=ax.pcolor(lon2, lat2, cur_index.cur_index, vmin=-1, vmax=1)
    u_arrow = urs.interp(time=ntime)
    v_arrow = vrs.interp(time=ntime)
#  U = (u_arrow / np.sqrt(u_arrow**2 + v_arrow**2)); #esse passo é necessário pra setas ficarem do mesmo tamanho
#  V = (v_arrow / np.sqrt(u_arrow**2 + v_arrow**2));
    U = (u_arrow); #esse passo é necessário pra setas ficarem do mesmo tamanho
    V = (v_arrow);    
#  ax.quiver(lon2[0:-1:sp, 0:-1:sp],lat2[0:-1:sp, 0:-1:sp],U[0:-1:sp, 0:-1:sp],V[0:-1:sp, 0:-1:sp],color='k', width = 0.0025, headwidth=7, headlength=4, headaxislength=3, scale=50, alpha=0.5)	   
    ax.quiver(lon2[0:-1:sp, 0:-1:sp],lat2[0:-1:sp, 0:-1:sp],U[0:-1:sp, 0:-1:sp],V[0:-1:sp, 0:-1:sp],color='k', width = 0.0030, headwidth=3, headlength=3, headaxislength=3, scale=10, alpha=0.5)	   
    ax.plot(lonship[:j+1], latship[:j+1], '-', color='black', linewidth=2)
    ax.plot(lonship[j], latship[j],marker= "^", color='black',  markersize=6)
    ax.plot(lonship[0], latship[0],marker= ".", color='black',  markersize=13)
    ax.plot(LON_F, LAT_F,marker= ".", color='red',  markersize=13)
    ax.add_feature(cfeature.LAND, zorder=1)
    ax.coastlines(zorder=1)
    gl = ax.gridlines(draw_labels=True, alpha=0.8, color='0.4', linestyle='dotted')
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    parallels = np.arange(-40,-10,1)
    meridians = np.arange(-60,-15,1)
    gl.xlocator = mticker.FixedLocator(meridians)
    gl.ylocator = mticker.FixedLocator(parallels)
    ax.set_xlim(-41, -20)
    ax.set_ylim(-35,-20.05)
    cax = fig.add_axes([0.92, 0.3, 0.02, 0.38])
    cbar=fig.colorbar(im, shrink=0.8, extend='both', ax=ax, cax=cax)
    cbar.ax.set_ylabel('Current index', rotation=270)
    cbar.ax.get_yaxis().labelpad = 11
    text = cbar.ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(family='times new roman', style='normal', size=11)
    text.set_font_properties(font)
    mp.scale_bar(ax,location=[0.07, 0.0825], length=200,unit_name='km', linewidth=2, metres_per_unit=1000)
    left, bottom, width, height = [0.9, 0.75, 0.09, 0.29]
    axins4 = fig.add_axes([left, bottom, width, height])
    axins4.set_xticks([]) 
    axins4.set_yticks([])
    axins4.axis("off")
    figdateslocal=timeship[j] - dt.timedelta(hours=3)
    axins4.text(0,0,figdateslocal.strftime("%Y/%m/%d - %H:%M") + ' Local Time', fontsize=10, fontweight='bold', bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 5})
    plt.savefig(f'/mnt/c/Users/fernando.barreto/Desktop/weather_routing/current/{j}_{cor_fac[j-1]}_{dirship_initial}.png', transparent=False, bbox_inches="tight")
    plt.close()
    timeh=0
  if endo==1:
    print('END OF JOURNAY')
    break



###########################################################TOTAL

pathwave = f'/mnt/c/Users/fernando.barreto/Desktop/boletins/wave_copernicus/previsao_preliminar_{year2}{month2}{day2}.nc'
pathwave = f'/mnt/c/Users/fernando.barreto/Desktop/boletins/wave_copernicus/previsao_preliminar_20220418.nc'


ds = xr.open_dataset(pathwave)
ds = ds.interp(time = times, longitude=lon, latitude=lat)
hrs = ds.VHM0


###bathymetry
bathymetry=False
bati=xr.open_dataset('/mnt/c/Users/fernando.barreto/Desktop/ROMS/make_grid/gebco_222.nc')
limbat=50 #m
limdist=100 #km
CS=plt.contour(bati.lon, bati.lat, bati.elevation, levels=[0])
dat0= CS.allsegs[0][0]
latct=dat0[:,1]
lonct=dat0[:,0]
limcoast=100
distcoast=False

#ict=np.argmin( np.abs(lonct-(long[i])) + np.abs(latct-(latg[i])) )
#distance_loc(latct[ict], lonct[ict], latg[i],long[i])
#latg[i], lon=long[i]
##


wgtw=50
wgtc=50
wgtww=50


veln=7.5   #knots
velkts=veln
veln=[veln*0.51]  #m/s

LAT_INI=-24.0960
LON_INI=-42.0632 

LAT_F=-22.9772  # rio de janeiro cidade
LON_F=-43.1172

LAT_INI=-22.9772  # rio de janeiro cidade
LON_INI=-43.1172

LAT_F=-24.480339
LON_F=-42.109280

LAT_F=-24.406921
LON_F=-42.150772

#LAT_F=-11.03   #aracaju
#LON_F=-36.9

##1
#LAT_INI=-22.1
#LON_INI=-41
#LAT_F=-27
#LON_F=-38.
#2
#LAT_INI=-22.1
#LON_INI=-41
#LAT_F=-23.03
#LON_F=-43.13

numdays=6

#dirship=140
dirship=get_bearing(LAT_INI, LON_INI, LAT_F, LON_F)

cor_fac=1
          #correction_factor
timen=3600/2  #s
#timen=60  #s
distance=veln[-1]*timen

timeinitial=dt.datetime(int(year2), int(month2), int(day2), 3)

timeinitial=dt.datetime(int(year2), int(month2), int('07'), 2)

timeinitial=dt.datetime(int(year2), int(month2), int(day2), 15)


timl=np.arange(0, numdays*24*60*60, timen)
cor_fac=np.linspace(0,cor_fac, timl.shape[0])

lonship=[LON_INI]
latship=[LAT_INI]
timeship=[timeinitial]
timeshipp=[timeinitial]


u_wind = u.interp(time=timeship[0], lon=360+LON_INI, lat=LAT_INI).values
v_wind = v.interp(time=timeship[0],  lon=360+LON_INI, lat=LAT_INI).values
wmag=np.sqrt((u_wind)**2 + (v_wind)**2)


wmagw = hrs.interp(time=timeship[0],  longitude=LON_INI, latitude=LAT_INI).values


u_arrow = urs.interp(time=timeship[0], longitude=LON_INI, latitude=LAT_INI).values
v_arrow = vrs.interp(time=timeship[0], longitude=LON_INI, latitude=LAT_INI).values

cmag=np.sqrt((u_arrow)**2 + (v_arrow)**2)

velship=[cmag]
waveh=[wmagw]
velwind=[wmag]

    
disini=distance_loc(LAT_INI, LON_INI, LAT_F, LON_F)*1000
timedist=disini/veln[-1]
timl2=np.arange(0, timedist, timen)

df=50   #degree
df=0   #degree
#df=np.linspace(df,0,timl2.shape[0]+50)
df=np.linspace(df,0,timl2.shape[0])
dd=1   #d degree

wtrand=0.5 #weight for transversal velocity


endo=0
# angles=np.arange(dirship_initial-df, dirship_initial+df+dd, dd)    #degree of freedom
# angles[np.where(angles>=360)]=angles[np.where(angles>=360)] -360
# angles[np.where(angles<0)]=angles[np.where(angles<0)] +360

# indexes=np.zeros(len(angles))
# latg=np.zeros(len(angles))
# long=np.zeros(len(angles))
#df[:]=0
distanceb2=10000
distanceb3=0
sp=10
j=0
jj=0
#for j in range(1,20):
timeh=0
timeh1=0

#for loop in timl:
while True:
  timeh=timeh+(timen/(60*60))
  timeh1=timeh1+(timen/(60*60))
  j=j+1
  jj=jj+1
  if j > len(df):
    jj=len(df)-1
  ntime=timeship[j-1] + dt.timedelta(seconds=timen)
  #timeship[j-1]
  timeship.append(ntime)
  angles=np.arange(dirship-df[jj-1], dirship+df[jj-1] + dd, dd)
  angles[np.where(angles>=360)]=angles[np.where(angles>=360)] -360
  angles[np.where(angles<0)]=angles[np.where(angles<0)] +360
  latg=np.zeros(len(angles))
  long=np.zeros(len(angles))
  indexes=np.zeros(len(angles))
  depthb=np.zeros(len(angles))
  coastdist=np.zeros(len(angles))
  ub,vb=vel_conv(1,dirship)
#  print('dirship', dirship, cor_uv_to_dir(ub, vb))
  velcom=np.array(list(map(lambda x,y:np.dot([x,y], [ub,vb]), np.array(urs.interp(time=ntime)).ravel(), np.array(vrs.interp(time=ntime)).ravel()))).reshape(urs.shape[1], urs.shape[2])
#  velcom[np.isnan(velcom)]=0
  mag=np.array(list(map(lambda x,y:np.sqrt(x**2 + y**2), np.array(urs.interp(time=ntime)).ravel(), np.array(vrs.interp(time=ntime)).ravel()))).reshape(urs.shape[1], urs.shape[2])
  tcom=np.array(list(map(lambda x,y:np.sqrt(x**2 - y**2), np.array(mag).ravel(), np.array(velcom).ravel()))).reshape(urs.shape[1], urs.shape[2])
  velcom=velcom - (np.abs(tcom))*wtrand
  velcomp=velcom.copy()    
#  velcom[np.where(velcom>coplim)]=coplim - velcom[np.where(velcom>coplim)]
  velcom[np.where((mag>coplim) &(velcom>=0))]=coplim - mag[np.where((mag>coplim) &(velcom>=0))]
  velcom[np.where(velcom>0)]=velcom[np.where(velcom>0)]/np.nanmax(velcom)
  velcom[np.where(velcom<0)]=velcom[np.where(velcom<0)]/np.abs(np.nanmin(velcom))
  cur_index = xr.Dataset(data_vars={'cur_index':(['lat', 'lon'], velcom)},
                 coords={ 'lat': lat,
                          'lon': lon})
  velcomp2VEL = xr.Dataset(data_vars={'velcomp':(['lat', 'lon'], velcomp)},
                 coords={ 'lat': lat,
                          'lon': lon})
  velcom=np.array(list(map(lambda x,y:np.dot([x,y], [ub,vb]), np.array(u.interp(time=ntime)).ravel(), np.array(v.interp(time=ntime)).ravel()))).reshape(u.shape[1], u.shape[2])
  mag=np.array(list(map(lambda x,y:np.sqrt(x**2 + y**2), np.array(u.interp(time=ntime)).ravel(), np.array(v.interp(time=ntime)).ravel()))).reshape(u.shape[1], u.shape[2])
  tcom=np.array(list(map(lambda x,y:np.sqrt(x**2 - y**2), np.array(mag).ravel(), np.array(velcom).ravel()))).reshape(u.shape[1], u.shape[2])
  velcom=velcom - (np.abs(tcom))*wtrand
  velcomp=velcom.copy()  
#  velcom[np.where(velcom>woplim)]=woplim - velcom[np.where(velcom>woplim)]
  velcom[np.where((mag>woplim) &(velcom>=0))]=(woplim - mag[np.where((mag>woplim) &(velcom>=0))])*3
  velcom[np.where(velcom>0)]=velcom[np.where(velcom>0)]/np.nanmax(velcom)
  velcom[np.where(velcom<0)]=velcom[np.where(velcom<0)]/np.abs(np.nanmin(velcom))
  wind_index = xr.Dataset(data_vars={'wind_index':(['lat', 'lon'], velcom)},
                 coords={ 'lat': lat,
                          'lon': lon})
  velcomp2WIND = xr.Dataset(data_vars={'velcomp':(['lat', 'lon'], velcomp)},
                 coords={ 'lat': lat,
                          'lon': lon})
  #print(1)
  velcom=np.array(hrs.interp(time=ntime))
  hrsv=np.array(hrs.interp(time=ntime))
  hrsv[np.isnan(hrsv)]=0
  velcom=velcom/-np.ma.masked_invalid(velcom).max()
  velcom[np.isnan(velcom)]=0
  wave_index = xr.Dataset(data_vars={'wave_index':(['lat', 'lon'], velcom)},
                 coords={ 'lat': lat,
                          'lon': lon})  
  tot=(wind_index.wind_index*wgtw + cur_index.cur_index*wgtc + wave_index.wave_index*wgtww)/(1*wgtw + 1*wgtc + 1*wgtww)
  valm=np.where((mag<10) & (hrsv<2))
  tot.values[valm] = cur_index.cur_index.values[valm]
  valm=np.where((mag<10) & (hrsv>=2))
  tot.values[valm] = (cur_index.cur_index.values[valm]*wgtc+ wave_index.wave_index.values[valm]*wgtww)/( 1*wgtc + 1*wgtww)
  valm=np.where(hrsv>=wavelim)
  tot.values[valm] =  wave_index.wave_index.values[valm]*3
  total_index = xr.Dataset(data_vars={'total_index':(['lat', 'lon'], tot)},
                 coords={ 'lat': lat,
                          'lon': lon})
  for i in range(len(angles)):
   distance=veln[-1]*timen
   xn, yn=distancex_y(distance, angles[i])
   long[i],latg[i]=coordangle(xn, yn, lonship[j-1], latship[j-1])
#   indexes[i]=cur_index.interp(time=ntime, lat=latg[i], lon=long[i]).cur_index.values
   indexes[i]=total_index.interp(lat=latg[i], lon=long[i]).total_index.values
#   print(3)
   if bathymetry:
#     print(9)
#     depthb[i]=bati.interp(lat=latg[i], lon=long[i]).elevation.values
     depthb[i]=bati.sel(lat=latg[i], lon=long[i], method='nearest').elevation.values
   if distcoast:
#     print(99)
     ict=np.argmin( np.abs(lonct-(long[i])) + np.abs(latct-(latg[i])) )
     coastdist[i] = distance_loc(latct[ict], lonct[ict], latg[i],long[i])
######
 # stop
#  if (bathymetry) & (distanceb2 > limdist) & (distanceb3 > 50):
#  indexcop=indexes.copy()
  if (bathymetry) & (distanceb2 > limdist):
    print('111111')
    indexes[depthb>-limbat]=np.nan
  if (distcoast) & (distanceb2 > limdist):
    print('111111')
    indexes[coastdist<limcoast]=np.nan
#  stop
  acs=0
  bcs=0
#  stop
  while np.sum(np.isnan(indexes)) == len(indexes):
#    stop
#    if ( (len(indexes[depthb>-limbat+bcs])==0) and (len(indexes[coastdist<limcoast-bcs])==0) ):
#    if np.sum(np.isnan(indexcop)) == len(indexcop):
    acs=acs+10
    angles=np.arange(dirship-df[jj-1]-acs, dirship+df[jj-1] + acs + dd, dd)
    angles[np.where(angles>=360)]=angles[np.where(angles>=360)] -360
    angles[np.where(angles<0)]=angles[np.where(angles<0)] +360
    latg=np.zeros(len(angles))
    long=np.zeros(len(angles))
    indexes=np.zeros(len(angles))
    depthb=np.zeros(len(angles))
    coastdist=np.zeros(len(angles))
    for i in range(len(angles)):
     distance=veln[-1]*timen
     xn, yn=distancex_y(distance, angles[i])
     long[i],latg[i]=coordangle(xn, yn, lonship[j-1], latship[j-1])
     indexes[i]=total_index.interp(lat=latg[i], lon=long[i]).total_index.values
     if bathymetry:
#      depthb[i]=bati.interp(lat=latg[i], lon=long[i]).elevation.values
       depthb[i]=bati.sel(lat=latg[i], lon=long[i], method='nearest').elevation.values
     if distcoast:
#     print(99)
      ict=np.argmin( np.abs(lonct-(long[i])) + np.abs(latct-(latg[i])) )
      coastdist[i] = distance_loc(latct[ict], lonct[ict], latg[i],long[i])
#    if (bathymetry) & (distanceb2 > limdist) & (distanceb3 > 50):
    if (bathymetry) & (distanceb2 > limdist):
     print('222')
#     print('rrr')
#     stop
#     print(depthb)
#     print(indexes)
#     print(-limbat+bcs)
     indexes[depthb>-limbat+bcs]=np.nan
    if (distcoast) & (distanceb2 > limdist):
     print('111111')
     indexes[coastdist<limcoast-bcs]=np.nan
    bcs=bcs+10
#####
  lonship.append(long[np.nanargmax(indexes)])
  latship.append(latg[np.nanargmax(indexes)])
  lastangle=angles[np.argmax(indexes)]
#  deltaangle=dirship_initial - lastangle
#  dirship=angle_correction(deltaangle, dirship_initial, cor_fac[j-1])
  dirship = get_bearing(latship[-1], lonship[-1], LAT_F, LON_F)
  distanceb2=distance_loc(latship[-1], lonship[-1], LAT_F, LON_F)
  distanceb3=distance_loc(latship[-1], lonship[-1], LAT_INI, LON_INI)
  veln.append(veln[0] + velcomp2VEL.interp(lat=latship[-1], lon=lonship[-1]).velcomp.values)  # ship velocity minus current
  print('distance2', distanceb2)
  print('df', df[jj-1])
  print('depth',depthb)
  print('depth2', depthb[np.nanargmax(indexes)])
  print('vel', velcomp2VEL.interp(lat=latship[-1], lon=lonship[-1]).velcomp.values, veln[0] + velcomp2VEL.interp(lat=latship[-1], lon=lonship[-1]).velcomp.values,veln[0])
  u_wind = u.interp(time=ntime, lon=360+lonship[-1], lat=latship[-1]).values
  v_wind = v.interp(time=ntime,  lon=360+lonship[-1], lat=latship[-1]).values   
  wmag=np.sqrt((u_wind)**2 + (v_wind)**2)
  wmagw = hrs.interp(time=ntime,  longitude=lonship[-1], latitude=latship[-1]).values
  u_arrow = urs.interp(time=ntime,longitude=lonship[-1], latitude=latship[-1]).values
  v_arrow = vrs.interp(time=ntime, longitude=lonship[-1], latitude=latship[-1]).values
  cmag=np.sqrt((u_arrow)**2 + (v_arrow)**2)
  velship.append(cmag)
  waveh.append(wmagw)
  velwind.append(wmag)
  timeshipp.append(ntime)
#  if distanceb2<100:
  if distanceb2<30:
    df[:]=0
    timen = 1000
  if distanceb2<5:
    endo=1
# if timeh1 >= 0.5 or endo ==1:  
#    timeh1=1
  if timeh >= 20 or endo ==1:
    fig = plt.figure(figsize=(10,7), dpi=200)
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
#    im=ax.pcolor(lon2, lat2, cur_index.interp(time=ntime).cur_index, vmin=-1, vmax=1)
    im=ax.pcolor(lon2, lat2, total_index.total_index, vmin=-1, vmax=1)
    u_wind = u.interp(time=ntime, lon=360+lonship[-1], lat=latship[-1]).values
    v_wind = v.interp(time=ntime,  lon=360+lonship[-1], lat=latship[-1]).values
    wmag=np.sqrt((u_wind)**2 + (v_wind)**2)
    left, bottom, width, height = [0.85, 0.70, 0.15, 0.15]
    axins = fig.add_axes([left, bottom, width, height])
    aa2=axins.quiver(0.5, 0.5, u_wind,v_wind, headwidth=10, scale=wmag*6)
    axins.quiverkey(aa2, 0.5, 0.7, 0, 'Wind - ' +  str(np.round(wmag,2)) + ' m/s', coordinates='axes')
    axins.set_xticks([]) 
    axins.set_yticks([])
    wmagw = hrs.interp(time=ntime,  longitude=lonship[-1], latitude=latship[-1]).values
    if np.isnan(wmagw):
      wmagw=0
    left, bottom, width, height = [0.97, 0.34, 0.15, 0.15]
    axinsw = fig.add_axes([left, bottom, width, height])
    aa3=axinsw.quiver(0.5, 0.5, 0,0)
    axinsw.quiverkey(aa3, 0.5, 0.5, 0, 'Wave - ' +  str(np.round(wmagw,2)) + ' m', coordinates='axes')
    axinsw.set_xticks([]) 
    axinsw.set_yticks([])
    u_arrow = urs.interp(time=ntime)
    v_arrow = vrs.interp(time=ntime)
#  U = (u_arrow / np.sqrt(u_arrow**2 + v_arrow**2)); #esse passo é necessário pra setas ficarem do mesmo tamanho
#  V = (v_arrow / np.sqrt(u_arrow**2 + v_arrow**2));
    U = (u_arrow); #esse passo é necessário pra setas ficarem do mesmo tamanho
    V = (v_arrow);    
#  ax.quiver(lon2[0:-1:sp, 0:-1:sp],lat2[0:-1:sp, 0:-1:sp],U[0:-1:sp, 0:-1:sp],V[0:-1:sp, 0:-1:sp],color='k', width = 0.0025, headwidth=7, headlength=4, headaxislength=3, scale=50, alpha=0.5)	   
    ax.quiver(lon2[0:-1:sp, 0:-1:sp],lat2[0:-1:sp, 0:-1:sp],U[0:-1:sp, 0:-1:sp],V[0:-1:sp, 0:-1:sp],color='k', width = 0.0030, headwidth=3, headlength=3, headaxislength=3, scale=10, alpha=0.5)	   
    ax.plot(lonship[:j+1], latship[:j+1], '-', color='black', linewidth=2)
    ax.plot(lonship[j], latship[j],marker= "^", color='black',  markersize=6)
    ax.plot(lonship[0], latship[0],marker= ".", color='black',  markersize=13)
#    ax.contour(bati.lon.values, bati.lat.values, bati.elevation.values, levels=[-50])
    ax.plot(LON_F, LAT_F,marker= ".", color='red',  markersize=13)
    ax.add_feature(cfeature.LAND, zorder=1)
    ax.coastlines(zorder=1)
    gl = ax.gridlines(draw_labels=True, alpha=0.8, color='0.4', linestyle='dotted')
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    parallels = np.arange(-40,-10,1)
    meridians = np.arange(-60,-15,1)
    gl.xlocator = mticker.FixedLocator(meridians)
    gl.ylocator = mticker.FixedLocator(parallels)
#    ax.set_xlim(-50, -28)
#    ax.set_ylim(-30,-12)
    ax.set_xlim(-45, -41)
    ax.set_ylim(-25,-21.5)
    cax = fig.add_axes([0.85, 0.2, 0.02, 0.38])
    cbar=fig.colorbar(im, shrink=0.8, extend='both', ax=ax, cax=cax)
    cbar.ax.set_ylabel('Total index', rotation=270)
    cbar.ax.get_yaxis().labelpad = 11
    text = cbar.ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(family='times new roman', style='normal', size=11)
    text.set_font_properties(font)
    mp.scale_bar(ax,location=[0.5, 0.0825], length=50,unit_name='km', linewidth=2, metres_per_unit=1000)
    left, bottom, width, height = [0.85, 0.65, 0.09, 0.29]
    axins4 = fig.add_axes([left, bottom, width, height])
    axins4.set_xticks([]) 
    axins4.set_yticks([])
    axins4.axis("off")
    figdateslocal=timeship[j] - dt.timedelta(hours=3)
#    figdateslocal=timeship[j]
    axins4.text(0,0,figdateslocal.strftime("%Y/%m/%d - %H:%M") + ' Local Time', fontsize=10, fontweight='bold', bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 5})
    plt.savefig(f'/mnt/c/Users/fernando.barreto/Desktop/weather_routing/total/sj/{j}_4.png', transparent=False, bbox_inches="tight")
    plt.close()
    timeh=0
  if endo==1:
    print('END OF JOURNAY')
    break




velshipn=np.array(velship)*1.94384
velwindn=np.array(velwind)*1.94384
waveh=np.array(waveh)
waveh[np.isnan(waveh)]=waveh[np.where(np.isnan(waveh))[0][0]-1]
time=pd.to_datetime(np.array(timeshipp) - dt.timedelta(hours=3)).strftime('%d/%m/%y - %H:%M') 
#time=pd.to_datetime(timeshipp).strftime('%d/%m/%y - %H:%M') # UTC

#d = {'Time (UTC)': time, 'Ship Vel (Kts)':np.zeros(len(waveh))+velkts, 'Latitude':latship, 'Longitude':lonship, 'Current (Kts)': velshipn, 'Wind (Kts)':velwindn, 'Wave (m)':waveh}
d = {'Time (Local)': time, 'Ship Vel (Kts)':np.zeros(len(waveh))+velkts, 'Latitude':latship, 'Longitude':lonship, 'Current (Kts)': velshipn, 'Wind (Kts)':velwindn, 'Wave (m)':waveh}



df=pd.DataFrame(data=d)


df.to_excel("/mnt/c/Users/fernando.barreto/Desktop/routing_meteoceano3.xlsx",sheet_name='first', index=False)












