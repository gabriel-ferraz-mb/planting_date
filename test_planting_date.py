# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:21:22 2023

@author: gabri
"""
import ee

# try to initalize an ee session
# if not authenticated then run auth workflow and initialize
try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()
#**************************************************************************
# Importando bibliotecas
#**************************************************************************
import sys
import eemont, geemap
import datetime
from datetime import time, timedelta, date
from datetime import datetime
#import geemap.colormaps as cm
import geopandas as gpd
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from sqlalchemy import create_engine
import psycopg2

from scipy import signal

import json



def conecta_db(host, port, db, user, password):
    con = psycopg2.connect(host=host, 
                         port = port,
                         database=db,
                         user=user, # DEFINE! 
                         password=password, # DEFINE!
                         keepalives=1,
                         keepalives_idle=0,
                         keepalives_interval=0,
                         keepalives_count=0)
    return con


def consultar_db(sql, con):
    #con = conecta_db_homolog()
    cur = con.cursor()
    cur.execute(sql)
    recset = cur.fetchall()
    registros = []
    for rec in recset:
        registros.append(rec)
        con.close()
    return registros


def outlineEdges(featureCollection, width):
    fc = ee.Image().byte().paint(**{
        'featureCollection': featureCollection,
        'color': 1, # black
        'width': width})
    return fc

#**************************************************************************
# FUNÇÕES PARA AQUISIÇÃO DAS IMAGENS NDVI
#**************************************************************************

def escala(image):
    b4 = image.select('B4').multiply(0.0001).float();
    b3 = image.select('B3').multiply(0.0001).float();
    b2 = image.select('B2').multiply(0.0001).float();
    b8 = image.select('B8').multiply(0.0001).float();
    return b4.addBands(b3).addBands(b2).addBands(b8).copyProperties(image,['system:time_start','system:time_end']);

def Ndvi(img):  
    Ndvi_image = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return img.addBands(Ndvi_image)

def Ndvi_ls(img):  
    Ndvi_image = img.normalizedDifference(['B5', 'B4']).rename('NDVI')
    return img.addBands(Ndvi_image)

def apply_scale_factors(image):
    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    return image.addBands(opticalBands, None, True).addBands(thermalBands, None, True)

def renomear(img) :
    img = ee.Image(img.select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']))
    img = img.rename(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
    #img = img.gt('properties:CLOUD_COVER_LAND').rename('CLOUD_COVERAGE_ASSESSMENT')
    return img


#**************************************************************************
# Agregando as informações do S2 e LS e ordenando por data 
#**************************************************************************
def serie_temporal_s2_ls(start_date, end_date, lat_long):
    point = lat_long.buffer(100)

    S2_ts = (ee.ImageCollection('COPERNICUS/S2_SR')
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .maskClouds()
            .scaleAndOffset()
            .spectralIndices(['NDVI'])
            .select(['NDVI']))
    
    dataset9_t1 = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
            .filterBounds(point)
            #.filter(ee.Filter.lt('CLOUD_COVER', CLOUD_FILTER))
            .map(apply_scale_factors)
            .map(renomear)
            .map(Ndvi_ls)
            .filterDate(start_date, end_date))

    dataset9_t2 = (ee.ImageCollection('LANDSAT/LC09/C02/T2_L2')
            .filterBounds(point)
            #.filter(ee.Filter.lt('CLOUD_COVER', CLOUD_FILTER))
            .map(apply_scale_factors)
            .map(renomear)
            .map(Ndvi_ls)
            .filterDate(start_date, end_date))

    dataset8_t1 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
          .filterBounds(point)
          #.filter(ee.Filter.lt('CLOUD_COVER', CLOUD_FILTER))
          .map(apply_scale_factors)
          .map(renomear)
          .map(Ndvi_ls)
          .filterDate(start_date, end_date))

    dataset8_t2 = (ee.ImageCollection('LANDSAT/LC08/C02/T2_L2')
          .filterBounds(point)
          #.filter(ee.Filter.lt('CLOUD_COVER', CLOUD_FILTER))
          .map(apply_scale_factors)
          .map(renomear)
          .map(Ndvi_ls)
          .filterDate(start_date, end_date))
    
    LS_ts = dataset9_t1.merge(dataset9_t2).merge(dataset8_t1).merge(dataset8_t2).sort("system:time_start")
    
    NDVI_merged = S2_ts.merge(LS_ts).sort('system:time_start')
    #LS_ts.size().getInfo()
    #S2_ts.size().getInfo()
    #NDVI_merged.size().getInfo()
    
    ts_NDVI_merged = NDVI_merged.getTimeSeriesByRegion(reducer = [ee.Reducer.mean()],
                              geometry = point,
                              bands = ['NDVI'],
                              scale = 30)

    ts_ndvi_merged = geemap.ee_to_pandas(ts_NDVI_merged)

    ts_ndvi_merged[ts_ndvi_merged == -9999] = np.nan
    return ts_ndvi_merged 

#**************************************************************************
# Tratamento da série temporal com os sensores S2 e LS
#**************************************************************************

def tratamento_serie_temporal(ts_ndvi_merged):
    ## série bruta

    #x_merged = ts_ndvi_merged['date'].to_numpy()
    #y_merged = ts_ndvi_merged['NDVI'].to_numpy()
    ts_ndvi_merged["id"] = ts_ndvi_merged.index

    ## série interpolada

    ts_spline_merged = ts_ndvi_merged.interpolate('spline', order=5)
    #x_spline_merged= ts_spline_merged['date'].to_numpy()
    y_spline_merged = ts_spline_merged['NDVI'].to_numpy()
    ts_spline_merged["id"] = ts_spline_merged.index

    ## série suavizada

    y_smooth_merged = signal.savgol_filter(y_spline_merged, window_length=11, polyorder=2, mode="nearest")
    
    return y_smooth_merged, ts_spline_merged, y_spline_merged

#**************************************************************************
#Definir a posição da emergência na série temporal --> Regra de negócio (incorporando outras regras aqui)
#**************************************************************************


def emergencia(serie):
    
    #Identifica a data de plantio
    p = 1
    plantio_def = np.nan
    for j in range(len(serie)-4):
        if float(serie[j]) > 0.5:
           # menos_1 = p-1 #0
            mais_1  = p+1 #2
            mais_2  = p+2 #3
            mais_3  = p+3 #4
            if serie[p] < serie[mais_1] and serie[p] < serie[mais_2] \
                                        and serie[p] < serie[mais_3]:
                plantio_def = p
                break      
            p+=1
        
    return plantio_def, j

user = 'ferraz'
password = '3ino^Vq3^R1!'
host = 'vps40890.publiccloud.com.br'
port = 5432
database = 'carbon'

global engine
engine = create_engine(
        url="postgresql+psycopg2://{0}:{1}@{2}:{3}/{4}".format(
            user, password, host, port, database
        )
    )

polygons  = gpd.read_file(r'C:\Projetos\penalizacao\LEM_dataset\LEM_dataset.shp')
t = polygons.apply(lambda row: row[row == 'Soybean'].index, axis=1)
polygons['p_date'] = np.nan

for i in range(len(polygons)):
    try:
        polygons.at[i, 'p_date'] = t.iloc[i][0]
    except:
        pass
    
soybean_farms = polygons[polygons['p_date'].notnull()][['geometry', 'p_date']]
soybean_farms['plantio'] =  pd.to_datetime(soybean_farms['p_date'], format='%b_%Y')
idList = soybean_farms.index.to_list()
     
soybean_farms['plantio_estimado'] = np.nan
for i in idList:
    start_date = '2019-10-01'
    end_date = '2020-09-30'

    talhao = soybean_farms.iloc[[i]]['geometry']
    geo_json = talhao.to_json()
    j = json.loads(geo_json)
    talhaoFc = ee.FeatureCollection(j)
    lat_long = talhaoFc.geometry().centroid()

    ts_ndvi_merged = serie_temporal_s2_ls(start_date, end_date, lat_long) ### serie temporal sentinel e landsat's
    y_smooth_merged, ts_spline_merged, y_spline_merged = tratamento_serie_temporal(ts_ndvi_merged)

    on, p = emergencia(y_smooth_merged) 
    plantio_data = ts_spline_merged.iloc[p]['date']
    soybean_farms.at[i, 'plantio_estimado'] = datetime.strptime(plantio_data, '%Y-%m-%dT%H:%M:%S')# - timedelta(days = 40)
    print('row {} done'.format(i))

soybean_farms = soybean_farms[soybean_farms['plantio_estimado'].notnull()][['plantio', 'plantio_estimado']]
soybean_farms['plantio_estimado'] =  pd.to_datetime(soybean_farms['plantio_estimado'], format='%Y-%m-%dT%H:%M:%S')
soybean_farms['diff'] = (soybean_farms['plantio'] - soybean_farms['plantio_estimado']).astype('timedelta64[D]')

late = soybean_farms.loc[(soybean_farms['diff'] < -30)]
early = soybean_farms.loc[(soybean_farms['diff'] > 0)] 

early['diff'].describe()
late['diff'].describe()
soybean_farms['diff'].between(-30,0).value_counts()




