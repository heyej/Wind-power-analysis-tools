#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import datetime
import os

import numpy as np
import pandas as pd
import iris.pandas
from dateutil.rrule import rrule, MONTHLY

def read(station,date_bgn,date_end,variables):
    date_bgn = datetime.datetime.strptime(date_bgn,'%d-%m-%Y').date()
    date_end = datetime.datetime.strptime(date_end,'%d-%m-%Y').date()

    if station == 'la_rumorosa':
        dir_name = 'M1'
        prefix = 'M01'
    if station == 'merida':
        dir_name = 'M2'
        prefix = 'M02'
    if station == 'ciudad_cuauhtemoc':
        dir_name = 'M3'
        prefix = 'M03'
    if station == 'certe':
        dir_name = 'M4'
        prefix = 'M04'
    if station == 'ojuelos':
        dir_name = 'M5'
        prefix = 'M05'
    if station == 'san_fernando':
        dir_name = 'M6'
        prefix = 'M06'
    if station == 'tepexi':
        dir_name = 'M7'
        prefix = 'M07'

    aem_dir = '/home/heyej/NAS/AEM/'+dir_name+'/'

    start_date_1 = date_bgn.replace(day=1)
    end_date_1 = date_end.replace(day=1)
    dates = [dt for dt in rrule(MONTHLY, dtstart=start_date_1, until=end_date_1)]

    df_list = [None] * len(dates)
    for i in range(len(dates)):
        file_name = prefix + '_' + dates[i].strftime("%Y") + dates[i].strftime("%m") + '.txt'
        path = aem_dir + file_name
        if os.path.isfile(path):
            df = pd.read_csv(path, skiprows=[1,2], sep=" ")
            df.insert(0,'Date',pd.DataFrame({'time': pd.to_datetime(dict(year=df.YYYY, month=df.MM, day=df.DD, hour=df.hh, minute=df.mm))}))
            df.set_index('Date',inplace=True)
            df = df[variables]
            df = df.loc[df.index.month == int(dates[i].strftime("%m"))]
            steps = pd.date_range(dates[i], dates[i]+pd.DateOffset(months=1), freq='10min', closed='left')
            df = df.reindex(steps, fill_value=np.NaN)
            df.index.name = 'Date'
            df_list[i] = df
        else:
            steps = pd.date_range(dates[i], dates[i]+pd.DateOffset(months=1), freq='10min', closed='left')
            nan_array = np.empty((len(steps),len(variables)))
            nan_array[:] = np.NaN
            df = pd.DataFrame(nan_array, columns=variables, index=steps)
            df.index.name = 'Date'
            df_list[i] = df
    df = pd.concat(df_list)
    date_end = datetime.datetime.combine(date_end, datetime.time(23, 50))
    df = df.loc[date_bgn : date_end.replace(hour=23, minute=50)]

    cube_list = [None] * df.shape[1]
    for i in range(df.shape[1]):
        df_cube = df.iloc[:,i]
        cube = iris.pandas.as_cube(df_cube)
        iris.coords.DimCoord.rename(cube.coord('index'),'time')
        cube.rename(df.columns[i])
        cube_list[i] = cube
    cube_list = iris.cube.CubeList(cube_list)

    return df, cube_list

def main(args):
    variables = ['WS_80mA_mean','WS_80mB_mean','WS_60m_mean','WS_40m_mean','WS_20m_mean','WD_78m_mean','WD_58m_mean','temp_80m_mean','temp_40m_mean','temp_15m_mean','RH_15m_mean','P_15m_mea','Rad_80m_mean']
    places = ['la_rumorosa','merida','ciudad_cuauhtemoc','certe','ojuelos','san_fernando','tepexi']

    txt = '"Extract data from Mexican Wind Atlas." \n Use {} to extract sevral variables, ie. {WS_80mA_mean,WD_78m_mean}'
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,description=txt,prog='gusiAEM')
    parser.add_argument('-c', '--csv', help='export to csv', action='store_true')
    parser.add_argument('-n', '--netcdf', help='export to netcdf', action='store_true')
    parser.add_argument('-fn', '--filename', help='name for output')

    req_group = parser.add_argument_group(title='required arguments')
    req_group.add_argument('-v', '--var', help='variables to extract', required=True, choices=variables, nargs='+')
    req_group.add_argument('-l', '--loc', help='name of station', required=True, choices=places)
    req_group.add_argument('-s', '--start', help='start date (dd-mm-yyyy)', required=True)
    req_group.add_argument('-e', '--end', help='end date (dd-mm-yyyy)', required=True)

    args = parser.parse_args()
    outputname = vars(args)['filename']
    variables = vars(args)['var']
    station = vars(args)['loc']
    date_bgn = vars(args)['start']
    date_end = vars(args)['end']

    print('\n << Reading station "'+station+'" from '+str(vars(args)['start'])+' to '+str(vars(args)['end'])+'... >>\n')
    df,cube_list = read(station,date_bgn,date_end,variables)

    if vars(args)['csv']:
        print(' << Exporting to "'+str(outputname)+'.csv"... >>\n')
        df.to_csv('%s' % (str(outputname)+'.csv'))

    if vars(args)['netcdf']:
        print(' << Exporting to "'+str(outputname)+'.nc"... >>\n')
        iris.save(cube_list, str(outputname)+'.nc', zlib=True, complevel=9, shuffle=True)
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
