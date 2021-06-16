#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl


def biasCorrect(observation,model):
	obs = observation
	mod = model

	ido = np.isfinite(obs)
	idm = np.isfinite(mod)

	#Quantiles
	perc = np.linspace(0,100,101)
	q_obs = np.percentile(obs[ido], perc, interpolation='linear')
	q_mod = np.percentile(mod[idm], perc, interpolation='linear')

	#Bias correction
	bc = q_mod-q_obs
	p = stats.rankdata(mod, method='average')/len(mod)
	f = interp1d(perc/100, bc)
	cor = f(p)
	mod_bc = mod-cor
	return mod_bc

def scatters(observation,model,plotdir,title,style,colormap):
	#Ignore nans
	idx = np.isfinite(observation)
	idy = np.isfinite(model)
	idxy = idx & idy
	xx = observation[idxy]
	yy = model[idxy]

	maxv = int(np.ceil(np.amax([np.amax(xx),np.amax(yy)])))
	x_id = np.linspace(0,maxv,maxv)

	#Quantiles
	perc = np.linspace(0,100,101)
	q_yy = np.percentile(yy, perc, interpolation='linear')
	q_xx = np.percentile(xx, perc, interpolation='linear')

	for i in range(2):
		if i == 0:
			prefix = 'scat_'
		if i == 1:
			prefix = 'scat_bc_'
			bc_data = biasCorrect(observation,model)
			yy = bc_data[idxy]

		#Linear Regression
		m,b = np.polyfit(xx,yy,1)
		y_lr = m*x_id + b
		#Pearson correlation coefficient
		r = stats.pearsonr(xx,yy)[0]

		#Probability density
		if style == 'points':
			xy = np.vstack([xx,yy])
			z = stats.gaussian_kde(xy)(xy)
			idd = z.argsort()
			x,y,z = xx[idd], yy[idd], z[idd]

		if style == 'contour1' or style == 'contour2' or style == 'map':
			xy = np.vstack([xx,yy])
			z = stats.gaussian_kde(xy)
			xgrid = np.linspace(0,maxv,100)
			ygrid = np.linspace(0,maxv,100)
			Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
			z = z.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

		maxx = round(0.005*math.ceil(float(np.amax(z.flatten()))/0.005),3)
		bounds = np.linspace(0.0001,maxx,20)
		norm = mpl.colors.BoundaryNorm(boundaries=bounds,ncolors=250)

		#Plot
		fig = plt.figure()
		axs = plt.axes()
		if style == 'points':
			nbc = axs.scatter(x,y,c=z,marker='.',alpha=0.8,s=100,edgecolor='',norm=norm,cmap=colormap)
			cbar = fig.colorbar(nbc, ax=axs, format='%.4f')
			cbar.set_label('Probability density estimate')
		if style == 'map':
			nbc = axs.imshow(z.reshape(Xgrid.shape),origin='lower',extent=[0,maxv,0,maxv],aspect='auto',cmap=colormap)
			cbar = fig.colorbar(nbc, ax=axs, format='%.4f')
			cbar.set_label('Probability')
		if style == 'contour1':
			nbc = axs.contourf(Xgrid,Ygrid,z.reshape(Xgrid.shape),bounds,origin='lower',extent=[0,maxv,0,maxv],cmap=colormap)
			cbar = fig.colorbar(nbc, ax=axs, format='%.4f')
			cbar.set_label('Probability density estimate')
		if style == 'contour2':
			axs.contour(Xgrid,Ygrid,z.reshape(Xgrid.shape),bounds,origin='lower',extent=[0,maxv,0,maxv],linewidths=0.1,colors='#083471',alpha=0.3)
			nbc = axs.contourf(Xgrid,Ygrid,z.reshape(Xgrid.shape),bounds,origin='lower',extent=[0,maxv,0,maxv],cmap=colormap)
			cbar = fig.colorbar(nbc, ax=axs, format='%.4f')
			cbar.set_label('Probability density estimate')

		axs.plot(x_id,x_id,label='y = x',color='black',linewidth=1)
		axs.plot(x_id,y_lr,color='red',label='y = '+str("%.2f"%m)+' x + '+str("%.2f"%b),linewidth=1)
		axs.plot(q_xx,q_yy,linestyle='--',linewidth=1,color='#083471',label='Q-Q')

		handles0,labels0= axs.get_legend_handles_labels()
		handles0.append(mpatches.Patch(color='none', label='r = '+str("%.3f" % r)))
		axs.legend(markerscale=0,handles=handles0)

		axs.set_aspect(1,adjustable='box')
		axs.set_xlim([0,maxv])
		axs.set_ylim([0,maxv])

		locs, labels = plt.yticks()
		plt.xticks(locs)

		plt.xlabel(xx.name)
		plt.ylabel(yy.name)

		plt.savefig(plotdir+'/'+prefix+str(title)+'.pdf',format='pdf',bbox_inches='tight')
		plt.close(fig)

def main(args):
    plotdir = 'Plots'
    wrf_file = 'Ojuelos/Ojuelos_WRF3_80_2018.csv'
    aem_file = '../datos_AEM/2018-082020/Ojuelos_AEM.csv'
    title = 'Ojuelos 2018'

	#YYYY,MM,DD,hh
    date_bgn = datetime.datetime(2018,1,1,0)
    date_end = datetime.datetime(2018,12,31,23)

	#Read WRF
    df_wrf = pd.read_csv(wrf_file,index_col='Date',parse_dates=True)
    df_wrf.index = df_wrf.index - pd.DateOffset(hours=6)        #Pasar WRF a hora local
    df_wrf = df_wrf.loc[date_bgn : date_end]

	#Read AEM
    df_aem = pd.read_csv(aem_file,index_col='Date',parse_dates=True)
    df_aem = df_aem.resample('1H').mean()
    df_aem = df_aem.loc[date_bgn : date_end]

	#Create dataframe
    df = df_wrf.join(df_aem)

    #Time series
    df.rename(columns = {'Wind speed':'WRF-80m', 'WS_80mA_mean':'AEM-80m'}, inplace=True)
    fig = plt.figure()
    df.plot(ylabel='Wind speed (ms-1)')
    plt.savefig(plotdir+'/ts_'+title+'.pdf',format='pdf',bbox_inches='tight')
    plt.close(fig)

	#Scatter plots
    df.rename(columns = {'WRF-80m':'Wind speed WRF-80m (m s-1)', 'AEM-80m':'Wind speed AEM-80m (m s-1)'}, inplace=True)
	##Styles: points, map, contour1, contour2
    scatters(df['Wind speed AEM-80m (m s-1)'],df['Wind speed WRF-80m (m s-1)'],plotdir,title,'contour1','ocean_r')
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
