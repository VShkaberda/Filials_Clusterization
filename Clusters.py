# -*- coding: utf-8 -*-
"""
Created on Wed Jan 04 17:59:02 2017
@author: Vadim Shkaberda
"""

from os import chdir
chdir('D:\Git\Filials_Clusterization')

from load_data import DBConnect
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import scale
from time import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# для корректного отображения кириллицы
font = {'family': 'Verdana',
        'weight': 'normal'}
rc('font', **font)

# Loading data
business = '' # business name

with DBConnect() as dbc:
    data = dbc.get_data(3, business)

    # Loading lists of id
    to_load = ('regions', 'filials')

    for cur_load in to_load:
        globals()['{}'.format(cur_load)] = dbc.get_id_lists(cur_load)

# number of monthes in data
feautures_num = data.shape[1] - 3 # number of ID after data import

#%%
# identify stable data
stable = np.ma.masked_equal(data[:, 2], 1)
data_scaled = scale(data[stable.mask, 3:], axis=1)
data_cleared = data[np.where(data[:, 2] == 1)]

# not used in current version
#data_scaled = data_scaled_all[stable.mask, :]

# staorage for data after PCA decomposition
reduced_data = {}


#%%

#################################################
#            Functions for plotting             #
#################################################

# plot functions aren't stored in separate module because they use global variables

def plot_clusters(n_clusters, reduced_data, labels, title, save_fig=False):
    ''' 2D plot of clustered data.
        Input:
        n_clusters: int - number of clusters;
        reduced_data: (x, 2) shape array - data to plot;
        labels: (x) shape array - list of clusters;
        title: string - title of plot;
        save_fig: boolean - save fig as png if True.
    '''
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.figure(1)
    plt.clf()

    plt.scatter(reduced_data[:-n_clusters, 0],
                reduced_data[:-n_clusters, 1],
                color=plt.cm.gist_ncar(labels[:-n_clusters] / float(n_clusters)),
                marker='o')

    # Plot the centroids as a white X
    #plt.scatter(reduced_data[-12:, 0], reduced_data[-12:, 1],
    #            marker='x', s=169, linewidths=2,
    #            color=plt.cm.spectral(labels[-12:] / 12.), zorder=10)

    # Plot the centroids as a numbers
    for i in range(reduced_data.shape[0]-n_clusters, reduced_data.shape[0]):
        plt.text(reduced_data[i, 0], reduced_data[i, 1],
                 str(labels[i]),
                 color=plt.cm.gist_ncar(labels[i] / float(n_clusters)),
                 fontdict={'weight': 'bold', 'size': 12})

    # Plot specific filial names/ID's
#    for r_data, filID, label in zip(reduced_data, filIDs, labels):
#        if label == 3:
#            plt.text(r_data[0]+0.05, r_data[1]+0.1, filials[int(filID)],
#                         #color=plt.cm.spectral(labels[i] / float(n_clusters)),
#                         fontdict={'weight': 'bold', 'size': 8})

    plt.title(title + u' кластеризация\n' + \
              u'Цифры являются центроидами\n'
              u'Кол-во кластеров: {}'.format(n_clusters))

    plt.show()
    if save_fig:
        plt.savefig('KMeans_{}_PCA_scaled.png'.format(clusters))
    plt.close(fig)


def plot_clusters_3D(n_clusters, reduced_data, labels, title, region=None):
    ''' 3D plot of clustered data.
        Input:
        n_clusters: int - number of clusters;
        reduced_data: (x, 3) shape array - data to plot;
        labels: (x) shape array - list of clusters;
        title: string - title of plot;
        region: int - region to plot, uses array "outliers". If mentioned, wull
            be plotted data with mask outliers[region], otherwise -
            all data from reduced_data will be plotted.
    '''
    # 3-d plot
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.clf()

    # angle from which we will be looking at plot
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    #ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=4, azim=54)
    #ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=38, azim=-74)

    if region:
        ax.scatter(reduced_data[outliers[region].mask][:, 0],
                   reduced_data[outliers[region].mask][:, 1],
                   reduced_data[outliers[region].mask][:, 2],
                   c=plt.cm.gist_ncar(labels[:-n_clusters] / float(n_clusters))
                   )

        ax.scatter(reduced_data[~outliers[region].mask][:, 0],
                   reduced_data[~outliers[region].mask][:, 1],
                   reduced_data[~outliers[region].mask][:, 2],
                   c='k', marker='x')
    else:
        ax.scatter(reduced_data[outliers.mask][:, 0],
                   reduced_data[outliers.mask][:, 1],
                   reduced_data[outliers.mask][:, 2],
                   c=plt.cm.gist_ncar(labels[:-n_clusters] / float(n_clusters))
                   )

        ax.scatter(reduced_data[~outliers.mask][:, 0],
                   reduced_data[~outliers.mask][:, 1],
                   reduced_data[~outliers.mask][:, 2],
                   c='k', marker='x')

    plt.title(title + u' кластеризация\n' + \
              u'Цифры являются центроидами\n'
              u'Кол-во кластеров: {}'.format(n_clusters))

    plt.show()
    plt.close(fig)


def plot_rc_bindings():
    ''' Function to plot all filials with color, respectively to binded RC.
    '''
    LABEL_COLOR_MAP = {1 : 'red',
                       2 : 'blue',
                       3 : 'green',
                       4 : 'purple',
                       5 : 'orange'}

    label_color = [LABEL_COLOR_MAP[l] for l in data_cleared[:, 1]]

    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.scatter(reduced_data[0][:, 0], reduced_data[0][:, 1], s=5, color=label_color)

    # labels for lowest 6 filials
    #lowest = np.where(reduced_data[:, 1]<-3.5)
    #for low in lowest[0]:
    #    skew = 0.05 if data_cleared[low, 0] != 518 else -0.25
    #    plt.text(reduced_data[low, 0]+skew, reduced_data[low, 1]+0.05, str(int(data_cleared[low, 0])),
    #                 #color=plt.cm.spectral(labels[i] / float(n_clusters)),
    #                 fontdict={'weight': 'bold', 'size': 10})

    # Legend
    classes = []
    recs = []
    for i in LABEL_COLOR_MAP.keys():
        recs.append(mpatches.Rectangle((0,0),1,1,fc=LABEL_COLOR_MAP[i]))
        classes.append(regions[i])
    plt.legend(recs, classes, loc=4)

    plt.title(u'Распределение филиалов по привязкам к РЦ.\n'
            u'Регионы (Бизнес --)')
    plt.show()
    plt.close(fig)


# plot data with outliers
def plot_with_outliers(reduced_data, outliers):
    ''' Plot reduced data with outliers.
        Input:
        reduced_data: (x, 2) shape array - data to plot;
        outliers: (x, 1) mask array - bit array of ouliers.
    '''
    fig = plt.gcf()
    fig.set_size_inches(15, 10)


    xy = reduced_data[0][outliers.mask]
    fils, = plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor='g',
                 markeredgecolor='k', markersize=16, label=u'Филиалы')

    xy = reduced_data[0][~outliers.mask]
    fils_out, = plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor='k',
                 markeredgecolor='k', markersize=10, label=u'Исключения')

    # labels for outliers
    for outlier, filID in zip(xy, data_cleared[~outliers.mask][:, 0]):
        #skew = 0.05 if data_cleared[low, 0] != 518 else -0.25
        plt.text(outlier[0]+0.05, outlier[1]+0.1, filials[int(filID)],
                     #color=plt.cm.spectral(labels[i] / float(n_clusters)),
                     fontdict={'weight': 'bold', 'size': 10})

    plt.legend(handles=(fils, fils_out), loc='lower right')

    plt.title(u'Филиалы-outliers, не подлежащие дальнейшей кластеризации\n'
            u'(Бизнес {})'.format(business))

    plt.show()
    plt.close(fig)


def plot_cluster_versions(cluster_data, region=None, dim=2):
    ''' Plots clusters, PCA reduced to dim dimentions.
        Input:
        cluster_data - tuple: tuple of (number of clusters, seed) tuples;
        region - int: None if i-th tuple in cluster_data corresponds to region,
            number = n - if all data in cluster_data are from the n-th cluster;
        dim - int: 2 or 3 - dimension of plot.
    '''
    for i, cdata in enumerate(cluster_data, 1):
        if region:
            i = region
        data_scaled_ready = data_scaled[reg[i].mask, :][outliers[i].mask]
        #filIDs = data_cleared[reg[i].mask, :][outliers[i].mask][:, 0]
        n_clusters = cdata[0]

        kmeans[i] = KMeans(init='k-means++', n_clusters=n_clusters, n_init=100, random_state=cdata[1])
        kmeans[i].fit(data_scaled_ready) # reduced_data or data_scaled
        silhouette_avg = silhouette_score(data_scaled_ready, kmeans[i].labels_)
        print ("Region:{}. The average silhouette_score is :{}".format(regions[i], silhouette_avg))

        labels = np.hstack((kmeans[i].labels_, np.arange(n_clusters)))
        Title = 'Регион {0}. K-means'.format(regions[i])

        if dim == 2:
            reduced_data = PCA(n_components=dim).fit_transform(
                    np.vstack((data_scaled_ready, kmeans[i].cluster_centers_)))
            plot_clusters(n_clusters, reduced_data, labels, Title)

        if dim == 3:
            reduced_data = PCA(n_components=dim).fit_transform(data_scaled[reg[region].mask, :])
            plot_clusters_3D(n_clusters, reduced_data, labels, Title, region=i)

#%%

#################################################
# Part for clustering without region separation #
#################################################

# detecting outliers - filials far away, which have less than 2 neighbors
db = DBSCAN(eps=2.5, min_samples=3, algorithm='brute').fit(data_scaled)
outliers = np.ma.masked_not_equal(db.labels_, -1)

# for plotting outliers after DBSCAN
reduced_data[0] = PCA(n_components=2).fit_transform(data_scaled)

#%%

# plot data with outliers
plot_with_outliers(reduced_data, outliers)

#%%

# compute clusters
start_time = time()

data_scaled_ready = data_scaled[outliers.mask]
cluster_limit = 12

silhouettes = np.zeros((cluster_limit - 2, 4))
clusters = range(2, cluster_limit)

for cluster in clusters:
    scores = {}
    seeds =  np.random.randint(1, 100000 + 1, size=100)
    for seed in seeds:
        kmeans = KMeans(init='k-means++', n_clusters=cluster, n_init=100, random_state=seed)
        kmeans.fit(data_scaled_ready) # reduced_data or data_scaled
        silhouette_avg = silhouette_score(data_scaled_ready, kmeans.labels_)
        scores[seed] = (silhouette_avg, kmeans.inertia_)
        #print seed, silhouette_avg

    best_silhouette = max(scores, key=scores.get)
    silhouettes[cluster-2] = [cluster] + [best_silhouette] + list(scores[max(scores, key=scores.get)])
    print('Завершено: {} кластеров'.format(cluster))

print("--- %f seconds ---" % (time() - start_time))


#%%

# plot silhouette and inertia
fig = plt.gcf()
fig.set_size_inches(10, 7.5)

plt.plot(clusters, silhouettes[:, 2])

plt.title(u'Показатель silhouette для алгоритма K-means\n'
        u'Бизнес {}, с исключением outliers'.format(business))

plt.xlabel(u'Количество кластеров')
plt.ylabel(u'Значение Silhouette')

plt.show()
plt.close(fig)

fig = plt.gcf()
fig.set_size_inches(10, 7.5)

plt.plot(clusters, silhouettes[:, 3], marker='s')

plt.title(u'Показатель inertia для алгоритма K-means\n'
        u'Бизнес {}, с исключением outliers'.format(business))

plt.xlabel(u'Количество кластеров, n')
plt.ylabel(u'Inertia $J(C_n)$')

plt.show()
plt.close(fig)


#%%

def plot_cluster_one_region(n_clusters, seed, dim=2):

    global kmeans

    data_scaled_ready = data_scaled[outliers.mask]

    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=100, random_state=seed)
    kmeans.fit(data_scaled_ready) # reduced_data or data_scaled
    silhouette_avg = silhouette_score(data_scaled_ready, kmeans.labels_)
    print ("The average silhouette_score is :", silhouette_avg)

    labels = np.hstack((kmeans.labels_, np.arange(n_clusters)))
    Title = u'Бизнес {}. K-means'.format(business)

    if dim == 2:
        reduced_data = PCA(n_components=dim).fit_transform(np.vstack((data_scaled_ready, kmeans.cluster_centers_)))
        plot_clusters(n_clusters, reduced_data, labels, Title)

    if dim == 3:
        reduced_data = PCA(n_components=dim).fit_transform(data_scaled)
        plot_clusters_3D(n_clusters, reduced_data, labels, Title)

# storage for current run after choosing appropriate seed and clustersnumber

# data from silhouettes (T)
#plot_cluster_one_region(n_clusters=2, seed=86905, dim=2)

# data from silhouettes (T)
#plot_cluster_one_region(n_clusters=3, seed=39572, dim=2)

# data from silhouettes (F)
plot_cluster_one_region(n_clusters=4, seed=97937, dim=2)

#%%
''' Plot all centroids.
'''

x_plot = np.arange(1, feautures_num+1)
lw = 2

fig = plt.gcf()
fig.set_size_inches(15, 10)

for clust_num in range(max(kmeans.labels_)+1):
    plt.plot(x_plot, kmeans.cluster_centers_[clust_num], linewidth=lw, label='Cluster {}'.format(clust_num))

Title = 'Динамика центроидов для бизнеcа {}'.format(business)

plt.legend()
plt.title(Title)
#plt.show()
plt.savefig('Centroids_{}.png'.format(business))
plt.close(fig)

#%%

u_clusters = {}


u_clusters[0] = np.zeros((max(kmeans.labels_) + 1, feautures_num))
for j in range((max(kmeans.labels_) + 1)):
    clust_mask = np.ma.masked_equal(kmeans.labels_, j)
    u_clusters[0][j] = np.average(data_cleared[outliers.mask][clust_mask.mask], axis=0)[3:]

# Array to store cluters of unstable filials
unstable_clusters = np.zeros(stable.count())

for i, fil in enumerate(data[~stable.mask, :]):
    # Mask for stable months
    fil_mask = np.ma.masked_not_equal(fil[3:], 0)
    # In case of NaN instead of zeros use next mask for stable months + invert
    #fil_mask = np.ma.masked_invalid(fil[3:])
    #fil_mask.mask = np.invert(fil_mask.mask)
    # Scaled data of stable months for filial
    fil_stable = scale(fil[3:][fil_mask.mask])
    # Creating scaled clusters
    scaled_clusters = scale(u_clusters[0][:, fil_mask.mask], axis=1)
    # Calculating the MSE from unstable filial to clusters
    dist = np.sum((scaled_clusters - fil_stable) ** 2, axis=1)
    # Save the nearest cluster into storage
    unstable_clusters[i] = np.argmin(dist)


#%%

# Writing final data (in brackets - number of clusters according to the last run)
file_to_write = 'output_Kmeans_m3_' + business + '_2019.csv'

with open(file_to_write, 'w') as f:
    f.write('FilID;FilialName;MacroRegionName;Stable;Outlier;Cluster;ClusterID\n')


#%%

# Stable filials and outliers
data_scaled_ready = data_scaled[~outliers.mask]

with open(file_to_write, 'a') as f:
    for dat, label in zip(data_cleared[outliers.mask], kmeans.labels_):
        f.write("{0:d};{1};{2};{3:n};0;{5} {4:d};{4:d}\n".format(int(dat[0]),
                                            filials[int(dat[0])],
                                            regions[int(dat[1])],
                                            dat[2],
                                            label,
                                            business))
    if not all(outliers.mask == True):
        Z = kmeans.predict(data_scaled_ready)
        for dat, label in zip(data_cleared[~outliers.mask], Z):
            f.write("{0:d};{1};{2};{3:n};1;{5} {4:d};{4:d}\n".format(int(dat[0]),
                                                filials[int(dat[0])],
                                                regions[int(dat[1])],
                                                dat[2],
                                                label,
                                                business))

#with open('output_Kmeans_reg_5_6_cleared_centroids.csv', 'w') as f:
#    for label, centroid in zip(np.arange(n_clusters), kmeans.cluster_centers_):
#        f.write(("{}"+15*";{}"+"\n").format(label, *centroid.tolist()))


#%%

# Unstable filials
with open(file_to_write, 'a') as f:
    for dat, label in zip(data[~stable.mask], unstable_clusters):
        f.write("{0:d};{1};{2};{3:n};0;{5} {4:n};{4:n}\n".format(int(dat[0]),
                                        filials[int(dat[0])],
                                        regions[int(dat[1])],
                                        dat[2],
                                        label,
                                        business))

#%%

''' Distinct Filials after K-means.
'''

label = 2

new_mask = np.ma.masked_equal(kmeans.labels_, label)
X = data_scaled[outliers.mask][new_mask.mask, :]

x_plot = np.arange(1, feautures_num+1)
lw = 2

for row, filid in zip(X, data_cleared[outliers.mask][new_mask.mask, 0]):

    fig = plt.gcf()

    plt.scatter(x_plot, row, color='navy', s=30, marker='o', label="training points")
    plt.plot(x_plot, kmeans.cluster_centers_[label], color='g', linewidth=lw)

    plt.title("Filial {}, label {}".format(int(filid), label))
#    plt.title("Total, label {1}".format(int(filid), label))
#    plt.show()
    plt.savefig('{}_{}.png'.format(label, int(filid)))
    plt.close(fig)

#%%

''' Outliers after K-means.
'''

data_scaled_ready = data_scaled[~outliers.mask]
Z = kmeans.predict(data_scaled_ready)

x_plot = np.arange(1, feautures_num+1)
lw = 2

for row, filid in zip(range(len(data_scaled_ready)), data_cleared[~outliers.mask, 0]):

    fig = plt.gcf()

    plt.scatter(x_plot, data_scaled_ready[row], color='navy', s=30, marker='o', label="training points")
    plt.plot(x_plot, kmeans.cluster_centers_[Z[row]], color='g', linewidth=lw)

    Title = 'FilID {0}, {1}. Label {2}'.format(int(filid), filials[int(filid)], Z[row])

    plt.title(Title)
    #plt.show()
    plt.savefig('{}_{}.png'.format(Z[row], int(filid)))
    plt.close(fig)

#%%

#################################################
# Part for clustering with region separation    #
#################################################

# Region bindings
reg = {}

for i in range(1, 6):
    reg[i] = np.ma.masked_equal(data_cleared[:, 1], i)
    reduced_data[i] = PCA(n_components=2).fit_transform(data_scaled[reg[i].mask, :])

#%%

# plot initial data, divided by regions
fig, axarr = plt.subplots(3, 2)
fig.set_size_inches(18, 12)
subplots = (((0, 0)), ((0, 1)), ((1, 0)), ((1, 1)), ((2, 0)))

for i, sp in enumerate(subplots, 1):
    axarr[sp].scatter(reduced_data[i][:, 0], reduced_data[i][:, 1])
    axarr[sp].set_title(u'Бизнес {1}, регион {0}'.format(regions[i], business))

# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
#plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
#plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

plt.show()

plt.close(fig)

#%%

# detecting outliers - filials far away, which have less than 2 neighbors
outliers = {}

for i in range(1, 6):
#    if i == 2:
#        eps = 3.
#    if i == 5:
#        eps = 2.25
#    else:
#        eps = 2.75
    eps = 2.7 if not (i == 5) else 2.4
    db = DBSCAN(eps=eps, min_samples=3, algorithm='brute').fit(data_scaled[reg[i].mask, :])
    outliers[i] = np.ma.masked_not_equal(db.labels_, -1)

outliers[1].mask[np.where(data_cleared[reg[1].mask, 0] == 2156)] = True
outliers[2].mask[np.where(data_cleared[reg[2].mask, 0] == 2218)] = True
outliers[2].mask[np.where(data_cleared[reg[2].mask, 0] == 2238)] = True
outliers[2].mask[np.where(data_cleared[reg[2].mask, 0] == 2248)] = True

#%%

# plot outliers
fig, axarr = plt.subplots(3, 2)
fig.set_size_inches(18, 12)

for i, sp in enumerate(subplots, 1):
    xy = reduced_data[i][outliers[i].mask]
    axarr[sp].plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor='g',
                 markeredgecolor='k', markersize=16)
    xy = reduced_data[i][~outliers[i].mask]
    axarr[sp].plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor='k',
                 markeredgecolor='k', markersize=10)
    for outlier, filID in zip(xy, data_cleared[reg[i].mask, :][~outliers[i].mask][:, 0]):
        skew = -0.35 if int(filID) == 2078 else -0.1 if int(filID) == 2072 else 0.1
        axarr[sp].text(outlier[0]+0.05, outlier[1]+skew, filials[int(filID)],
                     #color=plt.cm.spectral(labels[i] / float(n_clusters)),
                     fontdict={'weight': 'bold', 'size': 8})

    axarr[sp].set_title(u'Бизнес {1}, регион {0}'.format(regions[i], business))

# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
#plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
#plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

plt.show()

plt.close(fig)

#%%

# parameters for clusterization
region = 1
cluster_limit = 12

# compute clusters
start_time = time()

data_scaled_ready = data_scaled[reg[region].mask, :][outliers[region].mask]

silhouettes = np.zeros((cluster_limit - 2, 4))
clusters = range(2, cluster_limit)
for cluster in clusters:
    scores = {}
    seeds =  np.random.randint(1, 10000 + 1, size=150)
    for seed in seeds:
        kmeans = KMeans(init='k-means++', n_clusters=cluster, n_init=100, random_state=seed)
        kmeans.fit(data_scaled_ready) # reduced_data or data_scaled
        silhouette_avg = silhouette_score(data_scaled_ready, kmeans.labels_)
        scores[seed] = (silhouette_avg, kmeans.inertia_)

    best_silhouette = max(scores, key=scores.get)
    silhouettes[cluster-2] = [cluster] + [best_silhouette] + list(scores[max(scores, key=scores.get)])
    print('Завершено: {} кластеров'.format(cluster))

print("--- %f seconds ---" % (time() - start_time))

#kmeans = KMeans(init='k-means++', n_clusters=12, n_init=20, random_state=15)
#kmeans.fit(data_scaled) # reduced_data or data_scaled
#silhouette_avg = silhouette_score(data_scaled, kmeans.labels_)
#print ("The average silhouette_score is :", silhouette_avg)


#%%

# plot silhouette and inertia
fig = plt.gcf()
fig.set_size_inches(10, 7.5)

plt.plot(clusters, silhouettes[:, 2])

plt.title(u'Показатель silhouette для алгоритма K-means\n'
        u'Бизнес {1}, Регион {0} с исключением outliers'.format(regions[region], business))

plt.xlabel(u'Количество кластеров')
plt.ylabel(u'Значение Silhouette')

plt.show()
plt.close(fig)

fig = plt.gcf()
fig.set_size_inches(10, 7.5)

plt.plot(clusters, silhouettes[:, 3], marker='s')

plt.title(u'Показатель inertia для алгоритма K-means\n'
        u'Бизнес {1}, Регион {0} с исключением outliers'.format(regions[region], business))

plt.xlabel(u'Количество кластеров, n')
plt.ylabel(u'Inertia $J(C_n)$')

plt.show()
plt.close(fig)

#%%

# storage for current run after choosing appropriate seed and clustersnumber
# cluster_data[number of clusters, seed] - order of final decision is Region order
cluster_data = ((6, 1900),)

# Final data M3 V1, V2
#cluster_data = ((3, 9196), (3, 2697), (3, 6124), (3, 7067), (5, 5630))
cluster_data = ((3, 9196), (4, 9931), (5, 4075), (3, 7067), (6, 1900))

# storage of the trained K-means
kmeans = {}

# disable region for the final run
plot_cluster_versions(cluster_data, region=None, dim=2)

#%%

#################################################
# Calculating clusters for unstable filials     #
#################################################

u_clusters = {}

# for each region
for i in range(1, 6):
    u_clusters[i] = np.zeros((max(kmeans[i].labels_) + 1, feautures_num))
    for j in range((max(kmeans[i].labels_) + 1)):
        # mask of cluster j in region i
        clust_mask = np.ma.masked_equal(kmeans[i].labels_, j)
        # calculate centroid of cluster j in region i
        u_clusters[i][j] = np.average(data_cleared[reg[i].mask, :][outliers[i].mask][clust_mask.mask], axis=0)[3:]

#%%

# Array to store cluters of unstable filials
unstable_clusters = np.zeros(stable.count())

for i, fil in enumerate(data[~stable.mask, :]):
    # Mask for stable months
    fil_mask = np.ma.masked_not_equal(fil[3:], 0)
    # In case of NaN instead of zeros use next mask for stable months + invert
    #fil_mask = np.ma.masked_invalid(fil[3:])
    #fil_mask.mask = np.invert(fil_mask.mask)
    # Scaled data of stable months for filial
    fil_stable = scale(fil[3:][fil_mask.mask])
    # Creating scaled clusters
    scaled_clusters = scale(u_clusters[fil[1]][:, fil_mask.mask], axis=1)
    # Calculating the MSE from unstable filial to clusters
    dist = np.sum((scaled_clusters - fil_stable) ** 2, axis=1)
    # Save the nearest cluster into storage
    unstable_clusters[i] = np.argmin(dist)



#%%

# Writing final data (in brackets - number of clusters according to the last run)
file_to_write = ('output_Kmeans_m3_' + business +
                 '_2019{}.csv'.format(tuple(i[0] for i in cluster_data))
                 )

with open(file_to_write, 'w') as f:
    f.write('FilID;FilialName;MacroRegionName;Stable;Outlier;Cluster;ClusterID\n')


#%%

# Stable filials and outliers
with open(file_to_write, 'a') as f:
    for i in range(1, 6):
        data_scaled_ready = data_scaled[reg[i].mask, :][~outliers[i].mask]

        for dat, label in zip(data_cleared[reg[i].mask, :][outliers[i].mask], kmeans[i].labels_):
            f.write("{0:d};{1};{2};{3:n};0;{2} {4:d};{4:d}\n".format(int(dat[0]),
                                            filials[int(dat[0])],
                                            regions[int(dat[1])],
                                            dat[2],
                                            label,
                                            business))
        # Check if there exist outliers
        if not all(outliers[i].mask == True):
            Z = kmeans[i].predict(data_scaled_ready)
            for dat, label in zip(data_cleared[reg[i].mask, :][~outliers[i].mask], Z):
                f.write("{0:d};{1};{2};{3:n};1;{2} {4:d};{4:d}\n".format(int(dat[0]),
                                                filials[int(dat[0])],
                                                regions[int(dat[1])],
                                                dat[2],
                                                label,
                                                business))

#with open('output_Kmeans_reg_5_6_cleared_centroids.csv', 'w') as f:
#    for label, centroid in zip(np.arange(n_clusters), kmeans.cluster_centers_):
#        f.write(("{}"+15*";{}"+"\n").format(label, *centroid.tolist()))


#%%

# Unstable filials
with open(file_to_write, 'a') as f:
    for dat, label in zip(data[~stable.mask], unstable_clusters):
        f.write("{0:d};{1};{2};{3:n};0;{2} {4:n};{4:n}\n".format(int(dat[0]),
                                        filials[int(dat[0])],
                                        regions[int(dat[1])],
                                        dat[2],
                                        label,
                                        business))

