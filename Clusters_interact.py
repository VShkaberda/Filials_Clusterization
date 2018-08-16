# -*- coding: utf-8 -*-
"""
Created on Wed Jan 04 17:59:02 2017
@author: Vadim Shkaberda
"""

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
business = 'Сильпо' # business name

with DBConnect() as dbc:
    data = dbc.get_data(3, business)

    # Loading lists of id
    to_load = ('regions', 'filials')

    for cur_load in to_load:
        globals()['{}'.format(cur_load)] = dbc.get_id_lists(cur_load)

#%%

def plot_clusters(n_clusters, reduced_data, labels, title):

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
    #plt.savefig('KMeans_{}_PCA_scaled.png'.format(clusters))
    plt.close(fig)


def plot_clusters_3D(n_clusters, outliers, reduced_data, labels, title, region=None):

    # 3-d plot
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.clf()

    #ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    #ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=4, azim=54)
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=38, azim=-74)

    if region:
        ax.scatter(reduced_data[outliers[region].mask][:, 0],
                   reduced_data[outliers[region].mask][:, 1],
                   reduced_data[outliers[region].mask][:, 2],
                   c=plt.cm.gist_ncar(labels[:-n_clusters] / float(n_clusters))
                   #c=kmeans.labels_.astype(np.float)
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
                   #c=kmeans.labels_.astype(np.float)
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

#%%
# identify satable data
stable = np.ma.masked_equal(data[:, 2], 1)
data_scaled = scale(data[stable.mask, 3:], axis=1)
data_cleared = data[np.where(data[:, 2] == 1)]

# not used at this time
#data_diff = np.diff(data_cleared[:, 2:-2]) / data_cleared[:, 2:-3]

reduced_data = {}

#%%

#################################################
# Part for clustering without region separation #
#################################################

def clustering_one_region(cluster_data):

    # detecting filials far away
    db = DBSCAN(eps=2.5, min_samples=3, algorithm='brute').fit(data_scaled)
    outliers = np.ma.masked_not_equal(db.labels_, -1)

#%%
    # for plotting outliers after DBSCAN
    reduced_data[0] = PCA(n_components=2).fit_transform(data_scaled)

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

    plt.legend(handles=(fils, fils_out))

    plt.title(u'Филиалы-outliers, не подлежащие дальнейшей кластеризации\n'
            u'(Бизнес {})'.format(business))

    plt.show()
    plt.close(fig)

    # data from silhouettes
    plot_cluster_one_region(n_clusters=cluster_data[0],
                            seed=cluster_data[1],
                            outliers=outliers,
                            dim=3)


    #%%


def plot_cluster_one_region(n_clusters, seed, outliers, dim=2):

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
        plot_clusters_3D(n_clusters, outliers, reduced_data, labels, Title)


#%%

#################################################
# Part for clustering with region separation    #
#################################################

def clustering_with_region_separation(cluster_data, region):
# Reg bindings

    reg = {}

    for i in range(1, 6):
        reg[i] = np.ma.masked_equal(data_cleared[:, 1], i)
        reduced_data[i] = PCA(n_components=2).fit_transform(data_scaled[reg[i].mask, :])

    #%%

    fig, axarr = plt.subplots(3, 2)
    fig.set_size_inches(18, 12)
    subplots = (((0, 0)), ((0, 1)), ((1, 0)), ((1, 1)), ((2, 0)))

    for i, sp in enumerate(subplots, 1):
        axarr[sp].scatter(reduced_data[i][:, 0], reduced_data[i][:, 1])
        axarr[sp].set_title(u'Бизнес Сильпо, регион {}'.format(regions[i]))

    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    #plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    #plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    plt.show()

    plt.close(fig)

    #%%

    outliers = {}

    for i in range(1, 6):
    #    if i == 2 or i == 4:
    #        eps = 2.75
    #    else:
        eps = 2.25 if not (i == 4) else 2.4
        db = DBSCAN(eps=eps, min_samples=3, algorithm='brute').fit(data_scaled[reg[i].mask, :])
        outliers[i] = np.ma.masked_not_equal(db.labels_, -1)

    outliers[4].mask[np.where(data_cleared[reg[4].mask, 0] == 2272)] = False
    outliers[4].mask[np.where(data_cleared[reg[4].mask, 0] == 2273)] = False
    outliers[4].mask[np.where(data_cleared[reg[4].mask, 0] == 2280)] = False

    #%%


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
            skew = -0.35 if int(filID) == 2059 else 0.1
            axarr[sp].text(outlier[0]+0.05, outlier[1]+skew, filials[int(filID)],
                         #color=plt.cm.spectral(labels[i] / float(n_clusters)),
                         fontdict={'weight': 'bold', 'size': 8})

        axarr[sp].set_title(u'Бизнес Сильпо, регион {}'.format(regions[i]))

    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    #plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    #plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    plt.show()

    plt.close(fig)

    plot_cluster_versions(cluster_data, outliers, reg, region, dim=3)

#%%

def plot_cluster_versions(cluster_data, outliers, reg, region=None, dim=2):
    ''' Plots clusters, PCA reduced to dim dimentions.
        Input:
        cluster_data - tuple: tuple of (number of clusters, seed) tuples;
        region - int: None if i-th tuple in cluster_data corresponds to region,
            number = n - if all data in cluster_data are from the n-th cluster;
        dim - int: 2 or 3 - dimension of plot.
    '''
    kmeans = {}
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
            plot_clusters_3D(n_clusters, outliers, reduced_data, labels, Title, region=i)


#%%

if __name__ == '__main__':
    # cluster_data[number of clusters, seed] - order of final decision is Region order
    cluster_data = ((3, 5532),)
    clustering_with_region_separation(cluster_data, region=4)

    # cluster_data[number of clusters, seed] - order of final decision is Region order
    #cluster_data = (3, 81829)
    #clustering_one_region(cluster_data)