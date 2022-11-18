# Author: Giuseppe P Gava, 11/2022

# Libraries import
import numpy as np
import scipy.linalg as spl
from scipy.ndimage.filters import gaussian_filter1d
import scipy.signal as sig
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Union, Any


# Loading data functions


def loadStages(b):
    """
    Load "stages" information (mostly from desen- and resofs-file).
    INPUT:
    - [b]:       <str> containing "block base"
    OUTPUT:
    - [stages]:  <DataFrame>"""

    # Read desen- and resofs-file
    stages = pd.read_csv(b + '.desen', header=None, names=['desen'])
    resofs = pd.read_csv(b + '.resofs', header=None)
    # Add start- and end-time and filebase of each session
    stages['start_t'] = [0] + list(resofs.squeeze().values)[:-1]
    stages['end_t'] = resofs
    # Let the index of this dataframe start from 1 instead of 0
    stages.index += 1
    return stages


def loadUnits(b):
    """Load "units" information
    INPUT:
    - [b]:       <str> containing "block base"
    OUTPUT:
    - [trodes]:  <DataFrame>"""

    units = pd.read_csv(b + '.des', header=None, names=['des'])
    # the index of the units dataframe start from 2(!) instead of 0
    units.index += 2
    return units


def loadTracking(b, smoothing=1, ext='whl'):
    """Load position data (whl)"""
    trk = pd.read_csv(b + '.' + ext, sep='\s+', header=None).values
    trk[trk <= 0] = np.nan
    if smoothing is not None:
        trk = gaussian_filter1d(trk, smoothing, axis=0)
    return pd.DataFrame(trk, columns=['x', 'y'])


def trackSpeed(track):
    """obtain speed from track coordinates"""
    KEYS = track.columns
    vx = np.diff(track[KEYS[0]])
    vy = np.diff(track[KEYS[1]])
    return np.sqrt(vx ** 2 + vy ** 2)


def getActiveTrack(track, thrV):
    """get active samples indexes of active track
    - track is pd.Dataframe with x and y coordinates
    - thrV is the velocity threshold used to detect active times (pixels/samples)"""
    v = trackSpeed(track)
    ind1 = v > thrV
    ind2 = ~np.isnan(v)
    # add one sample at the beginning to match track
    return np.hstack((False, ind1 & ind2))


def loadSpikeTimes(b, minClu=2, res2eeg=(1250. / 20000)):
    """Load spike times information
    INPUT:
    - [b]:       <str> containing "block base"
    - minClu:  <int> from which to consider clusters
    - res2eeg:   <float> conversion rate from ephys to lfp sampling
    OUTPUT:
    - res:  <DataFrame> with all spike times
    - clu:  <DataFrame> with the cluster ID to which each spike time in `res` belongs"""

    res = pd.read_csv(b + '.res', header=None, squeeze=True).values
    clu = pd.read_csv(b + '.clu', squeeze=True).values
    if minClu is not None:
        mask = clu >= minClu
        clu = clu[mask]

        res = res[mask]
    res = np.round(res * res2eeg).astype(int)
    return res, clu


def bin_spikes(spiketrains, edges):
    num_bins = len(edges) - 1  # Number of bins
    num_neurons = spiketrains.shape[0]  # Number of neurons
    actmat = np.empty([num_bins, num_neurons])  # Initialize array for binned neural data
    # Count number of spikes in each bin for each neuron, and put in array
    for i in range(num_neurons):
        actmat[:, i] = np.histogram(spiketrains[i], edges)[0]
    return actmat


def matGaussianSmooth(mat, sigma, nPoints=0, normOperator=np.sum):
    """Smooth matrix row-wise with a Gaussian kernel
    INPUT:
    - mat: matrix to smooth (rows will be smoothed)
    - sigma: standard deviation of Gaussian kernel (unit has to be number of samples)
    - nPoints: number of points of kernel
    - normOperator: # defines how to normalise kernel
    OUTPUT:
    - smoothMat: smoothed matrix
    - kernel: kernel used
    """
    if nPoints < sigma:
        nPoints = int(4 * sigma)
    # define smoothing gaussian kernel
    kernel = sig.get_window(('gaussian', sigma), nPoints)
    kernel = kernel / normOperator(kernel)
    # apply smoothing
    smoothMat = np.ones(np.shape(mat)) * np.nan
    for row_i in range(len(mat)):
        smoothMat[row_i, :] = np.convolve(mat[row_i, :], kernel, 'same')
    return smoothMat, kernel


def get_actmat_the(actmat, theta):
    """output theta-binned activity matrix
    INPUT:
    - actmat: activity matrix sampled @ theta rate
    - theta: matrix containing theta-cycles timestamps (nCycles x 6)
    OUTPUT:
    - actmat_the: theta-binned activity matrix (sum spikes)"""
    num_bins = theta.shape[0]  # Number of bins
    num_neurons = actmat.shape[1]  # Number of neurons
    actmat_the = np.empty([num_bins, num_neurons])  # Initialize array for binned neural data
    for nt, t in enumerate(theta):
        actmat_the[nt] = actmat[t[0]:t[-1], :].sum(0)
    return actmat_the


# Neurons' activity functions

def getISI(spikeTrains, sampT=50):
    """INPUT:
    # spikeTrains: list of lists containing the raw spike times of the neurons (tetrode sampled)
    # sampT: sampling period in us of the spike times, default is 50
    # OUTPUT:
    # ISI: list of ISIs in ms
    """
    ISI = []; ID = []
    for i in range(len(spikeTrains)):
        if spikeTrains[i].shape[0]>1: # consider only spiketrains > 1 spike
            ID.append(i)
            ISI.append(np.zeros((spikeTrains[i].shape[0] - 1)))
        else: # there's no isi if neuron fired only one spike
            ISI.append(np.nan)
    # ISI is derivative (diff operator) of spike times
    for i in ID:
        ISI[i] = np.diff(spikeTrains[i]) * (sampT*1e-6)
    return ISI


def getPlaceMap(track, spikes, active, nE=21, spks2tracking=1/32,
                    mazeDim=37, smoothStdCm=2):
    """obtain the placemap for one cell
    """
    # convert pixels to cm
    track2cm = (np.max(track['y']) - np.min(track['y'])) / mazeDim  # from pixels to cm
    smoothStdPixels = smoothStdCm * track2cm # smoothing parameter
    # obtain bin edges in pixels
    bin2pixels = (nE - 1) / (np.max(track['y']) - np.min(track['y']))  # no pixels for 1 bin
    Xedges = np.linspace(np.min(track['x']), np.max(track['x']), nE)
    Yedges = np.linspace(np.min(track['y']), np.max(track['y']), nE)
    # obtain occupancy map
    totalPos = active
    y = track['y'][totalPos]
    y = y[~np.isnan(y)]
    x = track['x'][totalPos]
    x = x[~np.isnan(x)]
    y = y[~y.index.duplicated(keep='first')]
    x = x[~x.index.duplicated(keep='first')]
    OccMap, _, _ = np.histogram2d(x, y, [Xedges, Yedges])
    mask = (OccMap) > 0
    OccMap = (OccMap) / (spks2tracking * 1250.)  # type: Union[float, Any]
    # obtain the spike count map
    validSpikes = np.in1d(np.round(spikes * spks2tracking).astype(int), np.where(active)[0])
    spikesPos = np.round(spikes[validSpikes] * spks2tracking).astype(int)
    y = track['y'][spikesPos]
    y = y[~np.isnan(y)]
    x = track['x'][spikesPos]
    x = x[~np.isnan(x)]
    # if less than 2 spikes fired, return a zeros matrix
    if len(x) < 2:
        nSpikesPerSpace = np.zeros((int(nE - 1), int(nE - 1)))
    else:
        nSpikesPerSpace, _, _= np.histogram2d(x, y, [Xedges, Yedges])
    # obtain the placemap -- count map / occupancy map
    placemap = nSpikesPerSpace / OccMap
    placemap[np.isnan(placemap)] = 0
    placemapS, _ = matGaussianSmooth(placemap, bin2pixels * smoothStdPixels,
                                            int(bin2pixels * smoothStdPixels * 4), np.sum)
    placemapS, _ = matGaussianSmooth(placemapS.T, bin2pixels * smoothStdPixels,
                                            int(bin2pixels * smoothStdPixels * 4), np.sum)
    placemapS = placemapS.T
    # obtain placemap's spatial info and coherence
    meanRate = np.mean(placemap[mask])
    OccMapProb = OccMap / np.sum(OccMap[mask])
    information = 0 # initialise spatial info
    auxCoh = np.array([]).reshape(0, 2)
    for bini in range(np.size(OccMapProb, 0)):
        for binj in range(np.size(OccMapProb, 1)):
            if (mask[bini, binj]) & (placemap[bini, binj] > 0):
                information += placemap[bini, binj] * \
                               np.log2(placemap[bini, binj] / meanRate) * OccMapProb[bini, binj]
                try:
                    aux1 = np.nanmean(np.array([placemap[bini, binj + 1], placemap[bini + 1, binj], \
                                                placemap[bini + 1, binj + 1], placemap[bini, binj - 1], \
                                                placemap[bini - 1, binj], placemap[bini - 1, binj - 1], \
                                                placemap[bini + 1, binj - 1], placemap[bini - 1, binj + 1]]))

                    auxCoh = np.vstack((auxCoh, np.array([placemap[bini, binj], aux1]).T))
                except:
                    pass
    placeMapInfoPerSpike = information / meanRate
    spatialInfo = [information, placeMapInfoPerSpike]
    try: placeMapCoh = stats.pearsonr(auxCoh[:, 0], auxCoh[:, 1])[0]
    except: placeMapCoh = 0
    # return
    return OccMap, placemap, placemapS, spatialInfo, placeMapCoh


# Neuronal co-firing graphs functions

def corr_metric(A,B):
    """Obtain the correlation distance metric between the rows of A and B"""
    # Rowwise mean of input arrays & subtract from input arrays
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]
    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);
    # Finally get corr
    dist = np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))
    dist[np.isnan(dist)] = 0
    return dist


def corrGraph(dat, THR=0):
    """make undirected graph with correlation metric, self connections are set to 0
    - dat is the smoothed spiketrain matrix (time_samples x n_cells)
    - THR is used to set weak connections to 0 - with THR=0 a dense matrix is returned"""
    ccorr = np.zeros((dat.shape[1],dat.shape[1]))
    ccorr = corr_metric(dat.T,dat.T)
    np.fill_diagonal(ccorr, 0)
    corrmat = ccorr.copy()
    corrmat[np.abs(corrmat)<THR] = 0
    return corrmat


def GLMgraph_lin(actmat_, symm=1, z=1):
    """linear GLM graph"""
    from sklearn.linear_model import TweedieRegressor
    from sklearn.preprocessing import StandardScaler

    numc = actmat_.shape[1]
    graph = np.zeros((numc, numc))
    graph_pop = np.zeros((numc, numc))
    if z: actmat_ = StandardScaler().fit_transform(actmat_)
    for i in range(numc):
        if symm: range_ = range(i+1,numc)
        else: range_ = range(numc)
        for j in range_:
            x = actmat_[:,i]
            y = actmat_[:,j]
            sel = np.ones(actmat_.shape[1], dtype=bool)
            idx_ = [i,j]; sel[idx_] = False # exclude the selected cells
            xp = actmat_[:,sel].sum(1)
            xp = (xp - np.nanmean(xp)) / np.nanstd(xp)
            X = np.vstack((x,xp)).T
            # fit GLM
            model = TweedieRegressor(power=0)
            model.fit(X,y)
            betas = model.coef_
            # populate graph
            graph[i,j] = betas[0]
            graph_pop[i,j] = betas[1]
            if symm:
                graph[j,i] = graph[i,j]
                graph_pop[j,i] = graph_pop[i,j]
    # make sure diag is 0
    graph[np.diag_indices(numc)] = 0
    graph_pop[np.diag_indices(numc)] = 0
    return graph, graph_pop


# Clustering coefficient functions

def clustering(G, nodes=None, weight='weight'):
    """Adapted from NetworkX

    Compute the clustering coefficient for nodes.
    For weighted graphs,there are several ways to define clustering [1]_.
    the one used here is defined
    as the geometric average of the subgraph edge weights [2],
    Here it aimed specifically at undirected weighted graphs
    with negative edges.

    .. math::

       c_u = \frac{1}{deg(u)(deg(u)-1))}
             \sum_{vw} (\hat{w}_{uv} \hat{w}_{uw} \hat{w}_{vw})^{1/3}.

    The edge weights :math:`\hat{w}_{uv}` are normalized by the maximum weight
    in the network :math:`\hat{w}_{uv} = w_{uv}/\max(w)`.

    The value of :math:`c_u` is assigned to 0 if :math:`deg(u) < 2`.

    Parameters
    ----------
    G : graph

    nodes : container of nodes, optional (default=all nodes in G)
       Compute clustering for nodes in this container.

    weight : string or None, optional (default=None)
       The edge attribute that holds the numerical value used as a weight.
       If None, then each edge has weight 1.

    Returns
    -------
    out : float, or dictionary
       Clustering coefficient at specified nodes

    Notes
    -----
    Self loops are ignored.

    References
    ----------
    .. [1] Generalizations of the clustering coefficient to weighted
       complex networks by J. Saramäki, M. Kivelä, J.-P. Onnela,
       K. Kaski, and J. Kertész, Physical Review E, 75 027105 (2007).
       http://jponnela.com/web_documents/a9.pdf
    .. [2] Intensity and coherence of motifs in weighted complex
       networks by J. P. Onnela, J. Saramäki, J. Kertész, and K. Kaski,
       Physical Review E, 71(6), 065103 (2005).
    .. [3] Clustering in complex directed networks by G. Fagiolo,
       Physical Review E, 76(2), 026107 (2007).
    """
    td_iter = _weighted_triangles_and_degree_iter(G, nodes, weight)
    clusterc = {v: 0 if t == 0 else t / (d * (d - 1)) for
                v, d, t in td_iter}
    if nodes in G:
        # Return the value of the sole entry in the dictionary.
        return clusterc[nodes]
    return clusterc


def _weighted_triangles_and_degree_iter(G, nodes=None, weight='weight'):
    """ Adapted from NetworkX

    Return an iterator of (node, degree, weighted_triangles).

    Used for calculating the clustering coefficient of weighted graphs with -ve edges.

    """
    if weight is None or G.number_of_edges() == 0:
        max_weight = 1
    else:
        max_weight = max(d.get(weight, 1) for u, v, d in G.edges(data=True))
        if max_weight <= 0:
            max_weight = -min(d.get(weight, 1) for u, v, d in G.edges(data=True))
    if nodes is None:
        nodes_nbrs = G.adj.items()
    else:
        nodes_nbrs = ((n, G[n]) for n in G.nbunch_iter(nodes))

    def wt(u, v):
        return G[u][v].get(weight, 1) / max_weight

    def cubeRoot(x):
        if x >= 0:
            return x ** (1 / 3)
        else:
            return -(-x) ** (1 / 3)

    for i, nbrs in nodes_nbrs:
        inbrs = set(nbrs) - {i}
        weighted_triangles = 0
        seen = set()
        for j in inbrs:
            seen.add(j)
            # This prevents double counting.
            jnbrs = set(G[j]) - seen
            # Only compute the edge weight once, before the inner inner
            # loop.
            wij = wt(i, j)
            weighted_triangles += sum(cubeRoot(wij * wt(j, k) * wt(k, i))
                                      for k in inbrs & jnbrs)
        yield (i, len(inbrs), 2 * weighted_triangles)


# Geodesic path length

def wNetGeoPathLength(mat, directed=False):
    """
    Calculate the geodesic path length in a weighted network disregarding -ve edges

    Length between nodes (neurons) is set to inverse of weight (cofiring relation)
    i.e l_ij = 1/w_ij
    """
    posg = mat.copy()
    posg[posg < 0] = 0  # graph with +ve edges only
    invg = 1 / posg  # distance graph is reciprocal of the cofiring one
    invg[posg == 0] = 0  # set all the infs to 0
    if directed:
        ig = nx.DiGraph(invg)
    else:
        ig = nx.Graph(invg)
    fmd = np.asarray(nx.floyd_warshall_numpy(ig))
    fmd[(np.isinf(fmd)) | (fmd == 0)] = np.nan
    return fmd


# Riemmanian log-Euclidean distance

def distRiemLE(A, B):
    """compute the distance between the semi +ve definite matrices A and B"""
    return np.linalg.norm(spl.logm(A) - spl.logm(B))


def symmMatPerturb(g, scale=1):
    """perturb the matrix `g` by adding white noise of amplitude `scale*std(g)`"""
    perturb = np.random.random(g.shape) * scale * np.std(g)
    perturb[np.tril_indices(perturb.shape[0])] = 0
    perturb += perturb.T
    np.fill_diagonal(perturb, 0)
    return perturb


# Topological distance analysis

def fitEllipse(x, y):
    xm = x.mean();
    ym = y.mean()
    x -= xm;
    y -= ym
    U, S, V = np.linalg.svd(np.stack((x, y)))
    tt = np.linspace(0, 2 * np.pi, 1000)
    circle = np.stack((np.cos(tt), np.sin(tt)))  # unit circle
    transform = np.sqrt(2 / len(x)) * U.dot(np.diag(S))  # transformation matrix
    fit = transform.dot(circle) + np.array([[xm], [ym]])
    return fit


# Visualising cofiring graphs

def plotWeightedGraph(corrG, graphType='spring', scale=5, posC='r', negC='b',
                      nodeC=[0.9, 0.9, 0.9], Labels=False, nodeSize=500,
                      subset=None, subC='k', Alpha=0.8):
    G = nx.Graph(corrG / np.max(corrG))  # normalise the edges
    # select type of visualisation
    if graphType == 'spring':
        pos = nx.spring_layout(G)
    if graphType == 'circ':
        pos = nx.circular_layout(G)
    if Labels:  # plot nodes labels
        labels = {}
        for node in range(corrG.shape[0]):
            labels[node] = str(node + 1)
        nx.draw_networkx_labels(G, pos, labels, font_size=16)
    if subset is not None:  # highlight a subset of nodes if needed
        colors = []
        for i in range(corrG.shape[0]):
            if i in subset:
                colors.append(subC)
            else:
                colors.append(nodeC)
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=2 * nodeSize, alpha=Alpha)
    else:
        nx.draw_networkx_nodes(G, pos, node_color=[nodeC], node_size=nodeSize, alpha=Alpha)
    # Iterate through the graph nodes to gather all the weights
    all_weights = []
    for (node1, node2, data) in G.edges(data=True):
        all_weights.append(data['weight'])  # we'll use this when determining edge thickness
    # Get unique weights
    unique_weights = list(set(all_weights))

    # Plot the edges
    for weight in unique_weights:
        # Form a filtered list with just the weight you want to draw
        weighted_edges = [(node1, node2) for (node1, node2, edge_attr) in G.edges(data=True) \
                          if edge_attr['weight'] == weight]
        # define the width of the edges to draw
        width = scale * np.abs(weight ** 1.5) * corrG.shape[0]
        if weight > 0:
            nx.draw_networkx_edges(G, pos, edgelist=weighted_edges, width=width, \
                                   edge_color=[posC])
        else:
            nx.draw_networkx_edges(G, pos, edgelist=weighted_edges, width=width, \
                                   edge_color=[negC])
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.axis('off');