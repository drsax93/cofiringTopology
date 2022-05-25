### Author: Giuseppe P Gava, 02/2021

# Libraries import
import sys, os, fnmatch
import numpy as np
import scipy.linalg as spl
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

### Utility functions

### Clustering coefficient

def clustering(G, nodes=None, weight='weight'):
    """Taken from Network#X

    Compute the clustering coefficient for nodes.

    For unweighted graphs, the clustering of a node :math:`u`
    is the fraction of possible triangles through that node that exist,

    .. math::

      c_u = \frac{2 T(u)}{deg(u)(deg(u)-1)},

    where :math:`T(u)` is the number of triangles through node :math:`u` and
    :math:`deg(u)` is the degree of :math:`u`.

    For weighted graphs, there are several ways to define clustering [1]_.
    the one used here is defined
    as the geometric average of the subgraph edge weights [2]_,

    .. math::

       c_u = \frac{1}{deg(u)(deg(u)-1))}
             \sum_{vw} (\hat{w}_{uv} \hat{w}_{uw} \hat{w}_{vw})^{1/3}.

    The edge weights :math:`\hat{w}_{uv}` are normalized by the maximum weight
    in the network :math:`\hat{w}_{uv} = w_{uv}/\max(w)`.

    The value of :math:`c_u` is assigned to 0 if :math:`deg(u) < 2`.

    For directed graphs, the clustering is similarly defined as the fraction
    of all possible directed triangles or geometric average of the subgraph
    edge weights for unweighted and weighted directed graph respectively [3]_.

    .. math::

       c_u = \frac{1}{deg^{tot}(u)(deg^{tot}(u)-1) - 2deg^{\leftrightarrow}(u)}
             T(u),

    where :math:`T(u)` is the number of directed triangles through node
    :math:`u`, :math:`deg^{tot}(u)` is the sum of in degree and out degree of
    :math:`u` and :math:`deg^{\leftrightarrow}(u)` is the reciprocal degree of
    :math:`u`.

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

    Examples
    --------
    >>> G=nx.complete_graph(5)
    >>> print(nx.clustering(G,0))
    1.0
    >>> print(nx.clustering(G))
    {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

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
    if G.is_directed():
        if weight is not None:
            td_iter = _directed_weighted_triangles_and_degree_iter(
                G, nodes, weight)
            clusterc = {v: 0 if t == 0 else t / ((dt * (dt - 1) - 2 * db) * 2)
                        for v, dt, db, t in td_iter}
        else:
            td_iter = _directed_triangles_and_degree_iter(G, nodes)
            clusterc = {v: 0 if t == 0 else t / ((dt * (dt - 1) - 2 * db) * 2)
                        for v, dt, db, t in td_iter}
    else:
        if weight is not None:
            td_iter = _weighted_triangles_and_degree_iter(G, nodes, weight)
            clusterc = {v: 0 if t == 0 else t / (d * (d - 1)) for
                        v, d, t in td_iter}
        else:
            td_iter = _triangles_and_degree_iter(G, nodes)
            clusterc = {v: 0 if t == 0 else t / (d * (d - 1)) for
                        v, d, t, _ in td_iter}
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
        if max_weight<=0:
            max_weight = -min(d.get(weight, 1) for u, v, d in G.edges(data=True))
    if nodes is None:
        nodes_nbrs = G.adj.items()
    else:
        nodes_nbrs = ((n, G[n]) for n in G.nbunch_iter(nodes))

    def wt(u, v):
        return G[u][v].get(weight, 1) / max_weight

    def cubeRoot(x):
        if x >= 0:
            return x**(1/3)
        else:
            return -(-x)**(1/3)

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

### Geodesic path length

def wNetGeoPathLength(mat,directed=False):
    '''
    Calculate the geodesic path length in a weighted network disregarding -ve edges
    
    Length between nodes (neurons) is  set to inverse of weight (cofiring relation)
    i.e l_ij = 1/w_ij
    '''
    posg = mat; posg[posg<0] = 0
    invg = 1/posg; invg[posg==0] = 0
    if directed: ig = nx.DiGraph(invg)
    else: ig = nx.Graph(invg)
    fmd = np.asarray(nx.floyd_warshall_numpy(ig))
    fmd[(np.isinf(fmd)) | (fmd==0)] = np.nan
    return fmd

### Riemmanian log-Euclidean distance

def dist_riem_LE(A,B):
    # compute the distance between the =ve matrices A and B
    return np.linalg.norm(spl.logm(A) - spl.logm(B))

def symm_matPerturb(g,scale=1):
    # perturb the matrix `g` by adding white noise of amplitude `scale*std(g)`
    perturb = np.random.random(g.shape) * scale*np.std(g)
    perturb[np.tril_indices(perturb.shape[0])] = 0
    perturb += perturb.T
    np.fill_diagonal(perturb,0)
    return perturb

### Topological distance analysis

def fitEllipse(x,y):
    xm = x.mean(); ym = y.mean()
    x -= xm; y -= ym
    U, S, V = np.linalg.svd(np.stack((x, y)))
    tt = np.linspace(0, 2*np.pi, 1000)
    circle = np.stack((np.cos(tt), np.sin(tt)))    # unit circle
    transform = np.sqrt(2/len(x)) * U.dot(np.diag(S))   # transformation matrix
    fit = transform.dot(circle) + np.array([[xm], [ym]])
    return fit


### Visualising graphs

def plotWeightedGraph(corrG, graphType='spring', scale=5, posC='r', negC='b',
                     nodeC=[0.9,0.9,0.9], Labels=False, nodeSize=500,
                      subset=None, subC='k', Alpha=0.8):
    G = nx.Graph(corrG/np.max(corrG)) # normalise the edges
    # select type of visualisation
    if graphType=='spring': pos=nx.spring_layout(G)
    if graphType=='circ': pos=nx.circular_layout(G)
    if Labels: # plot nodes labels
        labels = {}
        for node in range(corrG.shape[0]):
            labels[node] = str(node+1)
        nx.draw_networkx_labels(G,pos,labels,font_size=16)
    if subset is not None: # highlight a subset of nodes if needed
        colors = []
        for i in range(corrG.shape[0]):
            if i in subset: colors.append(subC)
            else: colors.append(nodeC)
        nx.draw_networkx_nodes(G,pos,node_color=colors,node_size=2*nodeSize,alpha=Alpha)
    else:
        nx.draw_networkx_nodes(G,pos,node_color=[nodeC],node_size=nodeSize,alpha=Alpha)
    # Iterate through the graph nodes to gather all the weights
    all_weights = []
    for (node1,node2,data) in G.edges(data=True):
        all_weights.append(data['weight']) #we'll use this when determining edge thickness
    # Get unique weights
    unique_weights = list(set(all_weights))

    # Plot the edges
    for weight in unique_weights:
        # Form a filtered list with just the weight you want to draw
        weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in G.edges(data=True)\
                          if edge_attr['weight']==weight]
        # define the width of the edges to draw
        width = scale*np.abs(weight**1.5) * corrG.shape[0]
        if weight>0:
            nx.draw_networkx_edges(G,pos,edgelist=weighted_edges,width=width,\
                                   edge_color=[posC])
        else:
            nx.draw_networkx_edges(G,pos,edgelist=weighted_edges,width=width,\
                                   edge_color=[negC])
    plt.xlim(-1.2,1.2); plt.ylim(-1.2,1.2)
    plt.axis('off');