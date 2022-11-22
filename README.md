### README

The code in this directory performs some exemplar analysis used in [Gava, G.P. et al. *Integrating new memories into the hippocampal network activity space*. Nat Neurosci 24, 326–330 (2021)](https://doi.org/10.1038/s41593-021-00804-w).

The Jupyter notebook `cofiringTopology.ipynb` extracts the network parameters shown in Fig. 1 and Extended Data Fig. 2 of the paper (i.e. strength, clustering coefficient ad geodesic path length) of one exemplar pre-computed co-firing graph per experiment type (i.e. CPP, SPP, novel only and familiar reward).\
These data are stored in `/data/networks`.

In the notebook also the topological distance across co-firing graphs is computed and visualised (Fig. 2).

11/2022
The notebook has been expanded to perform some basic computation with one exemplar ephys recording from the CPP protocol (mhb10-161111, located in `/data/recday`).

These extra analyses were added for the MAIN Educational 2022 workshop (9-10/12/2022, Montreal, CA).

A Colab notebook version is also available [here](https://colab.research.google.com/drive/1eQjaabGHZFHwYjiM516S0UjeL9B3-2iZ#scrollTo=Rvuf55uzAKXn).