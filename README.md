PyHRG
====

PyHRG is a partial implemention of [Hierarchical Random Graphs](http://tuvalu.santafe.edu/~aaronc/hierarchy/).  As of February 2013, it supports fitting an HRG model to a network, finding a consensus dendrogram, and merging multiple consensus dendrograms into a new consensus dendrogram.  To date, PyHRG is the only HRG implementation that supports merging consensus dendrograms.

The core functionality of PyHRG requires networkx.  The plotting functionality, such as it is, requires matplotlib and only applies to plotting HRG dendrograms, not consensus dendrograms.  It is defective.

I will not be developing this code any further or fixing bugs.  If you wish to correct or enhance it, please fork this repository.
