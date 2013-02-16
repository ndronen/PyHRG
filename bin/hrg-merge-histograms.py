#!/usr/bin/env python3.2

import matplotlib.pylab as plt
import networkx as nx
import random
import pickle
import collections

import os
import sys
bindir = os.path.abspath(os.path.dirname(sys.argv[0]))
libdir = os.path.dirname(bindir) + "/lib"
sys.path.append(libdir)

from hrg import Dendrogram, ConsensusDendrogramBuilder

from optparse import OptionParser

import netbuilder

def main():
    parser = OptionParser(
        description="Merges the split histograms of two or more HRG consensus dendrograms.  Creates a new consensus dendrogram from the merged histograms.  Saves the new consensus dendrogram to a graph markup language (GML) file.",
        prog='hrg-merge-histograms.py',
        usage='%prog [options] GRAPH_EDGELIST_FILE PICKLED_HISTOGRAM ... PICKLED_HISTOGRAM OUTPUT_FILE')

    parser.add_option('-f', '--force', action='store', type=int, default=10000,
        help='Allow overwriting of existing GML dendrogram files')

    (options, args) = parser.parse_args()

    if len(args) < 4:
        parser.print_help()
        return 1

    graph_edgelist = args[0]
    G = nx.read_edgelist(graph_edgelist, nodetype=int)
    filename = os.path.basename(graph_edgelist)
    G.name = os.path.splitext(filename)[0]
    args.remove(graph_edgelist)

    outfile=args.pop()

    if os.path.exists(outfile) and not options.force:
        raise Exception("Output file " + outfile +
            " exists.  Won't overwrite without --force option.")

    n = 0

    histograms = []
    for histfile in args:
        f = open(histfile, 'rb')
        histogram = pickle.load(f)
        if not isinstance(histogram, collections.Mapping):
            raise Exception('Object in ' + histfile +
                ' is not a dictionary: ' + str(type(histogram)))
        if n == 0:
            n = histogram['num_samples']

        if histogram['num_samples'] != n:
            raise Exception('inconsistent number of samples, '
                'expected ' + str(n) + ', actual ' +
                histogram['num_samples'])

        del histogram['num_samples']

        histograms.append(histogram)

    nodes = G.nodes()
    nodes.sort()

    builder = ConsensusDendrogramBuilder()
    C = builder.build(nodes, histograms, n)

    # Save the consensus dendrogram to a GML file.
    nx.write_gml(C, outfile)
    print("Saved merged consensus dendrogram to " + outfile + ".")

    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
