#!/usr/bin/env python

import networkx as nx

import optparse
import os
import sys
bindir = os.path.abspath(os.path.dirname(sys.argv[0]))
libdir = os.path.dirname(bindir) + "/lib"
sys.path.append(libdir)

from hrg import Dendrogram

def print_status(*l):
    print("\t".join([str(x) for x in l]))

def main():
    parser = optparse.OptionParser(
        description='Fits a hierarchical random graph (HRG) model to a network.  Saves the model to a file in graph markup language (GML) format.',
        prog='hrg-fit.py',
        usage='%prog [options] GRAPH_EDGELIST_FILE')

    parser.add_option('-s', '--num-steps', action='store', type=int,
        default=100000,
        help='The number of MCMC steps to take (default=100000).')

    parser.add_option('-t', '--nodetype', action='store', type='choice',
        choices=[int,str],
        default=int,
        help='The type of the nodes in the edgelist file; "int" or "str" (default="int")')

    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.print_help()
        return 1

    G=nx.read_edgelist(args[0], nodetype=options.nodetype)
    name=os.path.splitext(args[0])[0]
    hrg_file = name + '-hrg.gml'
    print("HRG model will be saved as " + hrg_file + ".")

    D=Dendrogram.from_graph(G)

    bestL=initL=D.graph['L']
    prevL=bestL
    bestI=0

    print_status("step", "L", "best L", "MC step", "deltaL")

    for i in range(1, options.num_steps):
        taken=D.monte_carlo_move()
        t = ''
        if taken:
            t = '*'
        if D.graph['L'] > bestL:
            bestL=D.graph['L']
            bestI=i
            nx.write_gml(D, hrg_file)
            print_status("["+str(i)+"]", "%.3f" % bestL, "%.3f" % bestL, t, "%.3f"%D.deltaL)
        elif i % 4096 == 0:
            print_status("["+str(i)+"]", "%.3f" % D.graph['L'], "%.3f" % bestL, t, "%.3f"%D.deltaL)

        prevL=D.graph['L']

        if i % 10 == 0:
            sys.stdout.flush()

    print("Step number of last best fit "+str(bestI) + ".")
    print("HRG model was saved as " + hrg_file + ".")

    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
