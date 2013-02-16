import random
import networkx as nx
import numpy as np
import sys
import math
import re
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from collections import defaultdict

class RandomChooser(object):
    def __init__(self, dendrogram):
        self.D = dendrogram
        self.random = random.Random()

    def choose_edge(self):
        while True:
            parent = self.random.choice(self.D.dnode_list)
            child = self.random.choice(self.D.children(parent))
            if child in self.D.dnode_set:
                return parent, child
        
class Dendrogram(nx.DiGraph):
    """
    """
    @classmethod
    def from_graph(cls, G, layout='random'):
        """
        This is the first case.  The user provides a network G.,
        and we return an instance of Dendrogram.
        """
        D=cls(G)
        D.initialize(layout=layout)
        return D

    """
    """
    @classmethod
    def from_gml_file(cls, path, G):
        """
        This is the second case.  The user previously acquired
        an instance of Dengrogram using from_graph and saved
        the instance to disk using nx.write_gml.
        """
        D=cls(G)
        from_gml=nx.read_gml(path)

        # Relabel all of the dendrogram nodes.  There's a good chance
        # that in the GML file the identifiers of dendrogram nodes are
        # integers instead of strings (e.g. 42 vs "_D42").
        mapping={}
        for n,d in from_gml.nodes_iter(data=True):
            label=d['label']
            mapping[n]=label
        from_gml=nx.relabel_nodes(from_gml, mapping)
        D.add_nodes_from(from_gml.nodes(data=True))
        D.add_edges_from(from_gml.edges(data=True))

        D.initialize_graph_keys()
        D.initialize_dendrogram_node_structures()

        for n,d in D.dendrogram_nodes_iter(data=True):
            D.graph['L']+=d['L']

        return D

    """
    """
    @classmethod
    def from_dendro(cls, D):
        newD=cls()
        newD.add_nodes_from(D.nodes(data=True))
        newD.add_edges_from(D.edges(data=True))
        D.initialize_graph_keys()
        D.initialize_dendrogram_node_structures()
        return newD

    """
    """
    def __init__(self, G=None):
        """
        """
        super(Dendrogram, self).__init__()

        self.chooser = RandomChooser(self)
        self.random = random.Random()
        self.number_of_graph_nodes = 0
        self.graph['L'] = 0
        self.deltaL = 0
        self.graph_edges=set()
        self.graph_nodes=set()
        self.split_histogram=defaultdict(int)
        self.num_samples=0

        if G is not None:
            if len(G) < 2:
                msg = "graph must have at least 2 nodes"
                msg += ", this one has "+str(len(G))
                raise Exception(msg)
            self.number_of_graph_nodes = len(G)
            self.graph_edges=set(G.edges())
            self.graph_nodes_set=set(G.nodes())
            self.graph_nodes_list=G.nodes()
            self.graph_nodes_list.sort()
            self.G = nx.Graph(G)
            # When the network is stored in an edge list file, self
            # loops are required in order to preserve nodes of degree
            # 0.  Remove self loops here.
            self.G.remove_edges_from(self.G.selfloop_edges())

    """
    """
    def initialize_graph_keys(self):
        self.number_of_graph_nodes=len(self.G)
        self.graph_edges=set(self.G.edges())
        self.graph_nodes=set(self.G.nodes())
        self.graph_nodes_list=self.G.nodes()
        self.graph_nodes_list.sort()

    """
    """
    def initialize_dendrogram_node_structures(self):
        self.dnode_list=[]
        self.dnode_set=set()
        for node in self.dendrogram_nodes_iter():
            self.dnode_list.append(node)
            self.dnode_set.add(node)

    """
    """
    def initialize(self, layout='random'):
        self.initialize_graph_keys()

        # Add the root node of the dendrogram.
        root = "_D0"
        self.add_node(root, p=self.random.random())

        # And the remaining nodes of the dendrogram.
        for i in range(1, self.G.number_of_nodes()-1):
            name="_D"+str(i)
            self.add_node(name, p=self.random.random())
            self.insert_attr(root, name, 'p')

        self.gnodes=set(self.G.nodes())

        if layout == 'random':
            # Permute the graph nodes by randomly sampling all of
            # them without replacement.
            graph_nodes=self.random.sample(self.G.nodes(), len(self.G))
    
            # Add each graph node as a left or right child
            # of any dendrogram node that doesn't have a left
            # or right child.
            for n,d in self.nodes(data=True):
                try:
                    self.node[n]['left']
                except KeyError:
                    gnode=graph_nodes.pop()
                    l=self.node[n]['p']-0.0000000000001
                    self.add_node(gnode, p=l)
                    self.add_left_child(n, gnode)
    
                try:
                    self.node[n]['right']
                except KeyError:
                    gnode=graph_nodes.pop()
                    l=self.node[n]['p']+0.0000000000001
                    self.add_node(gnode, p=l)
                    self.add_right_child(n, gnode)

        # Label each dendrogram nodes with the name of its smallest
        # child graph node.
        sorted_graph_nodes=self.G.nodes()
        sorted_graph_nodes.sort()
        for graph_node in sorted_graph_nodes:
            # Each graph node's order label is itself.
            self.node[graph_node]['orderprop'] = graph_node
            parent = self.parent(graph_node)
            while parent != None:
                if 'orderprop' not in self.node[parent]:
                    self.node[parent]['orderprop'] = graph_node
                parent = self.parent(parent)

        # Swap children to enforce order property
        for node,d in self.dendrogram_nodes(data=True):
            # If the order label of the node's left child is greater
            # than the node's order label, swap them.
            left_child=self.node[node]['left']
            if self.node[left_child]['orderprop'] > self.node[node]['orderprop']:
                right_child=self.node[node]['right']
                self.remove_left_child(node, left_child)
                self.remove_right_child(node, right_child)

                self.add_left_child(node, right_child)
                self.add_right_child(node, left_child)

        # Compute nL and nR for each dendrogram node.
        self.compute_initial_numbers_of_leaf_nodes(self.G)

        for node,d in self.dendrogram_nodes(data=True):
            e, nL, nR, p, L = self.compute_likelihood(node)
            #self.node[node]['e'] = e
            d['e']=e

            #self.node[node]['nL'] = nL
            d['nL']=nL

            #self.node[node]['nR'] = nR
            d['nR']=nR

            #self.node[node]['p'] = p
            d['p']=p

            #self.node[node]['L'] = L
            d['L']=L

            self.graph['L']+=d['L']

        self.initialize_dendrogram_node_structures()

    def __hash__(self):
        raise TypeError()

    def __rep__(self):
        return "Dendrogram("+str(id(self))+"): " + \
            " nodes " + str(self.nodes(data=True)) + \
            " edges " + str(self.edges())

    def __str__(self):
        return self.__rep__()

    def clear(self):
        """
        """
        super(Dendrogram, self).__init__()
        super(Dendrogram, self).clear()
        #G.clear()
        
    def dendrogram_nodes_iter(self, data=False):
        """
        """
        if data == True:
            for node,d in self.nodes_iter(data=True):
                if self.is_dendrogram_node(node):
                    yield node, d
        else:
            for node in self.nodes_iter(data=False):
                if self.is_dendrogram_node(node):
                    yield node

    def dendrogram_nodes(self, data=False):
        """
        """
        if data == True:
            return [(n,d) for n,d in self.dendrogram_nodes_iter(data=True)]
        else:
            return [n for n in self.dendrogram_nodes_iter(data=False)]

    def dendrogram_edges_iter(self, data=False):
        """
        """
        if data == True:
            raise Exception("data keyword argument not yet implemented")
        for v,w in self.edges_iter():
            if self.is_dendrogram_node(v) and self.is_dendrogram_node(w):
                yield v,w

    def is_dendrogram_node(self, node):
        """
        """
        return str(node).startswith('_D')

    def is_graph_node(self, node):
        """
        """
        return not str(node).startswith('_D')

    def has_children(self, parent):
        """
        """
        # A node is a graph node if it has no children.
        try:
            self.node[parent]['left']
            return True
        except KeyError as e:
            pass

        try:
            self.node[parent]['right']
            return True
        except KeyError as e:
            pass

        return False

    def add_left_child(self, parent, child):
        """
        """
        self.add_edge(parent, child, side='left')
        self.node[parent]['left'] = child
    
    def add_right_child(self, parent, child):
        """
        """
        self.add_edge(parent, child, side='right')
        self.node[parent]['right'] = child

    def remove_child(self, parent, child, side):
        """
        """
        self.remove_edge(parent, child)
        del self.node[parent][side]

    def remove_left_child(self, parent, child):
        """
        """
        self.remove_child(parent, child, 'left')

    def remove_right_child(self, parent, child):
        """
        """
        self.remove_child(parent, child, 'right')
    
    def child_node(self, parent, side):
        """
        """
        return self.node[parent][side]

    def is_left_child(self, parent, child):
        try:
            return self.node[parent]['left'] == child
        except Exception:
            return False

    def is_right_child(self, parent, child):
        try:
            return self.node[parent]['right'] == child
        except Exception:
            return False

    def left_child(self, parent):
        """
        """
        return self.node[parent]['left']
    
    def right_child(self, parent):
        """
        """
        return self.node[parent]['right']

    def children(self, parent):
        """
        """
        children = []
        try:
            children.append(self.node[parent]['left'])
        except KeyError as e:
            pass

        try:
            children.append(self.node[parent]['right'])
        except KeyError as e:
            pass

        return children

    def parent(self, child):
        """
        """
        parents=self.predecessors(child)
        try:
            return parents[0]
        except Exception as e:
            return None
    
    def insert_attr(self, v, w, attr):
        """
        """
        if self.node[v][attr] < self.node[w][attr]:
            # check if left subtree is empty
            try:
                self.node[v]['left']
                nextv=self.node[v]['left']
                self.insert_attr(nextv, w, attr)
            except KeyError:
                # make w left child
                self.add_left_child(v, w)
        else:
            # check if right subtree is empty
            try:
                self.node[v]['right']
                nextv=self.node[v]['right']
                self.insert_attr(nextv, w, attr)
            except KeyError:
                self.add_right_child(v, w)

    def enforce_order_property(self, node):
        left_child=self.node[node]['left']
        node_label=self.node[node]['orderprop']
        left_child_label=self.node[node]['orderprop']
        if node_label > left_child_label:
            s=str(self)
            raise Exception("order property violated by node "+str(node) + " "+
                "label "+str(node_label) + " "+
                "left child label "+str(left_child_label) + " "+
                ": "+s)

    def verify_nkids(self, node):
        left=self.node[node]['nL']
        right=self.node[node]['nR']
        if left+right > self.number_of_graph_nodes:
            raise Exception("wrong number of kids "+
                "node "+str(node)+" "+
                "left "+str(left)+" "+
                "right "+str(right)+" "+
                "D "+str(self))

    def update_leaf_nodes(self, child, parent):
        self.update_numbers_of_leaf_nodes(child)
        self.verify_nkids(child)
        self.update_numbers_of_leaf_nodes(parent)
        self.verify_nkids(parent)

    def left_alpha_move(self, child, update_leaf_nodes=True):
        """
        1) Swap right edges of child and parent
        """
        parent = self.parent(child)

        parent_right_child = self.node[parent]['right']
        child_right_child = self.node[child]['right']

        self.remove_right_child(parent, parent_right_child)
        self.remove_right_child(child, child_right_child)

        self.add_right_child(parent, child_right_child)
        self.add_right_child(child, parent_right_child)

        if update_leaf_nodes:
            self.update_leaf_nodes(child, parent)

    def undo_left_alpha_move(self, child, update_leaf_nodes=True):
        """
        Undoing the left alpha move is equivalent to doing
        the left alpha move.
        """
        self.left_alpha_move(child, update_leaf_nodes=update_leaf_nodes)

    def left_beta_move(self, child, update_leaf_nodes=True):
        """
        1) Swap left and right edges of child
        2) Swap right edges of child and parent
        3) Swap left and right edges of parent
        """
        parent = self.parent(child) 

        parent_left_child = self.node[parent]['left']
        parent_right_child = self.node[parent]['right']

        child_left_child = self.node[child]['left']
        child_right_child = self.node[child]['right']

        # Swap left and right edges of child
        self.remove_left_child(child, child_left_child)
        self.remove_right_child(child, child_right_child)
        self.add_left_child(child, child_right_child)
        self.add_right_child(child, child_left_child)

        # Update local variables
        child_left_child = self.node[child]['left']
        child_right_child = self.node[child]['right']

        # Swap right edges of child and parent
        self.remove_right_child(parent, parent_right_child)
        self.remove_right_child(child, child_right_child)
        self.add_right_child(parent, child_right_child)
        self.add_right_child(child, parent_right_child)

        # Update local variables
        child_right_child = self.node[child]['right']
        parent_right_child = self.node[parent]['right']

        # Swap left and right edges of parent
        self.remove_left_child(parent, parent_left_child)
        self.remove_right_child(parent, parent_right_child)
        self.add_left_child(parent, parent_right_child)
        self.add_right_child(parent, parent_left_child)

        # If the child's left and right children don't adhere
        # to the order property, swap them.
        if self.node[child_right_child]['orderprop'] < self.node[child_right_child]['orderprop']:
            self.remove_left_child(child, child_left_child)
            self.remove_right_child(child, child_right_child)
            self.add_left_child(child, child_right_child)
            self.add_right_child(child, child_left_child)
            tmp = child_left_child
            child_left_child = child_right_child
            child_right_child = child_left_child

        # Set the chlid's order label to the order label of its left child.
        self.node[child]['orderprop'] = self.node[child_left_child]['orderprop']

        self.enforce_order_property(parent)
        self.enforce_order_property(child)

        if update_leaf_nodes:
            self.update_leaf_nodes(child, parent)

    def undo_left_beta_move(self, child, update_leaf_nodes=True):
        """
        1) Swap left and right edges of parent
        2) Swap right edges of child and parent
        3) Swap left and right edges of child
        """
        parent = self.parent(child) 

        parent_left_child = self.node[parent]['left']
        parent_right_child = self.node[parent]['right']

        child_left_child = self.node[child]['left']
        child_right_child = self.node[child]['right']

        # Swap left and right edges of parent
        self.remove_left_child(parent, parent_left_child)
        self.remove_right_child(parent, parent_right_child)
        self.add_left_child(parent, parent_right_child)
        self.add_right_child(parent, parent_left_child)

        # Update local variables
        parent_left_child = self.node[parent]['left']
        parent_right_child = self.node[parent]['right']

        # Swap right edges of child and parent
        self.remove_right_child(parent, parent_right_child)
        self.remove_right_child(child, child_right_child)
        self.add_right_child(parent, child_right_child)
        self.add_right_child(child, parent_right_child)

        # Update local variables
        parent_right_child = self.node[parent]['right']
        child_right_child = self.node[child]['right']

        # Swap left and right edges of child
        self.remove_left_child(child, child_left_child)
        self.remove_right_child(child, child_right_child)
        self.add_left_child(child, child_right_child)
        self.add_right_child(child, child_left_child)

        child_right_child = child_left_child

        # Set the child's order label to the order label of its left child.
        self.node[child]['orderprop'] = self.node[child_left_child]['orderprop']

        self.enforce_order_property(parent)
        self.enforce_order_property(child)

        if update_leaf_nodes:
            self.update_leaf_nodes(child, parent)

    def right_alpha_move(self, child, update_leaf_nodes=True):
        """
        1) Swap left and right edges of parent
        2) Swap left edge of child and right edge of parent
        """
        parent = self.parent(child) 

        parent_left_child = self.node[parent]['left']
        parent_right_child = self.node[parent]['right']

        child_left_child = self.node[child]['left']

        # Swap left and right edges of parent
        self.remove_left_child(parent, parent_left_child)
        self.remove_right_child(parent, parent_right_child)
        self.add_left_child(parent, parent_right_child)
        self.add_right_child(parent, parent_left_child)

        # Update local variables
        parent_right_child = self.node[parent]['right']

        # Swap left edge of child and right edge of parent
        self.remove_left_child(child, child_left_child)
        self.remove_right_child(parent, parent_right_child)
        self.add_left_child(child, parent_right_child)
        self.add_right_child(parent, child_left_child)

        self.node[child]['orderprop'] = self.node[parent]['orderprop']

        self.enforce_order_property(parent)
        self.enforce_order_property(child)

        if update_leaf_nodes:
            self.update_leaf_nodes(child, parent)

    def undo_right_alpha_move(self, child, update_leaf_nodes=True):
        """
        1) Swap left edge of child and right edge of parent
        2) Swap left and right edges of parent
        """
        parent = self.parent(child) 

        parent_left_child = self.node[parent]['left']
        parent_right_child = self.node[parent]['right']

        child_left_child = self.node[child]['left']

        # Swap left edge of child and right edge of parent
        self.remove_left_child(child, child_left_child)
        self.remove_right_child(parent, parent_right_child)
        self.add_left_child(child, parent_right_child)
        self.add_right_child(parent, child_left_child)

        # Update local variables
        parent_right_child = self.node[parent]['right']
        child_left_child = self.node[child]['left']

        # Swap left and right edges of parent
        self.remove_left_child(parent, parent_left_child)
        self.remove_right_child(parent, parent_right_child)
        self.add_left_child(parent, parent_right_child)
        self.add_right_child(parent, parent_left_child)

        self.enforce_order_property(parent)
        self.enforce_order_property(child)

        if update_leaf_nodes:
            self.update_leaf_nodes(child, parent)

    def right_beta_move(self, child, update_leaf_nodes=True):
        """
        1) Swap left and right edges of parent
        2) Swap right edge of child and right edge of parent
        3) Swap left and right edges of child
        """
        parent = self.parent(child) 

        parent_left_child = self.node[parent]['left']
        parent_right_child = self.node[parent]['right']

        child_left_child = self.node[child]['left']
        child_right_child = self.node[child]['right']

        # Swap left and right edges of parent 
        self.remove_left_child(parent, parent_left_child)
        self.remove_right_child(parent, parent_right_child)
        self.add_left_child(parent, parent_right_child)
        self.add_right_child(parent, parent_left_child)

        # Update local variables
        parent_right_child = self.node[parent]['right']

        # Swap right edge of child and right edge of parent
        self.remove_right_child(child, child_right_child)
        self.remove_right_child(parent, parent_right_child)
        self.add_right_child(child, parent_right_child)
        self.add_right_child(parent, child_right_child)

        # Update local variables
        child_right_child = self.node[child]['right']
        
        # Swap left and right edges of child
        self.remove_left_child(child, child_left_child)
        self.remove_right_child(child, child_right_child)
        self.add_left_child(child, child_right_child)
        self.add_right_child(child, child_left_child)

        self.enforce_order_property(parent)
        self.enforce_order_property(child)

        if update_leaf_nodes:
            self.update_leaf_nodes(child, parent)

    def undo_right_beta_move(self, child, update_leaf_nodes=True):
        """
        1) Swap left and right edges of child
        2) Swap right edge of child and right edge of parent
        3) Swap left and right edges of parent
        """
        parent = self.parent(child) 

        parent_left_child = self.node[parent]['left']
        parent_right_child = self.node[parent]['right']

        child_left_child = self.node[child]['left']
        child_right_child = self.node[child]['right']

        # Swap left and right edges of child
        self.remove_left_child(child, child_left_child)
        self.remove_right_child(child, child_right_child)
        self.add_left_child(child, child_right_child)
        self.add_right_child(child, child_left_child)

        # Update local variables
        child_left_child = self.node[child]['left']
        child_right_child = self.node[child]['right']

        # Swap right edge of child and right edge of parent
        self.remove_right_child(child, child_right_child)
        self.remove_right_child(parent, parent_right_child)
        self.add_right_child(child, parent_right_child)
        self.add_right_child(parent, child_right_child)

        # Update local variables
        parent_right_child = self.node[parent]['right']
        child_right_child = self.node[child]['right']

        # Swap left and right edges of parent 
        self.remove_left_child(parent, parent_left_child)
        self.remove_right_child(parent, parent_right_child)
        self.add_left_child(parent, parent_right_child)
        self.add_right_child(parent, parent_left_child)

        self.enforce_order_property(parent)
        self.enforce_order_property(child)

        if update_leaf_nodes:
            self.update_leaf_nodes(child, parent)

    def monte_carlo_move(self, T=1.0, debug=False):
        # Pick a random dendrogram edge.
        # Generate random number R to determine type of move (alpha or beta).
        # Compute change in likelihood DeltaL that would result.
        # Generate another random number S (for the next step).
        # If DeltaL is positive or S is less than exp(DeltaL), take the move.
        parent,child=self.chooser.choose_edge()

        child_type=''
        if self.is_left_child(parent, child):
            child_type='left'
        elif self.is_right_child(parent, child):
            child_type='right'
        else:
            raise Exception("node "+str(child)+" is neither left " +
                "nor right child of "+str(parent))
    
        move_type=''
        r=self.random.random()
        if r > 0.5:
            move_type='alpha'
        else:
            move_type='beta'

        doname=child_type+'_'+move_type+'_move'
        undoname='undo_'+doname

        dofunc=getattr(self, doname)
        undofunc=getattr(self, undoname)

        if debug:
            print("attempting "+doname+" on "+str((parent,child)))
    
        dofunc(child)

        potential = self.compute_potential_likelihood(parent, child, debug=False)

        if debug:
            print("likelihood before "+str(self.graph['L']))
            print("likelihood after "+str(potential['newL']))
            print("likelihood delta "+str(potential['deltaL']))

        r2=self.random.random()
        l2=0
        try:
            l2=math.exp(T*potential['deltaL'])
        except OverflowError as e:
            print(str(e) + " deltaL="+str(potential['deltaL']))
            potential['deltaL']=700
            l2=math.exp(T*potential['deltaL'])

        taken=False

        if math.isnan(potential['newL']) or math.isinf(potential['newL']):
            raise Exception("new likelihood is nan/inf: "+str(potential['newL']))
            if debug:
                print("undoing "+doname+" because nan/inf")
            undofunc(child)
            self.deltaL = 0
        elif potential['deltaL'] > 0:
            taken=True
            # Take the move by not undoing it.
            if debug:
                print("taking "+doname)
            self.update_likelihood(parent, child, potential)
        elif r2 < l2:
            taken=True
            if debug:
                print("taking "+doname+" because r2<l2 "+str(r2)+"<"+str(l2))
            self.update_likelihood(parent, child, potential)
        else:
            if debug:
                print("undoing "+doname)
            undofunc(child)
            self.deltaL = 0

        if debug:
            print("L          after "+str(self.graph['L']))

        if debug:
            self.validate_likelihood()

        return taken

    def validate_likelihood(self):
        l=0
        ls=[]
        for n,d in self.dendrogram_nodes_iter(data=True):
            ls.append(d['L'])
            l+=d['L']
        diff=abs(self.graph['L']-l)
        if diff > 0.0001:
            raise Exception("L != sum of likelihoods " +
                "L "+str(self.graph['L']) +
                "l "+str(l) + " " +
                "ls "+str(ls) +"\n" +
                "after  move "+str(self))
            
    def update_likelihood(self, parent, child, potential, debug=False):
        pprevL=self.node[parent]['L']
        cprevL=self.node[child]['L']

        self.node[parent]['L']=potential['pnewL']
        self.node[parent]['e']=potential['pnewE']
        self.node[parent]['p']=potential['pnewP']

        self.node[child]['L']=potential['cnewL']
        self.node[child]['e']=potential['cnewE']
        self.node[child]['p']=potential['cnewP']

        self.graph['L']=potential['newL']
        self.deltaL=potential['deltaL']

        if debug:
            print("update_likelihood L "+str(self.graph['L']) + \
                " parent "+str(parent) + " child "+str(child))
            print("              pnewL "+str(potential['pnewL']))
            print("             pprevL "+str(pprevL))
            print("              cnewL "+str(potential['cnewL']))
            print("             cprevL "+str(cprevL))

    def compute_potential_likelihood(self, parent, child, debug=False):
        potential = {}
        pprevL=self.node[parent]['L']
        pnewE,pnewNL,pnewNR,pnewP,pnewL=self.compute_likelihood(parent)
        potential['pnewE'] = pnewE
        potential['pnewNL'] = pnewNL
        potential['pnewNR'] = pnewNR
        potential['pnewP'] = pnewP
        potential['pnewL'] = pnewL

        cprevL=self.node[child]['L']
        cnewE,cnewNL,cnewNR,cnewP,cnewL=self.compute_likelihood(child)
        potential['cnewE'] = cnewE
        potential['cnewNL'] = cnewNL
        potential['cnewNR'] = cnewNR
        potential['cnewP'] = cnewP
        potential['cnewL'] = cnewL

        L=self.graph['L']

        deltaL=(pnewL-pprevL)+(cnewL-cprevL)
        L+=deltaL

        potential['deltaL']=deltaL
        potential['newL']=L

        if debug:
            print("compute_potential_likelihood ")
            print("    parent " +str(parent))
            print("    pprevL " +str(pprevL))
            print("    child " +str(child))
            print("    cprevL " +str(cprevL))
            print("    potential "+str(potential))

        return potential

    def compute_likelihood(self, node, debug=False):
        """
        \log(L) = \sum_{i=1}^{n} ( \
            ( e_i * \log[p_i] ) + ( (nL_i*nR_i - e_i) * \log[1-p_i] ) \
        )
        """
        nL = self.likelihood_L_fast(node)
        nR = self.likelihood_R_fast(node)
        e = self.likelihood_E_naive(node)
        p = 0.0

        if e == 0:
            p = 0.0
        elif e == nL*nR:
            p = 1.0
        else:
            p = e/float(nL*nR)
        
        try:
            L = self.likelihood(nL, nR, e, p, debug=debug)
            return e, nL, nR, p, L
        except ValueError as ve:
            print("not able to compute likelihood for node "+str(node))
            print("e "+str(e))
            print("p "+str(p))
            print("nL "+str(nL))
            print("nR "+str(nR))
            print(str(self))
            raise ve

    def likelihood(self, nL, nR, e, p, debug=False):
        """
        """
        if debug:
            print("likelihood "+" ".join([str(x) for x in [nL, nR, e, p]]))

        if e == 0:
            # If the number of edges with a given lowest common ancestor 
            # in the dendrogram is 0, then p=0/nL*nR, and evaluating
            # the liklihood function results in a ValueError from math.log.
            # We do not want to be biased towards assortative or disassortative
            # models, so don't assume that the lack of edges between the left
            # and right subtrees indicates a lack of fit.
            return 0.0
        elif e == nL*nR:
            # This is the other extreme.  The dendrogram's subtree fits the 
            # subgraph perfectly.  As with the case where e is 0, evaluating
            # the likelihood function in this case would result in a 
            # ValueError.  To avoid that, return the smallest possible float.
            return 0.0
        else:
            return e*math.log(p)+(nL*nR-e)*math.log(1-p)

    def likelihood_E_naive(self, parent, debug=False):
        """
        E_i is the number of edges in G that have lowest common
        ancestor i in D.  
        """
        # Do a depth-first search of the dendrogram starting at parent
        # to find all the graph nodes with the lowest common ancestor 
        # given by parent.  If a node has no successors, it is a graph
        # node, so count its edges.  We don't want to overcount edges,
        # so keep them in a set.

        left_child=self.node[parent]['left']
        right_child=self.node[parent]['right']


        left_graph_nodes=None
        right_graph_nodes=None

        if self.is_graph_node(left_child):
            left_graph_nodes=set([left_child])
        else:
            left_graph_nodes=self.graph_nodes_below(left_child)

        if self.is_graph_node(right_child):
            right_graph_nodes=set([right_child])
        else:
            right_graph_nodes=self.graph_nodes_below(right_child)

        if debug:
            print("likelihood_E_naive, parent "+str(parent))
            print("    left_graph_nodes "+str(left_graph_nodes))
            print("    right_graph_nodes "+str(right_graph_nodes))
            print("    graph_edges "+str(self.graph_edges))

        e=set()

        for left in left_graph_nodes:
            for right in right_graph_nodes:
                if left in self.G.edge[right]:
                    e.add(tuple((left,right)))

        return len(e)

    def get_number_of_leaf_nodes(self, parent, side):
        key='n'+side
        return self.node[parent][key]

    def set_number_of_leaf_nodes(self, parent, side, n):
        key='n'+side
        self.node[parent][key] = n

    def add_to_number_of_leaf_nodes(self, parent, side, n):
        """
        """
        key='n'+side
        if key in self.node[parent]:
            self.node[parent][key] += n
        else:
            self.node[parent][key] = n

    def compute_initial_numbers_of_leaf_nodes(self, G):
        """
        """
        for graph_node in G.nodes_iter():
            node=graph_node
            while True:
                parent=self.parent(node)
                if parent==None:
                    break
                if self.is_left_child(parent, node):
                    # node contributes to left count
                    self.add_to_number_of_leaf_nodes(parent, 'L', 1)
                else:
                    # node contributes to right count
                    self.add_to_number_of_leaf_nodes(parent, 'R', 1)
        
                node=parent

    def update_numbers_of_leaf_nodes(self, parent):
        left_child=self.node[parent]['left']
        left_count = 0

        if self.is_graph_node(left_child):
            left_count = 1
        else:
            left_count += self.get_number_of_leaf_nodes(
                left_child, 'L')
            left_count += self.get_number_of_leaf_nodes(
                left_child, 'R')

        self.set_number_of_leaf_nodes(parent, 'L', left_count)

        right_child=self.node[parent]['right']
        right_count = 0

        if self.is_graph_node(right_child):
            right_count = 1
        else:
            right_count += self.get_number_of_leaf_nodes(
                right_child, 'L')
            right_count += self.get_number_of_leaf_nodes(
                right_child, 'R')

        self.set_number_of_leaf_nodes(parent, 'R', right_count)

    def likelihood_L_fast(self, parent):
        return self.get_number_of_leaf_nodes(parent, 'L')

    def likelihood_L_naive(self, parent):
        """
        L_i is the number of leaves in the left subtree rooted at i.
        """
        left_child=self.node[parent]['left']
        return len(self.graph_nodes_below(left_child))

    def likelihood_R_fast(self, parent):
        return self.get_number_of_leaf_nodes(parent, 'R')

    def likelihood_R_naive(self, parent):
        """
        R_i is the number of leaves in the right subtree rooted at i.
        """
        right_child=self.node[parent]['right']
        return len(self.graph_nodes_below(right_child))

    def graph_nodes_below(self, parent):
        """
        Find the graph (leaf) nodes below the given dendrogram node.
        """
        try:
            left_child = self.node[parent]['left']
            right_child = self.node[parent]['right']

            graph_nodes = set()
            nodes = set([left_child, right_child])

            while len(nodes) > 0:
                node = nodes.pop()
                try:
                    left = self.node[node]['left']
                    right = self.node[node]['right']
                    nodes.update(set([left, right]))
                except KeyError:
                    graph_nodes.add(node)
    
            return graph_nodes
        except KeyError:
            return set([parent])

    def least_upper_bound(self, u, v):
        parent = self.parent(u)
        upath = []
        while parent != None:
            upath.append(parent)
            parent = self.parent(parent)

        parent = self.parent(v)
        vpath = []
        while parent != None:
            vpath.append(parent)
            parent = self.parent(parent)

        upath.reverse()
        vpath.reverse()

        lub = upath[0]

        for i, parent in enumerate(upath):
            try:
                if parent != vpath[i]:
                    break
                lub = parent
            except IndexError:
                break

        return lub

    def generate_graph(self):
        H = nx.Graph()
        H.add_nodes_from(self.G.nodes())

        nodes = H.nodes()

        for i,u in enumerate(nodes):
            for j in range(i + 1, len(nodes)):
                v = nodes[j]
                lub = self.least_upper_bound(u, v)
                p = self.node[lub]['p']
                if random.random() < p:
                    H.add_edge(u, v)

        return H

    def sample_splits(self):
        """
	    This method calls build_split, which is a precursor of the
	    existence of a consensus dendrogram, so it should be part
    	of the Dendrogram class.
        """
        for dnode in self.dendrogram_nodes_iter():
            split = self.build_split(dnode)
            self.split_histogram[split] += 1
        self.num_samples += 1

    
    def build_split(self, dnode):
        """
	    This method builds the splits that ultimately allow one to
	    generate a consensus dendrogram, so it should be part of
    	the Dendrogram class.
        """
        # Get the set of all graph nodes (self.graph_nodes).
        # Get the set C of graph nodes below the dendrogram node dnode.
        C = self.graph_nodes_below(dnode)
        # Get the set M of graph nodes not below the dendrogram node dnode.
        M = self.graph_nodes.difference(C)
        # Initialize split as all dashes.
        split = ['-'] * len(self.graph_nodes)

        for i, gnode in enumerate(self.graph_nodes_list):
            if gnode in C and gnode in M:
                raise Exception('Graph node '+str(gnode)+' exists on ' +
                    'both sides of the split for dendrogram node'+str(dnode))
            elif gnode not in C and gnode not in M:
                raise Exception('Graph node '+str(gnode)+' is not on ' +
                    'either side of the split for dendrogram node'+str(dnode))

            if gnode in C:
                split[i]='C'
            else:
                split[i]='M'

        return ''.join(split)

    @staticmethod
    def remove_infrequent_splits(histogram, num_samples, thresh=0.5):
        """
        This is a method of the Dendrogram class because when we
        start to use it to infer the hierarchical structure of
        large networks, we'll need to prune the split histogram
        while we're sampling in order to avoid running out of
        memory.
        """
        newhist = dict(histogram)

        for split in list(newhist.keys()):
            num_occurrences = newhist[split]
            if num_occurrences / num_samples < 0.5:
                del newhist[split]

        return newhist

    def get_dendrogram_height(self, root='_D0'):
        def _get_dendrogram_height(node):
            if str(node).startswith('_D'):
                lheight = _get_dendrogram_height(self.node[node]['left'])
                rheight = _get_dendrogram_height(self.node[node]['right'])
                return max(lheight, rheight) + 1
            else:
                # Leaf node
                return 0

        return _get_dendrogram_height(root)

    # FIXME: plotting doesn't work correctly quite yet.  IIRC, two things
    # can break: leaf nodes can be labeled incorrectly and the
    # structure of the dendrogram can be wrong.  It works on small dendrograms.
    # A good test case is to fit an HRG for Zachary's karate club, plot it,
    # and compare the result to the actual dendrogram in the GML file.
    def plot(self):
        L, labels, probabilities = self.linkage()
        dendrogram = sch.dendrogram(L, labels=labels,
            color_threshold=0,
            link_color_func=lambda x: "0.0")
        plt.gray()
        # The dictionary returned by scipy.cluster.hierarchy.dendrogram
        # has two lists of lists.  Each element of 'dcoord' is a 4-element
        # list [LB, LT, RT, RB], where:
        #
        #   LB: Y coordinate of the bottom of the left branch
        #   LT: Y coordinate of the top of the left branch
        #   RT: Y coordinate of the top of the right branch
        #   RB: Y coordinate of the bottom of the right branch.
        #
        # Each element of 'icoord' is a 4-element list [LB, LT, RT, RB],
        # where:
        #
        #   LB: X coordinate of the bottom of the left branch
        #   LT: X coordinate of the top of the left branch
        #   RT: X coordinate of the top of the right branch
        #   RB: X coordinate of the bottom of the right branch.
        #
        # We describe these coordinates as 'top' and 'bottom' because
        # we plot the fitted hierarchical model vertically with the
        # root of the tree at the top of the figure.
        ycoords = dendrogram['dcoord']
        xcoords = dendrogram['icoord']
        for i in range(len(ycoords)):
            y = ycoords[i][1]
            x = (xcoords[i][1] + xcoords[i][2])/float(2)
            #plt.plot(x, y, 'ro')
            plt.plot(x, y, 'o', c=str(1 - probabilities[i]))
            plt.annotate('%.2f' % probabilities[i], (x, y),
                xytext=(0, -8), textcoords='offset points',
                va='top', ha='center')
        #plt.xticks([])
        plt.yticks([])
        return dendrogram

    def linkage(self, root='_D0'):
        # Compute the left-right ordering of leaf nodes so we can use
        # 0-based indices instead of names in the linkage matrix and
        # tell scipy.cluster.hierarchy.dendrogram how to label them.
        lrorder = {}
        lrindex = 0
        lrq = [root]
        while len(lrq):
            node = lrq.pop(0)
            if self.is_dendrogram_node(node):
                lrq.insert(0, self.right_child(node))
                lrq.insert(0, self.left_child(node))
            else:
                lrorder[node] = lrindex
                lrindex += 1
    
        # Traverse the tree depth-first, following left edges before
        # right edges.  

        def _label_dendrogram_nodes(D, node, height, state):
            if str(node).startswith('_D'):
                # Get the left and right children.
                left = D.node[node]['left']
                right = D.node[node]['right']
                lheight = _label_dendrogram_nodes(
                    D, self.node[node]['left'], height, state)
                rheight = _label_dendrogram_nodes(
                    D, self.node[node]['right'], height, state)
                curheight = max(lheight, rheight) + 1
                if curheight == height:
                    state[node] = state['label-counter']
                    state['label-counter'] += 1
                    state['probabilities'].append(D.node[node]['p'])
                return curheight
            else:
                return 0
            
        # Get the height of the dendrogram
        height = self.get_dendrogram_height()

        # Apply a label to each internal dendrogram node that
        # is appropriate for building a linkage matrix (i.e.
        # increasing left-to-right, starting with all internal
        # dendrogram nodes at height 1, then height 2, up to
        # the root).
        state = {
            'label-counter': self.number_of_graph_nodes,
            'probabilities': []
        }

        for h in range(1, height+1):
            _label_dendrogram_nodes(self, root, h, state)
    
        # A list of lists.  Eventually, the ith list in tmp becomes the
        # ith row in the linkage matrix.
        tmp = []
    
        def build_linkage_matrix_rows(node, state):
            if str(node).startswith('_D'):
                # Get the left and right children.
                left = self.node[node]['left']
                right = self.node[node]['right']
    
                # Get their heights and leaf counts.
                left_height, left_n = build_linkage_matrix_rows(left, state)
    
                left_id = ''
                if str(left).startswith('_D'):
                    left_id = state[left]
                else:
                    left_id = lrorder[left]
    
                right_height, right_n = build_linkage_matrix_rows(right, state)
    
                right_id = ''
                if str(right).startswith('_D'):
                    right_id = state[right]
                else:
                    right_id = lrorder[right]
    
                height = max(left_height, right_height) + 1
                n = left_n + right_n
    
                if height > len(tmp):
                    tmp.append([])
    
                tmp[height - 1].append([left_id, right_id, height, n])
    
                return height, n
            else:
                height = 0
                n = 1
                return height, n

        build_linkage_matrix_rows(root, state)
    
        ntmp = 0
        for links in tmp:
            ntmp += len(links)
    
        L = np.zeros(shape=(ntmp, 4))
    
        linki = 0
    
        for links in tmp:
            for link in links:
                L[linki, :] = link
                linki += 1

        labels = [None]*len(lrorder)
        for key in lrorder.keys():
            val = lrorder[key]
            labels[val] = key
    
        return L, labels, state['probabilities']


class ConsensusDendrogramBuilder(object):

    def merge_histograms(self, histograms):
        """
        histograms is a list of split histograms (dict from split to count)
        """
        self.validate_histograms(histograms)

	    # Merge the histograms.  In the merged histogram, each split
    	# S that occurs in multiple histograms Hi, ..., Hk with counts
        # Ci, ..., Ck will have count sum(Ci, ..., Ck).
        merged = {}

        for hist in histograms:
            for split in hist.keys():
                try: 
                    merged[split] += hist[split]
                except KeyError:
                    merged[split] = hist[split]

        return merged

    def validate_histograms(self, histograms):
        if histograms is None or len(histograms) == 0:
            raise Exception('histograms argument is invalid ' +str(histograms))

        # Verify that all splits have the same length.
        n = 0
        for hist in histograms:
            for split in hist.keys():
                if n == 0:
                    n = len(split)

                if len(split) != n:
                    raise Exception('split ' + split +
                        ' has ' + str(len(split)) + ' nodes, ' +
                        'not the required ' + str(n))

    def get_splits_of_size(self, n, histogram, ch='M'):
        splits=[]
        for key in histogram.keys():
            if key.count(ch) == n:
                splits.append(key)
        return splits

    def build(self, graph_nodes, histograms, num_samples):
        """
        The information this function requires in order to build a
        consensus dendrogram:
            nodes:          A list of nodes from the original network.
                            The position of node i in this list must
                            correspond to the i-th character in a split.

            histograms:     A dictionary from splits to counts or a list
                            of dictionaries from splits to counts.  If it
                            is a list, each element is assumed to be a
                            histogram of splits on a dendrogram built atop
                            the same underlying network.

            num_samples:    The number of times the dendrogram was
                            sampled while building the split histogram.
        """

        histogram = {}

        if type(histograms) == list:
            self.validate_histograms(histograms)
            histogram = self.merge_histograms(histograms)
            num_samples = num_samples * len(histograms)
        elif type(histograms) in [dict, defaultdict]:
            self.validate_histograms([histograms])
            histogram = histograms
        else:
            raise Exception('unknown type of histograms ' +
                str(type(histograms)))

        histogram = Dendrogram.remove_infrequent_splits(histogram, num_samples)

        if len(histogram) == 1:
            print('histogram ' + str(histogram))

        consensus = nx.DiGraph()

        ################################################################# 
        # To build the majority consensus tree, we do the following:
        #
        # For each possible number of Ms in the split string (a
        # number that ranges from n-2 down to 0), and for each split
        # with that number of Ms, we create a new internal node of
        # the tree, and connect the oldest ancestor of each C to that
        # node (at most once). Then, we update our list of oldest
        # ancestors to reflect this new join, and proceed.
        ################################################################# 

        cnode_id = 0
        cnode = '_C' + str(cnode_id)
        ancestors = {}

        for i in range(len(graph_nodes) - 2, 0, -1):
            splits = self.get_splits_of_size(i, histogram)
            # Iterate over the split's containing exactly i M's.
            for split in splits:
                w = histogram[split]
                consensus.add_node(cnode, weight=w)

                # Iterate over each character in the split (either 'C' or 'M').
                for i, ch in enumerate(split):
                    if ch == 'M':
                        continue

                    graph_node = graph_nodes[i]

		            # This seems to be the same as asking whether
        		    # consensus.has_node(graph_node) is true.
                    if graph_node not in ancestors:
                        # This graph node has not been seen before.
                        consensus.add_node(graph_node)
                        consensus.add_edge(cnode, graph_node)
                    else:
                        # This graph node has been seen before.

                        # This is equivalent to
                        # p = consensus.predecessors(graph_node)[0]
                        p = ancestors[graph_node]
                        
                        predecessors = consensus.predecessors(p)

                        if len(predecessors) > 1:
                            raise Exception('node ' + str(p) +
                                ' has multiple predecessors ' +
                                str(predecessors))

                        try:
                            pprime = predecessors[0]

                        except IndexError:
                            pprime = None
                        
                        # Let the parent of this graph node be P and let
                        # the parent of P be P'.  If P' is not the current
                        # dendrogram node (the variable cnode), then change P
                        # to have cnode as its parent.  This entails removing
                        # the directed edge (P', P) and adding the directed
                        # edge (cnode, P).
                        if pprime != cnode:
                            try:
                                consensus.remove_edge(pprime, p)
                            except nx.NetworkXError:
                                pass
                            consensus.add_edge(cnode, p)

                        #consensus.add_edge(cnode, ancestors[graph_node])

                    # Remove the edge from (P, graph_node), where P is the
                    # current ancestor of graph_node; add the edge
                    # (cnode, graph_node).
                    ancestors[graph_node] = cnode
                
                cnode_id += 1
                cnode = '_C'+str(cnode_id)

        return consensus

    def consensus_tree_as_string(self, consensus):
        s=''

        for cnode, d in consensus.nodes_iter(data=True):
            if not str(cnode).startswith('_C'):
                continue
            weight = d['weight']
            parent = None
            parents = consensus.predecessors(cnode)
            if len(parents) == 0:
                pass
            elif len(parents) == 1:
                parent = str(parents[0])
            else:
                raise Exception('Consensus dendrogram node '+cnode+
                    ' has > 1 parent: '+str(parents))

            successors=consensus.successors(cnode)
            
            s += str([cnode, str(weight),
                str(len(successors)), str(successors)])
            s += '\n'

        return s

