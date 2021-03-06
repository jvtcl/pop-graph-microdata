import sys
import pandas as pd
import numpy as np
import networkx as nx
from networkx.linalg import spectrum
from scipy import stats
from scipy.spatial import distance
from sklearn import neighbors, metrics, cluster
from brewer2mpl import qualitative
import community
import pysal as ps
import matplotlib.pyplot as plt


def getpopw(tcl,v):

    """
    Helper function for computing the percent
    of an area's pop with one or more specified
    traits. If more than one trait is specified,
    result describes populations for whom those
    traits intersect.
    """

    s=tcl.groupby(v).size()

    if(len(v)>1):
        # check to ensure the intersection is being used
        # if no levels add up to length of labels, no intersection
        s_labs=np.array(s.index.labels)
        check=np.sum(s_labs,axis=0)
        intersects=any(check==len(s.index.labels))

        w=0
        if intersects:
            w+=float(s[len(s)-1])/float(tcl.shape[0])

    else:

            w=float(s[len(s)-1])/float(tcl.shape[0])

    return w


def popwmat(traits):

    popw=np.empty(([traits.shape[1]]*2))

    for i in traits.columns:

        ix=np.where(traits.columns==i)[0][0]
        pw=[getpopw(traits,[i,j]) for j in traits.columns]
        popw[ix]=pw

    return popw


def prep_traits(traits,subset=None,sub_op='union',exclude=None,drop_zero=False):

        ## subset by target population(s), if given
        if subset is not None:

            sub_traits=traits.loc[:,subset]

            # if length of subset >1
            if len(subset)>1:

                if sub_op=='union':
                    # logical 'OR' subset rows by sum of the traits of interest >=1
                    t=sub_traits.apply('sum',axis=1)
                    tix=np.where(t>=1)

                elif sub_op=='intersect':
                    # logical 'AND' subset rows by product of traits of interest ==1
                    t=np.prod(sub_traits,axis=1)
                    tix=np.where(t==1)

            else:
                # subset rows where trait == 1
                tix=np.where(sub_traits==1)[0]

            # subset the data row-wise
            traits=traits.iloc[tix]

            # remove the subset trait(s) from the df since they are default
            traits=traits.loc[:,~traits.columns.isin(subset)]

        ## exclude traits, if given
        if exclude is not None:
            traits=traits.loc[:,~traits.columns.isin(exclude)]

        if(drop_zero):
        ## drop all-zero columns ##
            traits=traits.loc[:,(traits != 0).any(axis=0)]

        return traits


def co_occurence_matrix(traits,subset=None,sub_op='union',exclude=None,drop_zero=False,pop_weights=None):

    """
    subset: an optional list of traits for subsetting observations
    sub_op: type of subset if more than one trait given, one of 'union'
        (any obs with matching traits) or 'intersect'
        (observations with all matching traits)

    exclude: an optional list of traits to exclude
    """

    tcl_t=np.transpose(traits)

    affmat=1-np.matrix(metrics.pairwise.pairwise_distances(tcl_t,metric='jaccard'))

    if pop_weights is not None:

        # popw=popwmat(traits)

        ## weight the co-occurence matrix by pop size ##
        affmat=np.array(affmat)*np.array(1+pop_weights)

    return affmat


def simplify_co_occurrence(co,nn=3):

    # quintile breaks (crude, still need to account for upper/diagonals)
    q = ps.Quantiles(co).yb
    q = np.reshape(q, co.shape)

    # nearest neighbors graph
    knn = neighbors.NearestNeighbors(n_neighbors = nn)
    neigh = knn.fit(co)
    knn_mat = neigh.kneighbors_graph(co).toarray()

    out = np.multiply(q,knn_mat)

    return out


def popGraph(affmat,labels,cut_links=None,exclude=None,verbose=False):

    """
    Generates a weighted graph of co-occurrences of population traits in
    an area of interest. Graph edges are weighted by the co-occurence of
    each i,j traits, scaled by the proportion of the population possessing
    both traits.

    affmat: co-occurrence matrix
    labels: node labels
    """

    ## cut links if specified ##
    if cut_links is not None:
        affmat=np.where(affmat>cut_links,affmat,0)

    ## Build the graph ##
    node=[i for i in range(affmat.shape[0])]
    # label=tcl.columns.tolist()
    labs=dict(zip(node,labels))

    G=nx.from_numpy_matrix(affmat)
    G=nx.relabel_nodes(G,labs)

    return G


def getCentrality(G):

    """
    Computes the eigenvector centrality of a weighted graph of population
    traits for an area of interest.
    """

    ## Compute centrality ##
    vw=nx.eigenvector_centrality(G,weight='weight',max_iter=1000)

    ## Write outputs ##
    vwd=pd.DataFrame.from_dict(vw,orient='index')
    vwd.columns=['cent']
    # vwd.sort_values('cent',ascending=False).round(2)
    return vwd


def getCentrality_all(loc_graphs,verbose=True):

    """
    Computes eigenvector centrality for a series of population traits in each
    location in a region of interest, where 'dat' is a dataframe of m pop traits
    and 'locs' is a list of target location GEOIDs.

    Returns a dataframe of n locations by m population traits.
    """

    locs=loc_graphs.keys()

    cent=pd.DataFrame()
    for loc in locs:

        if verbose:
            print loc

        try:
            G=loc_graphs[loc][0]
            c=getCentrality(G).transpose()
            c.index=[loc]
            cent=pd.concat([cent,c],axis=0)
        except:
            print 'Could not compute eigenvector centrality for '+str(loc)+'.'

    return cent


def drawPopGraph(G,tcl=None,part=None,edge_stand=True,emph=True):

    """
    Draws the weighted graph of population traits for an area of interest.

    If population traits data ('tcl') is provided, will size the nodes
    according to the proportion of that trait in the area's toal population.

    If a graph partitioning ('part') is provided (dict with keys=traits,
    values=membership), will plot the membership on the graph nodes.
    """

    # edge values, for plotting
    edgeval=np.array([i[2].values()[0] for i in G.edges(data=True)])

    # range-standardize the edges if specified
    if edge_stand:
        edgeval=(edgeval-min(edgeval))/(max(edgeval)-min(edgeval))

    if part is not None:
        # Colors for community visualization
        cols=qualitative.Set1['max'].hex_colors+qualitative.Dark2['max'].hex_colors
        node_cols=[cols[part[n]] for n in G.nodes()]
    else:
        node_cols=None

    if tcl is not None:
        popsize=tcl.apply('sum',axis=0)/tcl.shape[0]
        node_size=[1000*popsize[n] for n in G.nodes()]
    else:
        node_size=None

    if emph:
        nx.draw_spring(G,width=10*(edgeval**2),node_color=node_cols,node_size=node_size,with_labels=True) # with addl emphasis on edges
    else:
        nx.draw_spring(G,width=edgeval,node_color=node_cols,node_size=node_size,with_labels=True) # original



def build_pop_graphs(dat,locs,subset=None,sub_op='union',exclude=None,pop_weights=True,drop_zero=False,cut_links=None,part_method='louvain',verbose=True):

    """
    For a series of location GEOIDs 'locs',
    Generates a dictionary of key: GEOID
    value 0: pop graph
    value 1: louvain partition of pop graph
    """

    loc_graphs=dict()

    for loc in locs:

        if verbose:
            print loc

        # traits=dat[dat.GEOID.isin([loc])].iloc[:,2:]
        traits=dat[dat.index.isin([loc])]

        ## prep traits, if optional param's given
        if (subset is not None) or (exclude is not None) or (drop_zero==True):
            traits=prep_traits(traits,subset,sub_op,exclude,drop_zero)

        # Generate population weights
        if pop_weights:
            pw = popwmat(traits)
        else:
            pw = None

        # Generate co-occurrence matrix
        co=co_occurence_matrix(traits=traits,subset=subset,sub_op=sub_op,exclude=exclude,pop_weights=pw,drop_zero=drop_zero)

        co[np.where(np.isnan(co))] = 0 # give empty co-occurrences zeroes

        labels=traits.columns
        labels=labels.tolist()

        G=popGraph(co,labels=labels,cut_links=cut_links)

        if part_method=='louvain':

            part=community.best_partition(G)

        elif part_method=='affinity':

            afclust=cluster.AffinityPropagation(affinity='precomputed',max_iter=1000)
            part=dict(zip(labels,afclust.fit(co).labels_))

        loc_graphs[loc]=(G,part,co,pw)

    return loc_graphs


def part_dist(loc_graphs,loc1,loc2):

    g_loc1 = loc_graphs[loc1][0] # graph loc 1
    p_loc1 = loc_graphs[loc1][1].values() # partition loc 1

    g_loc2 = loc_graphs[loc2][0] # graph loc 2
    p_loc2 = loc_graphs[loc2][1].values() # partition loc 2

    # compute laplacian spectrum for each graph
    spec1 = spectrum.laplacian_spectrum(g_loc1)
    spec2 = spectrum.laplacian_spectrum(g_loc2)

    e = stats.energy_distance(spec1,spec2)
    m = metrics.adjusted_mutual_info_score(p_loc1,p_loc2)

    dist = (1-m)*(1+e)

    return dist


def part_dist_mat(loc_graphs,metric='info',pop_weights=True,verbose=True):

    """
    Generates a distance matrix based on label swapping among the
    louvain community partitions of n locations' pop graphs

    loc_graphs: a dict containing geoid keys, and values 0: pop graph,
        1:  community partition
    """

    locs=loc_graphs.keys()

    md=np.zeros((len(locs),len(locs)))

    for loc in locs:

        if verbose:
            print loc

        loc_pos=np.where(locs==loc)[0][0]

        compare=np.delete(locs,loc_pos)

        for comp in compare:

            comp_pos=np.where(locs==comp)[0][0]

            # if type=='info':
            #     md[loc_pos,comp_pos]=info_dist(loc_graphs,loc,comp)
            # elif type=='mod':
            #     md[loc_pos,comp_pos]=mod_dist(loc_graphs,loc,comp)
            md[loc_pos,comp_pos]=part_dist(loc_graphs,loc,comp)

    return md
