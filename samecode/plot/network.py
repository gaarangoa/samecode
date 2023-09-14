import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def bipartite_graph(left_nodes: dict, right_nodes: dict, edges: dict, direction: dict, ax: object, **kwargs):
    
    '''
    Plot attention map: 
    nodes: A list of all nodes in the data
    edges: A list of all edges in the data [source, target, score]
    direction: A dict where key is a positive and negative score (defines the direction)
    pos_color: color for positive scores (raw values of each feature in the population)
    neg_color: color for positive scores (raw values of each feature in the population)
    label_color: label color in boxes
    edge_color: edge color 
    offset_x: For edges and labels overlap
    offset_y: For edges and labels overlap
    alpha: alpha for edges
    
    Example: 
    
    nodes = ['<cls>',
             'Tumor proliferation rate',
             'Angiogenesis',
             'Matrix',
             'Cancer-associated fibroblasts',
             'Protumor cytokines',
             'Antitumor cytokines',
             'Th1 signature',
             'B cells',
             'NK cells',
             'T cells',
             'MHCI']
    edges = [['Matrix', 'NK cells', 8.823914508677685],
             ['Matrix', 'T cells', 8.06140887859505],
             ['Angiogenesis', 'Th1 signature', 6.868375098347111]]
             
    direction = {'Matrix': 0.30622423947951827,
                 'MHCI': -0.05945478997492155,
                 'B cells': -0.10101842522915781,
                 'NK cells': -0.06531603322426809,
                 'T cells': -0.34819624947872607,
                 'Tumor proliferation rate': 0.6231494073580373,
                 'Protumor cytokines': 0.2560214461158169,
                 'Th1 signature': 0.014631845496634476,
                 '<cls>': 0,
                 'Cancer-associated fibroblasts': 0.4457385288264749,
                 'Angiogenesis': 0.10007748649060139,
                 'Antitumor cytokines': -0.534504312182629}
    
    f, axs = subplots(cols=1, rows=2, w=4.5, h=6.4, return_f=True)
    attention_plot(
        nodes, edges, direction, axs[0], alpha=0.3, edge_color='black', offset_x=0.1, 
        rename = {'<cls>': 'Short-term Survivors', 'Cancer-associated fibroblasts': 'CAFs', 'Tumor proliferation rate': 'Proliferation'}
    )

    '''
    
    pos_color = kwargs.get('pos_color', 'red')
    neg_color = kwargs.get('neg_color', 'darkblue')
    label_color = kwargs.get('label_color', 'white')
    edge_color = kwargs.get('edge_color', 'gray')
    
    left_nodes = {i: ix for ix, i in enumerate(left_nodes)}
    right_nodes = {i: ix for ix, i in enumerate(right_nodes)}   
    
    edges = [[left_nodes[i], right_nodes[j], k, c] for [i, j, k, c] in edges]
    
    rename = kwargs.get('rename', {})
    
    # Add Nodes
    for node_name, y in left_nodes.items():
        y = y / len(left_nodes)
        fc_color_i = pos_color if direction.get(node_name, 0) > 0 else neg_color
        ec_color_i = fc_color_i
        label_color = 'white'
        if direction.get(node_name, 0) == 0:
            fc_color_i = 'white'
            ec_color_i = 'white'
            label_color = 'black'
        ax.annotate(rename.get(node_name, node_name), (0, y), (0, y), color=label_color, bbox=dict(boxstyle='round,pad=0.2', fc=fc_color_i, ec=ec_color_i), horizontalalignment='right')
        
    # Add Nodes
    for node_name, y in right_nodes.items():
        y = y / len(left_nodes)
        fc_color_i = pos_color if direction.get(node_name, 0) > 0 else neg_color
        ec_color_i = fc_color_i
        label_color = 'white'
        if direction.get(node_name, 0) == 0:
            fc_color_i = 'white'
            ec_color_i = 'white'
            label_color = 'black'
        ax.annotate(rename.get(node_name, node_name), (1, y), (1, y), color=label_color, bbox=dict(boxstyle='round,pad=0.2', fc=fc_color_i, ec=ec_color_i),)

    # Add edges
    alpha=kwargs.get('alpha', 0.5)
    offset_x = kwargs.get('offset_x', 0.03)
    offset_y = kwargs.get('offset_y', 0.01)
    for y1, y2, score, ecolor in edges:
        if ecolor == None:
            ecolor = edge_color
        ax.annotate(
            '', 
            xy=(    1-offset_x,  y2/len(left_nodes) + offset_y), 
            xytext=(0+offset_x,  y1/len(left_nodes) + offset_y), 
            arrowprops= {'arrowstyle': '-', 'color': ecolor, 'linewidth': score, 'alpha': alpha},
        )
        
    ax.set_xticklabels([]);
    ax.set_yticklabels([]);

    ax.axis('off')