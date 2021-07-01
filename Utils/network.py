#!/usr/bin/env python

from typing import Any, Callable
import os
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

from Model.bn import BayesNet


def draw_network(model: BayesNet,
                 ntype: str = "dag",
                 layout: Callable[[Any], dict] = nx.shell_layout,
                 save_path: str = "../Pics",
                 pic_name: str = "network") -> None:
    """Draw the Bayesian Network structure using NetworkX. """

    if ntype == "skel":

        graph = nx.Graph()
        graph.add_nodes_from(model.state_names)
        graph.add_node('Sp')
        graph.add_edges_from([(u, 'Sp' if v == model.lnode else v) for (u, v) in model.skel.edges])

    elif ntype == "dag":

        graph = nx.DiGraph()
        graph.add_nodes_from(model.state_names)
        graph.add_node('Sp')
        # Adjust the edges: modify the arrow direction & label node name
        graph.add_edges_from([(u, 'Sp' if v == model.lnode else v) for (u, v) in model.dag.edges])

    else: # ntype == "pdag"

        graph = nx.DiGraph()
        graph.add_nodes_from(model.state_names)
        graph.add_node('Sp')
        # Adjust the edges: modify the arrow direction & label node name
        graph.add_edges_from([(u, 'Sp' if v == model.lnode else v) for (u, v) in model.pdag.edges])

    layout = layout(graph)  # Set node layout
    nx.draw_networkx_nodes(graph, layout, node_color='silver', edgecolors='k',
                           node_size=np.repeat((500, 1000), (model.feat_length, 1)))
    nx.draw_networkx_edges(graph, layout, edge_color='k', arrows=not ntype == "skel", arrowsize=13,
                           min_source_margin=12, min_target_margin=13)
    nx.draw_networkx_labels(graph, layout, font_color='k')
    plt.savefig(os.path.join(save_path, pic_name + ".svg"))
    print("Done! Network graph saved in {}/{}.svg.".format(save_path, pic_name))
    plt.close()


if __name__ == "__main__":

    from Utils.extract import Sequence

    TRAINING_PATH = "../Data/Training Set"

    site_type = "donor"
    model_path = '../Model'

    seq = Sequence(filepath=TRAINING_PATH, type="train")

    bn = BayesNet(seq, site_type)
    bn.load_model(model_path, "BN")
    draw_network(bn, "skel", nx.shell_layout, pic_name="skeleton")
    draw_network(bn, "pdag", nx.shell_layout, pic_name="pdag")
    draw_network(bn, "dag", nx.shell_layout, pic_name="dag")
