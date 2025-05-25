from graphviz import Digraph
from typing import List
from .tree_node import TreeNode, LeafNode
from collections import defaultdict


class SynergyTreeVisualizer:
    """
    Visualizes synergy trees separately using Graphviz.
    """
    def __init__(self, root_nodes: List[TreeNode]):
        """
        root_nodes: List of root nodes of the synergy trees.
        """
        self.root_nodes = root_nodes
        self.dot = Digraph(format="svg", 
                           graph_attr={"size": "12", "ranksep": "1.5", "nodesep": "1.0"})

    def visualize(self, disease_name: str, filename="synergy_tree"):
        """
        Generates and renders separate synergy tree visualizations.
        """
        title = f"Synergy Trees Visualization for {disease_name})"
        self.dot.attr(label=title, labelloc="t", fontsize="20", fontcolor="black")

        for i, root in enumerate(self.root_nodes):
            with self.dot.subgraph(name=f"cluster_{i}") as sub:
                sub.attr(label=f"Tree {i+1}", fontsize="16", style="dashed")
                self._add_node(sub, root, tree_index=i)

        self.dot.render(filename, view=True)

    def _add_node(self, graph: Digraph, node: TreeNode, parent_id=None, tree_index=0):
        """
        Recursively adds nodes and edges to the Graphviz subgraph.
        """
       
        node_id = f"Tree{tree_index}-" + "-".join(map(str, node.get_feature_indices))

        if isinstance(node, LeafNode):
          
            leaf_id = f"{node_id}-leaf"

            label = f"Leaf: {node.get_feature_indices}\nLabel: {node.label}"  
            shape = "box"
            graph.node(leaf_id, label, shape=shape)

            if parent_id:
                graph.edge(parent_id, leaf_id)
        else:

            label = f"Internal: {node.get_feature_indices}\nSynergy: {node.synergy:.6f}"
            shape = "ellipse"
            graph.node(node_id, label, shape=shape)

            if parent_id:
                graph.edge(parent_id, node_id)

            for child in node.children:
                self._add_node(graph, child, node_id, tree_index)


class SynergyTreeVisualizerconnected:
    """
    Visualizes synergy trees separately using Graphviz.
    """
    def __init__(self, root_nodes: List[TreeNode]):
        """
        root_nodes: List of root nodes of the synergy trees.
        """
        self.root_nodes = root_nodes
        self.dot = Digraph(format="svg", 
                           graph_attr={"size": "12", "ranksep": "1.5", "nodesep": "1.0"})
        self.leaf_cache = defaultdict(str)  # A dictionary to store previously added leaf nodes

    def visualize(self, disease_name: str, filename="synergy_tree"):
        """
        Generates and renders separate synergy tree visualizations.
        """
        title = f"Synergy Trees Visualization for {disease_name}"
        self.dot.attr(label=title, labelloc="t", fontsize="20", fontcolor="black")

        for i, root in enumerate(self.root_nodes):
            with self.dot.subgraph(name=f"cluster_{i}") as sub:
                sub.attr(label=f"Tree {i+1}", fontsize="16", style="dashed")
                self._add_node(sub, root, tree_index=i)

        self.dot.render(filename, view=True)

    def _add_node(self, graph: Digraph, node: TreeNode, parent_id=None, tree_index=0):
        """
        Recursively adds nodes and edges to the Graphviz subgraph.
        """
        node_id = f"Tree{tree_index}-" + "-".join(map(str, node.get_feature_indices))

        if isinstance(node, LeafNode):
            # Create a unique identifier for the leaf node based on its feature indices
            leaf_key = "-".join(map(str, node.get_feature_indices))
            
            # If this leaf has already been added, reuse the existing ID
            if leaf_key not in self.leaf_cache:
                leaf_id = f"leaf_{leaf_key}"
                label = f"Leaf: {node.get_feature_indices}\nLabel: {node.label}"  
                shape = "box"
                graph.node(leaf_id, label, shape=shape)
                self.leaf_cache[leaf_key] = leaf_id  # Store the leaf ID for reuse
            else:
                leaf_id = self.leaf_cache[leaf_key]  # Use the cached ID
            
            if parent_id:
                graph.edge(parent_id, leaf_id)

        else:
            label = f"Internal: {node.get_feature_indices}\nSynergy: {node.synergy:.6f}"
            shape = "ellipse"
            graph.node(node_id, label, shape=shape)

            if parent_id:
                graph.edge(parent_id, node_id)

            for child in node.children:
                self._add_node(graph, child, node_id, tree_index)


