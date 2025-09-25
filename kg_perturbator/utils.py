"""
Utility functions for KG Perturbator.
"""

import os
import random
import networkx as nx
from typing import Dict, Any, Set
from .llm_integrations.provider_factory import get_llm_provider


def get_llm_provider_from_config(llm_config: Dict[str, Any]):
    """
    Helper method to get LLM provider from configuration.
    Uses environment variable LLM_PROVIDER if available, otherwise uses config.
    """
    # Get provider from environment variable or config
    provider_name = os.getenv("LLM_PROVIDER") or llm_config.get("provider", "vertexai")
    
    # Get args from config
    provider_args = llm_config.get("args", {})
    return get_llm_provider(provider_name, **provider_args)


def load_llm_config(llm_config_path: str = None) -> Dict[str, Any]:
    """Load LLM configuration from llm_config.yaml if it exists."""
    try:
        import yaml
        if llm_config_path is None:
            llm_config_path = "llm_config.yaml"
        if os.path.exists(llm_config_path):
            with open(llm_config_path, 'r') as f:
                return yaml.safe_load(f)
    except Exception:
        pass
    return {"provider": "vertexai", "args": {}}


def generate_unique_node_id(G: nx.MultiDiGraph) -> str:
    """
    Generate a unique node ID for a new entity.
    """
    i = 1
    while True:
        candidate = f"rand_{i}"
        if candidate not in G:
            return candidate
        i += 1


def reassign_entity_ids(G: nx.MultiDiGraph, removed_entities: Set[str], added_entities: Set[str], n_nodes: int) -> Dict[str, str]:
    """
    Reassigns IDs in Graph 2 (G) to be sequential starting from e(original_count + 1).
    Returns a mapping from original ID to new ID for surviving entities only.
    """
    # Get all current nodes in the graph
    current_nodes = list(G.nodes())
    
    # Identify surviving entities (not removed, not added)
    surviving_entities = [node_id for node_id in current_nodes 
                       if node_id not in removed_entities and node_id not in added_entities]
        
    # Create mapping for surviving entities only
    entity_mapping = {}
    new_id_counter = n_nodes + 1  # Start from e(original_count + 1)
    
    # Map surviving entities to new sequential IDs
    for original_id in surviving_entities:
        new_id = f"e{new_id_counter}"
        entity_mapping[original_id] = new_id
        new_id_counter += 1
    
    # Store all edges before node reassignment
    all_edges = list(G.edges(keys=True, data=True))
    
    # Reassign IDs in the graph
    # First, create new nodes with new IDs
    for original_id, new_id in entity_mapping.items():
        if original_id in G.nodes:
            # Copy node data to new ID
            G.add_node(new_id, **G.nodes[original_id])
    
    # Then remove old nodes
    for original_id in entity_mapping.keys():
        if original_id in G.nodes:
            G.remove_node(original_id)
    
    # Re-add edges with updated node IDs
    for src, tgt, key, data in all_edges:
        new_src = entity_mapping.get(src, src)
        new_tgt = entity_mapping.get(tgt, tgt)
        # Only add edges where both endpoints are still in the graph
        if new_src in G.nodes and new_tgt in G.nodes:
            G.add_edge(new_src, new_tgt, key, **data)
    
    return entity_mapping


def add_random_entities(G: nx.MultiDiGraph, n: int) -> Dict[str, str]:
    """
    Add n random entities (nodes) to the graph.
    Returns a mapping of None to new entity IDs.
    """
    added_entities = {}
    for _ in range(n):
        new_id = generate_unique_node_id(G)
        G.add_node(new_id, type="RandomEntity")
        added_entities[None] = new_id
    return added_entities


def remove_random_entities(G: nx.MultiDiGraph, n: int) -> Dict[str, None]:
    """
    Remove n random entities (nodes) from the graph.
    Returns a mapping of removed entity IDs to None.
    """
    nodes = list(G.nodes)
    to_remove = random.sample(nodes, min(n, len(nodes)))
    G.remove_nodes_from(to_remove)
    return {node_id: None for node_id in to_remove}


def add_random_edges(G: nx.MultiDiGraph, n: int) -> None:
    """
    Add n random edges (relations) to the graph.
    """
    nodes = list(G.nodes)
    for _ in range(n):
        src, tgt = random.sample(nodes, 2)
        G.add_edge(src, tgt, type="randomRelation")


def remove_random_edges(G: nx.MultiDiGraph, n: int) -> None:
    """
    Remove n random edges (relations) from the graph.
    """
    edges = list(G.edges)
    to_remove = random.sample(edges, min(n, len(edges)))
    G.remove_edges_from(to_remove) 