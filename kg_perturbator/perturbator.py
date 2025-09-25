from typing import Callable, Dict, Any, Tuple
import random
from .conversion import json_to_networkx, networkx_to_json
import networkx as nx
from . import utils

class KGPerturbator:
    """
    Main class for applying perturbations to Knowledge Graphs (KGs) in JSON format.
    Internally uses NetworkX for graph operations.
    """
    def __init__(self, config: Dict[str, Any], llm_config_path: str = None):
        self.config = config
        # Load LLM config if available
        self.llm_config = utils.load_llm_config(llm_config_path)

    def rename_entities_with_llm(self, G: nx.MultiDiGraph, llm_config: Dict[str, Any]) -> Dict[str, str]:
        """
        Rename entities using an LLM. Returns a mapping from entity ID to entity ID (same ID, entity was renamed).
        The LLM provider is determined by the configuration.
        """
        provider = utils.get_llm_provider_from_config(llm_config)
        
        renamed_entities = {}
        for node_id, attrs in G.nodes(data=True):
            if "name" in attrs and attrs["name"]:
                new_name = provider.rename_entity(attrs)
                if new_name:    
                    # Handle both list and string name formats
                    if isinstance(attrs["name"], list):
                        G.nodes[node_id]["name"][0] = new_name
                    else:
                        G.nodes[node_id]["name"] = new_name
                    # Entity was renamed but kept the same ID
                    renamed_entities[node_id] = node_id
        return renamed_entities

    def rename_relations_with_llm(self, G: nx.MultiDiGraph, llm_config: Dict[str, Any]) -> None:
        """
        Rename relation types using an LLM.
        The LLM provider is determined by the configuration.
        """
        provider = utils.get_llm_provider_from_config(llm_config)

        # Simply iterate over all edges and rename their types
        for src, tgt, key, attrs in G.edges(keys=True, data=True):
            old_type = attrs.get("type")
            if old_type:
                # Use the provider's rename_relation method
                new_type = provider.rename_relation(attrs)
                if new_type:
                    G.edges[src, tgt, key]["type"] = new_type

    def perturb_entities_with_llm(self, G: nx.MultiDiGraph, llm_config: Dict[str, Any]) -> Dict[str, str]:
        """
        Perturbs entities using an LLM to synthesize new descriptions and optionally update names.
        Returns a mapping from entity ID to entity ID (same ID, entity was perturbed).
        """
        provider = utils.get_llm_provider_from_config(llm_config)
        
        update_name = llm_config.get("update_name", False)
        update_description = llm_config.get("update_description", True)

        perturbed_entities = {}

        for node_id, attrs in G.nodes(data=True):
            if attrs.get("type") == "RandomEntity":
                continue

            # Ensure there are some attributes to describe, excluding the ID
            describable_attrs = {k: v for k, v in attrs.items() if k != 'id'}
            if not describable_attrs:
                continue

            # Skip entities without a name for description synthesis
            name_value = attrs.get("name")
            if not name_value or (isinstance(name_value, list) and len(name_value) == 0):
                continue
                
            new_description = provider.synthesize_description(attrs)

            if new_description:
                if update_description:
                    G.nodes[node_id]["description"] = [new_description]

                if update_name and name_value:
                    G.nodes[node_id]["name"] = [new_description]
                
                # Entity was perturbed but kept the same ID
                perturbed_entities[node_id] = node_id

        return perturbed_entities

    def perturb(self, kg_json: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Apply all configured perturbations to the KG (in JSON format).
        Returns the perturbed KG (JSON format) and a mapping from original to perturbed entity IDs.
        This mapping only includes entities that exist in both graphs (surviving entities).
        """
        G = json_to_networkx(kg_json)
        n_nodes = len(G.nodes)
        
        # Track which entities are removed and added
        removed_entities = set()
        added_entities = set()
        
        # Track removed entities
        if self.config.get("remove_entities"):
            removed = utils.remove_random_entities(G, self.config["remove_entities"])
            removed_entities.update(removed.keys())
        
        # Track added entities
        if self.config.get("add_entities"):
            added = utils.add_random_entities(G, self.config["add_entities"])
            added_entities.update(added.values())

        # Edge operations don't affect entity mapping
        if self.config.get("remove_edges"):
            utils.remove_random_edges(G, self.config["remove_edges"])
        if self.config.get("add_edges"):
            utils.add_random_edges(G, self.config["add_edges"])
        
        # Reassign IDs in Graph 2 and create mapping FIRST
        entity_mapping = utils.reassign_entity_ids(G, removed_entities, added_entities, n_nodes)
        
        # Apply LLM perturbations AFTER ID reassignment (these modify content, not structure)
        if self.config.get("llm_rename_entities"):
            self.rename_entities_with_llm(G, self.llm_config)

        if self.config.get("llm_rename_relations"):
            self.rename_relations_with_llm(G, self.llm_config)

        if self.config.get("llm_perturb_entities"):
            llm_config = self.llm_config # Use the loaded LLM config
            self.perturb_entities_with_llm(G, llm_config)
            
        perturbed_json = networkx_to_json(G)
        return perturbed_json, entity_mapping 