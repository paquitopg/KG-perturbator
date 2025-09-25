import networkx as nx
from typing import Dict, Any

def json_to_networkx(kg_json: Dict[str, Any]) -> nx.MultiDiGraph:
    """
    Convert a KG in dict format (with 'entities' and 'relations') to a NetworkX MultiDiGraph.
    Node attributes are copied; edge attributes are only 'type' by default.
    """
    G = nx.MultiDiGraph()
    for entity in kg_json.get("entities", []):
        node_id = entity["id"]
        G.add_node(node_id, **{k: v for k, v in entity.items() if k != "id"})
    for rel in kg_json.get("relations", []):
        src = rel["source"]
        tgt = rel["target"]
        edge_type = rel.get("type", "")
        edge_attrs = {k: v for k, v in rel.items() if k not in ("source", "target", "type")}
        G.add_edge(src, tgt, type=edge_type, **edge_attrs)
    return G

def networkx_to_json(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """
    Convert a NetworkX MultiDiGraph back to the JSON dict format with 'entities' and 'relations'.
    """
    entities = []
    for node, attrs in G.nodes(data=True):
        entity = {"id": node}
        entity.update(attrs)
        entities.append(entity)
    relations = []
    for src, tgt, key, attrs in G.edges(keys=True, data=True):
        rel = {"source": src, "target": tgt}
        rel.update(attrs)
        relations.append(rel)
    return {"entities": entities, "relations": relations} 