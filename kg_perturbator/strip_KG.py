import json

"""
This script is used to strip the sources from the KG.
llm_kg_extraction produces complicated KG with sources, which the perturbator does not need.
This script is used to simplify the KG for the perturbation step.
"""

def strip_sources_from_attribute(attr):
    """
    If attr is a list of dicts with 'value' and 'source_doc_id', return a list of just the 'value's.
    If attr is a list of dicts with only 'value', return a list of 'value's.
    If attr is a list of dicts with 'value' as a list, flatten accordingly.
    Otherwise, return attr unchanged.
    """
    if isinstance(attr, list) and all(isinstance(x, dict) and 'value' in x for x in attr):
        values = []
        for x in attr:
            v = x['value']
            # flatten if value is a list, else just append
            if isinstance(v, list):
                values.extend(v)
            else:
                values.append(v)
        return values
    return attr

def simplify_entity(entity):
    # Remove all keys that start with '_source' or are 'source_doc_id'
    new_entity = {}
    for k, v in entity.items():
        if k.startswith('_source') or k == 'source_doc_id':
            continue
        new_entity[k] = strip_sources_from_attribute(v)
    # Remove 'pekg:' from type if present
    if 'type' in new_entity and isinstance(new_entity['type'], str):
        new_entity['type'] = new_entity['type'].replace('pekg:', '')
    return new_entity

def simplify_relation(relation):
    # Same logic as for entities
    new_relation = {}
    for k, v in relation.items():
        if k.startswith('_source') or k == 'source_doc_id':
            continue
        new_relation[k] = strip_sources_from_attribute(v)
    # Remove 'pekg:' from type if present
    if 'type' in new_relation and isinstance(new_relation['type'], str):
        new_relation['type'] = new_relation['type'].replace('pekg:', '')
    return new_relation

def simplify_kg(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        kg = json.load(f)

    simplified = {
        "entities": [simplify_entity(e) for e in kg.get("entities", [])],
        "relations": [simplify_relation(r) for r in kg.get("relationships", [])]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(simplified, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    simplify_kg("C:/PE/REPOS/KG-perturbator/full_project_kg.json", "C:/PE/REPOS/KG-perturbator/sampleKG_deck.json")