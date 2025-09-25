#!/usr/bin/env python3
"""
Script to convert KG perturbation outputs into entity alignment training files.

This script takes the original KG, perturbed KG, and entity mapping files
and generates 9 files needed for entity alignment experiments:
- ent_ids_1/2: Entity IDs and names for both KGs
- ref_ent_ids: All aligned entity pairs
- ref_pairs: 57% of aligned pairs for testing
- sup_pairs: 43% of aligned pairs for training
- rel_ids_1/2: Relation type mappings for both KGs
- triples_1/2: Triples in (head_id, relation_id, tail_id) format
"""

import json
import random
from typing import Dict, List, Set, Tuple
from pathlib import Path


def load_json_file(file_path: str) -> Dict:
    """Load a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_entity_name(entity: Dict) -> str:
    """Extract the entity name from the entity dictionary."""
    if 'name' in entity and entity['name']:
        name_value = entity['name']
        if isinstance(name_value, list):
            return name_value[0] if name_value else entity['id']
        else:
            return str(name_value)
    elif 'fullName' in entity and entity['fullName']:
        full_name = entity['fullName']
        if isinstance(full_name, list):
            return full_name[0] if full_name else entity['id']
        else:
            return str(full_name)
    elif 'locationName' in entity and entity['locationName']:
        loc_name = entity['locationName']
        if isinstance(loc_name, list):
            return loc_name[0] if loc_name else entity['id']
        else:
            return str(loc_name)
    elif 'kpiName' in entity and entity['kpiName']:
        kpi_name = entity['kpiName']
        if isinstance(kpi_name, list):
            return kpi_name[0] if kpi_name else entity['id']
        else:
            return str(kpi_name)
    elif 'metricName' in entity and entity['metricName']:
        metric_name = entity['metricName']
        if isinstance(metric_name, list):
            return metric_name[0] if metric_name else entity['id']
        else:
            return str(metric_name)
    elif 'headcountName' in entity and entity['headcountName']:
        headcount_name = entity['headcountName']
        if isinstance(headcount_name, list):
            return headcount_name[0] if headcount_name else entity['id']
        else:
            return str(headcount_name)
    elif 'contextName' in entity and entity['contextName']:
        context_name = entity['contextName']
        if isinstance(context_name, list):
            return context_name[0] if context_name else entity['id']
        else:
            return str(context_name)
    elif 'titleName' in entity and entity['titleName']:
        title_name = entity['titleName']
        if isinstance(title_name, list):
            return title_name[0] if title_name else entity['id']
        else:
            return str(title_name)
    else:
        return entity['id']  # Fallback to ID if no name found


def create_type_id_file(kg1: Dict, kg2: Dict, output_dir: Path) -> Dict[str, int]:
    """
    Create a type_ids file mapping unique entity type strings (across KG1 and KG2)
    to integer IDs. Returns the mapping {type_str: type_id}.
    """
    unique_types: Set[str] = set()

    # Collect types from KG1
    for entity in kg1.get('entities', []):
        ent_type = entity.get('type', 'Unknown')
        if isinstance(ent_type, list):
            ent_type = ent_type[0] if ent_type else 'Unknown'
        if "pekg:" in ent_type:
            ent_type = ent_type.replace("pekg:", "")
        unique_types.add(ent_type)

    # Collect types from KG2
    for entity in kg2.get('entities', []):
        ent_type = entity.get('type', 'Unknown')
        if isinstance(ent_type, list):
            ent_type = ent_type[0] if ent_type else 'Unknown'
        if "pekg:" in ent_type:
            ent_type = ent_type.replace("pekg:", "")
        unique_types.add(ent_type)

    # Stable ordering for determinism
    sorted_types = sorted(unique_types)
    type_to_id: Dict[str, int] = {t: i for i, t in enumerate(sorted_types)}

    # Write mapping file
    with open(output_dir / 'type_ids', 'w', encoding='utf-8') as f:
        for t, tid in type_to_id.items():
            f.write(f"{tid}\t{t}\n")

    return type_to_id


def create_entity_files(kg1: Dict, kg2: Dict, output_dir: Path, type_to_id: Dict[str, int]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Create ent_ids_1, ent_ids_2, ent_types_1, and ent_types_2 files.

    - ent_ids_*: entity_id (int) -> entity_name
    - ent_types_*: entity_id (int) -> type_id (int)
    """
    
    # Create entity ID to name mappings
    ent1_id_to_name = {}
    ent2_id_to_name = {}
    
    # Create entity ID to type_id mappings
    ent1_id_to_type_id = {}
    ent2_id_to_type_id = {}
    
    # Process KG1 entities
    for entity in kg1['entities']:
        ent_id = entity['id']
        ent_id_as_integer = int(ent_id[1:]) - 1
        ent_name = get_entity_name(entity)
        ent_type = entity.get('type', 'Unknown')
        if isinstance(ent_type, list):
            ent_type = ent_type[0] if ent_type else 'Unknown'
        if "pekg:" in ent_type:
            ent_type = ent_type.replace("pekg:", "")
        ent1_id_to_name[ent_id_as_integer] = ent_name
        ent1_id_to_type_id[ent_id_as_integer] = type_to_id.get(ent_type, -1)
    
    # Process KG2 entities
    for entity in kg2['entities']:
        ent_id = entity['id']
        ent_id_as_integer = int(ent_id[1:]) - 1
        ent_name = get_entity_name(entity)
        ent_type = entity.get('type', 'Unknown')
        if isinstance(ent_type, list):
            ent_type = ent_type[0] if ent_type else 'Unknown'
        if "pekg:" in ent_type:
            ent_type = ent_type.replace("pekg:", "")
        ent2_id_to_name[ent_id_as_integer] = ent_name
        ent2_id_to_type_id[ent_id_as_integer] = type_to_id.get(ent_type, -1)
    
    # Write ent_ids_1.txt
    with open(output_dir / 'ent_ids_1', 'w', encoding='utf-8') as f:
        for ent_id, ent_name in ent1_id_to_name.items():
            f.write(f"{ent_id}\t{ent_name}\n")
    
    # Write ent_ids_2.txt
    with open(output_dir / 'ent_ids_2', 'w', encoding='utf-8') as f:
        for ent_id, ent_name in ent2_id_to_name.items():
            f.write(f"{ent_id}\t{ent_name}\n")
    
    # Write ent_types_1.txt (entity_id -> type_id)
    with open(output_dir / 'ent_types_1', 'w', encoding='utf-8') as f:
        for ent_id, type_id in ent1_id_to_type_id.items():
            f.write(f"{ent_id}\t{type_id}\n")
    
    # Write ent_types_2.txt (entity_id -> type_id)
    with open(output_dir / 'ent_types_2', 'w', encoding='utf-8') as f:
        for ent_id, type_id in ent2_id_to_type_id.items():
            f.write(f"{ent_id}\t{type_id}\n")
    
    return ent1_id_to_name, ent2_id_to_name


def create_alignment_files(entity_mapping: Dict[str, str], output_dir: Path) -> None:
    """Create ref_ent_ids, ref_pairs, and sup_pairs files."""
    
    # Create all aligned pairs
    aligned_pairs = []
    for kg1_id, kg2_id in entity_mapping.items():
        kg1_id_as_integer = int(kg1_id[1:]) - 1
        kg2_id_as_integer = int(kg2_id[1:]) - 1
        aligned_pairs.append((kg1_id_as_integer, kg2_id_as_integer))
    
    # Randomly split into test (57%) and train (43%) sets
    random.seed(42)  # Fixed seed for reproducibility
    random.shuffle(aligned_pairs)
    
    split_index = int(len(aligned_pairs) * 0.57)
    test_pairs = aligned_pairs[:split_index]
    train_pairs = aligned_pairs[split_index:]
    
    # Write ref_ent_ids (all aligned pairs)
    with open(output_dir / 'ref_ent_ids', 'w', encoding='utf-8') as f:
        for kg1_id, kg2_id in aligned_pairs:
            f.write(f"{kg1_id}\t{kg2_id}\n")
    
    # Write ref_pairs (test set)
    with open(output_dir / 'ref_pairs', 'w', encoding='utf-8') as f:
        for kg1_id, kg2_id in test_pairs:
            f.write(f"{kg1_id}\t{kg2_id}\n")
    
    # Write sup_pairs (train set)
    with open(output_dir / 'sup_pairs', 'w', encoding='utf-8') as f:
        for kg1_id, kg2_id in train_pairs:
            f.write(f"{kg1_id}\t{kg2_id}\n")


def create_relation_files(kg1: Dict, kg2: Dict, output_dir: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Create rel_ids_1.txt and rel_ids_2.txt files."""
    
    # Collect unique relation types from both KGs
    kg1_relation_types = set()
    kg2_relation_types = set()
    
    for relation in kg1['relations']:
        rel_type = relation['type']
        if "pekg:" in rel_type:
            rel_type = rel_type.replace("pekg:", "")
        kg1_relation_types.add(rel_type)
    
    for relation in kg2['relations']:
        rel_type = relation['type']
        if "pekg:" in rel_type:
            rel_type = rel_type.replace("pekg:", "")
        kg2_relation_types.add(rel_type)
    
    # Create relation type to ID mappings
    kg1_rel_type_to_id = {}
    kg2_rel_type_to_id = {}
    
    # Assign IDs for KG1 relations (0 to n1-1)
    for i, rel_type in enumerate(sorted(kg1_relation_types)):
        kg1_rel_type_to_id[rel_type] = i
    
    # Assign IDs for KG2 relations (n1 to n1+n2-1)
    n1 = len(kg1_relation_types)
    for i, rel_type in enumerate(sorted(kg2_relation_types)):
        kg2_rel_type_to_id[rel_type] = n1 + i
    
    # Write rel_ids_1.txt
    with open(output_dir / 'rel_ids_1', 'w', encoding='utf-8') as f:
        for rel_type, rel_id in kg1_rel_type_to_id.items():
            f.write(f"{rel_id}\t{rel_type}\n")
    
    # Write rel_ids_2.txt
    with open(output_dir / 'rel_ids_2', 'w', encoding='utf-8') as f:
        for rel_type, rel_id in kg2_rel_type_to_id.items():
            f.write(f"{rel_id}\t{rel_type}\n")
    
    return kg1_rel_type_to_id, kg2_rel_type_to_id


def create_triple_files(kg1: Dict, kg2: Dict, 
                       kg1_rel_type_to_id: Dict[str, int], 
                       kg2_rel_type_to_id: Dict[str, int],
                       output_dir: Path) -> None:
    """Create triples_1.txt and triples_2.txt files."""
    
    # Write triples_1.txt
    with open(output_dir / 'triples_1', 'w', encoding='utf-8') as f:
        for relation in kg1['relations']:
            head_id = int(relation['source'][1:])- 1
            tail_id = int(relation['target'][1:])- 1
            rel_type = relation['type']
            if "pekg:" in rel_type:
                rel_type = rel_type.replace("pekg:", "")
            rel_id = kg1_rel_type_to_id[rel_type]
            f.write(f"{head_id}\t{rel_id}\t{tail_id}\n")
    
    # Write triples_2.txt
    with open(output_dir / 'triples_2', 'w', encoding='utf-8') as f:
        for relation in kg2['relations']:
            head_id = int(relation['source'][1:])- 1
            tail_id = int(relation['target'][1:])- 1
            rel_type = relation['type']
            if "pekg:" in rel_type:
                rel_type = rel_type.replace("pekg:", "")
            rel_id = kg2_rel_type_to_id[rel_type]
            f.write(f"{head_id}\t{rel_id}\t{tail_id}\n")


def main():
    """Main function to generate all entity alignment files."""
    
    # Hardcoded paths for current use case
    kg1_path = "test_output/KG_deck.json"
    kg2_path = "test_output/perturbed_KG_deck.json"
    mapping_path = "test_output/entity_mapping_deck.json"
    output_dir = Path("test_output/entity_alignment_files")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Load the JSON files
    print("Loading JSON files...")
    kg1 = load_json_file(kg1_path)
    kg2 = load_json_file(kg2_path)
    entity_mapping = load_json_file(mapping_path)
    
    print(f"Loaded KG1 with {len(kg1['entities'])} entities and {len(kg1['relations'])} relations")
    print(f"Loaded KG2 with {len(kg2['entities'])} entities and {len(kg2['relations'])} relations")
    print(f"Loaded entity mapping with {len(entity_mapping)} aligned pairs")
    
    # Create type IDs and entity files
    print("Creating type IDs and entity files...")
    type_to_id = create_type_id_file(kg1, kg2, output_dir)
    ent1_id_to_name, ent2_id_to_name = create_entity_files(kg1, kg2, output_dir, type_to_id)
    
    # Create alignment files
    print("Creating alignment files...")
    create_alignment_files(entity_mapping, output_dir)
    
    # Create relation files
    print("Creating relation files...")
    kg1_rel_type_to_id, kg2_rel_type_to_id = create_relation_files(kg1, kg2, output_dir)
    
    # Create triple files
    print("Creating triple files...")
    create_triple_files(kg1, kg2, kg1_rel_type_to_id, kg2_rel_type_to_id, output_dir)
    
    print(f"All files created successfully in {output_dir}")
    print(f"Generated files:")
    for file_name in ['ent_ids_1', 'ent_ids_2', 'ent_types_1', 'ent_types_2', 'type_ids', 'ref_ent_ids', 'ref_pairs', 'sup_pairs', 
                      'rel_ids_1', 'rel_ids_2', 'triples_1', 'triples_2']:
        file_path = output_dir / file_name
        if file_path.exists():
            print(f"  - {file_name}")


if __name__ == "__main__":
    main() 