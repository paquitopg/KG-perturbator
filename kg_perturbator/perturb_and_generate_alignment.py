"""
Main method to perturb a knowledge graph and generate entity alignment files.
Combines the following steps:
- perturb the knowledge graph according to config
- generate entity alignment files from the perturbed graph and mapping

Parameters: 
--config: path to the perturbation config YAML file
--llm-config: path to the LLM configuration YAML file
--input-kg: path to the input knowledge graph JSON file
--output-kg: path to save the perturbed knowledge graph JSON file
--output-dir: directory to save the alignment files (optional, defaults to output-kg parent)

Usage:
python perturb_and_generate_alignment.py --config config.yaml --input-kg input.json --output-kg output.json
"""

import json
import yaml
import random
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

try:
    from .perturbator import KGPerturbator
    from .kg_to_entity_alignment_files import (
        create_entity_files, 
        create_alignment_files, 
        create_relation_files, 
        create_triple_files,
        create_type_id_file,
    )
except ImportError:
    # When running as standalone script
    from perturbator import KGPerturbator
    from kg_to_entity_alignment_files import (
        create_entity_files, 
        create_alignment_files, 
        create_relation_files, 
        create_triple_files,
        create_type_id_file,
    )


def perturb_kg(config_path: str, input_kg_path: str, output_kg_path: str, llm_config_path: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Perturb a knowledge graph according to the given config.
    
    Args:
        config_path: Path to the perturbation config YAML file
        input_kg_path: Path to the input knowledge graph JSON file
        output_kg_path: Path to save the perturbed knowledge graph JSON file
        llm_config_path: path to the LLM configuration file
        
    Returns:
        Tuple of (perturbed_kg, entity_mapping)
    """
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loading LLM config from {llm_config_path}")
    with open(llm_config_path, 'r') as f:
        llm_config = yaml.safe_load(f)

    # Display configuration summary
    print(f"   Seed: {config.get('seed', 'Not set')}")
    print(f"   Remove entities: {config.get('remove_entities', 0)}")
    print(f"   Add entities: {config.get('add_entities', 0)}")
    print(f"   Remove edges: {config.get('remove_edges', 0)}")
    print(f"   Add edges: {config.get('add_edges', 0)}")
    print(f"   LLM rename entities: {config.get('llm_rename_entities', False)}")
    print(f"   LLM rename relations: {config.get('llm_rename_relations', False)}")
    print(f"   LLM perturb entities: {config.get('llm_perturb_entities', False)}")

    print(f"Loading knowledge graph from {input_kg_path}")
    with open(input_kg_path, 'r', encoding='utf-8') as f:
        kg_json = json.load(f)
        
    if "seed" in config:
        random.seed(config["seed"])
        print(f"Random seed set to {config['seed']}")

    perturbator = KGPerturbator(config, llm_config)
    perturbed_kg, entity_mapping = perturbator.perturb(kg_json)

    print(f"Saving perturbed knowledge graph to {output_kg_path}")
    with open(output_kg_path, 'w', encoding='utf-8') as f:
        json.dump(perturbed_kg, f, indent=2, ensure_ascii=False)

    print(f"Perturbation complete. Generated {len(entity_mapping)} entity mappings.")
    return perturbed_kg, entity_mapping


def generate_alignment_files(original_kg_path: str, perturbed_kg_path: str, entity_mapping: Dict[str, str], output_dir: Path) -> None:
    """
    Generate entity alignment files from the original KG, perturbed KG, and entity mapping.
    
    Args:
        original_kg_path: Path to the original KG JSON file
        perturbed_kg_path: Path to the perturbed KG JSON file
        entity_mapping: Dictionary mapping original entity IDs to perturbed entity IDs
        output_dir: Directory to save the alignment files
    """
    print(f"Loading files for entity alignment generation...")
    with open(original_kg_path, 'r', encoding='utf-8') as f:
        kg1 = json.load(f)
    with open(perturbed_kg_path, 'r', encoding='utf-8') as f:
        kg2 = json.load(f)
    
    print(f"Loaded KG1 with {len(kg1['entities'])} entities and {len(kg1['relations'])} relations")
    print(f"Loaded KG2 with {len(kg2['entities'])} entities and {len(kg2['relations'])} relations")
    print(f"Loaded entity mapping with {len(entity_mapping)} aligned pairs")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Create type IDs mapping and entity files
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
    
    print(f"Entity alignment files created successfully in {output_dir}")
    print(f"Generated files:")
    for file_name in ['ent_ids_1', 'ent_ids_2', 'ent_types_1', 'ent_types_2', 'type_ids', 'ref_ent_ids', 'ref_pairs', 'sup_pairs', 
                      'rel_ids_1', 'rel_ids_2', 'triples_1', 'triples_2']:
        file_path = output_dir / file_name
        if file_path.exists():
            print(f"  - {file_name}")


def main(config_path: str, input_kg_path: str, output_kg_path: str, llm_config_path: str, output_dir: str = None) -> None:
    """
    Main function to perturb a knowledge graph and generate entity alignment files.
    
    Args:
        config_path: Path to the perturbation config YAML file
        input_kg_path: Path to the input knowledge graph JSON file
        output_kg_path: Path to save the perturbed knowledge graph JSON file
        output_dir: Directory to save the alignment files (optional)
        llm_config_path: path to the LLM configuration file
    """
    # 1. Perturb the knowledge graph
    print("=" * 50)
    print("STEP 1: Perturbing knowledge graph")
    print("=" * 50)
    perturbed_kg, entity_mapping = perturb_kg(config_path, input_kg_path, output_kg_path, llm_config_path)
    
    print("=" * 50)
    print("=" * 50)
    
    # 2. Generate entity alignment files
    print("STEP 2: Generating entity alignment files")
    print("=" * 50)
    
    # Determine output directory for alignment files
    if output_dir is None:
        output_dir = Path(output_kg_path).parent / "entity_alignment_files"
    else:
        output_dir = Path(output_dir)
    
    generate_alignment_files(input_kg_path, output_kg_path, entity_mapping, output_dir)
    
    print("=" * 50)
    print("=" * 50)
    print("All steps completed successfully!")
    print(f"Perturbed KG saved to: {output_kg_path}")
    print(f"Alignment files saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perturb a knowledge graph and generate entity alignment files.")
    parser.add_argument("--config", required=True, help="Path to the perturbation config YAML file")
    parser.add_argument("--input-kg", required=True, help="Path to the input knowledge graph JSON file")
    parser.add_argument("--output-kg", required=True, help="Path to save the perturbed knowledge graph JSON file")
    parser.add_argument("--llm-config", required=True, help="path to the LLM configuration file")
    parser.add_argument("--output-dir", help="Directory to save the alignment files (optional)")
    
    args = parser.parse_args()
    
    main(
        config_path=args.config,
        input_kg_path=args.input_kg,
        output_kg_path=args.output_kg,
        llm_config_path=args.llm_config,
        output_dir=args.output_dir
    ) 

"""
AS A STANDALONE SCRIPT:
poetry run python -m kg_perturbator.perturb_and_generate_alignment \
  --config configs/config.yaml \
  --input-kg test_output/KG_deck.json \
  --output-kg test_output/perturbed_KG_deck.json \
  --llm-config configs/llm_config.yaml \
  --output-dir test_output/alignment_files

example:
poetry run python -m kg_perturbator.perturb_and_generate_alignment --config configs/config.yaml --input-kg test_output/KG_airelle.json --output-kg test_output/perturbed_KG_airelle.json --llm-config configs/llm_config.yaml --output-dir test_output/alignment_files_airelle


AS A CLI COMMAND:
poetry run python -m kg_perturbator.cli perturb-and-align \
  configs/config.yaml \
  test_output/KG_deck.json \
  test_output/perturbed_KG_deck.json \
  --llm-config configs/llm_config.yaml \
  --output-dir test_output/alignment_files
"""