import json
import yaml
import argparse
import random
import os
from pathlib import Path
from .perturbator import KGPerturbator
from .kg_to_entity_alignment_files import create_entity_files, create_alignment_files, create_relation_files, create_triple_files
from .perturb_and_generate_alignment import main as perturb_and_generate_alignment_main

def perturb_kg_from_files(config_path: str, input_path: str, output_path: str, mapping_path: str = None, llm_config_path: str = None):
    """
    Loads a KG and configs from files, applies perturbations, and saves the results.

    Args:
        config_path (str): Path to the YAML configuration file for perturbations.
        input_path (str): Path to the input KG JSON file.
        output_path (str): Path where the perturbed KG JSON will be saved.
        mapping_path (str, optional): Path to save the entity mapping JSON. Defaults to None.
        llm_config_path (str, optional): Path to the LLM configuration file. Defaults to None.
    """
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Display configuration summary
    print(f"   Seed: {config.get('seed', 'Not set')}")
    print(f"   Remove entities: {config.get('remove_entities', 0)}")
    print(f"   Add entities: {config.get('add_entities', 0)}")
    print(f"   Remove edges: {config.get('remove_edges', 0)}")
    print(f"   Add edges: {config.get('add_edges', 0)}")
    print(f"   LLM rename entities: {config.get('llm_rename_entities', False)}")
    print(f"   LLM rename relations: {config.get('llm_rename_relations', False)}")
    print(f"   LLM perturb entities: {config.get('llm_perturb_entities', False)}")

    # Check for LLM config
    if llm_config_path and os.path.exists(llm_config_path):
        print(f"   LLM config: Found {llm_config_path}")
    elif os.path.exists("llm_config.yaml"):
        print(f"   LLM config: Found llm_config.yaml")
    else:
        print(f"   LLM config: Using defaults (LLM_PROVIDER env var or vertexai)")

    print(f"Loading knowledge graph from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        kg_json = json.load(f)
        
    if "seed" in config:
        random.seed(config["seed"])
        print(f"Random seed set to {config['seed']}")

    perturbator = KGPerturbator(config, llm_config_path)
    perturbed_kg, mapping = perturbator.perturb(kg_json)

    print(f"Saving perturbed knowledge graph to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(perturbed_kg, f, indent=2, ensure_ascii=False)

    if mapping_path and mapping:
        print(f"Saving entity mapping to {mapping_path}")
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

    print("Perturbation complete.")

def main():
    parser = argparse.ArgumentParser(description="Apply perturbations to a Knowledge Graph.")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Perturb command
    perturb_parser = subparsers.add_parser('perturb', help='Apply perturbations to a Knowledge Graph')
    perturb_parser.add_argument("config", help="Path to the perturbation config YAML file.")
    perturb_parser.add_argument("input_kg", help="Path to the input Knowledge Graph JSON file.")
    perturb_parser.add_argument("output_kg", help="Path to save the perturbed Knowledge Graph JSON file.")
    perturb_parser.add_argument("--mapping", help="Optional path to save the entity mapping JSON file.", default=None)
    perturb_parser.add_argument("--llm-config", help="Optional path to the LLM configuration file.", default=None)
    
    # Perturb and generate alignment files command
    perturb_align_parser = subparsers.add_parser('perturb-and-align', help='Perturb a KG and generate entity alignment files')
    perturb_align_parser.add_argument("config", help="Path to the perturbation config YAML file")
    perturb_align_parser.add_argument("input_kg", help="Path to the input knowledge graph JSON file")
    perturb_align_parser.add_argument("output_kg", help="Path to save the perturbed knowledge graph JSON file")
    perturb_align_parser.add_argument("--output-dir", help="Directory to save the alignment files (optional)")
    perturb_align_parser.add_argument("--llm-config", help="Optional path to the LLM configuration file")
    
    args = parser.parse_args()
    
    if args.command == 'perturb':
        perturb_kg_from_files(
            config_path=args.config,
            input_path=args.input_kg,
            output_path=args.output_kg,
            mapping_path=args.mapping,
            llm_config_path=args.llm_config
        )
    elif args.command == 'perturb-and-align':
        perturb_and_generate_alignment_main(
            config_path=args.config,
            input_kg_path=args.input_kg,
            output_kg_path=args.output_kg,
            output_dir=args.output_dir,
            llm_config_path=args.llm_config
        )
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 

#poetry run python -m kg_perturbator.cli configs/config.yaml test_output/KG_deck.json test_output/perturbed_KG_deck.json --mapping test_output/entity_mapping_deck.json --llm-config configs/llm_config.yaml