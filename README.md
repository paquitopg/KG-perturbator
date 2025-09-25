# KG-perturbator

A Python package for applying controlled perturbations to Knowledge Graphs (KGs) in JSON format. Useful for generating test/finetuning sets for entity alignment, robustness evaluation, and adversarial testing.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency%20manager-poetry-blue.svg)](https://python-poetry.org/)

## Overview

KG-perturbator enables systematic generation of synthetic knowledge graph datasets for entity alignment research by applying controlled structural and textual perturbations to existing knowledge graphs. This tool is particularly valuable for researchers working on entity alignment algorithms, as it provides a scalable way to create ground-truth alignment datasets without manual annotation.

## Features

- **Entity Modifications**: Add/remove entities (nodes) and modify entity attributes (names, descriptions, types)
- **Structural Perturbations**: Add/remove relations (edges) and triplets to simulate incomplete knowledge graphs
- **Textual Transformations**: Rename entities using LLM-based approaches (synonyms, abbreviations, alternative naming conventions)
- **Controlled Perturbation**: Configurable perturbation degree and random seed for reproducible experiments
- **Ground Truth Mapping**: Maintains precise mapping between original and perturbed KGs for evaluation
- **Hybrid Architecture**: Accepts JSON-format KGs while using NetworkX internally for efficient graph operations
- **Command-Line Interface**: Easy integration into data processing pipelines
- **LLM Integration**: Supports multiple LLM providers (VertexAI, OpenAI, etc.) for semantic entity transformations

## Installation

### Prerequisites

- Python 3.8 or higher
- Poetry (recommended) or pip

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/paquitopg/KG-perturbator.git
cd KG-perturbator

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/paquitopg/KG-perturbator.git
cd KG-perturbator

# Install the package
pip install .
```

## Basic Usage

### Python API

```python
from kg_perturbator import KGPerturbator
import json

# Load your knowledge graph
with open("sampleKG.json") as f:
    kg_json = json.load(f)

# Configure perturbations
perturbator = KGPerturbator(config={
    "seed": 42,                    # For reproducible results
    "remove_entities": 5,          # Remove 5 entities
    "add_entities": 2,             # Add 2 synthetic entities
    "remove_edges": 10,            # Remove 10 relationships
    "llm_perturb_entities": {      # LLM-based entity modifications
        "update_name": True,
        "update_description": True
    }
})

# Apply perturbations
perturbed_kg, mapping = perturbator.perturb(kg_json)

# Save results
with open("perturbedKG.json", "w") as f:
    json.dump(perturbed_kg, f, indent=2)

with open("entity_mapping.json", "w") as f:
    json.dump(mapping, f, indent=2)
```

### Understanding the Output

- **`perturbed_kg`**: The modified knowledge graph with applied perturbations
- **`mapping`**: Dictionary containing the correspondence between original and perturbed entities, essential for ground-truth alignment evaluation

## Command-Line Interface (CLI)

The package includes a command-line interface for easy integration into data processing pipelines and batch processing workflows.

### Step 1: Configure Environment Variables

Create a `.env` file in the project root to store your LLM provider credentials and configuration. This keeps sensitive keys separate from the main configuration.

```bash
# .env
LLM_PROVIDER="vertexai"                    # or "openai", "anthropic", etc.
VERTEX_MODEL_NAME="gemini-1.0-pro-preview"
GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
OPENAI_API_KEY="your-openai-key"           # if using OpenAI
```

### Step 2: Create a Perturbation Configuration File

Create a `config.yaml` file to define which perturbations to apply:

```yaml
# config.yaml
seed: 42                                    # Random seed for reproducibility
remove_entities: 5                         # Number of entities to remove
add_entities: 2                            # Number of synthetic entities to add
remove_edges: 10                           # Number of relationships to remove
add_edges: 5                               # Number of relationships to add

llm_perturb_entities:                      # LLM-based entity modifications
  update_name: true                        # Modify entity names
  update_description: true                 # Modify entity descriptions
  update_type: false                       # Modify entity types
```

### Step 3: Run the CLI

Use the `kg-perturb` command to apply perturbations:

```bash
# Basic usage
poetry run kg-perturb config.yaml input.json output.json

# With entity mapping output
poetry run kg-perturb config.yaml input.json output.json --mapping mapping.json

# Help and options
poetry run kg-perturb --help
```

**Command Arguments:**
- `config.yaml`: Perturbation configuration file
- `input.json`: Source Knowledge Graph (JSON format)
- `output.json`: Destination for perturbed Knowledge Graph
- `--mapping` (optional): Path to save entity mapping file for ground-truth alignment

## Configuration Options

### Perturbation Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `seed` | int | Random seed for reproducible results |
| `remove_entities` | int | Number of entities to remove |
| `add_entities` | int | Number of synthetic entities to add |
| `remove_edges` | int | Number of relationships to remove |
| `add_edges` | int | Number of relationships to add |

### LLM Entity Perturbation Options

| Parameter | Type | Description |
|-----------|------|-------------|
| `update_name` | bool | Modify entity names using LLM |
| `update_description` | bool | Modify entity descriptions using LLM |
| `update_type` | bool | Modify entity types using LLM |

## Use Cases

- **Entity Alignment Research**: Generate synthetic datasets with known ground-truth alignments
- **Robustness Testing**: Evaluate alignment algorithms under various perturbation scenarios
- **Data Augmentation**: Create additional training data for alignment models
- **Benchmarking**: Standardized perturbation protocols for fair algorithm comparison

## Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Development

### Running Tests

```bash
# Using Poetry
poetry run pytest

# Using pip
pytest
```

### Code Formatting

```bash
# Using Poetry
poetry run black .
poetry run isort .

# Using pip
black .
isort .
```

## Citation

If you use KG-perturbator in your research, please cite our work:

```bibtex
@software{kg_perturbator,
  title={KG-perturbator: Controlled Perturbation of Knowledge Graphs for Entity Alignment},
  author={Paco Goze},
  year={2025},
  url={https://github.com/paquitopg/KG-perturbator}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built for entity alignment research and evaluation
- Integrates with popular LLM providers for semantic entity transformations
- Inspired by the need for systematic knowledge graph perturbation tools in research