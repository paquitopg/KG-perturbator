"""
A simple python script to test the LLM implementation.
In particular, it focuses VertexAI and HuggingFace implementations, and test the following:
    - Rename name
    - Rename names batch
    - Synthesize description
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from kg_perturbator.llm_integrations.provider_factory import get_llm_provider


def test_rename_name():
    # Test entities
    entity_1 = {
        "name": "CNIM Group",
        "type": "Company",
        "foundedYear": 1950,
        "keyStrengths": ["industrial equipment", "industrial systems"],
        "location": "France",
        "revenue": 1000000000,
        "industry": "Industrial Equipment",
        "employees": 10000,
        "products": ["industrial equipment", "industrial systems"],
    }
    entity_2 = {
        "name": "Sam Altman",
        "type": "Person",
        "description": "Sam Altman is the CEO of OpenAI, a company that develops artificial intelligence technology.",
        "location": "United States",
        "industry": "Artificial Intelligence",
        "employees": 10000,
        "products": ["artificial intelligence", "artificial intelligence technology"],
    }
    entities_list = [entity_1, entity_2]
    
    # Test VertexAI
    print("=== Testing VertexAI ===")
    try:
        llm_provider_vertex = get_llm_provider("vertexai")
        perturbed_names_vertex = llm_provider_vertex.rename_entities_batch(entities_list)
        for entity, perturbed_name in zip(entities_list, perturbed_names_vertex):
            print(f"Original name: {entity['name']}")
            print(f"VertexAI perturbed name: {perturbed_name}")
    except Exception as e:
        print(f"VertexAI test failed: {e}")
    

def test_synthesize_description():
    print("\n=== Testing Description Synthesis ===")
    entity = {
        "name": "Tesla Inc",
        "type": "Company",
        "foundedYear": 2003,
        "keyStrengths": ["electric vehicles", "renewable energy"],
        "location": "United States",
        "industry": "Automotive",
        "employees": 127855,
        "products": ["electric vehicles", "solar panels", "energy storage"],
    }
    
    # Test VertexAI
    try:
        llm_provider_vertex = get_llm_provider("vertexai")
        description_vertex = llm_provider_vertex.synthesize_description(entity)
        print(f"VertexAI description: {description_vertex}")
    except Exception as e:
        print(f"VertexAI description test failed: {e}")

if __name__ == "__main__":
    test_rename_name()
    test_synthesize_description()
