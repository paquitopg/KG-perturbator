import os
from .base_llm_wrapper import BaseLLMWrapper
from .vertex_llm import VertexLLM
from .azure_llm import AzureLLM
from .huggingface_llm import HuggingFaceLLM
from typing import Dict, Any

"""
This factory is used to get the LLM provider based on the environment variable or the provider name.
It is used to get the LLM provider for the perturbator.
Input : 
    - provider_name: str : The name of the LLM provider.
    - **kwargs: Any : Additional keyword arguments for the LLM provider.
Output :
    - BaseLLMWrapper : The LLM provider.
"""

def get_llm_provider(provider_name: str, **kwargs: Any) -> BaseLLMWrapper:
    """
    Factory function to get an LLM provider based on the environment
    or specified provider name.
    """
    provider_name = provider_name or os.getenv("LLM_PROVIDER", "vertexai")
    
    if provider_name == "vertexai":
        return VertexLLM(**kwargs)
    elif provider_name == "azure":
        return AzureLLM(**kwargs)
    elif provider_name == "huggingface":
        return HuggingFaceLLM(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}") 