import os
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from .base_llm_wrapper import BaseLLMWrapper
from typing import Optional, List, Any, Dict, Union

class VertexLLM(BaseLLMWrapper):
    """
    Wrapper for Google's Vertex AI (Gemini) Large Language Models.
    """

    def __init__(self, model_name: Optional[str] = None, 
                 project_id: Optional[str] = None, 
                 location: Optional[str] = None, **kwargs):
        """
        Initializes the Vertex AI LLM wrapper.
        Configuration is loaded from environment variables if not provided.

        Args:
            model_name (Optional[str]): Model name to use. Defaults to GOOGLE_VERTEXAI_MODEL_NAME env var.
            project_id (Optional[str]): Google Cloud project ID. Defaults to GOOGLE_CLOUD_PROJECT env var.
            location (Optional[str]): Google Cloud location. Defaults to GOOGLE_CLOUD_LOCATION env var.
            **kwargs: Additional keyword arguments passed to BaseLLMWrapper.
        """
        # We rely on google-cloud-aiplatform for project/location discovery if not provided
        _model_name = model_name or os.getenv("GOOGLE_VERTEXAI_MODEL_NAME")
        if not _model_name:
            raise ValueError("VertexAI model name must be provided via argument or GOOGLE_VERTEXAI_MODEL_NAME env var.")

        super().__init__(model_name=_model_name, **kwargs)

        try:
            vertexai.init(
                project=project_id or os.getenv("GOOGLE_CLOUD_PROJECT"),
                location=location or os.getenv("GOOGLE_CLOUD_LOCATION")
            )
            self.gemini = GenerativeModel(self.model_name)
        except Exception as e:
            print(f"Error initializing Vertex AI client: {e}")
            self.gemini = None

    def generate_content(self, prompt: Union[str, List[Any]], temperature: float = 0.1, response_mime_type: Optional[str] = None, **kwargs) -> Optional[str]:
        if not self.gemini:
            print("Vertex AI (Gemini) client not initialized.")
            return None

        if isinstance(prompt, list) and all(isinstance(p, str) for p in prompt):
            content = " ".join(prompt)
        elif isinstance(prompt, str):
            content = prompt
        else:
            print(f"VertexLLM: Unsupported prompt type: {type(prompt)}")
            return None
        
        try:
            generation_config = GenerationConfig(temperature=float(temperature))
            response = self.gemini.generate_content(
                content,
                generation_config=generation_config,
                **kwargs
            )
            return response.text
        except Exception as e:
            print(f"An error occurred with Vertex AI: {e}")
            return None

    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.1, **kwargs) -> Optional[str]:
        prompt = "\n".join([msg.get("content", "") for msg in messages if msg.get("role") in ("system", "user")])
        return self.generate_content(prompt, temperature=temperature, **kwargs)