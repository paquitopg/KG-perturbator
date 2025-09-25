import os
from typing import List, Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Pipeline
import torch
from .base_llm_wrapper import BaseLLMWrapper
from dotenv import load_dotenv

load_dotenv()

class HuggingFaceLLM(BaseLLMWrapper):
    """
    Wrapper for local Hugging Face Transformer models.
    """
    _local_pipeline: Optional[Pipeline] = None
    _tokenizer: Optional[AutoTokenizer] = None

    def __init__(self, model_name: Optional[str] = None, **kwargs):
        """
        Initializes the Hugging Face LLM wrapper.
        The model is loaded from the local machine.

        Args:
            model_name (Optional[str]): The model name. Defaults to HUGGINGFACE_MODEL_NAME env var.
            **kwargs: Additional keyword arguments passed to BaseLLMWrapper.
        """
        _model_name = model_name or os.getenv("HUGGINGFACE_MODEL_NAME")
        if not _model_name:
            raise ValueError("HuggingFace model name must be provided via argument or HUGGINGFACE_MODEL_NAME env var.")
        
        super().__init__(model_name=_model_name, **kwargs)
        
        if HuggingFaceLLM._local_pipeline is None:
            try:
                print(f"Loading HuggingFace model: {self.model_name}")
                
                # Load tokenizer
                HuggingFaceLLM._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                
                # Add padding token if not present
                if HuggingFaceLLM._tokenizer.pad_token is None:
                    HuggingFaceLLM._tokenizer.pad_token = HuggingFaceLLM._tokenizer.eos_token
                
                # Load model with quantization for memory efficiency
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,  # Use half precision
                    device_map="auto",  # Automatically handle device placement
                    load_in_8bit=True,  # Use 8-bit quantization
                    trust_remote_code=True
                )
                
                # Create pipeline
                HuggingFaceLLM._local_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=HuggingFaceLLM._tokenizer,
                    device_map="auto"
                )
                
                print(f"Successfully loaded model: {self.model_name}")
                
            except Exception as e:
                print(f"Error loading Hugging Face model '{self.model_name}': {e}")
                print("Make sure you have sufficient GPU memory (8GB+ recommended for 7B models)")
                HuggingFaceLLM._local_pipeline = None

    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7, **kwargs) -> Optional[str]:
        # For chat models, format the messages properly
        if HuggingFaceLLM._tokenizer and hasattr(HuggingFaceLLM._tokenizer, 'apply_chat_template'):
            # Use the model's chat template if available
            prompt = HuggingFaceLLM._tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            # Fallback: concatenate user messages
            user_content = " ".join([m["content"] for m in messages if m["role"] == "user"])
            prompt = user_content
            
        return self.generate_content(prompt, temperature=temperature, **kwargs)

    def generate_content(self, prompt: Union[str, List[Any]], temperature: float = 0.7, response_mime_type: Optional[str] = None, **kwargs) -> Optional[str]:
        if not HuggingFaceLLM._local_pipeline:
            print("HuggingFace pipeline not initialized.")
            return None

        try:
            # Set generation parameters
            generation_kwargs = {
                "max_new_tokens": kwargs.get("max_new_tokens", 100),
                "temperature": temperature,
                "do_sample": True,
                "pad_token_id": HuggingFaceLLM._tokenizer.eos_token_id if HuggingFaceLLM._tokenizer else None,
                **kwargs
            }
            
            # Generate text
            results = HuggingFaceLLM._local_pipeline(prompt, **generation_kwargs)
            
            if results and isinstance(results, list) and "generated_text" in results[0]:
                generated_text = results[0]["generated_text"]
                
                # Extract only the new generated part (remove the original prompt)
                if generated_text.startswith(prompt):
                    new_text = generated_text[len(prompt):].strip()
                else:
                    # If prompt wasn't found at start, return the full generated text
                    new_text = generated_text
                
                return new_text
            else:
                print("Unexpected response format from HuggingFace pipeline")
                return None
                
        except Exception as e:
            print(f"Error during HuggingFace pipeline inference: {e}")
            return None 