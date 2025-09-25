import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

class BaseLLMWrapper(ABC):
    """
    Abstract base class for Large Language Model client wrappers.
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initializes the LLM wrapper.

        Args:
            model_name (str): The name of the model to be used.
            **kwargs: Additional provider-specific keyword arguments.
        """
        self.model_name = model_name
        self.additional_config = kwargs

    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.1, **kwargs) -> Optional[str]:
        """
        Generates a response based on a list of chat messages.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries,
                e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}].
            temperature (float): The sampling temperature.
            **kwargs: Additional provider-specific keyword arguments for the completion.

        Returns:
            Optional[str]: The content of the LLM's response message, or None if an error occurs.
        """
        pass

    @abstractmethod
    def generate_content(self, prompt: Union[str, List[Any]], temperature: float = 0.1, response_mime_type: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        Generates content based on a prompt (text or multimodal parts).

        Args:
            prompt (Union[str, List[Any]]): The prompt string or a list of content parts
                                            (e.g., for multimodal input with Vertex AI).
            temperature (float): The sampling temperature.
            response_mime_type (Optional[str]): The desired MIME type for the response (e.g., "application/json").
            **kwargs: Additional provider-specific keyword arguments for content generation.

        Returns:
            Optional[str]: The generated text content, or None if an error occurs.
        """
        pass

    def rename_entity(self, entity: Dict[str, Any], prompt_template: str = None) -> str:
        """
        Rename the name of an entity.
        """

        entity_type = entity.get("type", "entity")
        name = entity.get("name", "")

        prompt = prompt_template or f"""
        Generate one alternative name or description for the following {entity_type} that is commonly recognized and actually used.
        This alternative should help a model understand that the original entity and the generated alternative refer to the same real-world concept, person, or thing.

        Only provide alternatives that you are confident are:
        - Actually used in real contexts
        - Commonly recognized and understood
        - Verified to exist (not invented or speculative)

        Consider these types of verified alternatives:
        - **Official Abbreviations:** Commonly used official abbreviations (e.g., "International Business Machines" -> "IBM").
        - **Official Alternative Names:** Legally recognized alternative names or trade names.
        - **Common Descriptive References:** Widely used descriptive terms that are actually employed in practice.

        Constraints:
        - Generate **only one** alternative.
        - The alternative must refer to the **exact same entity**.
        - Only provide the alternative if you are confident it is real and commonly used.
        - If you are unsure about a good alternative, return the original name unchanged.
        - Do not include any additional text, explanations, or formatting beyond the alternative itself.

        Entity: {name}
        Alternative:
        """
        result = self.generate_content(prompt)
        return result.strip() if result else ""

    def rename_entities_batch(self, entities: List[Dict[str, Any]], prompt_template: str = None) -> List[str]:
        """
        Rename a list of entities.
        """
        return [self.rename_entity(entity, prompt_template) for entity in entities]

    def synthesize_description(self, entity: Dict[str, Any]) -> str:
        """
        Synthesizes a new description for an entity based on its attributes and the LLM's world knowledge.
        The entity is a dictionary with the following keys:
        - name: str
        - type: str
        - Other attributes (e.g., foundedYear, keyStrengths, description,etc.)
        """
        attributes_str = "\n".join([f"- {key}: {value}" for key, value in entity.items() if key != "name" and key != "type"])
        prompt = f"""
        Given the following entity, generate a concise, natural-sounding description that could appear in a different knowledge graph or context.
        The new description should refer to the same real-world entity but present it from a fresh perspective.

        Use your world knowledge to create a description that:
        - Captures the entity's essence and significance
        - May include additional context or historical background you know about this entity
        - Could appear in a different type of document (news article, academic paper, industry report, etc.)
        - Can focus on different aspects than what's explicitly provided in the attributes

        Constraints:
        - Generate **only the new description**.
        - Do not include any extra text, explanations, or formatting.
        - Ensure the description clearly refers to the same real-world entity.

        Entity Name: {entity["name"]}
        Entity Type: {entity["type"]}

        Available Attributes:
        {attributes_str}

        New Description:
        """
        result = self.generate_content(prompt, temperature=0.8) # Higher temp for more creativity
        return result.strip() if result else ""

    def rename_relation(self, relation: Dict[str, Any], prompt_template: str = None) -> str:
        """
        Rename the type of a relation.
        """
        relation_type = relation.get("type", "relation")
        prompt = prompt_template or f"""
        Generate one alternative name for the following relation type that is commonly recognized and actually used.

        Constraints:
        - Generate **only one** alternative.
        - Only provide the alternative if you are confident it is real and commonly used.
        - If you are unsure about a good alternative, return the original name unchanged.
        - Do not include any additional text, explanations, or formatting beyond the alternative itself.

        **Examples** : 
        - If the relation type is "competes_with", the alternative could be "competes_against".
        - If the relation type is "partners_with", the alternative could be "collaborates_with".
        - If the relation type is "is_located_in", the alternative could be "is_based_in".

        Relation Type: {relation_type}
        Alternative:
        """
        result = self.generate_content(prompt)
        return result.strip() if result else ""
