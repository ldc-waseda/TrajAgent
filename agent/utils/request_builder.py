"""
Request builder for API calls.

Provides shared logic for building API requests from trajectory data.
"""

from typing import Any, Dict, List, Tuple

from agent.utils.image_io import image_to_data_url
from agent.utils.prompt_template import render_template


class RequestBuilder:
    """Builds API requests from trajectory data."""
    
    @staticmethod
    def get_image_dimensions(img_any: Any) -> Tuple[int, int]:
        """
        Get image dimensions (width, height) from various image formats.
        
        Args:
            img_any: Image array (H, W, C) or (H, W)
            
        Returns:
            (width, height) tuple
        """
        height, width = img_any.shape[:2]
        return width, height
    
    @staticmethod
    def add_image_dimensions(template_data: Dict[str, Any], img_any: Any) -> Dict[str, Any]:
        """
        Add image dimensions to template data.
        
        Args:
            template_data: Template data dictionary
            img_any: Image array
            
        Returns:
            Updated template data with image_size field
        """
        width, height = RequestBuilder.get_image_dimensions(img_any)
        template_data['image_size'] = f"{width}x{height}"
        return template_data
    
    @staticmethod
    def build_messages(
        system_template_name: str,
        user_template_name: str,
        template_data: Dict[str, Any],
        img_any: Any,
        scenario_mask: Any = None
    ) -> List[Dict[str, Any]]:
        """
        Build chat messages for API request.
        
        Args:
            system_template_name: Name of system prompt template
            user_template_name: Name of user prompt template
            template_data: Data for template rendering
            img_any: Main image array
            scenario_mask: Optional scenario mask image
            
        Returns:
            List of message dictionaries
        """
        # Add image dimensions
        template_data = RequestBuilder.add_image_dimensions(template_data, img_any)
        
        # Render prompts
        system_prompt = render_template(system_template_name, template_data)
        user_prompt = render_template(user_template_name, template_data)
        
        # Convert main image to data URL
        data_url = image_to_data_url(img_any)
        
        # Build user content with images and text
        user_content = [
            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
            {"type": "text", "text": user_prompt},
        ]
        
        # Add scenario mask if available
        if scenario_mask is not None:
            mask_data_url = image_to_data_url(scenario_mask)
            user_content.insert(0, {"type": "image_url", "image_url": {"url": mask_data_url, "detail": "high"}})
        
        # Build messages
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        
        return messages
    
    @staticmethod
    def build_batch_request(
        custom_id: str,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_completion_tokens: int,
        response_format: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Build a batch API request object.
        
        Args:
            custom_id: Custom ID for the request (used for matching responses)
            messages: Chat messages
            model: Model name
            temperature: Temperature parameter
            max_completion_tokens: Max tokens parameter
            response_format: Response format specification
            **kwargs: Additional parameters
            
        Returns:
            Batch request object ready for JSONL serialization
        """
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": max_completion_tokens,
                "response_format": response_format,
                **kwargs
            }
        }

