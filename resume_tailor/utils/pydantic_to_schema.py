from typing import get_args, get_origin, List, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel


def pydantic_model_to_schema(model: BaseModel) -> Dict[str, Any]:
    """
    Converts a Pydantic model to a fully inlined JSON schema, suitable for LLM tool-calling.

    Args:
        model (BaseModel): The Pydantic model class.

    Returns:
        Dict[str, Any]: A JSON schema representation of the model.
    """
    properties = {}
    required_fields = []

    for field_name, model_field in model.model_fields.items():
        
        field_alias = model_field.alias or field_name
        required_fields.append(field_alias)
        field_type = model_field.annotation
        description = model_field.description or ""
        schema_type = get_schema_type(field_type, description)
        properties[field_alias] = schema_type

    return {
        "type": "object",
        "properties": properties,
        "required": required_fields,
    }


def get_schema_type(field_type, description: str) -> Dict[str, Any]:
    """
    Returns a JSON Schema representation for a single field.
    """
    origin = get_origin(field_type)

    if origin in (list, List):
        item_type = get_args(field_type)[0]
        return {
            "type": "array",
            "items": get_schema_type(item_type, ""),
            "description": description,
        }

    elif isinstance(field_type, type) and issubclass(field_type, Enum):
        return {
            "type": "string",
            "enum": [e.value for e in field_type],
            "description": description,
        }

    elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
        # Recursively generate schema for nested models
        return pydantic_model_to_schema(field_type)

    elif field_type in (str, int, float, bool):
        type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
        return {
            "type": type_map[field_type],
            "description": description,
        }

    else:
        # Default fallback
        return {
            "type": "string",
            "description": description,
        }