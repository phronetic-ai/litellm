from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class BedrockModelRegistryCreateRequest(BaseModel):
    model_id: str
    inference_profile_arn: Optional[str] = None
    aws_role_name: Optional[str] = None
    display_name: Optional[str] = None
    mode: Optional[str] = None
    input_cost_per_token: Optional[float] = None
    output_cost_per_token: Optional[float] = None
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    capabilities: Optional[Dict[str, bool]] = None
    tags: List[str] = Field(default_factory=list)
    enabled: bool = True


class BedrockModelRegistryUpdateRequest(BaseModel):
    inference_profile_arn: Optional[str] = None
    aws_role_name: Optional[str] = None
    display_name: Optional[str] = None
    mode: Optional[str] = None
    input_cost_per_token: Optional[float] = None
    output_cost_per_token: Optional[float] = None
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    capabilities: Optional[Dict[str, bool]] = None
    tags: Optional[List[str]] = None
    enabled: Optional[bool] = None


class BedrockModelInfo(BaseModel):
    """Public-safe model info — inference_profile_arn is intentionally excluded."""

    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str
    litellm_provider: Literal["bedrock"] = "bedrock"
    display_name: Optional[str] = None
    mode: Optional[str] = None
    input_cost_per_token: Optional[float] = None
    output_cost_per_token: Optional[float] = None
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    capabilities: Optional[Dict[str, bool]] = None
    tags: List[str] = Field(default_factory=list)
    enabled: bool = True


class BedrockModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[BedrockModelInfo]
