"""
Bedrock Model Registry endpoints.

Manages a whitelist of Bedrock models that map a human-facing model_id to an
inference_profile_arn. The ARN is used as the actual Bedrock API target while
model_id is retained in all logs and audit trails.

Endpoints:
  GET    /bedrock/models/ui                 — browser management UI
  GET    /bedrock/models                    — list enabled models (no ARNs)
  POST   /bedrock/models                    — add a model to the registry
  PATCH  /bedrock/models/{model_id}         — update pricing / capabilities
  DELETE /bedrock/models/{model_id}         — remove a model
  POST   /bedrock/models/reload             — re-sync in-memory dict from DB
"""

import json
from pathlib import Path
from typing import List, Optional
from urllib.parse import unquote

import litellm
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse

from litellm._logging import verbose_proxy_logger
from litellm.constants import LITELLM_PROXY_ADMIN_NAME
from litellm.proxy._types import (
    CommonProxyErrors,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.bedrock_registry import (
    BedrockModelInfo,
    BedrockModelListResponse,
    BedrockModelRegistryCreateRequest,
    BedrockModelRegistryUpdateRequest,
)

router = APIRouter()

# Tracks model_ids that THIS registry added to litellm.bedrock_converse_models so
# we can safely remove them on disable/delete without touching built-in entries.
_registry_converse_model_ids: set = set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_cost_entry(record) -> dict:
    """Build a litellm.model_cost entry from a DB record."""
    return {
        "input_cost_per_token": record.input_cost_per_token or 0,
        "output_cost_per_token": record.output_cost_per_token or 0,
        "max_input_tokens": record.max_input_tokens,
        "max_output_tokens": record.max_output_tokens,
        "litellm_provider": "bedrock",
        "mode": record.mode or "chat",
    }


def _sync_model_cost(model_id: str, record=None) -> None:
    """
    Keep litellm.model_cost in sync with the registry.

    Stores the entry under 'bedrock/{model_id}' so the cost calculator finds it
    via the 'model_with_provider' lookup path (highest-priority check).
    If record is None, the entry is removed.
    """
    key = f"bedrock/{model_id}"
    if record is None or not record.enabled:
        litellm.model_cost.pop(key, None)
    elif record.input_cost_per_token is not None or record.output_cost_per_token is not None:
        litellm.model_cost[key] = _build_cost_entry(record)


def _db_record_to_model_info(record) -> BedrockModelInfo:
    """Convert a DB row to the public-safe BedrockModelInfo (no ARN)."""
    capabilities: Optional[dict] = None
    if record.capabilities is not None:
        if isinstance(record.capabilities, str):
            capabilities = json.loads(record.capabilities)
        else:
            capabilities = dict(record.capabilities)

    return BedrockModelInfo(
        id=record.model_id,
        owned_by=record.model_id.split(".")[0],
        display_name=record.display_name,
        mode=record.mode,
        input_cost_per_token=record.input_cost_per_token,
        output_cost_per_token=record.output_cost_per_token,
        max_input_tokens=record.max_input_tokens,
        max_output_tokens=record.max_output_tokens,
        capabilities=capabilities,
        tags=list(record.tags) if record.tags else [],
        enabled=record.enabled,
    )


def _require_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only proxy admins can manage the Bedrock model registry.",
        )


def _register_converse_model(model_id: str) -> None:
    """Add model_id to bedrock_converse_models so LiteLLM routes it via Converse API."""
    _registry_converse_model_ids.add(model_id)
    litellm.bedrock_converse_models.add(model_id)


def _unregister_converse_model(model_id: str) -> None:
    """Remove model_id from bedrock_converse_models only if the registry added it."""
    if model_id in _registry_converse_model_ids:
        _registry_converse_model_ids.discard(model_id)
        litellm.bedrock_converse_models.discard(model_id)


async def load_bedrock_registry_from_db(prisma_client) -> int:
    """
    Load all enabled registry entries from DB into litellm.bedrock_model_registry.
    Called on proxy startup and by the /reload endpoint.
    Returns the number of models loaded.
    """
    if prisma_client is None:
        verbose_proxy_logger.warning(
            "Bedrock registry: prisma_client is None, skipping DB load"
        )
        return 0

    records = await prisma_client.db.litellm_bedrockmodelregistry.find_many(
        where={"enabled": True}
    )
    new_registry: dict = {
        r.model_id: {
            "arn": r.inference_profile_arn,
            "role": getattr(r, "aws_role_name", None),
        }
        for r in records
    }
    litellm.bedrock_model_registry.clear()
    litellm.bedrock_model_registry.update(new_registry)

    # All registry models route via Converse (provider-agnostic, supports ARN injection)
    litellm.bedrock_converse_models.difference_update(_registry_converse_model_ids)
    _registry_converse_model_ids.clear()
    for r in records:
        _register_converse_model(r.model_id)
        _sync_model_cost(r.model_id, r)

    verbose_proxy_logger.info(
        "Bedrock registry: loaded %d model(s) from DB", len(new_registry)
    )
    return len(new_registry)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


_UI_HTML = Path(__file__).parent / "bedrock_registry_ui.html"


@router.get("/bedrock/models/ui", response_class=HTMLResponse, tags=["bedrock"])
async def bedrock_models_ui():
    """Browser-based management UI for the Bedrock model registry."""
    return HTMLResponse(content=_UI_HTML.read_text(encoding="utf-8"))


@router.get(
    "/bedrock/models",
    response_model=BedrockModelListResponse,
    tags=["bedrock"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_bedrock_models(
    tag: Optional[str] = None,
    enabled_only: bool = True,
):
    """
    List whitelisted Bedrock models with pricing and capability metadata.

    - `tag`: filter to models that include this tag (e.g. `?tag=production`)
    - `enabled_only`: when false, also returns disabled models (default: true)

    The `inference_profile_arn` is never included in the response.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    where: dict = {}
    if enabled_only:
        where["enabled"] = True

    records = await prisma_client.db.litellm_bedrockmodelregistry.find_many(
        where=where
    )

    models: List[BedrockModelInfo] = [_db_record_to_model_info(r) for r in records]

    if tag is not None:
        models = [m for m in models if tag in m.tags]

    return BedrockModelListResponse(data=models)


@router.get(
    "/bedrock/models/prices",
    tags=["bedrock"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_model_prices(model_id: str):
    """
    Look up pricing and capabilities for a Bedrock model from LiteLLM's built-in
    model database (model_prices_and_context_window.json).

    Returns the data as-is from the built-in database so callers can compare
    against or pre-populate registry entries. The `inference_profile_arn` is
    never included — this is purely for pricing/capability discovery.
    """
    # The JSON file uses bare model IDs as keys (e.g. "anthropic.claude-3-5-sonnet-20241022-v2:0").
    # Some models use a "bedrock_converse/" prefix variant. We do NOT check
    # "bedrock/{model_id}" because that's where _sync_model_cost stores our own
    # registry data — we want the original built-in values only.
    entry = litellm.model_cost.get(model_id)
    if entry is None:
        entry = litellm.model_cost.get(f"bedrock_converse/{model_id}")
    if entry is None or entry.get("litellm_provider") not in ("bedrock", "bedrock_converse"):
        return {"found": False, "model_id": model_id}

    capabilities = {
        k: v for k, v in entry.items()
        if k.startswith("supports_") and isinstance(v, bool)
    }

    return {
        "found": True,
        "model_id": model_id,
        "input_cost_per_token": entry.get("input_cost_per_token"),
        "output_cost_per_token": entry.get("output_cost_per_token"),
        "max_input_tokens": entry.get("max_input_tokens"),
        "max_output_tokens": entry.get("max_output_tokens"),
        "mode": entry.get("mode"),
        "capabilities": capabilities or None,
    }


@router.get(
    "/bedrock/models/{model_id:path}",
    tags=["bedrock"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_bedrock_model(
    model_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get a single registry entry including the inference_profile_arn.
    Requires proxy admin role (ARN contains AWS account IDs).
    """
    from litellm.proxy.proxy_server import prisma_client

    _require_proxy_admin(user_api_key_dict)

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    model_id = unquote(model_id)
    record = await prisma_client.db.litellm_bedrockmodelregistry.find_unique(
        where={"model_id": model_id}
    )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found in the Bedrock registry.",
        )

    info = _db_record_to_model_info(record)
    return {
        **info.model_dump(),
        "inference_profile_arn": record.inference_profile_arn,
        "aws_role_name": getattr(record, "aws_role_name", None),
    }


@router.post(
    "/bedrock/models",
    response_model=BedrockModelInfo,
    tags=["bedrock"],
    dependencies=[Depends(user_api_key_auth)],
)
async def create_bedrock_model(
    request: BedrockModelRegistryCreateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Add a model to the Bedrock registry. Requires proxy admin role."""
    from litellm.proxy.proxy_server import litellm_proxy_admin_name, prisma_client

    _require_proxy_admin(user_api_key_dict)

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    existing = await prisma_client.db.litellm_bedrockmodelregistry.find_unique(
        where={"model_id": request.model_id}
    )
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{request.model_id}' already exists in the registry. Use PATCH to update it.",
        )

    actor = user_api_key_dict.user_id or litellm_proxy_admin_name
    create_data: dict = {
        "model_id": request.model_id,
        "display_name": request.display_name,
        "mode": request.mode,
        "input_cost_per_token": request.input_cost_per_token,
        "output_cost_per_token": request.output_cost_per_token,
        "max_input_tokens": request.max_input_tokens,
        "max_output_tokens": request.max_output_tokens,
        "tags": request.tags,
        "enabled": request.enabled,
        "created_by": actor,
        "updated_by": actor,
    }
    if request.inference_profile_arn is not None:
        create_data["inference_profile_arn"] = request.inference_profile_arn
    if request.aws_role_name is not None:
        create_data["aws_role_name"] = request.aws_role_name
    if request.capabilities is not None:
        create_data["capabilities"] = json.dumps(request.capabilities)
    record = await prisma_client.db.litellm_bedrockmodelregistry.create(data=create_data)

    if request.enabled:
        litellm.bedrock_model_registry[request.model_id] = {
            "arn": request.inference_profile_arn,
            "role": request.aws_role_name,
        }
        _register_converse_model(request.model_id)
    _sync_model_cost(request.model_id, record)

    return _db_record_to_model_info(record)


@router.patch(
    "/bedrock/models/{model_id:path}",
    response_model=BedrockModelInfo,
    tags=["bedrock"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_bedrock_model(
    model_id: str,
    request: BedrockModelRegistryUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Update pricing, capabilities, or enabled status for a registry entry. Requires proxy admin role."""
    from litellm.proxy.proxy_server import litellm_proxy_admin_name, prisma_client

    _require_proxy_admin(user_api_key_dict)

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    # model_id may be URL-encoded (e.g. if it contains colons)
    model_id = unquote(model_id)

    existing = await prisma_client.db.litellm_bedrockmodelregistry.find_unique(
        where={"model_id": model_id}
    )
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found in the Bedrock registry.",
        )

    update_data: dict = {
        "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
    }
    for field in (
        "inference_profile_arn",
        "aws_role_name",
        "display_name",
        "mode",
        "input_cost_per_token",
        "output_cost_per_token",
        "max_input_tokens",
        "max_output_tokens",
        "tags",
        "enabled",
    ):
        value = getattr(request, field)
        if value is not None:
            update_data[field] = value

    if request.capabilities is not None:
        update_data["capabilities"] = json.dumps(request.capabilities)

    record = await prisma_client.db.litellm_bedrockmodelregistry.update(
        where={"model_id": model_id},
        data=update_data,
    )

    # Keep in-memory registry consistent
    new_arn = update_data.get("inference_profile_arn", existing.inference_profile_arn)
    new_role = update_data.get("aws_role_name", getattr(existing, "aws_role_name", None))
    new_enabled = update_data.get("enabled", existing.enabled)
    if new_enabled:
        litellm.bedrock_model_registry[model_id] = {"arn": new_arn, "role": new_role}
        _register_converse_model(model_id)
    else:
        litellm.bedrock_model_registry.pop(model_id, None)
        _unregister_converse_model(model_id)
    _sync_model_cost(model_id, record)

    return _db_record_to_model_info(record)


@router.delete(
    "/bedrock/models/{model_id:path}",
    tags=["bedrock"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_bedrock_model(
    model_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Remove a model from the Bedrock registry. Requires proxy admin role."""
    from litellm.proxy.proxy_server import prisma_client

    _require_proxy_admin(user_api_key_dict)

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    model_id = unquote(model_id)

    existing = await prisma_client.db.litellm_bedrockmodelregistry.find_unique(
        where={"model_id": model_id}
    )
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found in the Bedrock registry.",
        )

    await prisma_client.db.litellm_bedrockmodelregistry.delete(
        where={"model_id": model_id}
    )

    litellm.bedrock_model_registry.pop(model_id, None)
    _unregister_converse_model(model_id)
    _sync_model_cost(model_id, None)

    return {"deleted": True, "model_id": model_id}


@router.post(
    "/bedrock/models/reload",
    tags=["bedrock"],
    dependencies=[Depends(user_api_key_auth)],
)
async def reload_bedrock_registry(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Re-sync the in-memory Bedrock model registry from DB.
    Useful after direct DB edits or to pick up changes made on another instance.
    Requires proxy admin role.
    """
    from litellm.proxy.proxy_server import prisma_client

    _require_proxy_admin(user_api_key_dict)

    count = await load_bedrock_registry_from_db(prisma_client)
    return {"status": "ok", "models_loaded": count}
