#!/usr/bin/env python3
"""
Bedrock Model Registry — Price Sync Script

Fetches current pricing for models already in the LiteLLM Bedrock registry and
updates them via the proxy API. Runs entirely outside the Docker container.

Dependencies (no litellm required):
    pip install boto3 requests

Usage:
    python scripts/sync_bedrock_prices.py \
        --litellm-url http://your-proxy:4000 \
        --litellm-api-key sk-... \
        --aws-region us-east-1 \
        [--dry-run]

AWS credentials are resolved via the standard boto3 chain
(env vars, ~/.aws/credentials, IAM role, etc.).

Pricing notes:
  - The AWS Pricing API only has complete data in us-east-1.
  - Not all models appear in the Pricing API; the script skips those gracefully.
  - Prices are in USD per token (not per 1K tokens).
"""

import argparse
import json
import sys
from typing import Optional

try:
    import boto3
    import requests
except ImportError:
    print("ERROR: Missing dependencies. Run: pip install boto3 requests", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# AWS helpers
# ---------------------------------------------------------------------------

def get_registered_models(litellm_url: str, api_key: str) -> list:
    """Fetch all models currently in the LiteLLM Bedrock registry."""
    resp = requests.get(
        f"{litellm_url}/bedrock/models",
        headers={"Authorization": f"Bearer {api_key}"},
        params={"enabled_only": "false"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def fetch_bedrock_foundation_models(region: str) -> dict:
    """
    Return a dict of modelId -> modelSummary for all text-output foundation models
    accessible in the given region.
    """
    client = boto3.client("bedrock", region_name=region)
    paginator_resp = client.list_foundation_models(byOutputModality="TEXT")
    return {m["modelId"]: m for m in paginator_resp.get("modelSummaries", [])}


def fetch_pricing_for_model(model_id: str, region: str) -> Optional[dict]:
    """
    Query the AWS Pricing API for a Bedrock model.
    Returns a dict with input_cost_per_token / output_cost_per_token (USD/token)
    or None if pricing is unavailable.

    The Pricing API endpoint is always us-east-1 regardless of model region.
    """
    pricing_client = boto3.client("pricing", region_name="us-east-1")

    # Derive a short model name for the pricing filter (e.g. "claude-3-5-sonnet-20241022-v2")
    short_name = model_id.split(".")[-1] if "." in model_id else model_id

    try:
        response = pricing_client.get_products(
            ServiceCode="AmazonBedrock",
            Filters=[
                {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
                {"Type": "TERM_MATCH", "Field": "modelId", "Value": model_id},
            ],
            MaxResults=10,
        )
    except Exception as exc:
        print(f"  [warn] Pricing API error for {model_id}: {exc}", file=sys.stderr)
        return None

    price_list = response.get("PriceList", [])
    if not price_list:
        # Try a partial match on the short model name
        try:
            response = pricing_client.get_products(
                ServiceCode="AmazonBedrock",
                Filters=[
                    {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
                    {"Type": "TERM_MATCH", "Field": "modelId", "Value": short_name},
                ],
                MaxResults=10,
            )
            price_list = response.get("PriceList", [])
        except Exception:
            pass

    if not price_list:
        return None

    input_cost: Optional[float] = None
    output_cost: Optional[float] = None

    for item_str in price_list:
        item = json.loads(item_str)
        attributes = item.get("product", {}).get("attributes", {})
        token_type = attributes.get("tokenType", "").lower()
        terms = item.get("terms", {}).get("OnDemand", {})
        for term in terms.values():
            for dim in term.get("priceDimensions", {}).values():
                usd_per_unit = float(dim.get("pricePerUnit", {}).get("USD", 0))
                unit = dim.get("unit", "").lower()
                # Pricing is typically in USD per 1000 tokens
                cost_per_token = usd_per_unit / 1000 if "1000" in unit or usd_per_unit > 0 else usd_per_unit
                if "input" in token_type:
                    input_cost = cost_per_token
                elif "output" in token_type:
                    output_cost = cost_per_token

    if input_cost is None and output_cost is None:
        return None

    result = {}
    if input_cost is not None:
        result["input_cost_per_token"] = input_cost
    if output_cost is not None:
        result["output_cost_per_token"] = output_cost
    return result


# ---------------------------------------------------------------------------
# LiteLLM API helpers
# ---------------------------------------------------------------------------

def update_model_pricing(
    litellm_url: str,
    api_key: str,
    model_id: str,
    pricing: dict,
) -> dict:
    """PATCH the registry entry with new pricing data."""
    resp = requests.patch(
        f"{litellm_url}/bedrock/models/{model_id}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=pricing,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync AWS Bedrock model pricing into the LiteLLM Bedrock registry."
    )
    parser.add_argument("--litellm-url", required=True, help="LiteLLM proxy base URL, e.g. http://localhost:4000")
    parser.add_argument("--litellm-api-key", required=True, help="LiteLLM proxy admin API key")
    parser.add_argument("--aws-region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be updated without making API calls")
    args = parser.parse_args()

    print(f"Fetching registered models from {args.litellm_url} ...")
    registered = get_registered_models(args.litellm_url, args.litellm_api_key)
    if not registered:
        print("No models found in the registry. Add models via POST /bedrock/models first.")
        return

    print(f"Found {len(registered)} registered model(s). Fetching AWS pricing for region {args.aws_region} ...")
    foundation_models = fetch_bedrock_foundation_models(args.aws_region)

    updated = 0
    skipped = 0

    for model in registered:
        model_id = model["id"]
        if model_id not in foundation_models:
            print(f"  SKIP  {model_id} — not found in AWS foundation model list for {args.aws_region}")
            skipped += 1
            continue

        pricing = fetch_pricing_for_model(model_id, args.aws_region)
        if not pricing:
            print(f"  SKIP  {model_id} — pricing data unavailable in AWS Pricing API")
            skipped += 1
            continue

        price_summary = ", ".join(f"{k}={v:.2e}" for k, v in pricing.items())
        if args.dry_run:
            print(f"  DRY   {model_id} — would update: {price_summary}")
        else:
            try:
                update_model_pricing(args.litellm_url, args.litellm_api_key, model_id, pricing)
                print(f"  OK    {model_id} — {price_summary}")
                updated += 1
            except requests.HTTPError as exc:
                print(f"  ERROR {model_id} — {exc}", file=sys.stderr)

    print(f"\nDone. Updated: {updated}  Skipped: {skipped}")
    if args.dry_run:
        print("(dry-run — no changes were made)")


if __name__ == "__main__":
    main()
