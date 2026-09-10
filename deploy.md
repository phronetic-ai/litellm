# Building and Pushing the LiteLLM Image (ARM64/Graviton)

## Overview

Images are built for `linux/arm64` and pushed to ECR as `278699821793.dkr.ecr.ap-south-1.amazonaws.com/devtools/litellm:main`. `devtools-infra` deploys ECS services from this tag — see `devtools-infra/llm/deploy.md` for running/sizing/restarting the ECS service that consumes it.

---

## One-time Setup (per build machine)

### Apple Silicon Mac (M1/M2/M3/M4)
No extra setup needed — the machine is already ARM64.

You still need a `docker-container` buildx builder (the default `docker` driver can't push multi-platform manifests):
```bash
docker buildx create --name arm-builder --driver docker-container --use
docker buildx inspect --bootstrap
```

### x86 Linux or Intel Mac
```bash
# 1. Register ARM emulation
docker run --privileged --rm tonistiigi/binfmt --install all

# 2. Create buildx builder
docker buildx create --name arm-builder --driver docker-container --use
docker buildx inspect --bootstrap
```

Verify ARM support is listed in the platforms output before continuing.

---

## Building and Pushing

Authenticate to ECR first:
```bash
aws ecr get-login-password --region ap-south-1 | \
  docker login --username AWS --password-stdin \
  278699821793.dkr.ecr.ap-south-1.amazonaws.com
```

Then build and push:
```bash
make push-image
```

This runs `docker compose build --push litellm`, which builds the `linux/arm64` image and pushes it directly to ECR in one step.

> **Note:** you do NOT run `docker compose push litellm` separately. Multi-platform images cannot be loaded into the local Docker store — they must be pushed during the build.

---

## Verification

After pushing, confirm the image is ARM64:
```bash
docker buildx imagetools inspect \
  278699821793.dkr.ecr.ap-south-1.amazonaws.com/devtools/litellm:main
```

Look for `Platform: linux/arm64` in the output.

Once pushed, deploy it via `devtools-infra/llm/deploy.md`.
