# Deployment Guide — LiteLLM on AWS ECS Fargate (ARM64/Graviton)

## Overview

Images are built for `linux/arm64` and pushed to ECR. ECS Fargate tasks run on Graviton instances (~20-40% cheaper than x86 equivalents).

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

> **Note:** Unlike the old workflow, you do NOT run `docker compose push litellm` separately. Multi-platform images cannot be loaded into the local Docker store — they must be pushed during the build.

---

## ECS Task Definition — ARM64 Configuration

Add the `runtimePlatform` field to the ECS task definition:

```json
{
  "family": "litellm",
  "runtimePlatform": {
    "cpuArchitecture": "ARM64",
    "operatingSystemFamily": "LINUX"
  },
  "cpu": "1024",
  "memory": "2048",
  ...
}
```

### Sizing Guide

| Traffic level | vCPU | Memory | When to use |
|--------------|------|--------|-------------|
| Dev / staging | 0.5 (512) | 1 GB (1024) | Low request volume, internal use |
| **Production baseline** | **1 (1024)** | **2 GB (2048)** | **Start here** |
| High traffic | 2 (2048) | 4 GB (4096) | Many concurrent requests or heavy streaming |

**Start at 1 vCPU / 2 GB.** Review CloudWatch metrics after a week:
- CPU consistently < 20% at peak → scale down to 0.5 vCPU / 1 GB
- Memory pressure > 80% → scale up to 2 vCPU / 4 GB

---

## Verification

After pushing, confirm the image is ARM64:
```bash
docker buildx imagetools inspect \
  278699821793.dkr.ecr.ap-south-1.amazonaws.com/devtools/litellm:main
```

Look for `Platform: linux/arm64` in the output.

After the ECS task restarts, hit the health endpoint:
```bash
curl http://<task-ip>:4000/health/liveliness
```

---

## Restart ECS Service

After pushing a new image, force a new ECS deployment:
```bash
aws ecs update-service \
  --cluster <your-cluster> \
  --service <your-service> \
  --force-new-deployment \
  --region ap-south-1
```
