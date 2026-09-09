-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_BedrockModelRegistry" (
    "model_id" TEXT NOT NULL,
    "inference_profile_arn" TEXT,
    "aws_role_name" TEXT,
    "display_name" TEXT,
    "mode" TEXT,
    "input_cost_per_token" DOUBLE PRECISION,
    "output_cost_per_token" DOUBLE PRECISION,
    "max_input_tokens" INTEGER,
    "max_output_tokens" INTEGER,
    "capabilities" JSONB,
    "tags" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "enabled" BOOLEAN NOT NULL DEFAULT true,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT NOT NULL,

    CONSTRAINT "LiteLLM_BedrockModelRegistry_pkey" PRIMARY KEY ("model_id")
);
