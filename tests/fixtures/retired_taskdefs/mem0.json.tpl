{
  "family": "__FAMILY__",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "__CPU__",
  "memory": "__MEMORY__",
  "runtimePlatform": {
    "cpuArchitecture": "X86_64",
    "operatingSystemFamily": "LINUX"
  },
  "executionRoleArn": "__EXEC_ROLE__",
  "taskRoleArn": "__TASK_ROLE__",
  "containerDefinitions": [
    {
      "name": "memrank",
      "image": "__MEMRANK_IMAGE__",
      "essential": true,
      "entryPoint": ["/bin/sh", "-c"],
      "command": ["__COMMAND__"],
      "environment": [
        {"name": "MEM0_HTTP_URL", "value": "http://localhost:8888"},
        {"name": "MEM0_ENGINE_VERSION", "value": "__MEM0_ENGINE_VERSION__"},
        {"name": "MEM0_ENGINE_IMAGE_DIGEST", "value": "__MEM0_ENGINE_IMAGE_DIGEST__"},
        {"name": "MEM0_ENGINE_IMAGE_REPO", "value": "__MEM0_ENGINE_IMAGE_REPO__"},
        {"name": "MEM0_ENGINE_SOURCE_SHA", "value": "__MEM0_ENGINE_SOURCE_SHA__"},
        {"name": "MEM0_ENGINE_SOURCE_DESCRIBE", "value": "__MEM0_ENGINE_SOURCE_DESCRIBE__"},
        {"name": "MEM0_ENGINE_SOURCE_DIRTY", "value": "__MEM0_ENGINE_SOURCE_DIRTY__"},
        {"name": "MEM0_TIMEOUT_S", "value": "__MEM0_TIMEOUT_S__"}
      ],
      "dependsOn": [
        {"containerName": "mem0", "condition": "HEALTHY"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "__LOG_GROUP__",
          "awslogs-region": "__REGION__",
          "awslogs-stream-prefix": "memrank"
        }
      }
    },
    {
      "name": "mem0",
      "image": "__MEM0_IMAGE__",
      "essential": false,
      "command": ["sh", "-c", "alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port 8888"],
      "environment": [
        {"name": "PORT", "value": "8888"},
        {"name": "AUTH_DISABLED", "value": "true"},
        {"name": "MEM0_TELEMETRY", "value": "false"},
        {"name": "HISTORY_DB_PATH", "value": "/tmp/mem0-history.db"},
        {"name": "APP_DB_NAME", "value": "mem0_app"},
        {"name": "POSTGRES_HOST", "value": "localhost"},
        {"name": "POSTGRES_PORT", "value": "5432"},
        {"name": "POSTGRES_DB", "value": "postgres"},
        {"name": "POSTGRES_USER", "value": "postgres"},
        {"name": "POSTGRES_PASSWORD", "value": "postgres"},
        {"name": "POSTGRES_COLLECTION_NAME", "value": "memories"},
        {"name": "MEM0_LLM_PROVIDER", "value": "anthropic"},
        {"name": "MEM0_LLM_MODEL", "value": "__MEM0_LLM_MODEL__"},
        {"name": "MEM0_EMBEDDER_PROVIDER", "value": "huggingface"},
        {"name": "MEM0_EMBEDDER_ENDPOINT", "value": "http://localhost/v1"},
        {"name": "MEM0_EMBEDDER_MODEL", "value": "BAAI/bge-small-en-v1.5"},
        {"name": "MEM0_EMBEDDING_DIMS", "value": "384"}
      ],
      "secrets": [
        {"name": "ANTHROPIC_API_KEY", "valueFrom": "__ANTHROPIC_SECRET_ARN__"},
        {"name": "OPENAI_API_KEY", "valueFrom": "__OPENAI_SECRET_ARN__"}
      ],
      "dependsOn": [
        {"containerName": "postgres", "condition": "HEALTHY"},
        {"containerName": "tei", "condition": "START"}
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -fsS http://localhost:8888/auth/setup-status && curl -fsS http://localhost/health || exit 1"],
        "interval": 15,
        "timeout": 10,
        "retries": 10,
        "startPeriod": 180
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "__LOG_GROUP__",
          "awslogs-region": "__REGION__",
          "awslogs-stream-prefix": "mem0"
        }
      }
    },
    {
      "name": "postgres",
      "image": "__MEM0_PG_IMAGE__",
      "essential": false,
      "environment": [
        {"name": "POSTGRES_USER", "value": "postgres"},
        {"name": "POSTGRES_PASSWORD", "value": "postgres"},
        {"name": "POSTGRES_DB", "value": "postgres"}
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "pg_isready -q -U postgres -d postgres || exit 1"],
        "interval": 10,
        "timeout": 5,
        "retries": 10,
        "startPeriod": 30
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "__LOG_GROUP__",
          "awslogs-region": "__REGION__",
          "awslogs-stream-prefix": "postgres"
        }
      }
    },
    {
      "name": "tei",
      "image": "__TEI_IMAGE__",
      "essential": false,
      "command": ["--model-id", "BAAI/bge-small-en-v1.5", "--auto-truncate"],
      "environment": [
        {"name": "NO_COLOR", "value": "1"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "__LOG_GROUP__",
          "awslogs-region": "__REGION__",
          "awslogs-stream-prefix": "tei"
        }
      }
    }
  ]
}
