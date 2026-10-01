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
        {"name": "ATOMICMEMORY_API_URL", "value": "http://localhost:17350"},
        {"name": "ATOMICMEMORY_API_KEY", "value": "__ATOMICMEMORY_API_KEY__"},
        {"name": "ATOMICMEMORY_ENGINE_VERSION", "value": "__ATOMICMEMORY_ENGINE_VERSION__"},
        {"name": "ATOMICMEMORY_ENGINE_IMAGE_DIGEST", "value": "__ATOMICMEMORY_ENGINE_IMAGE_DIGEST__"},
        {"name": "ATOMICMEMORY_ENGINE_IMAGE_REPO", "value": "__ATOMICMEMORY_ENGINE_IMAGE_REPO__"},
        {"name": "ATOMICMEMORY_ENGINE_SOURCE_SHA", "value": "__ATOMICMEMORY_ENGINE_SOURCE_SHA__"},
        {"name": "ATOMICMEMORY_TIMEOUT_S", "value": "__ATOMICMEMORY_TIMEOUT_S__"}
      ],
      "dependsOn": [
        {"containerName": "atomicmemory", "condition": "HEALTHY"}
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
      "name": "atomicmemory",
      "image": "__ATOMICMEMORY_IMAGE__",
      "essential": false,
      "environment": [
        {"name": "DATABASE_URL", "value": "embedded"},
        {"name": "RAW_STORAGE_DEPLOYMENT_ENV", "value": "local"},
        {"name": "CORE_API_KEY", "value": "__ATOMICMEMORY_API_KEY__"},
        {"name": "EMBEDDING_PROVIDER", "value": "__ATOMICMEMORY_EMBEDDING_PROVIDER__"},
        {"name": "EMBEDDING_MODEL", "value": "__ATOMICMEMORY_EMBEDDING_MODEL__"},
        {"name": "EMBEDDING_DIMENSIONS", "value": "__ATOMICMEMORY_EMBEDDING_DIMS__"},
        {"name": "LLM_PROVIDER", "value": "anthropic"},
        {"name": "LLM_MODEL", "value": "__ATOMICMEMORY_LLM_MODEL__"}
      ],
      "secrets": [
        {"name": "ANTHROPIC_API_KEY", "valueFrom": "__ATOMICMEMORY_SECRET_ARN__"}
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "node -e \"fetch('http://localhost:17350/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))\""],
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
          "awslogs-stream-prefix": "atomicmemory"
        }
      }
    }
  ]
}
