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
        {"name": "HINDSIGHT_API_URL", "value": "http://localhost:8888"},
        {"name": "HINDSIGHT_ENGINE_VERSION", "value": "__HINDSIGHT_ENGINE_VERSION__"},
        {"name": "HINDSIGHT_ENGINE_IMAGE_DIGEST", "value": "__HINDSIGHT_ENGINE_IMAGE_DIGEST__"},
        {"name": "HINDSIGHT_ENGINE_IMAGE_REPO", "value": "__HINDSIGHT_ENGINE_IMAGE_REPO__"},
        {"name": "HINDSIGHT_TIMEOUT_S", "value": "__HINDSIGHT_TIMEOUT_S__"}
      ],
      "dependsOn": [
        {"containerName": "hindsight", "condition": "HEALTHY"}
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
      "name": "hindsight",
      "image": "__HINDSIGHT_IMAGE__",
      "essential": false,
      "environment": [
        {"name": "HINDSIGHT_API_LLM_PROVIDER", "value": "__HINDSIGHT_LLM_PROVIDER__"},
        {"name": "HINDSIGHT_API_LLM_MODEL", "value": "__HINDSIGHT_LLM_MODEL__"},
        {"name": "HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES", "value": "__HINDSIGHT_RETAIN_MAX_RETRIES__"},
        {"name": "HINDSIGHT_DATA_DIR", "value": "/tmp/hindsight-data"}
      ],
      "secrets": [
        {"name": "HINDSIGHT_API_LLM_API_KEY", "valueFrom": "__HINDSIGHT_SECRET_ARN__"}
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -fsS http://localhost:8888/health || exit 1"],
        "interval": 15,
        "timeout": 10,
        "retries": 10,
        "startPeriod": 120
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "__LOG_GROUP__",
          "awslogs-region": "__REGION__",
          "awslogs-stream-prefix": "hindsight"
        }
      }
    }
  ]
}
