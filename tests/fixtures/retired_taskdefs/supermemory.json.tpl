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
        {"name": "SUPERMEMORY_BASE_URL", "value": "http://localhost:6767"},
        {"name": "SUPERMEMORY_ENGINE_VERSION", "value": "__SUPERMEMORY_ENGINE_VERSION__"},
        {"name": "SUPERMEMORY_ENGINE_IMAGE_DIGEST", "value": "__SUPERMEMORY_ENGINE_IMAGE_DIGEST__"},
        {"name": "SUPERMEMORY_ENGINE_IMAGE_REPO", "value": "__SUPERMEMORY_ENGINE_IMAGE_REPO__"},
        {"name": "SUPERMEMORY_ENGINE_BINARY_VERSION", "value": "__SUPERMEMORY_ENGINE_BINARY_VERSION__"},
        {"name": "SUPERMEMORY_TIMEOUT_S", "value": "__SUPERMEMORY_TIMEOUT_S__"}
      ],
      "dependsOn": [
        {"containerName": "supermemory", "condition": "HEALTHY"}
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
      "name": "supermemory",
      "image": "__SUPERMEMORY_IMAGE__",
      "essential": false,
      "environment": [
        {"name": "SUPERMEMORY_DATA_DIR", "value": "/tmp/supermemory"}
      ],
      "secrets": [
        {"name": "OPENAI_API_KEY", "valueFrom": "__SUPERMEMORY_SECRET_ARN__"}
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "node -e \"fetch('http://localhost:6767/').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))\""],
        "interval": 15,
        "timeout": 10,
        "retries": 10,
        "startPeriod": 150
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "__LOG_GROUP__",
          "awslogs-region": "__REGION__",
          "awslogs-stream-prefix": "supermemory"
        }
      }
    }
  ]
}
