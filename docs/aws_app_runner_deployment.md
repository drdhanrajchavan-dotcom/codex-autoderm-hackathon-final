# AWS App Runner Deployment Plan

This project is built for a local hackathon demo first. To make the demo publicly reachable, use a private Amazon ECR image deployed to AWS App Runner.

Do not launch or update paid AWS resources without an explicit operator go-ahead.

## Local AWS Readiness Check

Read-only checks on 2026-04-16 showed:

- AWS CLI is installed.
- Default region is `ap-south-1`.
- `aws sts get-caller-identity` succeeds from this machine.
- The configured IAM user has `AdministratorAccess`.
- App Runner is reachable in `ap-south-1`.
- ECR is reachable in `ap-south-1`.
- EC2 is reachable and a default VPC exists.
- Docker is installed and running.

Keep account IDs and credential details out of tracked files. Derive the account ID locally when needed:

```bash
REGION=ap-south-1
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
```

## Recommended Path

For the hackathon public demo:

1. Package the app as a Docker image.
2. Push the image to a private Amazon ECR repository.
3. Run the image on AWS App Runner.
4. Use the App Runner public HTTPS URL for judging.
5. Optionally attach a custom domain later.

App Runner is simpler than EC2 for this demo because it provides HTTPS, deployment management, logs, and autoscaling without managing a server. The caveat is that App Runner is CPU/container hosting, not GPU hosting. If YOLO inference latency needs a GPU, use EC2 G5/G6 with Docker and a reverse proxy instead.

## Implemented App Runner Readiness

Implemented on 2026-04-16:

- `scripts/start_prod.sh` starts FastAPI on `127.0.0.1:8000` and Next.js on `0.0.0.0:3000`.
- `web/next.config.mjs` proxies same-origin `/api/*` and `/healthz` requests to FastAPI.
- Frontend API calls default to same-origin paths instead of `localhost`.
- FastAPI exposes `/healthz` and rejects image uploads above the configured limit. The default public-demo cap is 8 MB per image.
- `Dockerfile` builds a CPU App Runner image with the official KEEP checkpoint only.
- `.dockerignore` excludes private data, full pullback artifacts, source datasets, training runs, and local environments.
- The image generates local-only `config/active_checkpoint.json` for official KEEP run `iter_018` during build.
- The image includes only the selected `iter_018` `best.pt`, preprocessing hash, and minimal non-PHI iteration metadata from `.pulled_artifacts/runs/`.

Local verification completed with:

```bash
npm run build
venv/bin/python -m pytest
venv/bin/python -m scripts.check_release
docker buildx build --platform linux/amd64 -t autoderm-demo:local --load .
```

## AWS Setup Flow

Set deployment variables:

```bash
REGION=ap-south-1
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REPO=autoderm-demo
IMAGE_TAG=hackathon-demo
IMAGE_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:$IMAGE_TAG"
```

Create ECR:

```bash
aws ecr create-repository \
  --repository-name "$REPO" \
  --region "$REGION"
```

Login Docker to ECR:

```bash
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin \
    "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
```

Build and push:

```bash
docker buildx build --platform linux/amd64 -t "$IMAGE_URI" .
docker push "$IMAGE_URI"
```

Create an App Runner ECR access role. The trust policy should allow `build.apprunner.amazonaws.com` to assume the role.

`apprunner-trust-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "build.apprunner.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

Create and attach the role:

```bash
aws iam create-role \
  --role-name AppRunnerECRAccessRole \
  --assume-role-policy-document file://apprunner-trust-policy.json

aws iam attach-role-policy \
  --role-name AppRunnerECRAccessRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
```

Create the App Runner service:

```bash
ACCESS_ROLE_ARN="arn:aws:iam::$ACCOUNT_ID:role/AppRunnerECRAccessRole"

aws apprunner create-service \
  --region "$REGION" \
  --service-name autoderm-demo \
  --source-configuration "{
    \"AuthenticationConfiguration\": {
      \"AccessRoleArn\": \"$ACCESS_ROLE_ARN\"
    },
    \"ImageRepository\": {
      \"ImageIdentifier\": \"$IMAGE_URI\",
      \"ImageRepositoryType\": \"ECR\",
      \"ImageConfiguration\": {
        \"Port\": \"3000\",
        \"RuntimeEnvironmentVariables\": {
          \"NODE_ENV\": \"production\",
          \"NEXT_PUBLIC_API_BASE_URL\": \"\"
        }
      }
    },
    \"AutoDeploymentsEnabled\": true
  }" \
  --instance-configuration '{
    "Cpu": "2 vCPU",
    "Memory": "4 GB"
  }'
```

Fetch the public URL:

```bash
aws apprunner list-services \
  --region "$REGION" \
  --query 'ServiceSummaryList[?ServiceName==`autoderm-demo`].[ServiceUrl,Status]' \
  --output table
```

## Safety Notes

- Do not upload PHI or patient photos unless there is explicit consent and a proper privacy setup.
- Prefer the bundled approved demo photos for the public hackathon page.
- Keep the disclaimer visible: research demonstration, not a medical diagnosis.
- Do not persist uploaded images unless storage is explicitly designed, private, and opt-in.
- Add request size limits and basic abuse protection.
- Pause or delete the App Runner service after the hackathon to avoid ongoing cost.

## AWS References

- AWS App Runner service creation: https://docs.aws.amazon.com/apprunner/latest/dg/manage-create.html
- App Runner ECR image source: https://docs.aws.amazon.com/apprunner/latest/dg/service-source-image.html
- App Runner IAM access role: https://docs.aws.amazon.com/apprunner/latest/dg/security_iam_service-with-iam.html
- Amazon ECR Docker push flow: https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html
- AWS App Runner pricing: https://aws.amazon.com/apprunner/pricing/
