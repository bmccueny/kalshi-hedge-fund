# Trading Bot Deployment

## Quick Deploy to Google Cloud

### 1. Install Google Cloud CLI
```bash
curl https://sdk.cloud.google.com | bash
gcloud init
gcloud auth login
```

### 2. Create Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
```

### 3. Deploy to Cloud Run
```bash
# Build and deploy in one command
gcloud run deploy kalshi-hedge-fund \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="ANTHROPIC_API_KEY=your_key,KALSHI_API_KEY=your_key"
```

### Or build manually
```bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/kalshi-hedge-fund

# Deploy
gcloud run deploy kalshi-hedge-fund \
  --image gcr.io/PROJECT_ID/kalshi-hedge-fund \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### 4. Set Environment Variables
```bash
gcloud run services update kalshi-hedge-fund \
  --update-env-vars="ANTHROPIC_API_KEY=xxx,KALSHI_API_KEY=xxx,NEWS_API_KEY=xxx"
```

## Scheduled Jobs (Cloud Scheduler)

```bash
# Create job to run bot every hour
gcloud scheduler jobs create http hourly-bot \
  --schedule="0 * * * *" \
  --uri="https://your-service-url/run" \
  --location=us-central1
```

## Cost Estimate
- Cloud Run: ~$0.000224 per vCPU-second + $0.000024 per GB-second
- Cloud Scheduler: $0.10 per job (free up to 3)
- Estimated: ~$10-30/month for low usage
