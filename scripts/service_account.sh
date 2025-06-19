#!/bin/bash
PROJECT_ID="hebrew-document"  # Replace with your actual project ID

echo "Creating service account for GitHub Actions..."

# Create service account
gcloud iam service-accounts create github-actions-sa \
    --display-name="GitHub Actions" \
    --description="Service account for GitHub Actions deployments"

# Grant permissions
SA_EMAIL="github-actions-sa@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/compute.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/iam.serviceAccountUser"

# Create JSON key
gcloud iam service-accounts keys create github-sa-key.json \
    --iam-account="${SA_EMAIL}"

echo ""
echo "✅ Service account created!"
echo ""
echo "📋 Copy this JSON content to GitHub secret 'GCP_SA_KEY':"
echo "================================================"
cat github-sa-key.json
echo "================================================"
echo ""
echo "🔐 Add these GitHub secrets:"
echo "GCP_SA_KEY: [paste the JSON above]"
echo "GCP_PROJECT_ID: $PROJECT_ID"
echo ""
echo "🗑️  Cleaning up local file..."
rm github-sa-key.json
echo "Done!"