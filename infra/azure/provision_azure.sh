#!/usr/bin/env bash
# One-time idempotent Microsoft Azure provisioning for ScholarMind AI on Azure Container Apps.
# Prerequisites: Azure CLI (`az login`) with Contributor + User Access Administrator rights.

set -euo pipefail

LOCATION="${AZURE_LOCATION:-eastus}"
RG="${AZURE_RESOURCE_GROUP:-rg-scholarmind-prod}"
ACR_NAME="${AZURE_ACR_NAME:-acrscholarmind$RANDOM}"
STORAGE_NAME="${AZURE_STORAGE_NAME:-stscholarmind$RANDOM}"
ENV_NAME="${AZURE_ACA_ENV:-cae-scholarmind-prod}"
API_APP="${AZURE_API_APP_NAME:-scholarmind-api}"
UI_APP="${AZURE_UI_APP_NAME:-scholarmind-ui}"

echo "=== 1. Creating Resource Group ($RG in $LOCATION) ==="
az group create --name "$RG" --location "$LOCATION"

echo "=== 2. Creating Azure Container Registry ($ACR_NAME) ==="
az acr create --resource-group "$RG" --name "$ACR_NAME" --sku Basic --admin-enabled true
ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv)
ACR_USER=$(az acr credential show --name "$ACR_NAME" --query username -o tsv)
ACR_PASS=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)

echo "=== 3. Creating Azure Storage Account & File Shares for Persistent SQLite/FAISS/HF Cache ==="
az storage account create \
  --name "$STORAGE_NAME" \
  --resource-group "$RG" \
  --location "$LOCATION" \
  --sku Standard_LRS
STORAGE_KEY=$(az storage account keys list -g "$RG" -n "$STORAGE_NAME" --query "[0].value" -o tsv)

az storage share create --name "scholarmind-data" --account-name "$STORAGE_NAME" --account-key "$STORAGE_KEY" --quota 20
az storage share create --name "scholarmind-hf-cache" --account-name "$STORAGE_NAME" --account-key "$STORAGE_KEY" --quota 20

echo "=== 4. Creating Azure Container Apps Environment ($ENV_NAME) ==="
az containerapp env create \
  --name "$ENV_NAME" \
  --resource-group "$RG" \
  --location "$LOCATION"

az containerapp env storage set \
  --name "$ENV_NAME" --resource-group "$RG" \
  --storage-name "data-share" \
  --azure-file-account-name "$STORAGE_NAME" \
  --azure-file-account-key "$STORAGE_KEY" \
  --azure-file-share-name "scholarmind-data" \
  --access-mode ReadWrite

az containerapp env storage set \
  --name "$ENV_NAME" --resource-group "$RG" \
  --storage-name "hf-cache-share" \
  --azure-file-account-name "$STORAGE_NAME" \
  --azure-file-account-key "$STORAGE_KEY" \
  --azure-file-share-name "scholarmind-hf-cache" \
  --access-mode ReadWrite

echo "=== 5. Building Initial Image in ACR ==="
az acr build --registry "$ACR_NAME" --image "scholarmind-ai:latest" .

echo "=== 6. Creating FastAPI Backend Container App ($API_APP) ==="
az containerapp create \
  --name "$API_APP" \
  --resource-group "$RG" \
  --environment "$ENV_NAME" \
  --image "${ACR_LOGIN_SERVER}/scholarmind-ai:latest" \
  --registry-server "$ACR_LOGIN_SERVER" \
  --registry-username "$ACR_USER" \
  --registry-password "$ACR_PASS" \
  --target-port 8000 \
  --ingress external \
  --cpu 2.0 --memory 4.0Gi \
  --min-replicas 1 --max-replicas 1 \
  --env-vars \
    APP_MODE=api \
    SCHOLARMIND_ENV=production \
    SCHOLARMIND_PROFILE=production \
    SCHOLARMIND_PORT=8000 \
    SCHOLARMIND_DATA_DIR=/app/data \
    SCHOLARMIND_ARTIFACTS_DIR=/app/artifacts \
    HF_HOME=/app/artifacts/hf_cache

API_FQDN=$(az containerapp show --name "$API_APP" --resource-group "$RG" --query "properties.configuration.ingress.fqdn" -o tsv)

echo "=== 7. Creating Streamlit UI Container App ($UI_APP) ==="
az containerapp create \
  --name "$UI_APP" \
  --resource-group "$RG" \
  --environment "$ENV_NAME" \
  --image "${ACR_LOGIN_SERVER}/scholarmind-ai:latest" \
  --registry-server "$ACR_LOGIN_SERVER" \
  --registry-username "$ACR_USER" \
  --registry-password "$ACR_PASS" \
  --target-port 8501 \
  --ingress external \
  --cpu 1.0 --memory 2.0Gi \
  --min-replicas 1 --max-replicas 2 \
  --env-vars \
    APP_MODE=ui \
    SCHOLARMIND_ENV=production \
    SCHOLARMIND_PROFILE=production \
    SCHOLARMIND_UI_PORT=8501 \
    SCHOLARMIND_API_URL="https://${API_FQDN}"

UI_FQDN=$(az containerapp show --name "$UI_APP" --resource-group "$RG" --query "properties.configuration.ingress.fqdn" -o tsv)

echo "================================================================"
echo "Azure Provisioning Complete!"
echo "  Resource Group : $RG"
echo "  ACR Name       : $ACR_NAME"
echo "  FastAPI URL    : https://${API_FQDN}"
echo "  FastAPI Docs   : https://${API_FQDN}/docs"
echo "  Streamlit UI   : https://${UI_FQDN}"
echo "Next: Mount Azure Files volumes & add GitHub Secrets (see docs/AZURE_DEPLOYMENT.md)."
echo "================================================================"
