terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "embedding_dimensions" {
  description = "Dimensions of your custom embedding model output"
  type        = number
  default     = 768
}

variable "deploy_index" {
  description = "Whether to deploy the index to endpoint (set false for cost savings during development)"
  type        = bool
  default     = false
}

variable "create_index_endpoint" {
  description = "Whether to create the index endpoint at all (set false during early development)"
  type        = bool
  default     = false
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "cairo-genizah"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "storage.googleapis.com",
    "compute.googleapis.com",
    "run.googleapis.com"
  ])
  
  project = var.project_id
  service = each.value
}

# Cloud Storage buckets
resource "google_storage_bucket" "raw_content" {
  name     = "${var.project_name}-multimodal-raw-${var.environment}"
  location = var.region
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }
  
  labels = {
    environment = var.environment
    purpose     = "multimodal-raw-content"
  }
}

resource "google_storage_bucket" "vector_index" {
  name     = "${var.project_name}-vector-index-${var.environment}"
  location = var.region
  
  uniform_bucket_level_access = true
  
  labels = {
    environment = var.environment
    purpose     = "vector-index-storage"
  }
}

resource "google_storage_bucket" "model_artifacts" {
  name     = "${var.project_name}-model-artifacts-${var.environment}"
  location = var.region
  
  uniform_bucket_level_access = true
  
  labels = {
    environment = var.environment
    purpose     = "custom-models"
  }
}

# Vertex AI Vector Search Index
resource "google_vertex_ai_index" "multimodal_index" {
  display_name = "${var.project_name}-multimodal-rag-index-${var.environment}"
  description  = "Vector index for multimodal RAG with custom embeddings"
  region       = var.region
  
  metadata {
    contents_delta_uri = "gs://${google_storage_bucket.vector_index.name}/index"
    config {
      dimensions                  = var.embedding_dimensions
      approximate_neighbors_count = 150
      distance_measure_type      = "COSINE_DISTANCE"
      
      algorithm_config {
        tree_ah_config {
          leaf_node_embedding_count    = 1000
          leaf_nodes_to_search_percent = 10
        }
      }
      
      shard_size = "SHARD_SIZE_SMALL"  # Adjust based on your data size
    }
  }
  
  labels = {
    environment = var.environment
    purpose     = "multimodal-rag"
  }
  
  depends_on = [google_project_service.required_apis]
}

# Index Endpoint
resource "google_vertex_ai_index_endpoint" "multimodal_endpoint" {
  count = var.create_index_endpoint ? 1 : 0
  
  display_name   = "${var.project_name}-multimodal-rag-endpoint-${var.environment}"
  description    = "Endpoint for multimodal RAG vector search"
  region         = var.region
  
  labels = {
    environment = var.environment
    purpose     = "multimodal-rag"
  }
  
  depends_on = [google_project_service.required_apis]
}

# Deploy Index to Endpoint
resource "google_vertex_ai_index_endpoint_deployed_index" "deployed_index" {
  count = var.create_index_endpoint && var.deploy_index ? 1 : 0  # Both flags must be true
  
  index_endpoint    = google_vertex_ai_index_endpoint.multimodal_endpoint[0].id
  index            = google_vertex_ai_index.multimodal_index.id
  deployed_index_id = "${var.project_name}-deployed-index-${var.environment}"
  display_name     = "Deployed Multimodal Index"
  
  dedicated_resources {
    machine_spec {
      machine_type = "n1-standard-2"  # Smaller for dev
    }
    min_replica_count = 1
    max_replica_count = 2  # Lower max for dev
  }
}

# Service Accounts
resource "google_service_account" "embedding_processor" {
  account_id   = substr("${var.project_name}-embed-${var.environment}", 0, 30)
  display_name = "Cairo Genizah Embedding Processor Service Account"
  description  = "Service account for processing and generating embeddings"
}

resource "google_service_account" "rag_application" {
  account_id   = substr("${var.project_name}-rag-${var.environment}", 0, 30)
  display_name = "Cairo Genizah RAG Application Service Account" 
  description  = "Service account for RAG application queries"
}

# IAM permissions for embedding processor
resource "google_project_iam_member" "embedding_processor_permissions" {
  for_each = toset([
    "roles/storage.objectAdmin",           # Read/write to buckets
    "roles/aiplatform.user",              # Use Vertex AI services
    "roles/aiplatform.indexUser"          # Manage vector index
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.embedding_processor.email}"
}

# IAM permissions for RAG application (read-only)
resource "google_project_iam_member" "rag_application_permissions" {
  for_each = toset([
    "roles/storage.objectViewer",         # Read from buckets
    "roles/aiplatform.user",              # Query vector search
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.rag_application.email}"
}

# Cloud Run service for embedding processing (optional)
resource "google_cloud_run_v2_service" "embedding_service" {
  name     = "${var.project_name}-embed-svc-${var.environment}"
  location = var.region
  
  template {
    service_account = google_service_account.embedding_processor.email
    
    containers {
      image = "gcr.io/${var.project_id}/embedding-processor:latest"  # Your custom image
      
      resources {
        limits = {
          cpu    = "2"
          memory = "8Gi"
        }
        cpu_idle = true
      }
      
      env {
        name  = "PROJECT_ID"
        value = var.project_id
      }
      
      env {
        name  = "INDEX_ENDPOINT"
        value = var.create_index_endpoint ? google_vertex_ai_index_endpoint.multimodal_endpoint[0].name : "not-created"
      }
      
      env {
        name  = "VECTOR_BUCKET"
        value = google_storage_bucket.vector_index.name
      }
    }
    
    scaling {
      min_instance_count = 0
      max_instance_count = 10
    }
  }
  
  depends_on = [google_project_service.required_apis]
}

# Monitoring and Alerting
resource "google_monitoring_notification_channel" "email" {
  display_name = "RAG System Alerts"
  type         = "email"
  
  labels = {
    email_address = "your-email@company.com"  # Update this
  }
}

resource "google_monitoring_alert_policy" "vector_search_errors" {
  display_name = "Vector Search High Error Rate"
  combiner     = "OR"
  
  conditions {
    display_name = "High error rate on vector search"
    
    condition_threshold {
      filter          = "resource.type=\"aiplatform.googleapis.com/IndexEndpoint\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0.1  # 10% error rate
      duration        = "300s"
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }
  
  notification_channels = [google_monitoring_notification_channel.email.id]
}

# Outputs
output "vector_index_id" {
  description = "Vertex AI Vector Index ID"
  value       = google_vertex_ai_index.multimodal_index.id
}

output "index_endpoint_id" {
  description = "Vertex AI Index Endpoint ID"
  value       = var.create_index_endpoint ? google_vertex_ai_index_endpoint.multimodal_endpoint[0].id : "Not created - set create_index_endpoint=true"
}

output "index_endpoint_domain" {
  description = "Index Endpoint Domain for API calls"
  value       = var.create_index_endpoint ? google_vertex_ai_index_endpoint.multimodal_endpoint[0].public_endpoint_domain_name : "Not created - set create_index_endpoint=true"
}

output "raw_content_bucket" {
  description = "Cloud Storage bucket for raw multimodal content"
  value       = google_storage_bucket.raw_content.name
}

output "vector_index_bucket" {
  description = "Cloud Storage bucket for vector index data"
  value       = google_storage_bucket.vector_index.name
}

output "model_artifacts_bucket" {
  description = "Cloud Storage bucket for your custom models"
  value       = google_storage_bucket.model_artifacts.name
}

output "embedding_processor_sa" {
  description = "Service account email for embedding processing"
  value       = google_service_account.embedding_processor.email
}

output "rag_application_sa" {
  description = "Service account email for RAG application"
  value       = google_service_account.rag_application.email
}

output "embedding_service_url" {
  description = "Cloud Run embedding service URL"
  value       = google_cloud_run_v2_service.embedding_service.uri
}