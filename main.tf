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
  default     = 128
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
