"""
Azure Blob Storage helpers for model artifacts and chart images.

Authentication priority:
  1. AZURE_STORAGE_CONNECTION_STRING (local dev / connection string)
  2. AZURE_STORAGE_ACCOUNT_URL + DefaultAzureCredential
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from azure.identity import DefaultAzureCredential
from azure.storage.blob import (
    BlobServiceClient,
    BlobSasPermissions,
    generate_blob_sas,
)

from shared import config

logger = logging.getLogger(__name__)



def _get_client() -> BlobServiceClient:
    conn_str = config.get_storage_conn_str()
    if conn_str:
        return BlobServiceClient.from_connection_string(conn_str)

    account_url = config.get_storage_account_url()
    if account_url:
        return BlobServiceClient(account_url, credential=DefaultAzureCredential())

    raise ValueError(
        "No Azure Storage credentials found."
        "Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_URL."
    )



def upload_model(local_path: str, blob_name: str, container: str | None = None) -> None:
    """Upload a model artifact .pkl file to Blob Storage (overwrites if exists)."""
    container = container or config.get_model_container()
    client = _get_client()

    with open(local_path, "rb") as fh:
        client.get_blob_client(container=container, blob=blob_name).upload_blob(
            fh, overwrite=True
        )
    logger.info("Model uploaded", extra={"blob_name": blob_name, "container": container})



def download_model(blob_name: str, local_path: str, container: str | None = None) -> None:
    """Download a model artifact .pkl from Blob Storage to a local path."""
    container = container or config.get_model_container()
    client = _get_client()

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

    with open(local_path, "wb") as fh:
        stream = client.get_blob_client(container=container, blob=blob_name).download_blob()
        stream.readinto(fh)
    logger.info("Model downloaded", extra={"blob_name": blob_name, "local_path": local_path})



def get_blob_last_modified(blob_name: str, container: str | None = None) -> Optional[datetime]:
    """Return the blob's last-modified timestamp, or None if it does not exist."""
    container = container or config.get_model_container()

    try:
        client = _get_client()
        props = client.get_blob_client(container=container, blob=blob_name).get_blob_properties()
        return props.last_modified
    
    except Exception:
        return None



def upload_chart(png_bytes: bytes, blob_name: str, container: str | None = None) -> str:
    """
    Upload a PNG byte string and return a 1-hour SAS URL (or plain URL for
    managed-identity deployments where the container has read access).
    """
    container = container or config.get_chart_container()
    client = _get_client()
    client.get_blob_client(container=container, blob=blob_name).upload_blob(
        png_bytes, overwrite=True
    )

    account_name = client.account_name
    cred = client.credential

    # Connection-string path: credential object exposes account_key
    account_key: str | None = getattr(cred, "account_key", None)

    if account_key:
        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=container,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        return (
            f"https://{account_name}.blob.core.windows.net"
            f"/{container}/{blob_name}?{sas_token}"
        )

    # Managed-identity path: return the direct URL
    return f"https://{account_name}.blob.core.windows.net/{container}/{blob_name}"
