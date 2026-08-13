"""
db.py
-----
Isolated database connection module for Google Cloud AlloyDB (PostgreSQL).

Supports TWO connection modes:

1. **AlloyDB Auth Proxy / Local Mode** (default):
   Set `ALLOYDB_HOST` (e.g. 127.0.0.1) and `ALLOYDB_PORT` (e.g. 5432).
   Connects directly over TCP to the proxy.

2. **AlloyDB Python Connector Mode**:
   Set `ALLOYDB_INSTANCE_URI` (e.g. projects/PROJECT/locations/REGION/clusters/slait-steel/instances/primary).
   Connects securely via `google-cloud-alloydb-connector` without requiring a local proxy process.

Environment variables required
-------------------------------
    ALLOYDB_INSTANCE_URI - (Optional) Full GCP resource path to the AlloyDB instance
    ALLOYDB_HOST         - Host address (default: 127.0.0.1)
    ALLOYDB_PORT         - Port number (default: 5432)
    ALLOYDB_DATABASE     - Database name (default: postgres)
    ALLOYDB_USER         - Database user (default: postgres)
    ALLOYDB_PASSWORD     - Database password
"""

import os
import logging

import psycopg
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env file if present (for local development)
load_dotenv()


def get_connection() -> psycopg.Connection:
    """
    Return a new psycopg (v3) connection to the AlloyDB instance.

    If `ALLOYDB_INSTANCE_URI` is set in environment, uses the official
    `google-cloud-alloydb-connector` to establish an IAM-authenticated connection.

    Otherwise, connects via TCP to `ALLOYDB_HOST`:`ALLOYDB_PORT` (AlloyDB Auth Proxy mode).

    Raises:
        psycopg.OperationalError: If connection cannot be established.
        ValueError: If required environment variables are missing.
    """
    user = os.environ.get("ALLOYDB_USER", "postgres")
    password = os.environ.get("ALLOYDB_PASSWORD")
    database = os.environ.get("ALLOYDB_DATABASE", "postgres")
    instance_uri = os.environ.get("ALLOYDB_INSTANCE_URI")

    if not password:
        raise ValueError(
            "ALLOYDB_PASSWORD environment variable is not set. "
            "Please configure your .env file."
        )

    # -----------------------------------------------------------------------
    # Mode 1: Google Cloud AlloyDB Python Connector (if ALLOYDB_INSTANCE_URI set)
    # -----------------------------------------------------------------------
    if instance_uri:
        try:
            from google.cloud.alloydbconnector import Connector

            logger.info("Connecting via google-cloud-alloydb-connector: %s", instance_uri)
            connector = Connector()
            conn = connector.connect(
                instance_uri,
                "psycopg",
                user=user,
                password=password,
                db=database,
            )
            return conn
        except ImportError:
            logger.error(
                "google-cloud-alloydb-connector is not installed. "
                "Install it with: pip install 'google-cloud-alloydb-connector[psycopg]'"
            )
            raise

    # -----------------------------------------------------------------------
    # Mode 2: AlloyDB Auth Proxy / Standard TCP Connection
    # -----------------------------------------------------------------------
    host = os.environ.get("ALLOYDB_HOST", "127.0.0.1")
    port = os.environ.get("ALLOYDB_PORT", "5432")

    logger.info("Connecting via TCP: %s:%s (database: %s)", host, port, database)
    conn = psycopg.connect(
        host=host,
        port=int(port),
        dbname=database,
        user=user,
        password=password,
        connect_timeout=10,
        autocommit=False,
    )
    return conn
