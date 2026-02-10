"""PostgreSQL connection manager for three-tier storage."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from psycopg_pool import ConnectionPool

from lib.utils.config import config


class PostgresClient:
    """PostgreSQL connection manager with connection pooling."""

    def __init__(self):
        self.pool: ConnectionPool
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize connection pool."""
        conninfo = (
            f"host={config.database.postgres_host} "
            f"port={config.database.postgres_port} "
            f"dbname={config.database.postgres_database} "
            f"user={config.database.postgres_user} "
            f"password={config.database.postgres_password}"
        )
        self.pool = ConnectionPool(
            conninfo=conninfo, min_size=2, max_size=10, open=False
        )
        # psycopg_pool does not automatically open when open=False.
        self.pool.open()

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)

    @contextmanager
    def get_cursor(self):
        """Get a cursor from a pooled connection."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cursor.close()

    def execute_query(
        self, query: str, params: tuple[Any, ...] | None = None
    ) -> list[tuple[Any, ...]]:
        """Execute a query and return results."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params or ())
            return cursor.fetchall()

    def execute_update(self, query: str, params: tuple[Any, ...] | None = None) -> int:
        """Execute an update/insert query and return affected rows."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params or ())
            return cursor.rowcount

    def execute_batch(
        self, query: str, params_list: list[tuple], page_size: int = 100
    ) -> None:
        """Execute batch insert/update."""
        with self.get_cursor() as cursor:
            cursor.executemany(query, params_list)

    def close(self):
        """Close connection pool."""
        # psycopg_pool ConnectionPool exposes .close(); older psycopg2 pools used .closeall().
        if getattr(self, "pool", None) is not None:
            self.pool.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
