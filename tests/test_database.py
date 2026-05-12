import unittest

from src.database import (
    CLASSICMODELS_INCLUDE_TABLES,
    build_database_connection_options,
    build_database_unavailable_message,
    build_database_kwargs,
    normalize_database_uri,
)


class TestNormalizeDatabaseUri(unittest.TestCase):
    def test_leaves_psycopg_uri_unchanged(self) -> None:
        uri = "postgresql+psycopg://postgres:postgres@localhost:5432/classicmodels"
        self.assertEqual(normalize_database_uri(uri), uri)

    def test_upgrades_plain_postgresql_uri_to_psycopg(self) -> None:
        uri = "postgresql://postgres:postgres@localhost:5432/classicmodels"
        self.assertEqual(
            normalize_database_uri(uri),
            "postgresql+psycopg://postgres:postgres@localhost:5432/classicmodels",
        )

    def test_leaves_non_postgres_uris_unchanged(self) -> None:
        uri = "sqlite:///chinook.db"
        self.assertEqual(normalize_database_uri(uri), uri)


class TestBuildDatabaseKwargs(unittest.TestCase):
    def test_scopes_classicmodels_tables_for_postgres_by_default(self) -> None:
        kwargs = build_database_kwargs(
            "postgresql://postgres:postgres@localhost:5432/classicmodels",
            {},
        )

        self.assertEqual(kwargs, {"include_tables": CLASSICMODELS_INCLUDE_TABLES})

    def test_respects_explicit_database_schema_override(self) -> None:
        kwargs = build_database_kwargs(
            "postgresql://postgres:postgres@localhost:5432/classicmodels",
            {"DATABASE_SCHEMA": "sales"},
        )

        self.assertEqual(kwargs, {"include_tables": CLASSICMODELS_INCLUDE_TABLES})

    def test_does_not_force_schema_for_non_postgres(self) -> None:
        kwargs = build_database_kwargs("sqlite:///chinook.db", {})
        self.assertEqual(kwargs, {})


class TestBuildDatabaseConnectionOptions(unittest.TestCase):
    def test_sets_search_path_via_connect_args_for_postgres(self) -> None:
        options = build_database_connection_options(
            "postgresql://postgres:postgres@localhost:5432/classicmodels",
            {"DATABASE_SCHEMA": "classicmodels"},
        )

        self.assertEqual(
            options,
            {"connect_args": {"options": "-csearch_path=classicmodels"}},
        )

    def test_ignores_search_path_for_non_postgres(self) -> None:
        self.assertEqual(build_database_connection_options("sqlite:///chinook.db", {}), {})


class TestBuildDatabaseUnavailableMessage(unittest.TestCase):
    def test_includes_connection_context_for_postgres_uri(self) -> None:
        message = build_database_unavailable_message(
            "postgresql://postgres:postgres@localhost:5432/classicmodels",
            RuntimeError("connection timeout expired"),
        )

        self.assertIn("Unable to connect to PostgreSQL.", message)
        self.assertIn("Host: localhost", message)
        self.assertIn("Port: 5432", message)
        self.assertIn("Database: classicmodels", message)
        self.assertIn("connection timeout expired", message)


if __name__ == "__main__":
    unittest.main()
