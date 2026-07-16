from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from ..config import Settings, settings
from ..ingestion.method_aliases import normalize_method
from ..persistence.database import Database
from .catalog_query import AggregateQuery, formula_elements, normalize_reactant_name
from .growth_records import normalize_growth_record


CATALOG_VERSION = "2"
REPRESENTATIVE_LIMIT = 12


@dataclass(frozen=True)
class CatalogSyncResult:
    status: str
    record_count: int
    source_hash: str


class FactCatalog:
    """Structured, source-derived facts for aggregate real-record queries.

    This catalog deliberately lives in the existing application database. It is not a new
    retrieval service and it never calls the embedding model. Milvus remains the path for
    individual material protocol retrieval.
    """

    def __init__(self, database_url: str = settings.memory_database_url) -> None:
        self.database = Database(database_url)
        self._schema_ready = False
        self._schema_lock = Lock()

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            connection = self.database.connect()
            try:
                cursor = connection.cursor()
                for statement in _SCHEMA:
                    self.database.execute(cursor, statement)
                connection.commit()
                self._schema_ready = True
            finally:
                connection.close()

    def sync_from_jsonl(self, path: str | Path) -> CatalogSyncResult:
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"Catalog source is missing: {input_path}")
        self.ensure_schema()
        source_hash = _sha256(input_path)
        connection = self.database.connect()
        try:
            cursor = connection.cursor()
            self.database.execute(
                cursor,
                "SELECT catalog_version, source_hash, record_count "
                "FROM growth_catalog_meta WHERE catalog_key = ?",
                ("growth_records",),
            )
            existing = cursor.fetchone()
            if (
                existing
                and str(existing[0]) == CATALOG_VERSION
                and str(existing[1]) == source_hash
            ):
                return CatalogSyncResult("ready", int(existing[2]), source_hash)

            for table in (
                "growth_catalog_reactants",
                "growth_catalog_elements",
                "growth_catalog_records",
            ):
                self.database.execute(cursor, f"DELETE FROM {table}")

            count = 0
            with input_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    raw = json.loads(line)
                    self._insert_raw_record(cursor, raw)
                    count += 1

            self.database.execute(
                cursor,
                """
                INSERT INTO growth_catalog_meta (catalog_key, catalog_version, source_hash, record_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (catalog_key) DO UPDATE SET
                    catalog_version = EXCLUDED.catalog_version,
                    source_hash = EXCLUDED.source_hash,
                    record_count = EXCLUDED.record_count
                """,
                ("growth_records", CATALOG_VERSION, source_hash, count),
            )
            connection.commit()
            return CatalogSyncResult("rebuilt", count, source_hash)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def aggregate(self, query: AggregateQuery) -> dict[str, Any]:
        self.ensure_schema()
        where, params = self._where_clause(query)
        connection = self.database.connect()
        try:
            cursor = connection.cursor()
            summary_sql = f"""
                SELECT COUNT(*), COUNT(DISTINCT r.formula), COUNT(DISTINCT NULLIF(r.doi, ''))
                FROM growth_catalog_records r
                WHERE {where}
            """
            self.database.execute(cursor, summary_sql, params)
            row = cursor.fetchone() or (0, 0, 0)
            total_records, total_formulas, total_dois = (int(row[0]), int(row[1]), int(row[2]))

            group_sql = self._group_query(query, where)
            self.database.execute(cursor, group_sql, params)
            groups = [self._group_from_row(query, row) for row in cursor.fetchall()]

            representative_sql = f"""
                SELECT r.record_id, r.formula, r.growth_method, r.temperature_program,
                       r.precursors_json, r.doi, r.source_text
                FROM growth_catalog_records r
                WHERE {where}
                ORDER BY r.growth_method ASC, r.formula ASC, r.record_id ASC
                LIMIT ?
            """
            self.database.execute(cursor, representative_sql, [*params, REPRESENTATIVE_LIMIT])
            representatives = [self._record_from_row(row, query) for row in cursor.fetchall()]
        finally:
            connection.close()

        return {
            "query": query,
            "total_records": total_records,
            "total_formulas": total_formulas,
            "total_dois": total_dois,
            "groups": groups,
            "representatives": representatives,
        }

    def _insert_raw_record(self, cursor: Any, raw: dict[str, Any]) -> None:
        record = normalize_growth_record(raw, "growth_catalog")
        record_id = record["record_id"]
        if not record_id:
            return
        method = normalize_method(str(raw.get("method") or "")).normalized
        self.database.execute(
            cursor,
            """
            INSERT INTO growth_catalog_records
                (record_id, formula, growth_method, temperature_program, precursors_json, doi, source_text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                record["material_formula"] or "",
                method,
                record["temperature_program"],
                json.dumps(record["precursors"], ensure_ascii=False, separators=(",", ":")),
                record["doi"] or "",
                record["source_text"],
            ),
        )
        for element in formula_elements(record["material_formula"] or ""):
            self.database.execute(
                cursor,
                "INSERT INTO growth_catalog_elements (record_id, element) VALUES (?, ?)",
                (record_id, element),
            )
        reactants = raw.get("reactants") if isinstance(raw.get("reactants"), list) else []
        for reactant in reactants:
            if not isinstance(reactant, dict):
                continue
            name = str(reactant.get("n") or "").strip()
            if not name:
                continue
            raw_role = str(reactant.get("type") or "").strip()
            role = {
                "raw": "raw",
                "adtv": "additive",
                "raw_adtv": "raw_and_additive",
            }.get(raw_role, "other")
            normalized = normalize_reactant_name(name)
            self.database.execute(
                cursor,
                """
                INSERT INTO growth_catalog_reactants (record_id, reactant_normalized, role)
                VALUES (?, ?, ?)
                """,
                (record_id, normalized, role),
            )

    def _where_clause(self, query: AggregateQuery) -> tuple[str, list[str]]:
        clauses = ["1 = 1"]
        params: list[str] = []
        if query.get("element"):
            clauses.append(
                "EXISTS (SELECT 1 FROM growth_catalog_elements ce "
                "WHERE ce.record_id = r.record_id AND ce.element = ?)"
            )
            params.append(str(query["element"]))
        if query.get("growth_method"):
            clauses.append("r.growth_method = ?")
            params.append(str(query["growth_method"]))
        for reactant in query.get("reactants", []):
            roles = reactant["roles"]
            placeholders = ", ".join("?" for _ in roles)
            clauses.append(
                "EXISTS (SELECT 1 FROM growth_catalog_reactants cr "
                "WHERE cr.record_id = r.record_id AND cr.reactant_normalized = ? "
                f"AND cr.role IN ({placeholders}))"
            )
            params.append(normalize_reactant_name(reactant["name"]))
            params.extend(roles)
        return " AND ".join(clauses), params

    def _group_query(self, query: AggregateQuery, where: str) -> str:
        if query["kind"] == "element_method_distribution":
            return f"""
                SELECT r.growth_method, COUNT(*), COUNT(DISTINCT r.formula),
                       COUNT(DISTINCT NULLIF(r.doi, ''))
                FROM growth_catalog_records r
                WHERE {where}
                GROUP BY r.growth_method
                ORDER BY COUNT(*) DESC, r.growth_method ASC
            """
        return f"""
            SELECT r.formula, r.growth_method, COUNT(*), COUNT(DISTINCT NULLIF(r.doi, ''))
            FROM growth_catalog_records r
            WHERE {where}
            GROUP BY r.formula, r.growth_method
            ORDER BY COUNT(*) DESC, r.formula ASC, r.growth_method ASC
            LIMIT 30
        """

    @staticmethod
    def _group_from_row(query: AggregateQuery, row: Any) -> dict[str, Any]:
        if query["kind"] == "element_method_distribution":
            return {
                "label": str(row[0] or "未提供"),
                "growth_method": str(row[0] or "") or None,
                "record_count": int(row[1]),
                "formula_count": int(row[2]),
                "doi_count": int(row[3]),
            }
        return {
            "label": str(row[0] or "未提供"),
            "growth_method": str(row[1] or "") or None,
            "record_count": int(row[2]),
            "formula_count": 1 if row[0] else 0,
            "doi_count": int(row[3]),
        }

    @staticmethod
    def _record_from_row(row: Any, query: AggregateQuery) -> dict[str, Any]:
        try:
            precursors = json.loads(str(row[4] or "[]"))
        except json.JSONDecodeError:
            precursors = []
        matched = [f"catalog:{query['kind']}"]
        if query.get("element"):
            matched.append(f"element={query['element']}")
        if query.get("growth_method"):
            matched.append(f"growth_method={query['growth_method']}")
        matched.extend(f"reactant={item['name']}" for item in query.get("reactants", []))
        return {
            "record_id": str(row[0]),
            "score": 1.0,
            "dense_score": None,
            "sparse_score": None,
            "material_formula": str(row[1] or "") or None,
            "material_name": None,
            "growth_method": str(row[2] or "") or None,
            "temperature_program": str(row[3] or "") or None,
            "atmosphere": None,
            "precursors": precursors if isinstance(precursors, list) else [],
            "doi": str(row[5] or "") or None,
            "source_text": str(row[6] or ""),
            "source_file": "postgres-catalog",
            "matched_fields": matched,
        }


_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS growth_catalog_meta (
        catalog_key TEXT PRIMARY KEY,
        catalog_version TEXT NOT NULL,
        source_hash TEXT NOT NULL,
        record_count INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS growth_catalog_records (
        record_id TEXT PRIMARY KEY,
        formula TEXT NOT NULL,
        growth_method TEXT NOT NULL,
        temperature_program TEXT,
        precursors_json TEXT NOT NULL,
        doi TEXT NOT NULL,
        source_text TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS growth_catalog_elements (
        record_id TEXT NOT NULL,
        element TEXT NOT NULL,
        PRIMARY KEY (record_id, element)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS growth_catalog_reactants (
        record_id TEXT NOT NULL,
        reactant_normalized TEXT NOT NULL,
        role TEXT NOT NULL,
        PRIMARY KEY (record_id, reactant_normalized, role)
    )
    """,
    "CREATE INDEX IF NOT EXISTS growth_catalog_records_method_idx ON growth_catalog_records (growth_method)",
    "CREATE INDEX IF NOT EXISTS growth_catalog_elements_element_idx ON growth_catalog_elements (element, record_id)",
    "CREATE INDEX IF NOT EXISTS growth_catalog_reactants_lookup_idx ON growth_catalog_reactants (reactant_normalized, role, record_id)",
)


_default_catalog: FactCatalog | None = None


def get_default_fact_catalog(config: Settings = settings) -> FactCatalog:
    global _default_catalog
    if _default_catalog is None:
        _default_catalog = FactCatalog(config.memory_database_url)
    return _default_catalog


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
