"""Atlas tools: semantic search across Praxis Atlas experiments.

Reads the sidecar `data/praxis_meta.db` (NOT `crypto_data.db`). The sidecar is
populated by `python -m engines.atlas_sync` from the canonical Atlas markdown
files (`TRADING_ATLAS.md`, `PREDICTION_MARKET_ATLAS.md`, `docs/REGIME_MATRIX.md`).

Both tools are read-only.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from servers.praxis_mcp.db import connect_ro

# Load .env so VOYAGE_API_KEY / OPENAI_API_KEY are available at query time
# regardless of how Claude Desktop launches the server subprocess.
_REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_REPO_ROOT / ".env")


def register(mcp, db_path: Path):
    """Register atlas tools.

    Args:
        mcp: FastMCP instance
        db_path: path to praxis_meta.db (NOT crypto_data.db)
    """

    @mcp.tool()
    def atlas_search(query: str, top_k: int = 5) -> dict:
        """Semantic similarity search across Praxis Atlas experiments.

        Embeds `query` using the same provider that was used at sync time and
        returns the top_k most similar experiments by cosine similarity. Use
        this when triaging a new trading idea against accumulated experimental
        evidence: e.g. atlas_search("BTC mean reversion at 1m timescale") will
        find the SP500 pairs MR experiment even though the words don't overlap.

        For full detail on a specific entry, call atlas_get(entry_id).

        Args:
            query: natural-language description of the strategy / idea / topic.
            top_k: number of results to return (default 5).

        Returns:
            Dict with:
              query: input echo
              model: embedding model used
              results: list ordered by descending similarity, each containing
                {id, source_file, source_section, signal_type, asset_class,
                 result_class, result_summary, similarity_score}
        """
        try:
            if not db_path.exists():
                return {
                    "error": (
                        f"Atlas DB not found at {db_path}. Run "
                        "`python -m engines.atlas_sync` to create it."
                    )
                }

            conn = connect_ro(db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT DISTINCT embedding_model, embedding_dim "
                "FROM atlas_embeddings"
            )
            rows = cursor.fetchall()
            if not rows:
                conn.close()
                return {
                    "error": (
                        "No embeddings present in the Atlas DB. Re-run "
                        "`python -m engines.atlas_sync` with VOYAGE_API_KEY "
                        "or OPENAI_API_KEY set."
                    )
                }
            if len(rows) > 1:
                models = sorted({r["embedding_model"] for r in rows})
                conn.close()
                return {
                    "error": (
                        "Atlas DB has embeddings from multiple models "
                        f"({models}); cannot mix at query time. Re-sync to "
                        "regenerate all under one model."
                    )
                }

            stored_model = rows[0]["embedding_model"]
            stored_dim = rows[0]["embedding_dim"]

            # Verify the matching API key is available at query time
            from engines.atlas_sync import (
                VOYAGE_MODEL,
                OPENAI_MODEL,
                embed_query,
            )

            if stored_model == VOYAGE_MODEL:
                if not os.getenv("VOYAGE_API_KEY"):
                    conn.close()
                    return {
                        "error": (
                            "Atlas DB was synced with Voyage embeddings but "
                            "VOYAGE_API_KEY is not set in the server's "
                            "environment. Add it to .env and relaunch Claude "
                            "Desktop."
                        )
                    }
                provider = "voyage"
            elif stored_model == OPENAI_MODEL:
                if not os.getenv("OPENAI_API_KEY"):
                    conn.close()
                    return {
                        "error": (
                            "Atlas DB was synced with OpenAI embeddings but "
                            "OPENAI_API_KEY is not set."
                        )
                    }
                provider = "openai"
            else:
                conn.close()
                return {"error": f"Unknown embedding model in DB: {stored_model}"}

            qvec = embed_query(provider, stored_model, query)
            if qvec.shape[0] != stored_dim:
                conn.close()
                return {
                    "error": (
                        f"Query embedding dim {qvec.shape[0]} does not match "
                        f"stored dim {stored_dim} for model {stored_model}."
                    )
                }
            qnorm = float(np.linalg.norm(qvec))
            if qnorm > 0:
                qvec = qvec / qnorm

            cursor.execute(
                """
                SELECT e.experiment_id, e.embedding,
                       x.source_file, x.source_section, x.signal_type,
                       x.asset_class, x.result_class, x.result_summary
                FROM atlas_embeddings e
                JOIN atlas_experiments x ON x.id = e.experiment_id
                """
            )

            scored = []
            for row in cursor.fetchall():
                vec = np.frombuffer(row["embedding"], dtype=np.float32)
                # Stored vectors are pre-normalized; cosine == dot product
                score = float(np.dot(vec, qvec))
                scored.append(
                    {
                        "id": row["experiment_id"],
                        "source_file": row["source_file"],
                        "source_section": row["source_section"],
                        "signal_type": row["signal_type"],
                        "asset_class": row["asset_class"],
                        "result_class": row["result_class"],
                        "result_summary": row["result_summary"],
                        "similarity_score": round(score, 4),
                    }
                )

            scored.sort(key=lambda r: r["similarity_score"], reverse=True)
            top = scored[: max(1, min(top_k, 50))]

            conn.close()
            return {"query": query, "model": stored_model, "results": top}

        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def atlas_get(entry_id: int) -> dict:
        """Retrieve full details for a single Atlas entry by id.

        Returns the complete parsed structure plus the original markdown,
        plus a citation pointer (source_file:line_start-line_end) suitable
        for human verification against the canonical markdown.

        Args:
            entry_id: integer id from atlas_search results.

        Returns:
            Dict with all fields from atlas_experiments plus a `citation`
            field formatted as 'TRADING_ATLAS.md:lines 602-749'.
        """
        try:
            if not db_path.exists():
                return {
                    "error": (
                        f"Atlas DB not found at {db_path}. Run "
                        "`python -m engines.atlas_sync` first."
                    )
                }
            conn = connect_ro(db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, source_file, source_section, source_line_start,
                       source_line_end, signal_type, asset_class, framework,
                       date_run, result_class, result_summary, full_markdown,
                       key_findings, atlas_principle, md_hash, synced_at
                FROM atlas_experiments WHERE id = ?
                """,
                (entry_id,),
            )
            row = cursor.fetchone()
            conn.close()
            if row is None:
                return {"error": f"No atlas entry with id={entry_id}"}

            result = {k: row[k] for k in row.keys()}
            result["citation"] = (
                f"{row['source_file']}:lines "
                f"{row['source_line_start']}-{row['source_line_end']}"
            )
            return result
        except Exception as e:
            return {"error": str(e)}
