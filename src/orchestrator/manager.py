"""
Orchestrator Manager -- Edu Nexus  (Phase 5 — Workspace-Scoped Tri-Hybrid)
===========================================================================
The central nervous system of Edu Nexus.  An **LLM Router**
analyses each query and *decides* which retrieval brain(s) to invoke:

  1. Fast Brain   — BM25 keyword search   (src.retrieval.bm25_index)
  2. Deep Brain   — NetworkX knowledge graph  (src.graph_engine)
  3. Semantic Brain — Qdrant vector search  (src.vector_engine.store)

Only the chosen brain(s) execute.  Results are fused into a
``context_block`` and sent to the Answer LLM for final synthesis.

All retrieval is workspace-scoped via workspace_id parameter.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from groq import Groq

from src.retrieval import bm25_index
from src.graph_engine import neo4j_ops
from src.vector_engine import store
from src.vector_engine import vector as vec

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("Orchestrator")

# ── LLM config ────────────────────────────────────────────────────────
# Ordered fallback lists — if first model is rate-limited, try next
ROUTER_MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
ANSWER_MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]

# NVIDIA API fallback (final resort when all Groq models are rate-limited)
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_MODEL = "qwen/qwen3.5-122b-a10b"

BM25_TOP_K = 3
VECTOR_TOP_K = 5
GRAPH_RESULT_LIMIT = 8
MAX_CHUNK_WORDS = 400
MAX_CONTEXT_CHARS = 18000

# ── Router System Prompt ──────────────────────────────────────────────
ROUTER_PROMPT = (
    "You are the Strategy Router for a Tri-Hybrid RAG system.\n"
    "Your job is to analyze the user's query and decide exactly which "
    "retrieval brain(s) should be invoked to answer it.\n\n"
    "Available brains:\n"
    "1. **keyword** — BM25 lexical search. Best for: exact term lookups, "
    "definitions, specific names, acronyms, keyword-heavy factual questions.\n"
    "2. **graph** — Knowledge graph traversal. Best for: relationship "
    "questions (\"how does X relate to Y?\"), entity connections, taxonomy, "
    "cause-effect chains, structural/hierarchical queries.\n"
    "3. **semantic** — Dense vector similarity. Best for: conceptual "
    "questions, paraphrased queries, thematic exploration, \"explain\" or "
    "\"describe\" style questions, broad topic summaries.\n\n"
    "Rules:\n"
    "- Pick 1-3 brains.  Fewer is better — only select what the query needs.\n"
    "- For a simple keyword lookup, just pick 'keyword'.\n"
    "- For relationship queries, always include 'graph'.\n"
    "- For broad/conceptual questions, include 'semantic'.\n"
    "- If unsure, pick 'keyword' + 'semantic' (the safest combo).\n\n"
    "Respond with ONLY a JSON object, nothing else:\n"
    '{"brains": ["keyword", "graph", "semantic"], "reasoning": "short explanation"}\n'
)

# ── Answer System Prompt ──────────────────────────────────────────────
ANSWER_PROMPT = (
    "You are **Edu Nexus**, an elite academic assistant powered by a "
    "Tri-Hybrid Retrieval-Augmented Generation engine.\n\n"
    "You will receive a `context_block` containing evidence from the "
    "retrieval systems that the orchestrator chose for this query.\n\n"
    "── STRICT RULES ──\n"
    "- You MUST answer using ONLY the information present in the "
    "context_block.  Do NOT use prior knowledge.\n"
    "- Cross-reference different sources to verify facts.\n"
    "- If no relevant context is provided, respond: "
    '"I don\'t have enough information in my knowledge base '
    'to answer this."\n'
    "- NEVER hallucinate facts, names, dates, or relationships.\n\n"
    "── FORMAT ──\n"
    "- Use clean Markdown for readability.\n"
    "- Use ## headers for major sections.\n"
    "- Use **bold** for key terms and concepts.\n"
    "- Use bullet points (- ) for lists of items.\n"
    "- Use numbered lists (1. ) for sequential steps or processes.\n"
    "- Be concise yet thorough. Write in a conversational but academic tone.\n"
    "- Cite the source document for each claim using the format "
    "[Source: document_name] where document_name is the filename "
    "shown in the context block headers.\n"
    "- If a context section is labeled with a document name, use that "
    "name in your citation. Do NOT use generic labels like "
    "'Chunk 1' or 'Semantic Chunk'.\n"
)

# ── Single-doc prompt (no citations needed, saves tokens) ────────────
ANSWER_PROMPT_SINGLE_DOC = (
    "You are **Edu Nexus**, an elite academic assistant.\n\n"
    "The user is viewing a SINGLE document and asking about it. "
    "All context comes from that one document — do NOT cite sources, "
    "do NOT add [Source: ...] tags. Just answer directly.\n\n"
    "── RULES ──\n"
    "- Answer using ONLY the context provided. No prior knowledge.\n"
    "- NEVER hallucinate facts.\n"
    "- Use clean Markdown: ## headers, **bold** key terms, bullet points.\n"
    "- Be concise yet thorough. Conversational but academic.\n"
)


# ====================================================================== #
#  ORCHESTRATOR                                                           #
# ====================================================================== #

class OrchestratorManager:
    """
    LLM-Routed Tri-Hybrid RAG Orchestrator.

    All retrieval is workspace-scoped via workspace_id parameter.
    The Router LLM analyzes each query and decides which brain(s) to invoke.
    Only the selected brains execute.

    Public API used by ``server.py``:
        - ``generate_answer(query, workspace_id, ...)`` — full RAG pipeline → dict
        - ``get_response(query, workspace_id)``   — alias
    """

    def __init__(self) -> None:
        # ── Groq LLM clients (Router + Answer) ───────────────────
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment.")
        self.llm = Groq(api_key=api_key)

        logger.info("Orchestrator initialized (Qdrant + NetworkX + BM25).")

    # ------------------------------------------------------------------ #
    #  LLM call with automatic fallback                                   #
    # ------------------------------------------------------------------ #

    async def _llm_call_with_fallback(
        self,
        models: List[str],
        messages: List[Dict],
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        """
        Try each Groq model in order. If all fail (rate limit, error),
        fall back to the NVIDIA API as last resort.
        Returns the raw response text.
        """
        last_error = None

        for model in models:
            try:
                completion = await asyncio.to_thread(
                    lambda m=model: self.llm.chat.completions.create(
                        model=m,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False,
                    )
                )
                text = completion.choices[0].message.content or ""
                logger.info(f"LLM call succeeded with model: {model}")
                return text.strip()
            except Exception as e:
                last_error = e
                logger.warning(f"Model {model} failed ({e}), trying next...")
                continue

        # NVIDIA API fallback
        try:
            logger.info("All Groq models failed. Falling back to NVIDIA API...")
            payload = {
                "model": NVIDIA_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.95,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            headers = {
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Accept": "application/json",
            }
            resp = await asyncio.to_thread(
                lambda: requests.post(
                    NVIDIA_API_URL, headers=headers, json=payload, timeout=60
                )
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"] or ""
            logger.info(f"NVIDIA API fallback succeeded with {NVIDIA_MODEL}")
            return text.strip()
        except Exception as nvidia_err:
            logger.error(f"NVIDIA API fallback also failed: {nvidia_err}")

        raise RuntimeError(
            f"All LLM models failed. Last Groq error: {last_error}"
        )

    # ================================================================== #
    #  LLM ROUTER — decides which brain(s) to invoke                     #
    # ================================================================== #

    async def _route_query(self, query: str, workspace_id: str) -> dict:
        """
        Call the Router LLM to decide which brain(s) to use.

        Returns
        -------
        dict with keys: brains (list[str]), reasoning (str)
        """
        # Build available list based on what's ready
        available = ["keyword", "semantic"]  # always available with Qdrant + BM25

        # Graph available if workspace has a graph
        if neo4j_ops.workspace_graph_exists(workspace_id):
            available.append("graph")

        user_msg = (
            f"Available brains: {available}\n\n"
            f"User query: \"{query}\"\n\n"
            f"Which brain(s) should handle this query? "
            f"Respond with ONLY the JSON object, no markdown, no explanation."
        )

        try:
            raw = await self._llm_call_with_fallback(
                models=ROUTER_MODELS,
                messages=[
                    {"role": "system", "content": ROUTER_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                max_tokens=200,
            )
            logger.info(f"Router LLM raw response: {raw}")

            data = self._parse_router_json(raw, available)
            chosen = [b for b in data.get("brains", []) if b in available]
            reasoning = data.get("reasoning", "No reasoning provided.")

            if not chosen:
                chosen = available
                reasoning += " (Fallback: using all available brains.)"

            return {"brains": chosen, "reasoning": reasoning}

        except Exception as e:
            logger.error(f"Router LLM failed: {e}. Falling back to all brains.")
            return {
                "brains": available,
                "reasoning": f"Router error ({e}). Using all available brains as fallback.",
            }

    @staticmethod
    def _parse_router_json(raw: str, available: List[str]) -> dict:
        """Robustly parse the Router LLM response into a dict."""
        if not raw:
            return {"brains": available, "reasoning": "Empty router response."}

        # 1. Try direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # 2. Strip markdown code fences
        cleaned = re.sub(r'```(?:json)?\s*', '', raw)
        cleaned = cleaned.strip().rstrip('`').strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 3. Find first { ... } block
        match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # 4. Find { ... } allowing nested braces
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # 5. Last resort — look for brain names in text
        found = []
        for brain in ["keyword", "graph", "semantic"]:
            if brain in raw.lower():
                found.append(brain)

        if found:
            return {"brains": found, "reasoning": f"Parsed from text: {raw[:100]}"}

        logger.warning(f"Could not parse router response at all: {raw}")
        return {"brains": available, "reasoning": "Unparseable response. Fallback to all."}

    # ================================================================== #
    #  RETRIEVAL — one method per brain (workspace-scoped)                #
    # ================================================================== #

    def _retrieve_bm25(self, query: str, workspace_id: str) -> List[dict]:
        """Fast Brain: keyword-matched chunks."""
        try:
            results = bm25_index.search(workspace_id, query, top_k=BM25_TOP_K)
            logger.info(f"Fast Brain returned {len(results)} chunks.")
            return results
        except Exception as e:
            logger.error(f"Fast Brain search failed: {e}")
            return []

    def _retrieve_graph(self, query: str, workspace_id: str) -> List[Dict]:
        """Deep Brain: knowledge-graph related nodes matching query keywords."""
        stopwords = {
            "what", "is", "the", "a", "an", "of", "to", "and", "in",
            "for", "on", "how", "does", "do", "are", "was", "were",
            "who", "which", "can", "about", "tell", "me", "explain",
            "this", "that", "it", "they", "them", "their", "its",
        }
        keywords = [
            w.lower() for w in query.split()
            if w.lower() not in stopwords and len(w) > 1
        ]
        if not keywords:
            return []

        try:
            results = neo4j_ops.search_graph(workspace_id, keywords)
            logger.info(f"Deep Brain returned {len(results)} graph nodes.")
            return results
        except Exception as e:
            logger.error(f"Deep Brain query failed: {e}")
            return []

    def _retrieve_vector(self, query: str, workspace_id: str) -> List[dict]:
        """Semantic Brain: dense-vector similar chunks."""
        try:
            query_vec = vec.embed_query(query)
            results = store.search(workspace_id, query_vec, top_k=VECTOR_TOP_K)
            logger.info(f"Semantic Brain returned {len(results)} chunks.")
            return results
        except Exception as e:
            logger.error(f"Semantic Brain search failed: {e}")
            return []

    # ================================================================== #
    #  CONTEXT FORMATTING                                                 #
    # ================================================================== #

    @staticmethod
    def _truncate(text: str, max_words: int = MAX_CHUNK_WORDS) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + " ..."

    def _format_keyword_context(self, chunks: List[dict]) -> str:
        if not chunks:
            return "_No keyword chunks retrieved._"
        lines = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "") if isinstance(chunk, dict) else chunk
            label = f"[Keyword Result {i}]"
            lines.append(f"**{label}**\n{self._truncate(text)}")
        return "\n\n".join(lines)

    @staticmethod
    def _format_graph_context(graph_nodes: List[Dict]) -> str:
        if not graph_nodes:
            return "_No graph nodes retrieved._"
        lines = []
        for node in graph_nodes:
            nid = node.get("id", "?")
            label = node.get("label", "entity")
            freq = node.get("frequency", 0)
            lines.append(f"- **{nid}** (type: _{label}_, frequency: {freq})")
        return "\n".join(lines)

    def _format_semantic_context(self, results: List[dict]) -> str:
        if not results:
            return "_No semantic chunks retrieved._"
        lines = []
        for i, r in enumerate(results, 1):
            text = r.get("text", "")
            score = r.get("score", 0.0)
            doc_id = r.get("doc_id", "unknown")
            label = f"[Source: {doc_id}]"
            lines.append(
                f"**{label}** (relevance: {score:.2f})\n{self._truncate(text)}"
            )
        return "\n\n".join(lines)

    def _build_context_block(
        self,
        bm25_chunks: List[dict],
        graph_nodes: List[Dict],
        vector_results: List[dict],
        chosen_brains: List[str],
    ) -> str:
        sections = []

        if "keyword" in chosen_brains:
            keyword_ctx = self._format_keyword_context(bm25_chunks)
            sections.append(
                f"## [Keyword Context]  (Fast Brain -- BM25)\n\n{keyword_ctx}"
            )

        if "graph" in chosen_brains:
            graph_ctx = self._format_graph_context(graph_nodes)
            sections.append(
                f"## [Graph Context]  (Deep Brain -- Knowledge Graph)\n\n{graph_ctx}"
            )

        if "semantic" in chosen_brains:
            semantic_ctx = self._format_semantic_context(vector_results)
            sections.append(
                f"## [Semantic Context]  (Semantic Brain -- Qdrant)\n\n{semantic_ctx}"
            )

        block = "\n\n---\n\n".join(sections)

        if len(block) > MAX_CONTEXT_CHARS:
            logger.warning(
                f"Context block too large ({len(block)} chars), "
                f"truncating to {MAX_CONTEXT_CHARS} chars."
            )
            block = block[:MAX_CONTEXT_CHARS] + "\n\n... [context truncated]"

        return block

    # ================================================================== #
    #  MULTI-FILE QUERY DETECTION                                         #
    # ================================================================== #

    @staticmethod
    def _is_multi_file_query(query: str) -> bool:
        q = query.lower()
        multi_markers = [
            "both file", "all file", "all document", "every file",
            "every document", "each file", "each document",
            "these file", "these document", "the files", "the documents",
            "compare", "contrast", "differences between",
            "similarities between", "overview of everything",
            "summarize everything", "summarise everything",
            "all of them", "both of them",
            "across file", "across document",
        ]
        return any(marker in q for marker in multi_markers)

    # ================================================================== #
    #  MAIN ANSWER PIPELINE                                               #
    # ================================================================== #

    async def generate_answer(
        self,
        query: str,
        workspace_id: str = "default",
        source_filter: Optional[List[str]] = None,
        single_doc: bool = False,
    ) -> dict:
        """
        LLM-Routed Tri-Hybrid RAG pipeline:

          1. Router LLM analyses the query and picks brain(s).
          2. Only the chosen brains execute (concurrently).
          3. Results are fused into a ``context_block``.
          4. Answer LLM synthesises the final answer.
          5. Return a structured dict.

        Parameters
        ----------
        workspace_id : the workspace to search in
        source_filter : list of filenames to restrict retrieval to
        single_doc : if True, use the shorter prompt (no citations)
        """
        logger.info(f"Query received: {query} (workspace={workspace_id})")

        # ── Step 1: Router LLM decides which brain(s) ─────────────────
        router_decision = await self._route_query(query, workspace_id)
        chosen_brains = router_decision["brains"]
        routing_reasoning = router_decision["reasoning"]

        # If multi-file query, force all brains
        doc_count = store.count_docs(workspace_id)
        if doc_count > 1 and self._is_multi_file_query(query):
            available = ["keyword", "semantic"]
            if neo4j_ops.workspace_graph_exists(workspace_id):
                available.append("graph")
            chosen_brains = available
            routing_reasoning += (
                " [Override: multi-file query detected — "
                "using all available brains for full coverage.]"
            )
            router_decision["brains"] = chosen_brains
            router_decision["reasoning"] = routing_reasoning

        logger.info(f"Router chose: {chosen_brains} — {routing_reasoning}")

        # ── Step 2: Run ONLY chosen brains concurrently ───────────────
        tasks = {}
        if "keyword" in chosen_brains:
            tasks["bm25"] = asyncio.to_thread(self._retrieve_bm25, query, workspace_id)
        if "graph" in chosen_brains:
            tasks["graph"] = asyncio.to_thread(self._retrieve_graph, query, workspace_id)
        if "semantic" in chosen_brains:
            tasks["vector"] = asyncio.to_thread(self._retrieve_vector, query, workspace_id)

        results = {}
        if tasks:
            keys = list(tasks.keys())
            values = await asyncio.gather(*tasks.values())
            results = dict(zip(keys, values))

        bm25_chunks = results.get("bm25", [])
        graph_nodes = results.get("graph", [])
        vector_results = results.get("vector", [])

        # ── Step 2.5: Filter by source if workspace-scoped ────────────
        if source_filter:
            vector_results = [
                r for r in vector_results
                if r.get("doc_id", "") in source_filter
            ]

        # ── Step 3: Build context block ───────────────────────────────
        context_block = self._build_context_block(
            bm25_chunks, graph_nodes, vector_results, chosen_brains
        )

        # ── Step 4: Answer LLM synthesis ──────────────────────────────
        has_context = bool(bm25_chunks or graph_nodes or vector_results)

        if not has_context:
            answer_text = (
                "I don't have any knowledge base loaded yet. "
                "Please upload a document first using the attachment button."
            )
        else:
            prompt = ANSWER_PROMPT_SINGLE_DOC if single_doc else ANSWER_PROMPT
            messages = [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"### Context Block\n\n{context_block}\n\n"
                        f"---\n\n### Question\n\n{query}"
                    ),
                },
            ]
            try:
                answer_text = await self._llm_call_with_fallback(
                    models=ANSWER_MODELS,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1024,
                )
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                answer_text = (
                    "I encountered an error while generating the answer. "
                    "Please try again."
                )

        return {
            "answer": answer_text,
            "bm25_chunks": bm25_chunks,
            "graph_triples": graph_nodes,
            "vector_results": vector_results,
            "strategy": ",".join(chosen_brains),
            "router_decision": router_decision,
            "chosen_brains": chosen_brains,
        }

    # Backward-compatible alias
    async def get_response(self, query: str, workspace_id: str = "default") -> dict:
        return await self.generate_answer(query, workspace_id=workspace_id)

    # ── Stateless context mode (for deployed/IndexedDB clients) ──────
    async def answer_with_context(self, query: str, context_block: str) -> dict:
        """
        Answer a query using a pre-built context block (no retrieval).
        Used by the /api/query-with-context endpoint where chunks come
        from the client's IndexedDB, not from server-side indexes.
        """
        if not context_block.strip():
            return {
                "answer": "No context was provided. Please upload documents first.",
                "engine_used": "none",
                "chosen_brains": [],
                "sources": [],
                "confidence": 0.0,
                "chain_of_thought": [],
                "router_reasoning": "No context provided.",
            }

        messages = [
            {"role": "system", "content": ANSWER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"### Context Block\n\n{context_block}\n\n"
                    f"---\n\n### Question\n\n{query}"
                ),
            },
        ]

        try:
            answer_text = await self._llm_call_with_fallback(
                models=ANSWER_MODELS,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )
        except Exception as e:
            logger.error(f"answer_with_context LLM failed: {e}")
            answer_text = (
                "I encountered an error while generating the answer. "
                "Please try again."
            )

        return {
            "answer": answer_text,
            "engine_used": "context-provided",
            "chosen_brains": ["client-context"],
            "sources": [],
            "confidence": 0.85,
            "chain_of_thought": [
                {"step": "Context", "detail": "Using client-provided chunks", "status": "done"}
            ],
            "router_reasoning": "Context provided by client (IndexedDB mode).",
        }
