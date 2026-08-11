import operator
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from app.services.embeddings import embed_one
from app.services.llm import chat, grade_yes_no
from app.services.vectorstore import RetrievedDoc, search

MAX_RETRIEVAL_RETRIES = 2
MAX_GENERATION_RETRIES = 2


@dataclass
class RagStep:
    node: str
    detail: str


@dataclass
class RagResult:
    answer: str
    sources: list[dict]
    steps: list[RagStep] = field(default_factory=list)


# --- Graph state ---------------------------------------------------------------
# `steps` uses operator.add as its reducer so every node's returned step list is
# appended to the running trace instead of overwriting it — each node only needs
# to return its own step(s), not the whole history.
class GraphState(TypedDict, total=False):
    question: str
    route: str
    query: str
    docs: list[RetrievedDoc]
    sufficient: bool
    retrieval_attempts: int
    generation_attempts: int
    answer: str
    grounded: bool
    relevant: bool
    steps: Annotated[list[RagStep], operator.add]


# --- Node: route_query -------------------------------------------------------
# The agentic decision: does this question even need retrieval?
async def route_query(state: GraphState) -> dict:
    grade = grade_yes_no(
        "You decide whether answering the user's question requires looking up private/internal "
        "documents in a knowledge base. Say true if retrieval would help (specific facts, internal "
        "docs, anything you can't be fully sure about). Say false only for pure greetings, chit-chat, "
        "or general knowledge you're fully confident about.",
        state["question"],
    )
    route = "retrieve" if grade.decision else "direct"
    return {
        "route": route,
        "steps": [RagStep("route_query", f'Routed to "{route}"')],
    }


def route_query_edge(state: GraphState) -> str:
    return state["route"]


# --- Node: direct_answer ------------------------------------------------------
async def direct_answer(state: GraphState) -> dict:
    answer = chat(
        [
            {"role": "system", "content": "Answer directly and concisely."},
            {"role": "user", "content": state["question"]},
        ]
    )
    return {
        "answer": answer,
        "docs": [],
        "steps": [RagStep("generate", "Answered directly, no retrieval needed")],
    }


# --- Node: retrieve -----------------------------------------------------------
async def retrieve(state: GraphState) -> dict:
    query = state.get("query") or state["question"]
    vector = await embed_one(query, "retrieval.query")
    docs = search(vector, top_k=5)
    attempts = state.get("retrieval_attempts", 0) + 1
    return {
        "query": query,
        "docs": docs,
        "retrieval_attempts": attempts,
        "steps": [RagStep("retrieve", f'Retrieved {len(docs)} docs for "{query}"')],
    }


# --- Node: grade_documents -----------------------------------------------------
async def grade_documents(state: GraphState) -> dict:
    question = state["question"]
    docs = state["docs"]
    relevant_docs = []
    for doc in docs:
        grade = grade_yes_no(
            "You grade whether a retrieved document is relevant to the user's question. "
            "Be strict — only true if it actually helps answer it.",
            f"Question: {question}\n\nDocument:\n{doc.text[:2000]}",
        )
        if grade.decision:
            relevant_docs.append(doc)

    sufficient = len(relevant_docs) > 0
    step = RagStep("grade_documents", f"{len(relevant_docs)}/{len(docs)} docs graded relevant")

    # Only narrow down to the relevant subset once we actually have enough to work
    # with — mirrors the original: on a forced best-effort break we keep the full
    # (unfiltered) doc list rather than an empty one.
    update: dict = {"sufficient": sufficient, "steps": [step]}
    if sufficient:
        update["docs"] = relevant_docs
    return update


def grade_documents_edge(state: GraphState) -> str:
    if state["sufficient"]:
        return "generate"
    if state["retrieval_attempts"] >= MAX_RETRIEVAL_RETRIES + 1:
        return "generate"  # retries exhausted — proceed with best effort
    return "rewrite_query_retrieval"


# --- Node: rewrite_query (shared helper) ---------------------------------------
async def _rewrite_query(question: str, previous_query: str) -> str:
    rewritten = chat(
        [
            {
                "role": "system",
                "content": (
                    "Rewrite the search query to retrieve better results from a vector database. "
                    "Fix ambiguity, add likely synonyms/keywords, keep it short. "
                    "Return ONLY the rewritten query, nothing else."
                ),
            },
            {"role": "user", "content": f"Original question: {question}\nPrevious search query: {previous_query}"},
        ],
        temperature=0.3,
    )
    return rewritten.strip("\"'")


async def rewrite_query_retrieval(state: GraphState) -> dict:
    query = await _rewrite_query(state["question"], state["query"])
    return {
        "query": query,
        "steps": [RagStep("rewrite_query", f'Rewrote query to "{query}"')],
    }


async def rewrite_query_regenerate(state: GraphState) -> dict:
    # The retrieved context itself was probably the problem — go back further and
    # re-retrieve without re-grading, exactly as the original "not relevant" branch did.
    query = await _rewrite_query(state["question"], state["query"])
    vector = await embed_one(query, "retrieval.query")
    docs = search(vector, top_k=5)
    return {
        "query": query,
        "docs": docs,
        "steps": [RagStep("rewrite_query", f'Answer off-topic, re-retrieved with "{query}"')],
    }


# --- Node: generate ---------------------------------------------------------------
async def generate(state: GraphState) -> dict:
    docs = state["docs"]
    context = "\n\n".join(f"[{i + 1}] {d.text}" for i, d in enumerate(docs))
    answer = chat(
        [
            {
                "role": "system",
                "content": (
                    "Answer the user's question using ONLY the provided context. Cite sources inline "
                    "like [1], [2]. If the context doesn't contain the answer, say so plainly instead "
                    "of guessing."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {state['question']}"},
        ]
    )
    attempts = state.get("generation_attempts", 0) + 1
    return {
        "answer": answer,
        "generation_attempts": attempts,
        "steps": [RagStep("generate", f"Generated answer (attempt {attempts})")],
    }


# --- Node: grade_answer -------------------------------------------------------------
async def grade_answer(state: GraphState) -> dict:
    docs = state["docs"]
    context = "\n\n".join(d.text for d in docs)
    grounded = grade_yes_no(
        "You check whether an answer is fully grounded in (supported by) the given context, "
        "with no hallucinated claims. Be strict.",
        f"Context:\n{context}\n\nAnswer:\n{state['answer']}",
    ).decision
    relevant = grade_yes_no(
        "You check whether an answer actually addresses the user's question.",
        f"Question: {state['question']}\n\nAnswer:\n{state['answer']}",
    ).decision
    return {
        "grounded": grounded,
        "relevant": relevant,
        "steps": [RagStep("grade_answer", f"Grounded: {grounded}, relevant: {relevant}")],
    }


def grade_answer_edge(state: GraphState) -> str:
    if state["grounded"] and state["relevant"]:
        return "done"
    if state["generation_attempts"] >= MAX_GENERATION_RETRIES + 1:
        return "exhausted"
    if not state["relevant"]:
        return "rewrite_query_regenerate"
    return "regenerate"  # not grounded, still on-topic — retry with same context


def _exhausted_step(state: GraphState) -> dict:
    return {"steps": [RagStep("grade_answer", "Generation retries exhausted, returning best effort answer")]}


# --- Build the graph ------------------------------------------------------------
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("route_query", route_query)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query_retrieval", rewrite_query_retrieval)
    graph.add_node("rewrite_query_regenerate", rewrite_query_regenerate)
    graph.add_node("generate", generate)
    graph.add_node("grade_answer", grade_answer)
    graph.add_node("exhausted", _exhausted_step)

    graph.set_entry_point("route_query")
    graph.add_conditional_edges(
        "route_query",
        route_query_edge,
        {"direct": "direct_answer", "retrieve": "retrieve"},
    )
    graph.add_edge("direct_answer", END)

    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        grade_documents_edge,
        {"generate": "generate", "rewrite_query_retrieval": "rewrite_query_retrieval"},
    )
    graph.add_edge("rewrite_query_retrieval", "retrieve")

    graph.add_edge("generate", "grade_answer")
    graph.add_conditional_edges(
        "grade_answer",
        grade_answer_edge,
        {
            "done": END,
            "exhausted": "exhausted",
            "rewrite_query_regenerate": "rewrite_query_regenerate",
            "regenerate": "generate",
        },
    )
    graph.add_edge("exhausted", END)
    graph.add_edge("rewrite_query_regenerate", "generate")

    return graph.compile()


_compiled_graph = build_graph()


# --- Orchestration entry point ---------------------------------------------------
async def run_agentic_rag(question: str) -> RagResult:
    final_state: GraphState = await _compiled_graph.ainvoke(
        {"question": question, "steps": []}
    )

    docs = final_state.get("docs", [])
    return RagResult(
        answer=final_state.get("answer", ""),
        sources=[{"text": d.text[:300], "source": d.source} for d in docs],
        steps=final_state.get("steps", []),
    )
