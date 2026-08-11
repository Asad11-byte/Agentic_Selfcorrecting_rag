import json
from dataclasses import dataclass
from functools import lru_cache

from groq import Groq

from app.config import settings


@lru_cache
def get_groq() -> Groq:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return Groq(api_key=settings.groq_api_key)


def chat(messages: list[dict], model: str | None = None, temperature: float = 0.2) -> str:
    client = get_groq()
    completion = client.chat.completions.create(
        model=model or settings.groq_generation_model,
        messages=messages,
        temperature=temperature,
    )
    return (completion.choices[0].message.content or "").strip()


@dataclass
class Grade:
    decision: bool
    reason: str


def grade_yes_no(system_prompt: str, user_prompt: str) -> Grade:
    """Ask the grader model a yes/no question and parse a strict JSON response.

    Used by grade_documents and grade_answer — small, fast, cheap calls, deliberately
    on the smaller GROQ_GRADER_MODEL since grading is a simple classification task.
    """
    client = get_groq()
    completion = client.chat.completions.create(
        model=settings.groq_grader_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": f'{system_prompt}\nRespond ONLY with JSON: {{"decision": true|false, "reason": "<one short sentence>"}}',
            },
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = completion.choices[0].message.content or '{"decision": false, "reason": "no response"}'
    try:
        parsed = json.loads(raw)
        return Grade(decision=bool(parsed.get("decision")), reason=str(parsed.get("reason", "")))
    except (json.JSONDecodeError, AttributeError):
        return Grade(decision=False, reason="failed to parse grader output")
