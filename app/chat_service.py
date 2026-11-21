import re
from datetime import datetime
from typing import List, Tuple

from google import genai
from google.genai import types

from app.rag import ClinicRetriever
from config import (
    FALLBACK_GENERATION_MODEL,
    GEMINI_API_KEY,
    GENERATION_MODEL,
    MAX_HISTORY_TURNS,
    REQUEST_TIMEOUT_MS,
    SIMILARITY_THRESHOLD,
)

SYSTEM_INSTRUCTION = """
You are a helpful bilingual FAQ assistant for a medical clinic.

RULES:
1. STRICT LANGUAGE MATCH: Look ONLY at the language of the latest User Message. If the latest User Message is in English, respond strictly in English. If the latest User Message is in Arabic (MSA or Egyptian dialect), respond in that same Arabic style.
2. GROUNDING: Answer strictly using the facts in <faq_context>. Every fact in <faq_context> (working hours, location, doctors, specialties, schedules, consultation fees, follow-up policy, payment methods, insurance providers, appointment/walk-in/cancellation rules, visit preparation, facilities/parking, and medical safety) is valid clinic FAQ information. State the matching facts directly from <faq_context>.
3. MISSING INFO: Only if a topic is genuinely not mentioned anywhere in <faq_context>, say politely that it is not listed in the clinic FAQ. Never invent or include phone numbers.
4. MEDICAL SAFETY: Do not diagnose conditions or recommend medications. Advise consulting a qualified doctor or seeking emergency medical care for urgent symptoms.
5. FORMAT: Keep responses concise, clear, and well-structured with short bullet points. Do not start lines with a leading dot or colon.
"""


def _clean_reply_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^[:.\s]+", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


class ClinicChatService:
    """Orchestrates retrieval, grounding guardrails, and response generation."""

    def __init__(self, retriever: ClinicRetriever) -> None:
        self.retriever = retriever
        self.client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        )

    def _format_history(self, history: List[dict]) -> str:
        recent = history[-(MAX_HISTORY_TURNS * 2) :]
        if not recent:
            return "None"
        lines = []
        for item in recent:
            role = "User" if item.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {item.get('content', '')}")
        return "\n".join(lines)

    def _call_model(self, model_name: str, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,
                max_output_tokens=450,
            ),
        )
        return _clean_reply_text(response.text or "")

    def generate_reply(self, user_message: str, history: List[dict]) -> Tuple[str, str]:
        timestamp = datetime.now().strftime("%I:%M %p")
        debug_lines: List[str] = []

        retrieved, mode = self.retriever.retrieve(user_message)
        best_score = retrieved[0][1] if retrieved else 0.0

        debug_lines.append(f"**Retrieval Mode:** `{mode}`")
        debug_lines.append(
            f"**Top Score:** `{best_score:.3f}` (Threshold: `{SIMILARITY_THRESHOLD}`)"
        )
        for doc, score in retrieved:
            debug_lines.append(f"- `[{score:.3f}]` **{doc['category']}** (`{doc['id']}`)")

        context_block = "\n\n".join(
            [f"[{doc['category']}]\n{doc['content']}" for doc, _ in retrieved]
        )

        has_arabic = bool(re.search(r"[\u0600-\u06FF]", user_message))
        lang_hint = "Arabic" if has_arabic else "English"

        prompt = f"""
<faq_context>
{context_block}
</faq_context>

<conversation_history>
{self._format_history(history)}
</conversation_history>

Required Response Language: {lang_hint}
User Message: {user_message}
"""

        reply_text = ""
        last_error = ""
        for model_name in [GENERATION_MODEL, FALLBACK_GENERATION_MODEL, "gemini-4-hydrogen"]:
            try:
                reply_text = self._call_model(model_name, prompt)
                if reply_text:
                    debug_lines.append(f"**Model Used:** `{model_name}`")
                    break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                debug_lines.append(f"- `{model_name}` error: `{last_error}`")

        if not reply_text:
            reply_text = (
                "Could not get a response from the model right now. Please try again or open "
                "'RAG Retrieval Inspector' below for details."
            )
            if last_error:
                debug_lines.append(f"\n**Last API Error:** `{last_error}`")

        formatted_reply = f"{reply_text}\n\n<sub>\u2713\u2713 {timestamp}</sub>"
        return formatted_reply, "\n".join(debug_lines)
