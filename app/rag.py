import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from google import genai
from google.genai import types

from config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    REQUEST_TIMEOUT_MS,
    TOP_K,
)

DEFAULT_CLINIC_FAQS: List[dict] = [
    {
        "id": "hours_location",
        "category": "Working Hours & Location",
        "content": (
            "Location: 42 Tahrir Street, Dokki, Giza. Working hours: Saturday to Thursday from 10:00 AM to 10:00 PM; closed on Fridays. "
            "الموقع ومكان العيادة والعنوان: 42 شارع التحرير، الدقي، الجيزة. "
            "مواعيد العمل ومواعيد العيادة وساعات العمل: من السبت إلى الخميس من الساعة 10 صباحاً إلى الساعة 10 مساءً، والجمعة إجازة."
        ),
    },
    {
        "id": "specialties_doctors",
        "category": "Specialties & Doctors Schedules",
        "content": (
            "Available specialties and doctors schedules: "
            "1) Internal Medicine & Cardiology (الباطنة والقلب): Dr. Tariq Mansour (Sun & Tue, 4:00 PM - 9:00 PM). "
            "2) Dermatology & Laser (الجلدية والليزر): Dr. Nadia El-Sayed (Sat, Mon & Wed, 12:00 PM - 6:00 PM). "
            "3) Pediatrics (طب الأطفال): Dr. Karim Adel (Sat-Thu, 11:00 AM - 5:00 PM). "
            "4) Dentistry (طب الأسنان): Dr. Salma Hamed (Sun, Tue & Thu, 2:00 PM - 8:00 PM). "
            "5) Orthopedics (العظام): Dr. Hany Samir (Mon & Wed, 5:00 PM - 9:30 PM). "
            "6) ENT (الأنف والأذن والحنجرة): Dr. Mona Kamel (Sat & Thu, 1:00 PM - 6:00 PM). "
            "التخصصات والأطباء والمواعيد: الباطنة والقلب د. طارق منصور، الجلدية والليزر د. نادية السيد، "
            "طب الأطفال د. كريم عادل، طب الأسنان د. سلمى حامد، العظام د. هاني سمير، الأنف والأذن د. منى كامل."
        ),
    },
    {
        "id": "consultation_fees_followup",
        "category": "Consultation Fees & Follow-Up",
        "content": (
            "Standard consultation fee (سعر الكشف) is 400 EGP. "
            "Follow-up visit (الاستشارة) within 14 days is free (مجانية خلال 14 يوماً). "
            "Dental checkup and basic cleaning start at 600 EGP. "
            "سعر الكشف 400 جنيه مصري، والاستشارة خلال 14 يوماً مجانية، وتنظيف الأسنان يبدأ من 600 جنيه."
        ),
    },
    {
        "id": "payment_methods",
        "category": "Payment Methods",
        "content": (
            "Accepted payment methods: Cash (كاش), Visa/Mastercard (فيزا / ماستر كارد), "
            "InstaPay (إنستاباي), and Vodafone Cash (فودافون كاش). "
            "طرق الدفع وطرق السداد المتاحة: كاش، فيزا أو ماستر كارد، إنستاباي، وفودافون كاش."
        ),
    },
    {
        "id": "insurance_coverage",
        "category": "Medical Insurance",
        "content": (
            "Accepted medical insurance providers: AXA, MetLife, Bupa, Allianz, and GlobeMed. "
            "Bring a physical or digital insurance card and national ID. "
            "نقبل التأمين الطبي وشركات التأمين: أكسا (AXA)، ميتلايف (MetLife)، بوبا (Bupa)، أليانز (Allianz)، وجلوب ميد (GlobeMed)."
        ),
    },
    {
        "id": "appointments_walkins_cancellation",
        "category": "Appointments, Walk-Ins & Cancellation",
        "content": (
            "Pre-booked appointments are recommended and have priority, while walk-ins (بدون موعد مسبق) are accepted based on availability. "
            "Free cancellation or rescheduling is available up to 3 hours before the appointment. "
            "الحجز والمواعيد وحجز ميعاد: يُفضل الحجز المسبق وله الأولوية، وتتوفر الكشوفات بدون موعد مسبق حسب الإتاحة، ويمكن الإلغاء أو التعديل قبل الموعد بـ 3 ساعات على الأقل."
        ),
    },
    {
        "id": "visit_preparation_records",
        "category": "Visit Preparation & Medical Records",
        "content": (
            "Bring prior medical reports, lab tests, scans, and current medications or allergies. "
            "Fasting blood tests generally require 8-12 hours of fasting (water allowed) unless instructed otherwise. "
            "التحضير للزيارة والسجلات الطبية: يُرجى إحضار التقارير والتحاليل والأشعات السابقة وقائمة الأدوية، وصيام 8-12 ساعة للتحاليل التي تتطلب صياماً."
        ),
    },
    {
        "id": "facilities_accessibility_parking",
        "category": "Facilities, Accessibility & Parking",
        "content": (
            "Elevator access and wheelchair-friendly entry are available, with limited street/nearby parking. "
            "المرافق والوصول ومواقف السيارات: يتوفر مصعد ومدخل مناسب للكراسي المتحركة مع أماكن انتظار محدودة في الشارع أو بالقرب من العيادة."
        ),
    },
    {
        "id": "medical_safety_emergencies",
        "category": "Medical Safety & Emergencies",
        "content": (
            "General clinic FAQ information only; no medical diagnosis or medication recommendations. "
            "For urgent or severe symptoms (such as chest pain, severe bleeding, or breathing difficulty), seek emergency care immediately. "
            "السلامة الطبية والطوارئ: معلومات عامة فقط دون تشخيص أو وصف أدوية، وفي الحالات الطارئة يجب التوجه للطوارئ فوراً."
        ),
    },
]

ARABIC_SYNONYMS = {
    "مواعيد": {"مواعيد", "ميعاد", "ساعات", "العمل", "العياده", "العيادة", "وقت", "اوقات"},
    "مكان": {"مكان", "المكان", "عنوان", "العنوان", "موقع", "الموقع", "العياده", "العيادة", "فين"},
    "دكاتره": {"دكاتره", "دكاترة", "دكتور", "اطباء", "أطباء", "تخصصات", "التخصصات", "تخصص"},
    "سعر": {"سعر", "السعر", "كشف", "الكشف", "رسوم", "استشاره", "استشارة"},
    "تامين": {"تامين", "تأمين", "التامين", "التأمين", "اكسا", "axa"},
    "حجز": {"حجز", "احجز", "ميعاد", "موعد", "الغاء", "إلغاء", "تعديل"},
    "دفع": {"دفع", "الدفع", "كاش", "فيزا", "انستاباي", "فودافون"},
}


def _normalize_arabic(text: str) -> str:
    s = text.lower()
    s = re.sub(r"[إأآا]", "ا", s)
    s = re.sub(r"[ىي]", "ي", s)
    s = re.sub(r"ة", "ه", s)
    s = re.sub(r"[\u064B-\u065F\u0670]", "", s)
    return s


def _tokenize(text: str) -> set[str]:
    norm = _normalize_arabic(text)
    raw = set(re.findall(r"[\w\u0600-\u06FF]+", norm))
    expanded = set(raw)
    for tok in list(raw):
        for _, group in ARABIC_SYNONYMS.items():
            norm_group = {_normalize_arabic(w) for w in group}
            if tok in norm_group:
                expanded |= norm_group
    return {t for t in expanded if len(t) > 1}


def _load_documents(kb_path: Path) -> List[dict]:
    if kb_path.exists():
        try:
            with open(kb_path, "r", encoding="utf-8-sig") as f:
                loaded = json.load(f)
            if isinstance(loaded, list) and len(loaded) > 0:
                existing_ids = {str(d.get("id", "")) for d in loaded}
                merged = list(loaded)
                for d in DEFAULT_CLINIC_FAQS:
                    if d["id"] not in existing_ids:
                        merged.append(d)
                return merged
        except Exception:
            pass
    return list(DEFAULT_CLINIC_FAQS)


class ClinicRetriever:
    """Semantic retriever with synonym-aware lexical fallback."""

    def __init__(self, kb_path: Path) -> None:
        self.client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        )
        self.documents: List[dict] = _load_documents(kb_path)
        self._doc_tokens = [
            _tokenize(f"{doc.get('id', '')} {doc.get('category', '')} {doc.get('content', '')}")
            for doc in self.documents
        ]
        self.embeddings: np.ndarray | None = None
        self.active_embedding_model: str | None = None
        self.index_error: str | None = None
        self._try_build_index()

    def _embed_texts(self, model_name: str, texts: List[str], task_type: str) -> np.ndarray:
        response = self.client.models.embed_content(
            model=model_name,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )
        vectors = np.array([item.values for item in response.embeddings], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-10, None)

    def _try_build_index(self) -> None:
        corpus = [
            f"{doc.get('id', '')} - {doc.get('category', '')}: {doc.get('content', '')}"
            for doc in self.documents
        ]
        candidates = [EMBEDDING_MODEL, "gemini-embedding-001", "text-embedding-004"]
        seen = set()
        for model_name in candidates:
            if not model_name or model_name in seen:
                continue
            seen.add(model_name)
            try:
                self.embeddings = self._embed_texts(model_name, corpus, task_type="RETRIEVAL_DOCUMENT")
                self.active_embedding_model = model_name
                self.index_error = None
                return
            except Exception as exc:
                self.index_error = f"{model_name}: {type(exc).__name__}: {exc}"

    def _boost_scores(self, query: str, base_scores: np.ndarray) -> np.ndarray:
        q_norm = _normalize_arabic(query.strip())
        q_tokens = _tokenize(query)
        boosted = base_scores.copy()
        for idx, (doc, doc_toks) in enumerate(zip(self.documents, self._doc_tokens)):
            doc_id = _normalize_arabic(str(doc.get("id", "")))
            doc_cat = _normalize_arabic(str(doc.get("category", "")))
            if q_norm and (q_norm == doc_id or q_norm == doc_cat):
                boosted[idx] = max(boosted[idx], 0.99)
            elif q_tokens and (q_tokens & doc_toks):
                overlap_ratio = len(q_tokens & doc_toks) / max(1, len(q_tokens))
                boosted[idx] = max(
                    boosted[idx],
                    min(0.95, float(boosted[idx]) + 0.45 * overlap_ratio),
                )
        return boosted

    def _lexical_retrieve(self, query: str, top_k: int) -> List[Tuple[dict, float]]:
        base = np.zeros(len(self.documents), dtype=np.float32)
        q_tokens = _tokenize(query)
        for idx, doc_toks in enumerate(self._doc_tokens):
            if q_tokens and doc_toks:
                base[idx] = min(1.0, len(q_tokens & doc_toks) / max(1, len(q_tokens)))
        scores = self._boost_scores(query, base)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[int(idx)], float(scores[int(idx)])) for idx in top_indices]

    def retrieve(self, query: str, top_k: int = TOP_K) -> Tuple[List[Tuple[dict, float]], str]:
        k = max(1, min(int(top_k), len(self.documents)))
        if self.embeddings is not None and len(self.embeddings) == len(self.documents) and self.active_embedding_model:
            try:
                query_vec = self._embed_texts(self.active_embedding_model, [query], task_type="RETRIEVAL_QUERY")[0]
                raw_scores = np.dot(self.embeddings, query_vec)
                scores = self._boost_scores(query, raw_scores)
                top_indices = np.argsort(scores)[::-1][:k]
                results = [(self.documents[int(idx)], float(scores[int(idx)])) for idx in top_indices]
                return results, "dense_cosine"
            except Exception as exc:
                self.index_error = f"{type(exc).__name__}: {exc}"

        return self._lexical_retrieve(query, k), f"lexical_fallback ({self.index_error or 'no embeddings'})"