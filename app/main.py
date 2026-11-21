import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel

from app.chat_service import ClinicChatService
from app.rag import ClinicRetriever
from app.whatsapp_ui import build_whatsapp_ui
from config import DATA_PATH

retriever = ClinicRetriever(DATA_PATH)
chat_service = ClinicChatService(retriever)
whatsapp_demo = build_whatsapp_ui(chat_service)

api = FastAPI(title="Clinic WhatsApp FAQ Bot API", version="1.0.0")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    rag_debug: str


@api.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": len(retriever.documents)}


@api.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    reply, debug_info = chat_service.generate_reply(payload.message, [])
    return ChatResponse(reply=reply, rag_debug=debug_info)


app = gr.mount_gradio_app(api, whatsapp_demo, path="/")