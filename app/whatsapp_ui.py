import gradio as gr

from app.chat_service import ClinicChatService

WHATSAPP_CSS = """
.gradio-container {
    max-width: 540px !important;
    margin: 16px auto !important;
    padding-top: 8px !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
#wa-header {
    background-color: #075E54;
    color: white;
    padding: 16px 18px;
    border-radius: 12px 12px 0 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
#wa-header h3 { margin: 0; color: white; font-size: 16px; line-height: 1.3; }
#wa-header p { margin: 2px 0 0 0; color: #D9FDD3; font-size: 12px; line-height: 1.3; }
#wa-chatbot {
    background-color: #EFEAE2 !important;
    border: 1px solid #d1d7db;
    border-top: none;
}
#wa-chatbot .user,
#wa-chatbot .bot {
    unicode-bidi: plaintext;
    text-align: start;
}
#wa-chatbot .user {
    background-color: #D9FDD3 !important;
    color: #111B21 !important;
    border-radius: 10px 0px 10px 10px !important;
    box-shadow: 0 1px 1px rgba(0,0,0,0.08);
}
#wa-chatbot .bot {
    background-color: #FFFFFF !important;
    color: #111B21 !important;
    border-radius: 0px 10px 10px 10px !important;
    box-shadow: 0 1px 1px rgba(0,0,0,0.08);
}
#send-btn {
    background-color: #25D366 !important;
    color: white !important;
    border: none !important;
    font-weight: 600;
}
"""

INITIAL_MESSAGES = [
    {
        "role": "assistant",
        "content": (
            "مرحباً بك في مساعد الأسئلة الشائعة للعيادة 🏥\n"
            "Welcome to the Clinic FAQ Assistant!\n\n"
            "Ask about working hours, doctors, consultation fees, or insurance in English or Arabic.\n\n"
            "<sub>✓✓ Online</sub>"
        ),
    }
]

SAMPLE_PROMPTS = [
    "What are the working hours and location?",
    "سعر الكشف كام وهل بتقبلوا تأمين AXA؟",
    "I have chest pain, what medication should I take?",
]


def _make_sample_handler(prompt_text: str, run_fn):
    def _handler(history: list):
        yield from run_fn(prompt_text, history)

    return _handler


def build_whatsapp_ui(chat_service: ClinicChatService) -> gr.Blocks:
    with gr.Blocks(title="Clinic WhatsApp FAQ Bot") as demo:
        gr.HTML(
            """
            <div id="wa-header">
                <div style="width:40px;height:40px;min-width:40px;border-radius:50%;background:#25D366;display:flex;align-items:center;justify-content:center;font-size:20px;">🏥</div>
                <div>
                    <h3>Clinic FAQ Assistant • مساعد العيادة</h3>
                    <p>● Online | WhatsApp-Style Demo</p>
                </div>
            </div>
            """
        )

        chatbot = gr.Chatbot(
            value=INITIAL_MESSAGES,
            elem_id="wa-chatbot",
            height=440,
            show_label=False,
        )

        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Type a message or اكتب سؤالك هنا...",
                show_label=False,
                scale=4,
                container=False,
            )
            send_btn = gr.Button("Send ➤", elem_id="send-btn", scale=1)

        def run_message(user_msg: str, history: list):
            if not user_msg or not user_msg.strip():
                yield "", history, ""
                return
            clean_msg = user_msg.strip()
            history_with_user = list(history) + [{"role": "user", "content": clean_msg}]
            yield "", history_with_user, "*Retrieving FAQ context and generating reply...*"

            reply, debug_info = chat_service.generate_reply(clean_msg, history)
            final_history = list(history_with_user) + [{"role": "assistant", "content": reply}]
            yield "", final_history, debug_info

        gr.Markdown("**Recommended Messages (Click to Send):**")
        with gr.Row():
            sample_buttons = [gr.Button(prompt, size="sm") for prompt in SAMPLE_PROMPTS]

        with gr.Accordion("RAG Retrieval Inspector", open=False):
            debug_output = gr.Markdown("*Send a message to inspect retrieved chunks and similarity scores.*")

        for prompt_text, btn in zip(SAMPLE_PROMPTS, sample_buttons):
            btn.click(
                fn=_make_sample_handler(prompt_text, run_message),
                inputs=[chatbot],
                outputs=[msg_input, chatbot, debug_output],
            )

        send_btn.click(run_message, [msg_input, chatbot], [msg_input, chatbot, debug_output])
        msg_input.submit(run_message, [msg_input, chatbot], [msg_input, chatbot, debug_output])

    return demo