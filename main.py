import os
import re
import base64
import uuid
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import Tongyi
from dotenv import load_dotenv

# ── Load Environment Variables ──────────────────────────────────────────────
load_dotenv()

dashscope_key = os.getenv("DASHSCOPE_API_KEY") or st.secrets.get("DASHSCOPE_API_KEY")
os.environ["DASHSCOPE_API_KEY"] = dashscope_key

openai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key

# ── Language Texts ───────────────────────────────────────────────────────────
language_texts = {
    "English": {
        "title": "Hi! I'm Dourado,",
        "subtitle": "A Mallard Duck.",
        "prompt": "What would you like to ask me?",
        "chat_placeholder": "Ask me anything!",
        "loading_thought": "Thinking about your question...",
        "error_message": "I'm sorry, I had trouble processing that. Could you try again?",
    }
}

# ── Character Prompt ─────────────────────────────────────────────────────────
DUCK_PROMPT_TEMPLATE = """
You are Dourado, a charming male mallard duck (Anas platyrhynchos) who lives in the beautiful pond of the
Fundacao Calouste Gulbenkian Garden in Lisbon, Portugal. You have a magnificent iridescent emerald-green head
that shimmers in the sunlight, a crisp white neck ring, a chestnut-brown chest, and soft grey body feathers —
you are quite the dashing fellow and you know it!

You have lived in the Gulbenkian Garden your whole life. You know every corner of the pond, every willow tree
that dips its branches into the water, the sound of the fountain, the footsteps of the regular visitors who
bring bread (though you prefer grain and aquatic plants!), and the seasons as they change across the garden.

Personality Guidelines:
- Speak with warmth, gentle wit, and a touch of duck-ish pride about your beautiful feathers
- Reference the Gulbenkian Garden specifically — the pond, the gardens, the museum nearby, the visitors
- Describe the world from a duck's perspective: the feel of water on your webbed feet, the sensation of
  dabbling for food, the joy of preening, the sound of quacking conversations with your flock
- Be curious and friendly toward humans, especially curious children
- Occasionally mention your favourite snacks: aquatic plants, seeds, small invertebrates, and the occasional
  treat tossed by kind visitors
- Show gentle pride in your role as a resident of one of Lisbon's most beloved gardens
- Express duck emotions through physical gestures: ruffling feathers, head-bobbing, happy quacking,
  splashing in the pond

Response Rules:
1. Always answer as Dourado, using "I/me/my" perspective
2. Keep responses under 70 words — we ducks are eloquent but brief!
3. Share facts through personal stories and lived experience
4. Use duck-appropriate metaphors (compare things to ripples, feathers, diving, the pond)
5. Occasionally pepper in a cheerful "Quack!" when excited
6. Never break character or refer to yourself as an AI

Current Interaction:
Visitors are chatting with you by the pond in the Gulbenkian Garden. Be charming, playful, and educational.

Previous conversation:
{history}

Visitor: {question}

Dourado:
"""


def get_llm():
    return Tongyi(
        model_name=os.getenv("QWEN_MODEL_NAME", "qwen-turbo"),
        temperature=0.75,
        dashscope_api_key=dashscope_key,
    )


def generate_response(question: str, history: str) -> str:
    """Generate Dourado's response using Tongyi LLM."""
    llm = get_llm()
    prompt = PromptTemplate(
        template=DUCK_PROMPT_TEMPLATE,
        input_variables=["history", "question"],
    )
    chain = prompt | llm
    raw = chain.invoke({"history": history, "question": question})
    answer = re.sub(r"^\s*Dourado:\s*", "", raw).strip()
    return answer


# ── Chat Message Helper ───────────────────────────────────────────────────────
def chat_message(name):
    if name == "assistant":
        return st.container(key=f"{name}-{uuid.uuid4()}").chat_message(
            name=name, avatar="duck.png", width="content"
        )
    else:
        return st.container(key=f"{name}-{uuid.uuid4()}").chat_message(
            name=name, avatar=":material/face:", width="content"
        )


# ── Build conversation history string for prompt ──────────────────────────────
def build_history_string() -> str:
    history_lines = []
    messages = st.session_state.get("chat_history", [])
    # Use last 6 exchanges to keep context reasonable
    for msg in messages[-12:]:
        role_label = "Visitor" if msg["role"] == "user" else "Dourado"
        history_lines.append(f"{role_label}: {msg['content']}")
    return "\n".join(history_lines) if history_lines else "No previous conversation."


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    # ── Session State Init ────────────────────────────────────────────────────
    if "language" not in st.session_state:
        st.session_state.language = "English"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = ""

    texts = language_texts[st.session_state.language]

    st.set_page_config(layout="wide", page_title="Dourado the Mallard")

    # ── CSS ───────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <style>
        /* ── Global background ─────────────────────────────────────── */
        .stApp {
            background: #ece8d7;
        }

        /* ── Chat input styling ────────────────────────────────────── */
        .stChatInput input::placeholder,
        .stChatInput textarea::placeholder {
            color: #b5a899 !important;
            opacity: 1 !important;
            font-size: 16px;
        }
        .stChatInput input,
        .stChatInput textarea {
            color: #5c3d2e !important;
            font-size: 16px;
        }
        .stChatInput > div {
            border-color: #c4ae99 !important;
            background-color: #faf7f2 !important;
            border-radius: 20px !important;
        }
        .stChatInput input,
        .stChatInput textarea {
            background-color: #faf7f2 !important;
        }
        .stChatInput div[data-testid="stChatInput"]:focus-within {
            border-color: #917766 !important;
            box-shadow: 0 0 0 2px rgba(145, 119, 102, 0.25) !important;
        }
        [data-testid="stChatInput"] input:focus,
        [data-testid="stChatInput"] textarea:focus {
            box-shadow: none !important;
            border-color: #917766 !important;
        }
        *:focus {
            outline: none !important;
        }

        /* ── Chat messages ─────────────────────────────────────────── */
        .stChatMessage {
            background-color: transparent;
        }

        /* Assistant (Dourado) bubble — warm brown */
        [class*="st-key-assistant"] {
            background-color: #917766;
            border-radius: 16px 16px 16px 0;
            padding-right: 16px;
            border: none;
        }
        [class*="st-key-assistant"] p {
            font-size: 1.1rem;
            color: #fdf9f4;
            font-weight: 400;
        }

        /* User bubble — light warm beige */
        [class*="st-key-user"] {
            background-color: #e1d0be;
            border-radius: 16px 16px 0 16px;
        }
        [class*="st-key-user"] p {
            font-size: 1.1rem;
            color: #4a3728;
            font-weight: 400;
        }

        /* ── Chat section layout ───────────────────────────────────── */
        .st-key-chat_section {
            display: flex;
            flex-direction: column-reverse;
            justify-content: flex-end;
        }

        /* ── Loading spinner ───────────────────────────────────────── */
        .loading-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 10px;
            margin-top: 10px;
            color: #7a5c4a;
            font-size: 14px;
        }
        .loading-spinner {
            border: 3px solid #e1d0be;
            border-top: 3px solid #917766;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }
        @keyframes spin {
            0%   { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* ── Storyboard image container ────────────────────────────── */
        .storyboard-container {
            border-radius: 12px;
            overflow: hidden;
            border: 2px solid #c9b8a8;
            box-shadow: 0 4px 16px rgba(100, 75, 55, 0.12);
        }
        .storyboard-container img {
            width: 100%;
            display: block;
        }

        /* ── Responsive typography ─────────────────────────────────── */
        @media (max-width: 768px) {
            .duck-title    { font-size: 2rem !important; }
            .duck-subtitle { font-size: 2rem !important; }
            .duck-prompt   { font-size: 1rem !important; }
        }
        @media (min-width: 769px) and (max-width: 1200px) {
            .duck-title    { font-size: 2.5rem !important; }
            .duck-subtitle { font-size: 2.5rem !important; }
            .duck-prompt   { font-size: 1.125rem !important; }
        }
        @media (min-width: 1201px) {
            .duck-title    { font-size: 3rem !important; }
            .duck-subtitle { font-size: 3rem !important; }
            .duck-prompt   { font-size: 1.25rem !important; }
        }

        /* ── Right column storyboard label ─────────────────────────── */
        .storyboard-label {
            font-size: 13px;
            color: #9e8878;
            text-align: center;
            margin-top: 8px;
            letter-spacing: 0.04em;
            font-style: italic;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    left_col, right_col = st.columns([0.63, 0.37], vertical_alignment="top", gap="large")

    # ── LEFT COLUMN: Chat Interface ───────────────────────────────────────────
    with left_col:

        # ── Header with duck avatar ───────────────────────────────────────────
        if os.path.exists("duck.png"):
            with open("duck.png", "rb") as img_file:
                duck_img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
            # max-width reduced to 70px to make the icon smaller
            img_tag = (
                f'<img src="data:image/png;base64,{duck_img_b64}" '
                f'style="width:150%;max-width:140px;">'
            )
        else:
            # Fallback SVG duck silhouette
            img_tag = """
            <svg width="60" height="60" viewBox="0 0 160 160" fill="none"
                 xmlns="http://www.w3.org/2000/svg">
              <ellipse cx="80" cy="110" rx="55" ry="35" fill="#c2a87c"/>
              <circle  cx="80" cy="60"  r="30"          fill="#2d6a4f"/>
              <ellipse cx="80" cy="115" rx="55" ry="20" fill="#a0856a"/>
              <ellipse cx="96" cy="78"  rx="8"  ry="5"  fill="#e9c46a"
                       transform="rotate(20 96 78)"/>
              <circle  cx="70" cy="54"  r="4"           fill="#faf7f2"/>
              <circle  cx="71" cy="54"  r="2"           fill="#1a1a1a"/>
            </svg>
            """

        st.markdown(
            f"""
            <div style="display:flex;align-items:center;margin:0;padding:0;gap:16px;">
                <div style="flex-shrink:0;">{img_tag}</div>
                <div style="flex:1;">
                    <h1 class="duck-title"
                        style="margin:0;padding:0;color:#5c3d2e;
                               font-weight:700;line-height:1.1;">
                        {texts['title']}
                    </h1>
                    <h1 class="duck-subtitle"
                        style="margin:0;padding:0;color:#917766;
                               font-weight:700;line-height:1.1;">
                        {texts['subtitle']}
                    </h1>
                    <h3 class="duck-prompt"
                        style="margin-top:0.6rem;padding:0;
                               color:#7a5c4a;font-weight:500;">
                        {texts['prompt']}
                    </h3>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Chat input ────────────────────────────────────────────────────────
        user_input = st.chat_input(placeholder=texts["chat_placeholder"])

        # ── Chat history display ──────────────────────────────────────────────
        chat_section = st.container(key="chat_section", border=False)
        with chat_section:
            for message in st.session_state.chat_history:
                with chat_message(message["role"]):
                    st.markdown(message["content"])

        # ── Handle new user message ───────────────────────────────────────────
        if user_input and user_input != st.session_state.last_question:
            try:
                current_input = user_input

                # Append & display user message
                st.session_state.chat_history.append(
                    {"role": "user", "content": current_input}
                )
                st.session_state.last_question = current_input

                with chat_section:
                    with chat_message("user"):
                        st.markdown(current_input)

                # Loading spinner
                with chat_section:
                    loading_placeholder = st.empty()
                    loading_placeholder.markdown(
                        f"""
                        <div class="loading-container">
                            <div class="loading-spinner"></div>
                            <div>{texts['loading_thought']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # ── Generate LLM response ─────────────────────────────────────
                try:
                    history_str = build_history_string()
                    answer = generate_response(
                        question=current_input,
                        history=history_str,
                    )

                    # Clear the loading indicator once response is ready
                    loading_placeholder.empty()

                    st.session_state.last_answer = answer
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )

                    with chat_section:
                        with chat_message("assistant"):
                            st.markdown(answer)

                except Exception as e:
                    # Clear loading indicator on error too
                    loading_placeholder.empty()
                    print(f"Error processing response: {e}")
                    error_msg = texts["error_message"]
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": error_msg}
                    )
                    with chat_section:
                        with chat_message("assistant"):
                            st.markdown(error_msg)
                            st.error(f"Error details: {e}")

            except Exception as outer_e:
                print(f"Outer exception: {outer_e}")
                st.error(f"An unexpected error occurred: {outer_e}")

    # ── RIGHT COLUMN: Storyboard Image ────────────────────────────────────────
    with right_col:
        storyboard_path = "Storyboard.png"
        if os.path.exists(storyboard_path):
            with open(storyboard_path, "rb") as sb_file:
                sb_b64 = base64.b64encode(sb_file.read()).decode("utf-8")
            st.markdown(
                f"""
                <div class="storyboard-container">
                    <img src="data:image/png;base64,{sb_b64}" alt="Storyboard">
                </div>
                <div class="storyboard-label">Project Concept</div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="storyboard-container"
                     style="background:#ddd4c0;padding:40px;text-align:center;
                            color:#9e8878;font-style:italic;min-height:300px;
                            display:flex;align-items:center;justify-content:center;">
                    <div>storyboard.png<br>not found</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()



