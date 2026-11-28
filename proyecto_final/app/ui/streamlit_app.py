"""Streamlit interface of the Cas'aguacate guest assistant. It only talks to the Flask API over HTTP.

Two modes:
- Integrated (no API_URL): the app starts the same Flask API in a background thread on 127.0.0.1 with a random
  per-process token and consumes it over HTTP (see api_integrada.py). This is how it runs on Streamlit Community Cloud.
- External (API_URL set): the UI talks to that API, exactly as before.

Run:  streamlit run ui/streamlit_app.py                                   (integrated)
      API_URL=http://127.0.0.1:5000 streamlit run ui/streamlit_app.py     (external API)
Variables: API_URL (external API; leave it unset for integrated mode), API_TOKEN (token of the external API).
On Streamlit Community Cloud set them in the app's Secrets; locally they can be environment variables.
"""
import os
import sys
import uuid

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api_integrada  # noqa: E402
from cliente_api import ApiClient  # noqa: E402


def setting(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:  # no secrets file: fine, use the default
        return default


TEXTS = {
    "es": {
        "title": "Asistente de huéspedes de Cas'aguacate",
        "subtitle": "Hotel boutique en Belén, Heredia, Costa Rica. Escribe en español o en inglés; puedes adjuntar una foto de lo que no funciona.",
        "ai_notice": "Soy una inteligencia artificial. Si prefieres hablar con una persona, escribe «hablar con Lore».",
        # Fixed banner: this public app is the academic demo, not Cas'aguacate's real guest channel
        "demo_notice": "Demo académica (Proyecto Final, Maestría en IA, Tecmilenio). No es el canal oficial de Cas'aguacate y no envía avisos reales a la anfitriona. Si eres huésped, escríbele a Lore por WhatsApp.",
        "chat_tab": "Asistente", "how_tab": "Cómo funciona", "metrics_tab": "Estado del servicio", "ethics_tab": "Privacidad y ética",
        "input": "Escribe tu mensaje (por ejemplo: ¿a qué hora es el check-in?)",
        "welcome": "¡Hola! Soy el asistente virtual de Cas'aguacate. Pregúntame por tarifas, ubicación, check-in, acceso o reglas. Si algo no abre o no funciona, puedes mandarme una foto.",
        "photo": "Foto (opcional)", "photo_help": "JPG o PNG, hasta 5 MB. Se envía junto con tu siguiente mensaje.",
        "send_photo": "Enviar solo la foto", "clear": "Nueva conversación", "language": "Idioma de la interfaz",
        "check": "Revisar conexión", "connected": "Conectado al asistente", "degraded": "Conectado, pero una parte del asistente no está disponible",
        "disconnected": "Sin conexión con el asistente", "advanced": "Avanzado", "api_url": "Dirección de la API",
        "thinking": "Pensando...", "trace": "Ver cómo decidió el asistente", "action": "Acción", "notified": "Se avisó a la anfitriona",
        "step": "Paso", "component": "Componente", "event": "Qué hizo", "data": "Datos", "ms": "ms",
        "refresh_metrics": "Cargar métricas", "metrics_hint": "Pulsa el botón para cargar las métricas del servicio.", "no_metrics": "Aún no hay métricas o la API no responde.", "requests": "Peticiones", "errors": "Errores 5xx", "uptime": "Tiempo activo (s)",
        "per_route": "Por ruta", "latency": "Latencia",
        "mode_integrated": "API REST integrada en la app (misma API Flask, en este servidor)",
        "mode_external": "API REST externa",
        "starting_api": "Iniciando el asistente...",
        "actions": {"RESPONDER": "Respondió", "RESPONDER_Y_AVISAR": "Respondió y avisó a la anfitriona", "PEDIR_FOTO": "Pidió una foto",
                    "ACLARAR": "Pidió aclarar", "ESCALAR": "Pasó el caso a la anfitriona", "ESCALAR_URGENTE": "Urgencia: avisó a la anfitriona",
                    "PROTEGER_DATOS": "Protegió tus datos"},
        "how": "Cada mensaje pasa primero por unas **reglas de seguridad** (urgencias, datos sensibles, petición de hablar con una persona). "
               "Si no se activan, el **chatbot NLP** identifica la intención y el **clasificador visual** reconoce la foto. "
               "La **capa deliberativa** combina ambos resultados con el estado de la conversación y decide: responder, pedir una foto, aclarar o pasar el caso a la anfitriona.",
        "ethics": "- **Eres tú quien decide**: soy una IA y lo digo desde el primer mensaje; siempre puedes pedir hablar con una persona.\n"
                  "- **No guardo tus datos**: ni tus mensajes ni tus fotos se almacenan en el servidor; el estado de la conversación vive solo en tu sesión.\n"
                  "- **No pidas ni compartas** números de tarjeta, documentos, códigos de acceso o claves de Wi-Fi: no los necesito y no los entrego. Los recibes en tu mensaje de bienvenida.\n"
                  "- **Puedo equivocarme**: cuando no estoy seguro, no adivino; paso el caso a la anfitriona.\n"
                  "- **Emergencias**: en peligro inmediato llama al 911; este asistente no sustituye a los servicios de emergencia.",
    },
    "en": {
        "title": "Cas'aguacate guest assistant",
        "subtitle": "Boutique hotel in Belén, Heredia, Costa Rica. Write in Spanish or English; you can attach a photo of what is not working.",
        "ai_notice": "I am an artificial intelligence. If you prefer a person, type “talk to Lore”.",
        "demo_notice": "Academic demo (Final Project, M.Sc. in AI, Tecmilenio). This is not Cas'aguacate's official channel and it does not send real alerts to the host. If you are a guest, message Lore on WhatsApp.",
        "chat_tab": "Assistant", "how_tab": "How it works", "metrics_tab": "Service status", "ethics_tab": "Privacy and ethics",
        "input": "Type your message (for example: what time is check-in?)",
        "welcome": "Hi! I am Cas'aguacate's virtual assistant. Ask me about rates, location, check-in, access or house rules. If something will not open or does not work, you can send me a photo.",
        "photo": "Photo (optional)", "photo_help": "JPG or PNG, up to 5 MB. It is sent with your next message.",
        "send_photo": "Send the photo only", "clear": "New conversation", "language": "Interface language",
        "check": "Check connection", "connected": "Connected to the assistant", "degraded": "Connected, but part of the assistant is unavailable",
        "disconnected": "No connection to the assistant", "advanced": "Advanced", "api_url": "API address",
        "thinking": "Thinking...", "trace": "See how the assistant decided", "action": "Action", "notified": "The host was notified",
        "step": "Step", "component": "Component", "event": "What it did", "data": "Data", "ms": "ms",
        "refresh_metrics": "Load metrics", "metrics_hint": "Press the button to load the service metrics.", "no_metrics": "No metrics yet or the API is not answering.", "requests": "Requests", "errors": "5xx errors", "uptime": "Uptime (s)",
        "per_route": "By route", "latency": "Latency",
        "mode_integrated": "REST API integrated in the app (same Flask API, on this server)",
        "mode_external": "External REST API",
        "starting_api": "Starting the assistant...",
        "actions": {"RESPONDER": "Answered", "RESPONDER_Y_AVISAR": "Answered and notified the host", "PEDIR_FOTO": "Asked for a photo",
                    "ACLARAR": "Asked to rephrase", "ESCALAR": "Passed the case to the host", "ESCALAR_URGENTE": "Emergency: host alerted",
                    "PROTEGER_DATOS": "Protected your data"},
        "how": "Every message first goes through **safety rules** (emergencies, sensitive data, request to talk to a person). "
               "If none fires, the **NLP chatbot** identifies the intent and the **visual classifier** recognizes the photo. "
               "The **deliberative layer** combines both results with the conversation state and decides: answer, ask for a photo, ask to rephrase or pass the case to the host.",
        "ethics": "- **You decide**: I am an AI and I say so from the first message; you can always ask to talk to a person.\n"
                  "- **I do not keep your data**: neither your messages nor your photos are stored on the server; the conversation state lives only in your session.\n"
                  "- **Do not ask for or share** card numbers, documents, access codes or Wi-Fi passwords: I do not need them and I do not give them out. You get them in your welcome message.\n"
                  "- **I can be wrong**: when I am not sure I do not guess; I pass the case to the host.\n"
                  "- **Emergencies**: in immediate danger call 911; this assistant does not replace emergency services.",
    },
}

FLOW = """digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fillcolor="#eef3ee"];
 huesped [label="Huésped\\n(texto + foto)"]; ui [label="Streamlit\\n(interfaz)"]; api [label="Flask API\\n/agent"]; reactiva [label="Capa reactiva\\nreglas de seguridad", fillcolor="#fbe3e0"];
 nlp [label="Chatbot NLP\\nTF-IDF + SVM"]; vis [label="Clasificador visual\\nMobileNetV2"]; delib [label="Capa deliberativa\\nestado + política", fillcolor="#fff2cc"]; kb [label="Base de conocimiento\\n(datos reales)"];
 huesped -> ui -> api -> reactiva; reactiva -> nlp [label="si no hay regla"]; reactiva -> vis [label="si hay foto"]; nlp -> delib; vis -> delib; delib -> kb; kb -> api [label="respuesta + traza"]; api -> ui; }"""


def init_state() -> None:
    st.session_state.setdefault("lang", "es")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("agent_state", {})
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("api_url_dev", "")
    st.session_state.setdefault("session_id", uuid.uuid4().hex)
    st.session_state.setdefault("metrics", None)
    st.session_state.setdefault("health", None)


def is_integrated() -> bool:
    """True when no API_URL is configured (env var or Secrets): the app then hosts its own API. Used by every
    function that needs the API address, the token or the mode label."""
    return not setting("API_URL")


@st.cache_resource(show_spinner=False, validate=lambda api: api.alive())
def embedded_api():
    """Starts the embedded API once per process (cached across sessions and reruns) and returns it; if its thread
    ever died the cache entry is discarded and a new one is started. Called by configured_api_url and get_client."""
    return api_integrada.start_embedded_api()


def configured_api_url() -> str:
    """The API address comes only from the server configuration: API_URL (external) or the embedded API's loopback URL."""
    return embedded_api().url if is_integrated() else setting("API_URL")


def api_token() -> str:
    """Token the UI sends: the random one of the embedded API, or API_TOKEN for an external API."""
    return embedded_api().token if is_integrated() else setting("API_TOKEN")


def dev_override_enabled() -> bool:
    """Local-development override of the API address: explicit flag, no access token configured, external mode."""
    return setting("ALLOW_API_URL_OVERRIDE") == "1" and not setting("API_TOKEN") and not is_integrated()


def get_client() -> ApiClient:
    url = configured_api_url()
    if dev_override_enabled() and st.session_state.get("api_url_dev"):
        url = st.session_state["api_url_dev"]
    # the token is only attached to the configured address, never to a different one
    return ApiClient(url, api_token(), token_url=configured_api_url(),
                     session_id=st.session_state["session_id"])


def mode_label(t: dict) -> str:
    """Honest one-line description of where the REST API runs (integrated or external). Shown in the sidebar and in
    the service-status tab."""
    return t["mode_integrated"] if is_integrated() else t["mode_external"]


def send(text: str, photo, t: dict, lang: str) -> None:
    """Sends one turn to the agent and appends both messages to the conversation."""
    photo_bytes = photo.getvalue() if photo is not None else None
    st.session_state["messages"].append({"role": "user", "content": text or "(foto)", "photo": photo_bytes})
    with st.spinner(t["thinking"]):
        result = get_client().agent(text, st.session_state["agent_state"], photo_bytes, photo.name if photo else "foto.jpg", lang)
    if result.ok:
        decision = result.data.get("decision", {})
        st.session_state["agent_state"] = result.data.get("estado", {})
        st.session_state["messages"].append({"role": "assistant", "content": decision.get("respuesta", ""), "decision": decision,
                                             "trace": result.data.get("traza", [])})
    else:
        st.session_state["messages"].append({"role": "assistant", "content": result.error, "error": True})
    st.session_state["uploader_key"] += 1  # clears the attached photo after sending


def render_message(message: dict, t: dict) -> None:
    with st.chat_message(message["role"], avatar="🥑" if message["role"] == "assistant" else None):
        if message.get("photo"):
            st.image(message["photo"], width=220)
        if message.get("error"):
            st.warning(message["content"])
            return
        st.markdown(message["content"])
        decision = message.get("decision")
        if decision:
            st.caption(f"{t['action']}: {t['actions'].get(decision['accion'], decision['accion'])}"
                       + (f" · {t['notified']}" if decision.get("avisar_anfitriona") else ""))
            with st.expander(t["trace"]):
                st.write(decision.get("motivo", ""))
                rows = [{t["step"]: s["paso"], t["component"]: s["componente"], t["event"]: s["evento"], t["ms"]: s["ms"],
                         t["data"]: str(s["datos"])} for s in message.get("trace", [])]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def sidebar(t: dict) -> None:
    with st.sidebar:
        st.selectbox(t["language"], ["es", "en"], key="lang", format_func=lambda c: {"es": "Español", "en": "English"}[c])
        st.file_uploader(t["photo"], type=["jpg", "jpeg", "png"], help=t["photo_help"], key=f"photo_{st.session_state['uploader_key']}")
        photo = st.session_state.get(f"photo_{st.session_state['uploader_key']}")
        if photo is not None:
            st.image(photo, width="stretch")
            if st.button(t["send_photo"], width="stretch"):
                send("", photo, t, st.session_state["lang"])
                st.rerun()
        if st.button(t["clear"], width="stretch"):
            st.session_state.update(messages=[], agent_state={}, uploader_key=st.session_state["uploader_key"] + 1)
            st.rerun()
        if st.button(t["check"], width="stretch") or st.session_state["health"] is None:
            with st.spinner(t["starting_api"]):
                st.session_state["health"] = get_client().health(st.session_state["lang"])
        health = st.session_state["health"]
        if not health.ok:
            st.error(f"{t['disconnected']}. {health.error}")
        elif health.data.get("estado") != "ok":
            st.warning(t["degraded"])
        else:
            st.success(t["connected"])
        st.caption(mode_label(t))
        if dev_override_enabled():
            with st.expander(t["advanced"]):
                st.text_input(t["api_url"], key="api_url_dev", placeholder=configured_api_url())


def metrics_tab(t: dict) -> None:
    st.caption(mode_label(t))
    # fetched only on demand, not on every rerun: the metrics call must not compete with the guests' chat
    if st.button(t["refresh_metrics"]):
        st.session_state["metrics"] = get_client().metrics(st.session_state["lang"])
    result = st.session_state["metrics"]
    if result is None:
        st.caption(t["metrics_hint"])
        return
    if not result.ok:
        st.info(t["no_metrics"])
        return
    data = result.data
    c1, c2, c3 = st.columns(3)
    c1.metric(t["requests"], data["peticiones_total"])
    c2.metric(t["errors"], data["errores_5xx_total"])
    c3.metric(t["uptime"], data["uptime_s"])
    rows = [{"ruta": route, "peticiones": v["peticiones"], "4xx": v["errores_cliente_4xx"], "5xx": v["errores_servidor_5xx"],
             **{f"{t['latency']} {k} ({t['ms']})": val for k, val in v.get("latencia_ms", {}).items() if k in ("media", "p95")}}
            for route, v in data["por_ruta"].items()]
    st.subheader(t["per_route"])
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if data.get("eventos"):
        st.bar_chart(pd.Series(data["eventos"]))


def main() -> None:
    st.set_page_config(page_title="Cas'aguacate · Asistente", page_icon="🥑", layout="wide")
    init_state()
    t = TEXTS[st.session_state["lang"]]
    sidebar(t)
    t = TEXTS[st.session_state["lang"]]   # the language may have just changed
    # Shown before everything else on every page load so a real guest never mistakes the demo for the live service
    st.warning(t["demo_notice"])
    st.title(f"🥑 {t['title']}")
    st.caption(t["subtitle"])
    chat_tab, how_tab, metrics_tab_, ethics_tab = st.tabs([t["chat_tab"], t["how_tab"], t["metrics_tab"], t["ethics_tab"]])
    with chat_tab:
        st.info(t["ai_notice"])
        with st.chat_message("assistant", avatar="🥑"):
            st.markdown(t["welcome"])
        for message in st.session_state["messages"]:
            render_message(message, t)
        prompt = st.chat_input(t["input"])
        if prompt:
            photo = st.session_state.get(f"photo_{st.session_state['uploader_key']}")
            send(prompt, photo, t, st.session_state["lang"])
            st.rerun()
    with how_tab:
        st.markdown(t["how"])
        st.graphviz_chart(FLOW)
    with metrics_tab_:
        metrics_tab(t)
    with ethics_tab:
        st.markdown(t["ethics"])


main()
