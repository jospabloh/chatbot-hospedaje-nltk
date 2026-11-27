"""Knowledge base: maps the 14 corpus intents to the replies of the Avance 1 chatbot (real Cas'aguacate data).

The replies live in the repository root (`chatbot_hospedaje.py`) so the business data has a single source.
Access codes, Wi-Fi passwords and property numbers are never written there: the guest receives them in the
welcome message, and the replies only say where to find them.
"""
from __future__ import annotations

import sys

from . import config
from .texto import normalize

if str(config.REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(config.REPO_ROOT))

import chatbot_hospedaje as base  # noqa: E402  (the Avance 1 chatbot: knowledge base and reply texts)

HOST = base.KNOWLEDGE_BASE["host"]
HOST_EMAIL = base.KNOWLEDGE_BASE["reservations_email"]

# Intent label (Spanish, as in the corpus) -> key in chatbot_hospedaje.REPLIES
INTENT_TO_REPLY = {
    "tarifas": "rates", "ubicacion": "location", "cerradura": "door_lock", "porton_estacionamiento": "parking_gate",
    "acceso_fraccionamiento": "community_access", "servicios": "amenities", "reglas": "house_rules",
    "llegada_tarde": "checkin_arrival", "contactar_anfitrion": "contact_host",
    "cancelacion": "cancellation",   # cancel / refund / change dates: general reply, no deadlines or amounts (see chatbot_hospedaje.py)
}
ACCESS_INTENTS = {"cerradura", "porton_estacionamiento", "acceso_fraccionamiento"}
ESCALATE_INTENTS = {"early_late_checkout", "contactar_anfitrion"}
PROBLEM_INTENTS = ACCESS_INTENTS | {"reportar_problema"}
IMAGE_TO_INTENT = {"cerradura": "cerradura", "porton": "porton_estacionamiento", "caseta": "acceso_fraccionamiento",
                   "controles": "reportar_problema", "interior": "reportar_problema"}

PAYMENT_WORDS = {"pago", "pagos", "pagar", "sinpe", "transferencia", "tarjeta", "factura", "facturar", "tiquete",
                 "comprobante", "payment", "payments", "pay", "card", "transfer", "invoice", "receipt"}

MESSAGES = {
    "es": {
        "escalate": "Le paso tu mensaje a {host}, la anfitriona, para que te responda personalmente.",
        "with_photo": " Incluyo la foto que enviaste.",
        "photo_issue": "Gracias por la foto. Le aviso a {host} para que lo revise contigo.",
        "ask_photo": "Para ayudarte mejor, ¿me mandas una foto de lo que no funciona? (la puerta, el portón, la caseta o el aparato).",
        "unclear_photo": "No logro identificar qué muestra la foto. Le aviso a {host} para que la revise y te escriba.",
        "notice": " Además le aviso a {host} para que te contacte.",
        "fallback": "Perdón, no entendí bien. Puedo ayudarte con tarifas, pagos, disponibilidad, ubicación, check-in, check-out o hablar con {host}.",
        "urgent": "Si hay peligro inmediato, llama al 911 ahora. Ya avisé a {host} para que te contacte de inmediato; también puedes escribir a {email}.",
        "privacy": "Por seguridad no compartas números de tarjeta ni documentos por este chat; no los guardo. Los datos de pago llegan con tu confirmación de reserva.",
        "empty": "No recibí ningún mensaje. Escríbeme tu pregunta o adjunta una foto.",
        "hello": "¡Hola! Soy el asistente virtual de Cas'aguacate (una IA). Puedo ayudarte con tarifas, ubicación, check-in y acceso, o avisar a {host}.",
    },
    "en": {
        "escalate": "I am passing your message to {host}, the host, so she can answer you personally.",
        "with_photo": " I am including the photo you sent.",
        "photo_issue": "Thanks for the photo. I am letting {host} know so she can check it with you.",
        "ask_photo": "To help you better, could you send a photo of what is not working? (the door, the gate, the security booth or the appliance).",
        "unclear_photo": "I cannot tell what the photo shows. I am letting {host} know so she can look at it and write to you.",
        "notice": " I am also letting {host} know so she can contact you.",
        "fallback": "Sorry, I did not quite get that. I can help with rates, payments, availability, location, check-in, check-out or talking to {host}.",
        "urgent": "If there is immediate danger, call 911 now. I have already alerted {host} so she can contact you right away; you can also write to {email}.",
        "privacy": "For your safety please do not share card numbers or documents in this chat; I do not store them. Payment details come with your booking confirmation.",
        "empty": "I did not receive any message. Type your question or attach a photo.",
        "hello": "Hi! I am Cas'aguacate's virtual assistant (an AI). I can help with rates, location, check-in and access, or let {host} know.",
    },
}


def message(key: str, lang: str) -> str:
    return MESSAGES[lang][key].format(host=HOST, email=HOST_EMAIL)


def _clean(reply: str) -> str:
    return reply.replace(" [HANDOFF TO HUMAN]", "").replace(" [TRANSFERENCIA A HUMANO]", "")


def reply_for(intent: str, text: str, lang: str) -> str | None:
    """Reply text for a resolved intent; None when the policy is to hand over to the host."""
    if intent == "saludo_cortesia":
        tokens = set(normalize(text).split())
        if tokens & {"adios", "bye", "goodbye", "vemos", "chao"}:
            return base.REPLIES[lang]["goodbye"]
        if tokens & {"gracias", "thanks", "thank", "agradezco", "muchas"}:
            return base.REPLIES[lang]["thanks"]
        return base.REPLIES[lang]["greeting"]
    if intent == "disponibilidad":
        return base.availability_reply(text, lang)
    if intent == "horarios":
        t = normalize(text)
        key = "checkout" if any(w in t for w in ("check out", "checkout", "salida", "salir", "leave", "departure")) else "checkin_arrival"
        return base.REPLIES[lang][key]
    if intent == "tarifas" and set(normalize(text).split()) & PAYMENT_WORDS:
        return base.REPLIES[lang]["payment"]  # the corpus has no "payments" intent: it lives under rates
    key = INTENT_TO_REPLY.get(intent)
    return _clean(base.REPLIES[lang][key]) if key else None
