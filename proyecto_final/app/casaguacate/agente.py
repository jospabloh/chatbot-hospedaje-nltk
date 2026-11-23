"""Hybrid intelligent agent (reactive + deliberative) that coordinates the chatbot and the visual classifier.

Perception -> reactive layer (safety rules, no models) -> deliberative layer (conversation state + services)
-> action. The agent never stores anything: the conversation state travels with every request (the client keeps
it), so the API stays stateless and no guest data stays on the server.

Actions: RESPONDER, RESPONDER_Y_AVISAR, PEDIR_FOTO, ACLARAR, ESCALAR, ESCALAR_URGENTE, PROTEGER_DATOS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import config, conocimiento as kb
from .reglas import check_blocking, check_tone
from .texto import detect_language

ACTIONS = ("RESPONDER", "RESPONDER_Y_AVISAR", "PEDIR_FOTO", "ACLARAR", "ESCALAR", "ESCALAR_URGENTE", "PROTEGER_DATOS")
NOTIFIES_HOST = {"RESPONDER_Y_AVISAR", "ESCALAR", "ESCALAR_URGENTE"}


@dataclass
class Services:
    """What the agent can call. `tone` is optional; the others are required for a full decision."""
    intents: object            # .predict(text, lang) -> TextPrediction
    vision: object = None      # .predict(PIL image) -> ImagePrediction
    tone: object = None        # .predict(text) -> TextPrediction


@dataclass
class ConversationState:
    idioma: str = "es"
    turnos: int = 0
    fallos: int = 0                      # consecutive messages the assistant could not understand
    pidio_foto: bool = False             # the assistant already asked for a photo in this problem
    intencion_pendiente: str | None = None
    escalada: bool = False               # the host was already notified in this conversation

    @classmethod
    def from_dict(cls, data: dict | None) -> "ConversationState":
        if not data:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("estado debe ser un objeto")
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"campos desconocidos en estado: {', '.join(sorted(unknown))}")
        state = cls(**{k: data[k] for k in data})
        if state.idioma not in ("es", "en"):
            raise ValueError("estado.idioma debe ser 'es' o 'en'")
        for name in ("turnos", "fallos"):
            value = getattr(state, name)
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 10_000:
                raise ValueError(f"estado.{name} debe ser un entero entre 0 y 10000")
        for name in ("pidio_foto", "escalada"):
            if not isinstance(getattr(state, name), bool):
                raise ValueError(f"estado.{name} debe ser booleano")
        if state.intencion_pendiente is not None and not isinstance(state.intencion_pendiente, str):
            raise ValueError("estado.intencion_pendiente debe ser texto o nulo")
        if state.intencion_pendiente is not None and len(state.intencion_pendiente) > 40:
            raise ValueError("estado.intencion_pendiente es demasiado largo")
        return state

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class AgentResult:
    action: str
    final_intent: str | None
    decided_by: str
    reply: str
    lang: str
    notify_host: bool
    reason: str
    trace: list = field(default_factory=list)
    state: ConversationState = field(default_factory=ConversationState)
    total_ms: float = 0.0

    def as_dict(self) -> dict:
        return {"decision": {"accion": self.action, "intencion_final": self.final_intent, "decidido_por": self.decided_by,
                             "respuesta": self.reply, "idioma": self.lang, "avisar_anfitriona": self.notify_host,
                             "motivo": self.reason},
                "traza": self.trace, "estado": self.state.as_dict(), "tiempos_ms": {"total": round(self.total_ms, 1)}}


class HybridAgent:
    def __init__(self, services: Services):
        self.services = services

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _step(trace: list, component: str, event: str, data: dict, started: float) -> None:
        trace.append({"paso": len(trace) + 1, "componente": component, "evento": event, "datos": data,
                      "ms": round((time.perf_counter() - started) * 1000, 1)})

    def _finish(self, started, trace, state, lang, action, intent, decided_by, reply, reason) -> AgentResult:
        state.idioma = lang
        if action in NOTIFIES_HOST:
            state.escalada = True
        if action in ("RESPONDER", "RESPONDER_Y_AVISAR", "ESCALAR", "ESCALAR_URGENTE", "PROTEGER_DATOS"):
            state.fallos, state.pidio_foto, state.intencion_pendiente = 0, False, None
        return AgentResult(action, intent, decided_by, reply, lang, action in NOTIFIES_HOST, reason, trace, state,
                           (time.perf_counter() - started) * 1000)

    # ------------------------------------------------------------------ main loop
    def run(self, text: str = "", image=None, state: ConversationState | None = None) -> AgentResult:
        started = time.perf_counter()
        state = ConversationState.from_dict(state.as_dict()) if state else ConversationState()
        state.turnos += 1
        trace: list = []
        text = (text or "").strip()
        lang = detect_language(text, default=state.idioma) if text else state.idioma
        t = time.perf_counter()
        self._step(trace, "Agente", "percepcion",
                   {"idioma": lang, "tiene_texto": bool(text), "tiene_foto": image is not None, "turno": state.turnos}, t)

        if not text and image is None:
            return self._finish(started, trace, state, lang, "ACLARAR", None, "regla", kb.message("empty", lang), "mensaje vacio")

        # ---- reactive layer: rules that decide by themselves, before any model
        t = time.perf_counter()
        hit = check_blocking(text) if text else None
        self._step(trace, "Capa reactiva", "reglas de seguridad", {"regla": hit.as_dict() if hit else None}, t)
        if hit:
            if hit.rule == "urgencia":
                return self._finish(started, trace, state, lang, "ESCALAR_URGENTE", None, "regla: urgencia",
                                    kb.message("urgent", lang), hit.detail)
            if hit.rule == "datos_sensibles":
                return self._finish(started, trace, state, lang, "PROTEGER_DATOS", None, "regla: datos sensibles",
                                    kb.message("privacy", lang), hit.detail)
            if hit.rule == "pide_humano":
                return self._finish(started, trace, state, lang, "ESCALAR", "contactar_anfitrion", "regla: pide humano",
                                    kb.reply_for("contactar_anfitrion", text, lang), hit.detail)
            if hit.rule == "politica_anfitriona":
                return self._finish(started, trace, state, lang, "ESCALAR", "early_late_checkout", "regla: politica de la anfitriona",
                                    kb.message("escalate", lang), hit.detail)
            return self._finish(started, trace, state, lang, "ESCALAR", "fuera_de_alcance", "regla: fuera de alcance",
                                kb.message("escalate", lang), hit.detail)

        # ---- deliberative layer: call the services and combine their evidence with the conversation state
        text_pred = None
        if text:
            t = time.perf_counter()
            text_pred = self.services.intents.predict(text, lang)
            self._step(trace, "Chatbot NLP", "clasificar intencion", text_pred.as_dict(), t)
        image_pred = None
        if image is not None and self.services.vision is not None:
            t = time.perf_counter()
            image_pred = self.services.vision.predict(image)
            self._step(trace, "Clasificador visual", "clasificar foto", image_pred.as_dict(), t)

        upset = check_tone(text) if text else None
        tone_source = "regla de lexico" if upset else None
        if not upset and text and self.services.tone is not None:
            t = time.perf_counter()
            tone = self.services.tone.predict(text)
            self._step(trace, "Chatbot NLP", "clasificar tono", tone.as_dict(), t)
            if tone.label == "negativo" and tone.confidence >= config.TONE_THRESHOLD:
                upset, tone_source = True, "modelo de tono"

        intent = text_pred.label if text_pred else None
        p = text_pred.confidence if text_pred else 0.0
        low_text = text_pred is None or p < config.TEXT_THRESHOLD
        photo = image_pred.label if image_pred and image_pred.reliable else None      # reliable photo class
        suggested = kb.IMAGE_TO_INTENT.get(photo) if photo else None
        if intent is None and state.intencion_pendiente and photo:    # photo sent alone after being asked for it
            intent, low_text = state.intencion_pendiente, False

        final, decided_by = intent, "texto"
        if suggested and (intent is None or p < config.AMBIGUOUS_TEXT or intent == "reportar_problema"
                          or (intent in kb.ACCESS_INTENTS and suggested in kb.ACCESS_INTENTS)):
            final = suggested
            decided_by = "texto e imagen coinciden" if suggested == intent else "imagen"
        if upset and tone_source:
            decided_by += " y tono"

        action, reply, reason = self._deliberate(state, lang, text, final, p, low_text, image_pred, photo)
        if upset and action == "RESPONDER":
            action, reply = "RESPONDER_Y_AVISAR", reply + kb.message("notice", lang)
            reason += "; el huesped suena molesto, tambien se avisa a la anfitriona"
        elif upset and action in ("ACLARAR", "PEDIR_FOTO"):
            action, reply = "ESCALAR", kb.message("escalate", lang)
            reason += "; huesped molesto sin resolver: se pasa a la anfitriona"
        t = time.perf_counter()
        self._step(trace, "Capa deliberativa", "decision",
                   {"accion": action, "intencion_final": final, "decidido_por": decided_by, "motivo": reason,
                    "estado": {"fallos": state.fallos, "pidio_foto": state.pidio_foto}}, t)
        return self._finish(started, trace, state, lang, action, final, decided_by, reply, reason)

    def _deliberate(self, state, lang, text, final, p, low_text, image_pred, photo):
        """Policy of the deliberative layer. Returns (action, reply, reason); may update the state."""
        host_photo = kb.message("with_photo", lang) if image_pred is not None else ""
        if image_pred is not None and photo is None and (low_text or final in kb.PROBLEM_INTENTS):
            return "ESCALAR", kb.message("unclear_photo", lang), "la foto no supera el umbral y el texto no basta"
        if final is None or (low_text and photo is None):
            state.fallos += 1
            if state.fallos >= config.MAX_FALLBACKS:
                return ("ESCALAR", kb.message("escalate", lang) + host_photo,
                        f"no se entendio el mensaje {state.fallos} veces seguidas")
            return "ACLARAR", kb.message("fallback", lang), f"confianza del texto {p:.2f} menor que {config.TEXT_THRESHOLD}"
        if final in kb.ESCALATE_INTENTS:
            return "ESCALAR", kb.message("escalate", lang) + host_photo, "la politica del negocio deja esta decision a la anfitriona"
        if final == "reportar_problema":
            if image_pred is None and not state.pidio_foto:
                state.pidio_foto, state.intencion_pendiente = True, final
                return "PEDIR_FOTO", kb.message("ask_photo", lang), "un reporte de problema se entiende mejor con foto"
            key = "photo_issue" if image_pred is not None else "escalate"
            return "ESCALAR", kb.message(key, lang), "los reportes de fallas los revisa la anfitriona"
        if final in kb.ACCESS_INTENTS and image_pred is None and p < config.AMBIGUOUS_TEXT and not state.pidio_foto:
            state.pidio_foto, state.intencion_pendiente = True, final
            return "PEDIR_FOTO", kb.message("ask_photo", lang), f"texto ambiguo ({p:.2f}) sobre un acceso: se pide una foto"
        reply = kb.reply_for(final, text, lang)
        if reply is None:
            return "ESCALAR", kb.message("escalate", lang), "sin respuesta en la base de conocimiento"
        return "RESPONDER", reply, "respuesta de la base de conocimiento"


def answer_text(services: Services, text: str, lang: str | None = None) -> dict:
    """Chatbot-only component (no agent): intent, confidence and the reply, or a hand-over below the threshold."""
    lang = lang or detect_language(text)
    pred = services.intents.predict(text, lang)
    if pred.confidence < config.TEXT_THRESHOLD or pred.label in kb.ESCALATE_INTENTS or pred.label == "reportar_problema":
        reply, handled = kb.message("escalate", lang), False
    else:
        reply, handled = kb.reply_for(pred.label, text, lang), True
    return {"idioma": lang, "intencion": pred.as_dict(), "respuesta": reply, "resuelto_por_el_bot": handled}
