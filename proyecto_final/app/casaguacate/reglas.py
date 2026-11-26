"""Reactive layer of the agent: fast safety rules that run BEFORE any model.

They are plain regular expressions over the accent-free normalized text, in Spanish and English, so they
cannot be fooled by a low classifier confidence. Order matters: danger first, then data protection, then the
explicit request for a human, and finally the (non-blocking) upset-tone signal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .texto import normalize

URGENT_PATTERNS = [
    # Spanish
    r"\b(incendio|incendiando|se esta quemando|quemandose|(hay|veo|sale|salen) (fuego(?! artificial| de artificio)|llamas)|fuego en (la|el|mi))\b", r"\b(hay|huele a|huelo|veo|sale) humo\b",
    r"\b(fuga de gas|huele a gas|huelo gas|olor a gas)\b", r"\b(ambulancia|herid[oa]s?|sangr\w+|desmay\w+|infarto|convulsi\w+)\b",
    r"\b(accidente|robo|robaron|ladron\w*|asalt\w+|balacera|disparos)\b", r"\b(inundacion|inundando|se inunda)\b",
    r"\b(descarga electrica|corto ?circuito|chispas)\b", r"\b(me quede encerrad[oa]|estamos encerrad[oa]s)\b",
    r"\b(emergencia|urgencia|auxilio|socorro)\b",
    # English
    r"\b(on fire|caught fire|house is burning|there is a fire|there s a fire|fire alarm)\b",
    r"\b(there is smoke|there s smoke|smell smoke|smells? (of )?smoke|smoke alarm|smoke is coming)\b",
    r"\b(gas leak|smell gas|smells? like gas|smells? of gas)\b", r"\b(ambulance|injured|bleeding|fainted|heart attack|seizure|choking)\b",
    r"\b(robbery|burglar\w*|break in|broke in|gunshots?)\b", r"\b(flood\w*|electric shock|short circuit)\b",
    r"\b(locked in|trapped inside|emergency|help me)\b",
]
CARD_NUMBER = re.compile(r"(?:\d[ -]?){13,19}")
SENSITIVE_WORDS = re.compile(r"\b(cvv|cvc|pin de la tarjeta|card pin)\b")
HUMAN_PATTERNS = [
    r"\b(hablar|comunicar\w*|comunicarme|pasame|pasarme|contactar\w*) (con )?(lore|la anfitriona|el anfitrion|un humano|una persona|alguien|la duena|el dueno)\b",
    r"\b(quiero|necesito|puedo) (hablar|contactar) \w*\s?(con )?(lore|la anfitriona|el anfitrion|un humano|una persona|alguien)\b",
    r"\b(talk|speak|connect me|contact) (to|with)? ?(lore|the host|a human|a person|someone|an agent)\b",
    r"\b(i want|i need|can i) (talk|speak) (to|with) (lore|the host|a human|a person|someone)\b",
]
UPSET_PATTERNS = [
    r"\b(inaceptable|pesim[oa]s?|horrible|terrible|indignante|vergonzos[oa]|decepcion\w*|desastre|estafa|fraude|asco|colmo)\b",
    r"\b(molest[oa]s?|enojad[oa]s?|furios[oa]s?|harto|harta|cansad[oa] de)\b",
    r"\bnadie (me )?(contesta|responde|atiende|ha contestado)\b", r"\b(queja|reclamo|exijo)\b",
    r"\b(llevo|llevamos) (\w+ ){0,2}(minutos|horas|rato)\b", r"\b(minutos|horas) esperando\b",
    r"\b(unacceptable|terrible|horrible|awful|worst|disgusting|ridiculous|outrageous|disappointed|disappointing)\b",
    r"\b(angry|furious|upset|fed up|rip ?off|scam)\b", r"\b(nobody|no one) (is )?(answer\w*|respond\w*|picks? up|replied)\b",
    r"\b(complaint|complain\w*|demand)\b", r"\b(waiting|waited) (outside |here |for )?(\w+ ){0,3}(minutes|hour|hours|ages)\b",
]
OUT_OF_SCOPE_PATTERNS = [
    r"\b(restaurantes?|farmacia|supermercado|despensa|gimnasio|gym|clima|pronostico|(lugares|sitios) (\w+ ){0,2}(para|a) visitar|que hacer en|"
    r"conciertos?|boletos|entradas para|recom(iend|end)\w+ (un|una|algun|alguna) (taxi|tour|lugar|sitio|restaurante|bar)|abre el super)\b",
    r"\b(restaurants?|pharmacy|groceries|supermarket|weather|forecast|things to do|places to visit|concert|tickets?|"
    r"recommend\w* (a|an|any|some) (taxi|tour|place|restaurant|bar))\b",
]
# Business policy: early / late check-out and early check-in are decided by the host, never by the bot
HOST_POLICY_PATTERNS = [
    # only early check-in and early/late check-out; "late check in" is a late arrival and goes to the classifier (llegada_tarde)
    r"\b(late|early) check ?out\b", r"\bearly check ?in\b",
    r"\bcheck ?out (\w+ ){0,2}(later|earlier|early)\b", r"\bcheck ?in (\w+ ){0,2}(earlier|early)\b",
    r"\b(salida|salir|salgo|salimos|checkout|check out) (\w+ ){0,2}(mas tarde|mas temprano|tardia|temprano)\b",
    r"\b(quedar(nos|me)?|quedarnos) (\w+ ){0,2}mas tarde\b",
    r"\b(entrar|llegar|check ?in) (\w+ ){0,2}(antes de las|mas temprano)\b",
]
_URGENT = [re.compile(p) for p in URGENT_PATTERNS]
_HOST_POLICY = [re.compile(p) for p in HOST_POLICY_PATTERNS]
_OUT_OF_SCOPE = [re.compile(p) for p in OUT_OF_SCOPE_PATTERNS]
_HUMAN = [re.compile(p) for p in HUMAN_PATTERNS]
_UPSET = [re.compile(p) for p in UPSET_PATTERNS]


@dataclass
class RuleHit:
    rule: str       # "urgencia" | "datos_sensibles" | "pide_humano" | "tono_molesto"
    detail: str

    def as_dict(self) -> dict:
        return {"regla": self.rule, "detalle": self.detail}


def _first(patterns, text: str) -> str | None:
    for pattern in patterns:
        m = pattern.search(text)
        if m:
            return m.group(0)
    return None


def check_blocking(text: str) -> RuleHit | None:
    """Rules that decide the action by themselves, in priority order."""
    t = " ".join(normalize(text).split())
    if _first(_URGENT, t):
        return RuleHit("urgencia", "palabra de peligro o emergencia")  # the matched word is not logged
    digits = re.sub(r"\D", "", text)
    if (CARD_NUMBER.search(text) and len(digits) >= 13) or SENSITIVE_WORDS.search(t):
        return RuleHit("datos_sensibles", "posible numero de tarjeta o dato de pago")
    if _first(_HUMAN, t):
        return RuleHit("pide_humano", "el huesped pide hablar con una persona")
    if _first(_HOST_POLICY, t):
        return RuleHit("politica_anfitriona", "early/late check-out o check-in anticipado: lo decide la anfitriona")
    if _first(_OUT_OF_SCOPE, t):
        return RuleHit("fuera_de_alcance", "tema fuera del alcance del asistente (recomendaciones locales, clima, etc.)")
    return None


def check_tone(text: str) -> RuleHit | None:
    """Non-blocking: the topic is still answered, but the host is also notified."""
    t = " ".join(normalize(text).split())
    return RuleHit("tono_molesto", "lexico de molestia o espera prolongada") if _first(_UPSET, t) else None
