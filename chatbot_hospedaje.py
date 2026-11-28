# -*- coding: utf-8 -*-
"""
Rule-based bilingual (Spanish/English) chatbot for Cas'aguacate, a boutique aparthotel in
La Asunción, Belén, Heredia, Costa Rica (host: Lore).
Desarrollo de aplicaciones con inteligencia artificial (Tecmilenio) - Avance 1, updated for the final project.
Author: José Pablo Herrera (07317961)

Business data (hours, rules, services, payment methods, reservations e-mail) come from the guest
information that Cas'aguacate shared, used with the hosts' permission. No guest data is stored.
Account numbers, Wi-Fi passwords and access codes are never written here: each guest gets them
in the welcome message. Booked nights below are SAMPLE dates to demo the availability check.
Requirements: Python 3.10+ and NLTK (pip install nltk). No corpus downloads needed.
"""
import re
import unicodedata
from datetime import date

import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.chat.util import Chat, reflections

# ----------------------------------------------------------------------------
# 1. Knowledge base (Cas'aguacate)
# ----------------------------------------------------------------------------
KNOWLEDGE_BASE = {
    "name": "Cas'aguacate",
    "host": "Lore",
    "reservations_email": "reservations.casaguacate@gmail.com",
    "units": "Apartamento Morfo y suites Tucán, Tortuga y Colibrí",
    "units_en": "the Morfo apartment and the Tucán, Tortuga and Colibrí suites",
    # SAMPLE booked nights (year 2026), only to demo the availability check
    "booked_dates": [date(2026, 10, 24), date(2026, 10, 25), date(2026, 10, 31),
                     date(2026, 11, 1), date(2026, 11, 2), date(2026, 11, 14),
                     date(2026, 11, 15), date(2026, 12, 24), date(2026, 12, 25)],
    "maps_name": "CasAguacate Hotel Boutique",
    "checkin_time": "15:00",
    "checkout_time": "11:00",
}

MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
    "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# ----------------------------------------------------------------------------
# 2. NLTK preprocessing
# ----------------------------------------------------------------------------
STEMMERS = {"es": SnowballStemmer("spanish"), "en": PorterStemmer()}

def normalize(text: str) -> str:
    """Lowercase and strip accents (keeps ñ)."""
    text = text.lower().replace("ñ", "\0")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return text.replace("\0", "ñ")

def tokenize(text: str) -> list[str]:
    """Tokenize with NLTK (wordpunct_tokenize) and keep alphanumeric tokens only."""
    return [t for t in wordpunct_tokenize(normalize(text)) if t.isalnum()]

def stem_tokens(tokens, lang: str) -> set[str]:
    """Return the stems of the tokens using the stemmer for the given language."""
    stemmer = STEMMERS[lang]
    return {stemmer.stem(t) for t in tokens}

# ----------------------------------------------------------------------------
# 3. Language detection (function-word overlap, Spanish or English only)
# ----------------------------------------------------------------------------
# Short lists of very frequent function words (similar to a stopwords list),
# written without accents because the text is normalized first.
SPANISH_MARKERS = {
    "el", "la", "los", "las", "de", "del", "que", "y", "en", "un", "una", "es", "por", "para",
    "con", "no", "se", "su", "al", "lo", "como", "mas", "pero", "sus", "le", "ya", "o", "si",
    "me", "mi", "tu", "te", "hay", "esta", "estoy", "cual", "cuanto", "cuando", "donde",
    "quiero", "tienen", "puedo", "hola", "gracias", "adios", "buenas", "buenos", "dias",
    "tardes", "noches", "puede", "dejo", "abro", "aceptan", "pasas", "favor", "hablar",
}
ENGLISH_MARKERS = {
    "the", "a", "an", "and", "of", "to", "in", "is", "are", "it", "for", "on", "with", "at",
    "do", "does", "you", "your", "i", "my", "me", "we", "our", "can", "what", "where", "when",
    "how", "much", "is", "there", "this", "that", "have", "has", "be", "will", "would", "want",
    "hello", "hi", "thanks", "thank", "goodbye", "please", "good", "morning", "afternoon",
    "evening", "night", "who", "which", "am", "not", "available", "speak", "talk",
}

def detect_language(text: str, default: str = "es") -> str:
    """Return 'es' or 'en' by counting marker words; ties keep the default."""
    tokens = tokenize(text)
    es_score = sum(t in SPANISH_MARKERS for t in tokens)
    en_score = sum(t in ENGLISH_MARKERS for t in tokens)
    if es_score > en_score:
        return "es"
    if en_score > es_score:
        return "en"
    return default

# ----------------------------------------------------------------------------
# 4. Intent rules (keywords in both languages). Priority: list order on ties.
# ----------------------------------------------------------------------------
INTENTS = [
    ("contact_host", {"anfitrion", "anfitriona", "lore", "humano", "persona", "hablar", "dueña", "dueño", "agente", "llamar", "queja", "urgente", "emergencia",
                      "host", "human", "person", "owner", "agent", "call", "speak", "talk", "complaint", "urgent", "emergency"}),
    ("parking_gate", {"porton", "estacionamiento", "estacionar", "cochera", "auto", "carro", "coche", "control",
                      "gate", "parking", "park", "garage", "car", "remote"}),
    ("door_lock", {"cerradura", "codigo", "puerta", "chapa", "llave", "llaves", "abrir", "contraseña",
                   "lock", "code", "door", "key", "keys", "open", "keypad"}),
    ("community_access", {"fraccionamiento", "caseta", "vigilancia", "guardia", "privada", "residencial", "pluma",
                          "guard", "security", "booth", "gated", "community", "neighborhood"}),
    ("checkout", {"checkout", "salida", "salir", "entregar", "irnos", "leave", "leaving", "departure"}),
    ("checkin_arrival", {"checkin", "llegada", "llegar", "llego", "llegamos", "entrada", "instrucciones",
                         "arrival", "arrive", "arriving", "instructions"}),
    ("availability", {"disponible", "disponibilidad", "libre", "fechas", "fecha", "ocupado", "reservar", "reserva", "apartar",
                      "available", "availability", "free", "dates", "date", "booked", "book", "booking", "reserve", "reservation", "vacancy"}),
    ("payment", {"pago", "pagos", "pagar", "sinpe", "transferencia", "tarjeta", "factura", "facturar", "tiquete",
                 "comprobante", "payment", "payments", "pay", "card", "transfer", "invoice", "receipt"}),
    ("rates", {"precio", "precios", "tarifa", "tarifas", "costo", "cuesta", "cobran", "noche", "cuanto",
               "price", "prices", "rate", "rates", "cost", "costs", "charge", "night", "much"}),
    ("location", {"ubicacion", "donde", "direccion", "mapa", "ubicados", "llego", "como", "centro",
                  "location", "where", "address", "map", "located", "downtown", "directions"}),
    ("amenities", {"wifi", "internet", "alberca", "piscina", "cocina", "servicios", "incluye", "aire", "asador",
                   "pool", "swimming", "kitchen", "amenities", "include", "includes", "ac", "grill", "bbq", "password"}),
    ("house_rules", {"mascota", "mascotas", "perro", "gato", "fumar", "vapear", "fiesta", "reglas", "ruido", "visitas",
                     "pet", "pets", "dog", "cat", "smoke", "smoking", "vape", "party", "rules", "noise", "visitors"}),
    ("thanks", {"gracias", "agradezco", "excelente", "perfecto", "thanks", "thank", "great", "perfect", "awesome"}),
    ("goodbye", {"adios", "bye", "hasta", "nos", "vemos", "salu2", "goodbye", "later", "see"}),
    ("greeting", {"hola", "buenas", "buenos", "dias", "tardes", "noches", "hey", "saludos",
                  "hello", "hi", "morning", "afternoon", "evening"}),
]

# Stemmed keyword sets, so inflected forms ("parked", "booking") also match.
STEMMED_INTENTS = {name: stem_tokens(keywords, "en") | stem_tokens(keywords, "es")
                   for name, keywords in INTENTS}

# Words that explicitly close the conversation (thanks/goodbye tie-break)
STRONG_GOODBYE = {"adios", "bye", "vemos", "salu2", "goodbye"}
AVAILABILITY_HINTS = {"disponible", "libre", "hay", "fechas", "reservar", "available", "free", "book", "open"}

# Regex fallback patterns with nltk.chat.util.Chat, one set per language
CHAT_PAIRS = {
    "es": [
        (r"(.*)(quien eres|que eres|eres un bot|eres robot)(.*)",
         ["Soy el asistente virtual de Cas'aguacate. Respondo dudas sobre tarifas, pagos, "
          "disponibilidad, ubicación y check-in. Si lo prefieres, escribe 'hablar con Lore'."]),
        (r"(.*)\b(ok|va|sale|de acuerdo|entendido)\b(.*)",
         ["¡Perfecto! ¿Te ayudo con algo más? (tarifas, disponibilidad, ubicación, check-in)"]),
    ],
    "en": [
        (r"(.*)(who are you|what are you|are you a bot|are you a robot)(.*)",
         ["I'm Cas'aguacate's virtual assistant. I can help with rates, payments, availability, "
          "location and check-in. If you prefer, type 'talk to Lore'."]),
        (r"(.*)\b(ok|okay|sure|got it|understood|alright)\b(.*)",
         ["Great! Anything else I can help with? (rates, availability, location, check-in)"]),
    ],
}
REGEX_CHATS = {lang: Chat(pairs, reflections) for lang, pairs in CHAT_PAIRS.items()}

# ----------------------------------------------------------------------------
# 5. Intent identification and responses
# ----------------------------------------------------------------------------
def extract_date(text: str):
    """Find a day + month in Spanish ('14 de noviembre') or English ('November 14', '14 Nov')."""
    t = normalize(text)
    month_names = "|".join(sorted(MONTHS, key=len, reverse=True))
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s*(?:de\s+|of\s+)?(" + month_names + r")\b", t)
    if m:
        day, month = m.group(1), m.group(2)
    else:
        m = re.search(r"\b(" + month_names + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", t)
        if not m:
            return None
        month, day = m.group(1), m.group(2)
    try:
        return date(2026, MONTHS[month], int(day))
    except ValueError:
        return None

def identify_intent(text: str) -> tuple[str, int]:
    tokens = set(tokenize(text))
    t = normalize(text)
    if "check in" in t or "check-in" in t:
        tokens.add("checkin")
    if "check out" in t or "check-out" in t:
        tokens.add("checkout")
    stems = stem_tokens(tokens, "en") | stem_tokens(tokens, "es")
    best, score = "fallback", 0
    for name, keywords in INTENTS:
        s = len(tokens & keywords)
        if s > score:
            best, score = name, s
    # Exact keywords come first; stems are only a backup for inflected forms.
    if best == "fallback":
        for name, _ in INTENTS:
            s = len(stems & STEMMED_INTENTS[name])
            if s > score:
                best, score = name, s
    # Priority rule: if the message thanks and says goodbye ("Gracias, adiós"),
    # goodbye wins because it closes the conversation.
    # A message that only mentions a date ("¿y el 20 de noviembre?") is a follow-up availability question.
    if best == "fallback" and extract_date(text):
        best = "availability"
    if best == "thanks" and tokens & STRONG_GOODBYE:
        best = "goodbye"
    if best in ("rates", "location", "checkin_arrival") and extract_date(text) and tokens & AVAILABILITY_HINTS:
        best = "availability"
    return best, score

def availability_reply(text: str, lang: str) -> str:
    d = extract_date(text)
    host = KNOWLEDGE_BASE["host"]
    if d is None:
        if lang == "en":
            return "Which date are you interested in? Write it like 'November 14'."
        return "¿Para qué fecha te interesa? Escríbela así: '14 de noviembre'."
    if d in KNOWLEDGE_BASE["booked_dates"]:
        if lang == "en":
            return (f"Sorry, the night of {d.strftime('%m/%d/%Y')} is already booked. "
                    f"Would you like to check another date or have {host} suggest options?")
        return (f"Lo siento, la noche del {d.strftime('%d/%m/%Y')} ya está ocupada. "
                f"¿Quieres revisar otra fecha o que {host} te proponga opciones?")
    if lang == "en":
        return (f"Yes! The night of {d.strftime('%m/%d/%Y')} is available (subject to the host's "
                f"confirmation). To book it, type 'talk to {host}'.")
    return (f"¡Sí! La noche del {d.strftime('%d/%m/%Y')} está disponible (sujeto a confirmación "
            f"de la anfitriona). Para apartar, escribe 'hablar con {host}'.")

def build_replies(lang: str) -> dict:
    """Fixed replies per intent, filled in from the knowledge base."""
    kb = KNOWLEDGE_BASE
    email, host = kb["reservations_email"], kb["host"]
    if lang == "en":
        return {
            "greeting": f"Hi! 👋 Welcome to {kb['name']}, a boutique hotel in Belén, Heredia. I can help with rates and payments, availability, location, check-in or put you in touch with {host}.",
            "rates": (f"Rates depend on the room ({kb['units_en']}) and on your dates. "
                      f"Write to {email} or type 'talk to {host}' and we'll send you a quote."),
            "payment": ("You can pay by SINPE Móvil, BAC bank transfer (colones or dollars) or card; "
                        "payment details are sent with your booking confirmation. "
                        "We issue an electronic invoice (factura electrónica) if you have Costa Rican tax details, "
                        "or an electronic receipt (tiquete electrónico) for international guests."),
            "location": (f"📍 Bosques de Doña Rosa, La Asunción, Belén, Heredia, Costa Rica, a few minutes from "
                         f"Juan Santamaría International Airport. Search for '{kb['maps_name']}' on Waze or Google Maps. "
                         "If you come by Uber, DiDi or taxi, follow the map so you aren't dropped off at the wrong place."),
            "checkin_arrival": (f"Check-in from {kb['checkin_time']} (3:00 p.m.); check-out at {kb['checkout_time']} a.m. "
                                "Before you arrive, please send us a photo of each guest's ID or passport (covering sensitive data). "
                                "Your welcome message includes your 4-digit code: the same code opens the pedestrian gate and your room. "
                                "Steps: 1) security at the gated community, 2) pedestrian gate, 3) your room's lock. Ask me about any of them."),
            "community_access": ("🛂 At roundabout #2 of the gated community, the guard will ask for the driver's ID. "
                                 "Tell them the property number from your welcome message."),
            "door_lock": ("🔐 Pedestrian gate: place your hand on the keypad to wake it up (it beeps), enter your 4-digit code "
                          "and press ✓. Your room uses the same code. To lock: from inside, turn the knob until it locks; "
                          "from outside, pull the door closed and place your palm on the keypad until it locks."),
            "parking_gate": ("🚗 If you come by car, the parking gate opens from the inside: turn the knob and pull the gate. "
                             "To close it, push it until it is fully closed. There is one parking space per room; "
                             "your space number is in your welcome message."),
            "checkout": (f"Check-out is at {kb['checkout_time']} a.m. Before leaving, turn off the A/C, lights and appliances, "
                         "and make sure the main door and gates are closed. Late check-out is subject to availability. Thank you!"),
            "amenities": ("Amenities: high-speed Wi-Fi (network and password are on the QR codes inside your room), A/C, "
                          "TV with streaming, mini fridge, shared kitchen, dining area, TV lounge, terraces, BBQ area "
                          "and private parking. On request: breakfast, airport transportation, laundry and luggage storage. "
                          "There is no swimming pool."),
            "house_rules": ("House rules: pets and visitors are not allowed; no parties or events; no smoking or vaping "
                            "(there is a smoking booth in the parking area); quiet hours from 10 p.m. to 8 a.m. "
                            "All guests must be registered before check-in. Breaking the rules carries a fine starting at USD 100."),
            "contact_host": f"I'm connecting you with {host}. 🙋‍♀️ I've let her know and she'll message you shortly. You can also write to {email}. [HANDOFF TO HUMAN]",
            # Cancellation / refund / date change. Used by the proyecto_final agent (intent "cancelacion"). The repo has no
            # cancellation policy, so this reply states NO deadlines or amounts: it points to the booking channel and the host.
            "cancellation": (f"To cancel or change the dates of your reservation, please do it through the same channel you booked with "
                             f"(the booking platform, or directly with {host}). Cancellation and refund conditions depend on your booking "
                             f"and its policy, and I cannot confirm deadlines or amounts. If you prefer, type 'talk to {host}' or write to {email}."),
            "thanks": "You're welcome! Anything else I can help with?",
            "goodbye": "See you soon! Enjoy your stay. 🥑",
        }
    return {
        "greeting": f"¡Hola! 👋 Bienvenido a {kb['name']}, hotel boutique en Belén, Heredia. Puedo ayudarte con tarifas y pagos, disponibilidad, ubicación, check-in o contactarte con {host}.",
        "rates": (f"La tarifa depende de la habitación ({kb['units']}) y de tus fechas. "
                  f"Escríbenos a {email} o escribe 'hablar con {host}' y te enviamos la cotización."),
        "payment": ("Puedes pagar por SINPE Móvil, transferencia BAC (colones o dólares) o tarjeta; los datos de pago "
                    "llegan con tu confirmación de reserva. Emitimos factura electrónica si tienes datos fiscales en "
                    "Costa Rica, o tiquete electrónico para huéspedes extranjeros."),
        "location": (f"📍 Bosques de Doña Rosa, La Asunción, Belén, Heredia, Costa Rica, a pocos minutos del Aeropuerto "
                     f"Internacional Juan Santamaría. Búscanos en Waze o Google Maps como '{kb['maps_name']}'. "
                     "Si llegas en Uber, DiDi o taxi, sigue el mapa para no quedar en un lugar equivocado."),
        "checkin_arrival": (f"Check-in desde las {kb['checkin_time']}; check-out a las {kb['checkout_time']}. "
                            "Antes de llegar, envíanos una foto de la identificación o pasaporte de cada huésped (cubriendo los datos sensibles). "
                            "En tu bienvenida recibes tu código de 4 dígitos: el mismo abre el acceso peatonal y tu habitación. "
                            "Pasos: 1) caseta del residencial, 2) acceso peatonal, 3) cerradura de tu habitación. Pregúntame por cualquiera."),
        "community_access": ("🛂 En la rotonda #2 del residencial el guardia solicitará la cédula del chofer. "
                             "Indica el número de propiedad que viene en tu bienvenida."),
        "door_lock": ("🔐 Acceso peatonal: coloca la mano sobre el teclado para activarlo (sonará), marca tu código de 4 dígitos "
                      "y presiona la palomita ✓. Tu habitación usa el mismo código. Para cerrar: desde adentro gira la perilla "
                      "hasta que quede bloqueada; desde afuera jala la puerta y coloca la palma sobre el teclado hasta que se bloquee."),
        "parking_gate": ("🚗 Si llegas en vehículo, el portón se abre por dentro: gira la perilla y jala el portón. "
                         "Para cerrarlo solo empújalo y verifica que quede completamente cerrado. Hay un espacio por "
                         "habitación; tu número de espacio viene en la bienvenida."),
        "checkout": (f"Check-out a las {kb['checkout_time']}. Antes de salir apaga el aire acondicionado, las luces y los aparatos, "
                     "y verifica que la puerta principal y los portones queden cerrados. Check-out tardío según disponibilidad. ¡Gracias!"),
        "amenities": ("Servicios: Internet de alta velocidad (la red y la clave están en los códigos QR de tu habitación), "
                      "aire acondicionado, TV con streaming, mini refri, cocina compartida, comedor, sala de TV, terrazas, "
                      "área de parrilla y parqueo privado. Bajo solicitud: desayuno, transporte al aeropuerto, lavandería "
                      "y guarda de maletas. No tenemos piscina."),
        "house_rules": ("Reglas: no se permiten mascotas ni visitantes; no se permiten fiestas ni eventos; prohibido fumar o vapear "
                        "(hay un área para fumar en el parqueo); silencio de 22:00 a 8:00. Todos los huéspedes deben "
                        "registrarse antes del check-in. Incumplir las reglas tiene una multa desde USD 100."),
        "contact_host": f"Te comunico con {host}. 🙋‍♀️ Le dejé aviso y te escribirá en breve. También puedes escribir a {email}. [TRANSFERENCIA A HUMANO]",
        # Cancelacion / reembolso / cambio de fechas. Lo usa el agente de proyecto_final (intencion "cancelacion"). El repo no tiene
        # politica de cancelacion, asi que esta respuesta NO da plazos ni montos: remite al canal de reserva y a la anfitriona.
        "cancellation": (f"Para cancelar o cambiar las fechas de tu reservación, hazlo por el mismo canal donde reservaste "
                         f"(la plataforma de reserva, o directamente con {host}). Las condiciones de cancelación y reembolso dependen de tu "
                         f"reserva y de su política, y yo no puedo confirmar plazos ni montos. Si prefieres, escribe 'hablar con {host}' o a {email}."),
        "thanks": "¡Con gusto! ¿Algo más en lo que te ayude?",
        "goodbye": "¡Hasta pronto! Que disfrutes tu estancia. 🥑",
    }

REPLIES = {"es": build_replies("es"), "en": build_replies("en")}

FALLBACK = {
    "es": ("Perdón, no entendí. 🤔 Puedo ayudarte con: tarifas, pagos, disponibilidad, "
           "ubicación, check-in (caseta, portón, cerradura), check-out o hablar con Lore."),
    "en": ("Sorry, I didn't get that. 🤔 I can help with: rates, payments, availability, location, "
           "check-in (security booth, gate, door lock), check-out or talking to Lore."),
}
ESCALATION = {
    "es": "No logro entender tu consulta. Para no hacerte esperar: ",
    "en": "I can't understand your request. So you don't have to wait: ",
}

def reply(intent: str, text: str, lang: str = "es") -> str:
    if intent == "availability":
        return availability_reply(text, lang)
    return REPLIES[lang][intent]

class RentalChatbot:
    """Keeps minimal state: the guest's last language and consecutive fallbacks to escalate to the host."""
    def __init__(self):
        self.fallbacks = 0
        self.lang = "es"

    def process(self, text: str) -> tuple[str, str]:
        self.lang = detect_language(text, default=self.lang)
        intent, _ = identify_intent(text)
        if intent != "fallback":
            self.fallbacks = 0
            return intent, reply(intent, text, self.lang)
        r = REGEX_CHATS[self.lang].respond(normalize(text))
        if r:
            return "regex_chat", r
        self.fallbacks += 1
        if self.fallbacks >= 2:
            self.fallbacks = 0
            return "fallback→contact_host", ESCALATION[self.lang] + reply("contact_host", text, self.lang)
        return "fallback", FALLBACK[self.lang]

def chat():
    """Interactive loop (Colab or terminal). Type 'salir' or 'exit' to quit."""
    bot = RentalChatbot()
    print("Bot: ¡Hola! Soy el asistente de Cas'aguacate. / Hi! I'm Cas'aguacate's assistant. "
          "Escribe 'salir' / type 'exit' para terminar.")
    while True:
        try:
            message = input("Tú/You: ")
        except EOFError:
            break
        if normalize(message).strip() in ("salir", "exit", "quit"):
            print("Bot:", "See you soon!" if normalize(message).strip() != "salir" else "¡Hasta pronto!")
            break
        print("Bot:", bot.process(message)[1])

# ----------------------------------------------------------------------------
# 6. Functional tests: conversations with expected intent and language per turn
# ----------------------------------------------------------------------------
# Each turn: (message, expected_intent, expected_language)
CONVERSATIONS = {
    1: ("Saludo y tarifas", [("Hola, buenas tardes", "greeting", "es"), ("¿Cuánto cuesta la noche?", "rates", "es"), ("Gracias", "thanks", "es")]),
    2: ("Disponibilidad libre", [("Hola", "greeting", "es"), ("¿Está disponible el 14 de diciembre?", "availability", "es"),
                                 ("Perfecto, quiero reservar, hablar con Lore", "contact_host", "es")]),
    3: ("Disponibilidad ocupada", [("¿Tienen libre el 31 de octubre?", "availability", "es"),
                                   ("¿y el 7 de noviembre está disponible?", "availability", "es")]),
    4: ("Ubicación", [("¿Dónde están ubicados?", "location", "es"), ("¿Me pasas la dirección?", "location", "es")]),
    5: ("Llegada / check-in", [("¿A qué hora es el check-in?", "checkin_arrival", "es"),
                               ("¿Qué instrucciones hay para la llegada?", "checkin_arrival", "es")]),
    6: ("Acceso al fraccionamiento", [("Ya estoy en la caseta del fraccionamiento, ¿qué le digo al guardia?", "community_access", "es")]),
    7: ("Cerradura electrónica", [("No puedo abrir la puerta", "door_lock", "es"), ("¿Cuál es el código de la cerradura?", "door_lock", "es")]),
    8: ("Portón del estacionamiento", [("¿Dónde dejo el carro?", "parking_gate", "es"), ("¿Cómo abro el portón?", "parking_gate", "es")]),
    9: ("Servicios y reglas", [("¿Tienen wifi?", "amenities", "es"), ("¿Aceptan mascotas?", "house_rules", "es"),
                               ("¿Se puede fumar?", "house_rules", "es")]),
    10: ("Check-out y despedida", [("¿A qué hora es la salida?", "checkout", "es"), ("Gracias, adiós", "goodbye", "es")]),
    11: ("Transferencia directa a la anfitriona", [("Tengo una emergencia, quiero hablar con la anfitriona", "contact_host", "es")]),
    12: ("Mensajes no reconocidos → escalamiento", [("¿Eres un bot?", "regex_chat", "es"), ("asdfgh", "fallback", "es"),
                                                    ("¿venden boletos para el concierto?", "fallback→contact_host", "es")]),
    13: ("EN: Greeting and rates", [("Hello, good afternoon", "greeting", "en"), ("How much is it per night?", "rates", "en"),
                                    ("Thank you", "thanks", "en")]),
    14: ("EN: Available date", [("Hi", "greeting", "en"), ("Is December 14 available?", "availability", "en"),
                                ("Perfect, I want to book, talk to Lore", "contact_host", "en")]),
    15: ("EN: Booked date", [("Is the 31st of October free?", "availability", "en"), ("What about Nov 7, is it available?", "availability", "en"),
                             ("And December 24th?", "availability", "en")]),
    16: ("EN: Location", [("Where are you located?", "location", "en"), ("Can you send me the address?", "location", "en")]),
    17: ("EN: Arrival / check-in", [("What time is check-in?", "checkin_arrival", "en"), ("What are the arrival instructions?", "checkin_arrival", "en")]),
    18: ("EN: Gated community", [("I'm at the security booth, what do I tell the guard?", "community_access", "en")]),
    19: ("EN: Door lock and parking", [("I can't open the door", "door_lock", "en"), ("What is the lock code?", "door_lock", "en"),
                                       ("Where do I park the car?", "parking_gate", "en")]),
    20: ("EN: Amenities and rules", [("Do you have wifi?", "amenities", "en"), ("Are pets allowed?", "house_rules", "en"),
                                     ("Can I smoke inside?", "house_rules", "en")]),
    21: ("EN: Check-out and goodbye", [("What time is check-out?", "checkout", "en"), ("Thanks, goodbye", "goodbye", "en")]),
    22: ("EN: Emergency handoff", [("I have an emergency, I need to speak with the host", "contact_host", "en")]),
    23: ("EN: Unrecognized → escalation", [("Are you a bot?", "regex_chat", "en"), ("qwerty", "fallback", "en"),
                                           ("Do you sell concert tickets?", "fallback→contact_host", "en")]),
    24: ("Cambio de idioma ES → EN → ES", [("Hola, ¿tienen alberca?", "amenities", "es"), ("Sorry, do you speak English? What is the wifi password?", "amenities", "en"),
                                           ("¿Cuánto cuesta la noche?", "rates", "es")]),
    25: ("Pagos y factura", [("¿Cómo puedo pagar?", "payment", "es"), ("¿Aceptan SINPE móvil?", "payment", "es"),
                             ("¿Me pueden dar factura electrónica?", "payment", "es")]),
    26: ("EN: Payment and pool", [("Can I pay by card?", "payment", "en"), ("Do you have a swimming pool?", "amenities", "en"),
                                  ("Can I bring my dog?", "house_rules", "en")]),
}

# Business-data checks: what the bot must and must not say about Cas'aguacate.
# (question, language, texts that must appear, texts that must NOT appear)
BUSINESS_CHECKS = [
    ("¿Aceptan mascotas?", "es", ["no se permiten mascotas"], ["Mascotas pequeñas", "con aviso"]),
    ("Can I bring my dog?", "en", ["pets and visitors are not allowed"], ["Small pets", "prior notice"]),
    ("¿Tienen alberca?", "es", ["No tenemos piscina"], ["alberca,"]),
    ("Do you have a pool?", "en", ["There is no swimming pool"], ["pool,"]),
    ("¿Cuánto cuesta la noche?", "es", ["reservations.casaguacate@gmail.com"], ["MXN", "$"]),
    ("How much is it per night?", "en", ["reservations.casaguacate@gmail.com"], ["MXN", "$"]),
    ("¿Cómo puedo pagar?", "es", ["SINPE Móvil", "BAC", "tarjeta", "factura electrónica", "tiquete electrónico"], ["MXN"]),
    ("Can I pay by card?", "en", ["SINPE Móvil", "BAC", "card"], ["MXN"]),
    ("¿Dónde están ubicados?", "es", ["La Asunción, Belén, Heredia, Costa Rica"], ["Las Palmas", "Fraccionamiento"]),
    ("Where are you located?", "en", ["La Asunción, Belén, Heredia, Costa Rica"], ["Las Palmas"]),
    ("¿A qué hora es el check-in?", "es", ["15:00", "4 dígitos", "mismo"], []),
    ("What time is check-out?", "en", ["11:00"], []),
    ("Quiero hablar con la anfitriona", "es", ["Lore", "reservations.casaguacate@gmail.com"], ["Marina", "+52"]),
]
FORBIDDEN_ANYWHERE = ["MXN", "Las Palmas", "Almendro", "Marina", "+52", "alberca,", "Mascotas pequeñas", "Small pets"]


def check_business_data() -> bool:
    ok_all = True
    for question, lang, must, must_not in BUSINESS_CHECKS:
        bot = RentalChatbot()
        bot.lang = lang
        _, answer = bot.process(question)
        ok = all(m in answer for m in must) and not any(m in answer for m in must_not)
        ok_all &= ok
        print(f"{'OK   ' if ok else 'FALLA'} {question}")
    every_reply = [r for lang in REPLIES for r in REPLIES[lang].values()] + list(FALLBACK.values())
    leaks = sorted({w for w in FORBIDDEN_ANYWHERE for r in every_reply if w in r})
    print(f"{'OK   ' if not leaks else 'FALLA'} Ninguna respuesta menciona: {', '.join(FORBIDDEN_ANYWHERE)}"
          + (f" (aparece: {', '.join(leaks)})" if leaks else ""))
    return ok_all and not leaks

# Language detection unit cases: (message, expected_language)
LANGUAGE_CASES = [
    ("Hola, ¿cómo estás?", "es"), ("¿Dónde está la casa?", "es"), ("Quiero reservar para el fin de semana", "es"),
    ("Hello, how are you?", "en"), ("Where is the house?", "en"), ("I want to book for the weekend", "en"),
    ("Gracias", "es"), ("Thanks", "en"), ("Can I bring my dog?", "en"), ("¿Puedo llevar a mi perro?", "es"),
]

def run_tests() -> bool:
    import platform
    print(f"Python {platform.python_version()} | NLTK {nltk.__version__}")
    print("Datos del negocio: Cas'aguacate (con permiso de los anfitriones); noches ocupadas de ejemplo\n")
    turns, fallbacks, passed = 0, 0, 0
    for n, (title, messages) in CONVERSATIONS.items():
        bot = RentalChatbot()
        print("=" * 78)
        print(f"Conversación {n}: {title}")
        print("-" * 78)
        for message, expected_intent, expected_lang in messages:
            intent, r = bot.process(message)
            ok = intent == expected_intent and bot.lang == expected_lang
            turns += 1
            passed += ok
            fallbacks += intent.startswith("fallback")
            print(f"Usuario  : {message}")
            print(f"Idioma   : {bot.lang} | Intención: {intent} | Esperado: {expected_intent}/{expected_lang} | {'OK' if ok else 'FALLA'}")
            print(f"Bot      : {r}\n")
    print("=" * 78)
    print("Pruebas de detección de idioma")
    print("-" * 78)
    lang_passed = 0
    for message, expected in LANGUAGE_CASES:
        got = detect_language(message)
        lang_passed += got == expected
        print(f"{'OK   ' if got == expected else 'FALLA'} {got} (esperado {expected}): {message}")
    print("=" * 78)
    print(f"Resumen: {len(CONVERSATIONS)} conversaciones, {turns} turnos, {fallbacks} turnos en fallback (esperados).")
    print(f"Intención + idioma correctos: {passed}/{turns}")
    print(f"Detección de idioma: {lang_passed}/{len(LANGUAGE_CASES)}")
    print("=" * 78)
    print("Datos del negocio (Cas'aguacate)")
    print("-" * 78)
    business_ok = check_business_data()
    return passed == turns and lang_passed == len(LANGUAGE_CASES) and business_ok

if __name__ == "__main__":
    import sys
    if "--chat" in sys.argv:
        chat()
    else:
        sys.exit(0 if run_tests() else 1)
