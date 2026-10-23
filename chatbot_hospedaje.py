# -*- coding: utf-8 -*-
"""
Rule-based bilingual (Spanish/English) chatbot for a vacation rental host ("Casa Almendro").
Project milestone 1 - Desarrollo de aplicaciones con inteligencia artificial (Tecmilenio)
Author: José Pablo Herrera (07317961)

IMPORTANT: all knowledge base data is SAMPLE / FICTITIOUS data.
It does not belong to any real rental, client or person.
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
# 1. Knowledge base (SAMPLE / FICTITIOUS DATA)
# ----------------------------------------------------------------------------
KNOWLEDGE_BASE = {
    "name": "Casa Almendro",
    "host": "Marina",
    "host_phone": "+52 000 000 0000",  # fictitious
    "rates": {
        "weekday": 1200,      # MXN per night, Sunday to Thursday
        "weekend": 1500,      # MXN per night, Friday and Saturday
        "extra_guest": 200,   # MXN per night, from the 5th guest on
        "cleaning": 350,      # MXN per stay
        "capacity": 6,
        "min_nights": 2,
    },
    # Sample booked nights (year 2026)
    "booked_dates": [date(2026, 10, 24), date(2026, 10, 25), date(2026, 10, 31),
                     date(2026, 11, 1), date(2026, 11, 2), date(2026, 11, 14),
                     date(2026, 11, 15), date(2026, 12, 24), date(2026, 12, 25)],
    "map_pin": "https://maps.example.com/casa-almendro",
    "wifi": ("CasaAlmendro_Ejemplo", "ejemplo2026"),
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
    ("contact_host", {"anfitrion", "anfitriona", "marina", "humano", "persona", "hablar", "dueña", "dueño", "agente", "llamar", "queja", "urgente", "emergencia",
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
    ("rates", {"precio", "precios", "tarifa", "tarifas", "costo", "cuesta", "cobran", "noche", "pagar", "cuanto",
               "price", "prices", "rate", "rates", "cost", "costs", "charge", "night", "pay", "much"}),
    ("location", {"ubicacion", "donde", "direccion", "mapa", "ubicados", "llego", "como", "centro",
                  "location", "where", "address", "map", "located", "downtown", "directions"}),
    ("amenities", {"wifi", "internet", "alberca", "cocina", "servicios", "incluye", "aire", "asador",
                   "pool", "kitchen", "amenities", "include", "includes", "ac", "grill", "bbq", "password"}),
    ("house_rules", {"mascota", "mascotas", "perro", "fumar", "fiesta", "reglas", "ruido",
                     "pet", "pets", "dog", "smoke", "smoking", "party", "rules", "noise"}),
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
         ["Soy el asistente virtual de Casa Almendro (bot de ejemplo). Respondo dudas sobre tarifas, "
          "disponibilidad, ubicación y check-in. Si lo prefieres, escribe 'hablar con Marina'."]),
        (r"(.*)\b(ok|va|sale|de acuerdo|entendido)\b(.*)",
         ["¡Perfecto! ¿Te ayudo con algo más? (tarifas, disponibilidad, ubicación, check-in)"]),
    ],
    "en": [
        (r"(.*)(who are you|what are you|are you a bot|are you a robot)(.*)",
         ["I'm the virtual assistant for Casa Almendro (sample bot). I can help with rates, availability, "
          "location and check-in. If you prefer, type 'talk to Marina'."]),
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
    min_nights = KNOWLEDGE_BASE["rates"]["min_nights"]
    if d is None:
        if lang == "en":
            return f"Which date are you interested in? Write it like 'November 14'. Minimum stay: {min_nights} nights."
        return f"¿Para qué fecha te interesa? Escríbela así: '14 de noviembre'. Estancia mínima: {min_nights} noches."
    if d in KNOWLEDGE_BASE["booked_dates"]:
        if lang == "en":
            return (f"Sorry, the night of {d.strftime('%m/%d/%Y')} is already booked. "
                    "Would you like to check another date or have Marina suggest options?")
        return (f"Lo siento, la noche del {d.strftime('%d/%m/%Y')} ya está ocupada. "
                "¿Quieres revisar otra fecha o que Marina te proponga opciones?")
    if lang == "en":
        return (f"Yes! The night of {d.strftime('%m/%d/%Y')} is available (subject to the host's "
                "confirmation). To book it, type 'talk to Marina'.")
    return (f"¡Sí! La noche del {d.strftime('%d/%m/%Y')} está disponible (sujeto a confirmación "
            "de la anfitriona). Para apartar, escribe 'hablar con Marina'.")

def build_replies(lang: str) -> dict:
    """Fixed replies per intent, filled in from the knowledge base."""
    kb, r = KNOWLEDGE_BASE, KNOWLEDGE_BASE["rates"]
    ssid, wifi_key = kb["wifi"]
    if lang == "en":
        return {
            "greeting": f"Hi! 👋 Welcome to {kb['name']} (sample data). I can help with rates, availability, location, check-in or put you in touch with Marina.",
            "rates": (f"Nightly rates: Sunday to Thursday ${r['weekday']} MXN; Friday and Saturday ${r['weekend']} MXN. "
                      f"Up to 4 guests included; extra guest ${r['extra_guest']} MXN/night (max. {r['capacity']}). "
                      f"Cleaning fee ${r['cleaning']} MXN per stay. Minimum {r['min_nights']} nights."),
            "location": f"📍 Las Palmas gated community (fictitious), 123 Ejemplo St., 10 minutes from downtown. Map pin: {kb['map_pin']}",
            "checkin_arrival": (f"Check-in from {kb['checkin_time']}. Please message this chat 30 minutes before you arrive; "
                                "you'll get a confirmation with your access code for the day. Steps: 1) community security booth, "
                                "2) parking gate, 3) electronic door lock. Ask me about any of them."),
            "community_access": ("🛂 At the security booth, tell the guard you're going to Casa Almendro, lot 12, and show your ID. "
                                 "The guard will register your license plate."),
            "door_lock": ("🔐 The door has an electronic lock: enter the 4-digit code we send you on arrival day "
                          "(example: 1234) and press the ✓ key."),
            "parking_gate": ("🚗 The parking gate opens with the remote in the box next to the main entrance. "
                             "There's room for 2 cars."),
            "checkout": (f"Check-out is at {kb['checkout_time']}. Leave the keys/remotes in the box, turn off the A/C "
                         "and close the gate. Thank you!"),
            "amenities": f"Amenities: Wi-Fi (network: {ssid}, password: {wifi_key}), A/C, fully equipped kitchen, pool, grill.",
            "house_rules": "House rules: no parties and no smoking indoors. Small pets with prior notice. Quiet hours after 10:00 p.m.",
            "contact_host": f"I'm connecting you with {kb['host']}. 🙋‍♀️ I've let her know and she'll message you shortly. Phone: {kb['host_phone']} (fictitious). [HANDOFF TO HUMAN]",
            "thanks": "You're welcome! Anything else I can help with?",
            "goodbye": "See you soon! Enjoy your stay. 🌴",
        }
    return {
        "greeting": f"¡Hola! 👋 Bienvenido a {kb['name']} (datos de ejemplo). Puedo ayudarte con tarifas, disponibilidad, ubicación, check-in o contactarte con Marina.",
        "rates": (f"Tarifas por noche: domingo a jueves ${r['weekday']} MXN; viernes y sábado ${r['weekend']} MXN. "
                  f"Incluye hasta 4 personas; persona extra ${r['extra_guest']} MXN/noche (máx. {r['capacity']}). "
                  f"Limpieza ${r['cleaning']} MXN por estancia. Mínimo {r['min_nights']} noches."),
        "location": f"📍 Fraccionamiento Las Palmas (ficticio), Calle Ejemplo 123, a 10 minutos del centro. Pin de ubicación: {kb['map_pin']}",
        "checkin_arrival": (f"Check-in desde las {kb['checkin_time']}. Al llegar, avisa por este chat 30 minutos antes. "
                            "Recibirás la confirmación con tu código de acceso del día. Pasos: 1) caseta del fraccionamiento, "
                            "2) portón del estacionamiento, 3) cerradura electrónica. Pregúntame por cualquiera."),
        "community_access": ("🛂 En la caseta de vigilancia del fraccionamiento indica que vas a Casa Almendro, lote 12, "
                             "y muestra tu identificación. El guardia registrará tu placa."),
        "door_lock": ("🔐 La puerta tiene cerradura electrónica: marca el código de 4 dígitos que te enviamos el día "
                      "de llegada (ejemplo: 1234) y presiona la palomita ✓."),
        "parking_gate": ("🚗 El portón del estacionamiento se abre con el control que está en la caja junto a la "
                         "entrada principal. Hay lugar para 2 autos."),
        "checkout": (f"Check-out a las {kb['checkout_time']}. Para el check-out deja las llaves/controles en la caja, "
                     "apaga el aire acondicionado y cierra el portón. ¡Gracias!"),
        "amenities": f"Servicios incluidos: Wi-Fi (red: {ssid}, clave: {wifi_key}), A/C, cocina equipada, alberca, asador.",
        "house_rules": "Reglas de la casa: No se permiten fiestas ni fumar dentro. Mascotas pequeñas con aviso previo. Silencio después de las 22:00.",
        "contact_host": f"Te comunico con {kb['host']}. 🙋‍♀️ Le dejé aviso y te escribirá en breve. Teléfono: {kb['host_phone']} (ficticio). [TRANSFERENCIA A HUMANO]",
        "thanks": "¡Con gusto! ¿Algo más en lo que te ayude?",
        "goodbye": "¡Hasta pronto! Que disfrutes tu estancia. 🌴",
    }

REPLIES = {"es": build_replies("es"), "en": build_replies("en")}

FALLBACK = {
    "es": ("Perdón, no entendí. 🤔 Puedo ayudarte con: tarifas, disponibilidad, "
           "ubicación, check-in (caseta, portón, cerradura), check-out o hablar con Marina."),
    "en": ("Sorry, I didn't get that. 🤔 I can help with: rates, availability, location, "
           "check-in (security booth, gate, door lock), check-out or talking to Marina."),
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
    print("Bot: ¡Hola! Soy el asistente de Casa Almendro. / Hi! I'm the Casa Almendro assistant. "
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
                                 ("Perfecto, quiero reservar, hablar con Marina", "contact_host", "es")]),
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
                                ("Perfect, I want to book, talk to Marina", "contact_host", "en")]),
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
}

# Language detection unit cases: (message, expected_language)
LANGUAGE_CASES = [
    ("Hola, ¿cómo estás?", "es"), ("¿Dónde está la casa?", "es"), ("Quiero reservar para el fin de semana", "es"),
    ("Hello, how are you?", "en"), ("Where is the house?", "en"), ("I want to book for the weekend", "en"),
    ("Gracias", "es"), ("Thanks", "en"), ("Can I bring my dog?", "en"), ("¿Puedo llevar a mi perro?", "es"),
]

def run_tests() -> bool:
    import platform
    print(f"Python {platform.python_version()} | NLTK {nltk.__version__}")
    print("Datos: DATOS DE EJEMPLO / FICTICIOS\n")
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
    return passed == turns and lang_passed == len(LANGUAGE_CASES)

if __name__ == "__main__":
    import sys
    if "--chat" in sys.argv:
        chat()
    else:
        sys.exit(0 if run_tests() else 1)
