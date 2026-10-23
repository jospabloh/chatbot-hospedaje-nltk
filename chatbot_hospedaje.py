# -*- coding: utf-8 -*-
"""
Chatbot basado en reglas para un anfitrión de hospedaje vacacional ("Casa Almendro").
Avance de proyecto 1 - Desarrollo de aplicaciones con inteligencia artificial (Tecmilenio)
Autor: José Pablo Herrera (07317961)

IMPORTANTE: todos los datos de la base de conocimiento son DATOS DE EJEMPLO / FICTICIOS.
No corresponden a ningún hospedaje, cliente ni persona real.
Requisitos: Python 3.10+ y NLTK (pip install nltk). No requiere descargas de corpus.
"""
import re
import unicodedata
from datetime import date

import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.chat.util import Chat, reflections

# ----------------------------------------------------------------------------
# 1. Base de conocimiento (DATOS DE EJEMPLO / FICTICIOS)
# ----------------------------------------------------------------------------
BASE_CONOCIMIENTO = {
    "nombre": "Casa Almendro (datos de ejemplo)",
    "anfitriona": "Marina",
    "telefono_anfitriona": "+52 000 000 0000 (ficticio)",
    "tarifas": {
        "entre_semana": 1200,   # MXN por noche, domingo a jueves
        "fin_de_semana": 1500,  # MXN por noche, viernes y sábado
        "persona_extra": 200,   # MXN por noche, a partir de la 5.ª persona
        "limpieza": 350,        # MXN por estancia
        "capacidad": 6,
        "minimo_noches": 2,
    },
    # Noches ocupadas de ejemplo (año 2026)
    "fechas_ocupadas": [date(2026, 10, 24), date(2026, 10, 25), date(2026, 10, 31),
                        date(2026, 11, 1), date(2026, 11, 2), date(2026, 11, 14),
                        date(2026, 11, 15), date(2026, 12, 24), date(2026, 12, 25)],
    "ubicacion": ("Fraccionamiento Las Palmas (ficticio), Calle Ejemplo 123, "
                  "a 10 minutos del centro. Pin de ubicación: https://maps.example.com/casa-almendro"),
    "checkin": {
        "hora_entrada": "15:00",
        "hora_salida": "11:00",
        "llegada": ("Al llegar, avisa por este chat 30 minutos antes. Recibirás la "
                    "confirmación con tu código de acceso del día."),
        "fraccionamiento": ("En la caseta de vigilancia del fraccionamiento indica que vas a "
                            "Casa Almendro, lote 12, y muestra tu identificación. El guardia "
                            "registrará tu placa."),
        "cerradura": ("La puerta tiene cerradura electrónica: marca el código de 4 dígitos que "
                      "te enviamos el día de llegada (ejemplo: 1234) y presiona la palomita ✓."),
        "porton": ("El portón del estacionamiento se abre con el control que está en la "
                   "caja junto a la entrada principal. Hay lugar para 2 autos."),
        "salida": ("Para el check-out deja las llaves/controles en la caja, apaga el aire "
                   "acondicionado y cierra el portón. ¡Gracias!"),
    },
    "servicios": "Wi-Fi (red: CasaAlmendro_Ejemplo, clave: ejemplo2026), A/C, cocina equipada, alberca, asador.",
    "reglas": "No se permiten fiestas ni fumar dentro. Mascotas pequeñas con aviso previo. Silencio después de las 22:00.",
}

MESES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
         "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}

# ----------------------------------------------------------------------------
# 2. Preprocesamiento con NLTK
# ----------------------------------------------------------------------------
def normalizar(texto: str) -> str:
    """Minúsculas y sin acentos (conserva la ñ)."""
    texto = texto.lower().replace("ñ", "\0")
    texto = "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")
    return texto.replace("\0", "ñ")

def tokenizar(texto: str) -> list[str]:
    """Tokeniza con NLTK (wordpunct_tokenize) y deja solo tokens alfanuméricos."""
    return [t for t in wordpunct_tokenize(normalizar(texto)) if t.isalnum()]

# ----------------------------------------------------------------------------
# 3. Reglas de intención (palabras clave). Prioridad: orden de la lista ante empate.
# ----------------------------------------------------------------------------
INTENCIONES = [
    ("contactar_anfitrion", {"anfitrion", "anfitriona", "marina", "humano", "persona", "hablar", "dueña", "dueño", "agente", "llamar", "queja", "urgente", "emergencia"}),
    ("porton_estacionamiento", {"porton", "estacionamiento", "estacionar", "cochera", "auto", "carro", "coche", "control"}),
    ("cerradura", {"cerradura", "codigo", "puerta", "chapa", "llave", "llaves", "abrir", "contraseña"}),
    ("acceso_fraccionamiento", {"fraccionamiento", "caseta", "vigilancia", "guardia", "privada", "residencial", "pluma"}),
    ("checkout", {"checkout", "salida", "salir", "entregar", "irnos"}),
    ("checkin_llegada", {"checkin", "llegada", "llegar", "llego", "llegamos", "entrada", "instrucciones"}),
    ("disponibilidad", {"disponible", "disponibilidad", "libre", "fechas", "fecha", "ocupado", "reservar", "reserva", "apartar"}),
    ("tarifas", {"precio", "precios", "tarifa", "tarifas", "costo", "cuesta", "cobran", "noche", "pagar", "cuanto"}),
    ("ubicacion", {"ubicacion", "donde", "direccion", "mapa", "ubicados", "llego", "como", "centro"}),
    ("servicios", {"wifi", "internet", "alberca", "cocina", "servicios", "incluye", "aire", "asador"}),
    ("reglas", {"mascota", "mascotas", "perro", "fumar", "fiesta", "reglas", "ruido"}),
    ("agradecimiento", {"gracias", "agradezco", "excelente", "perfecto"}),
    ("despedida", {"adios", "bye", "hasta", "nos", "vemos", "salu2"}),
    ("saludo", {"hola", "buenas", "buenos", "dias", "tardes", "noches", "hey", "saludos"}),
]

# Palabras que indican cierre explícito de la conversación (desempate agradecimiento/despedida)
DESPEDIDA_FUERTE = {"adios", "bye", "vemos", "salu2"}

# Patrones de respaldo con nltk.chat.util.Chat (expresiones regulares)
PARES_CHAT = [
    (r"(.*)(quien eres|que eres|eres un bot|eres robot)(.*)",
     ["Soy el asistente virtual de Casa Almendro (bot de ejemplo). Respondo dudas sobre tarifas, "
      "disponibilidad, ubicación y check-in. Si lo prefieres, escribe 'hablar con Marina'."]),
    (r"(.*)(ok|va|sale|de acuerdo|entendido)(.*)",
     ["¡Perfecto! ¿Te ayudo con algo más? (tarifas, disponibilidad, ubicación, check-in)"]),
]
chat_regex = Chat(PARES_CHAT, reflections)

# ----------------------------------------------------------------------------
# 4. Identificación de intención y respuestas
# ----------------------------------------------------------------------------
def extraer_fecha(texto: str):
    m = re.search(r"(\d{1,2})\s*(?:de\s+)?(" + "|".join(MESES) + r")", normalizar(texto))
    if not m:
        return None
    try:
        return date(2026, MESES[m.group(2)], int(m.group(1)))
    except ValueError:
        return None

def identificar_intencion(texto: str) -> tuple[str, int]:
    tokens = set(tokenizar(texto))
    t = normalizar(texto)
    if "check in" in t or "check-in" in t:
        tokens.add("checkin")
    if "check out" in t or "check-out" in t:
        tokens.add("checkout")
    mejor, puntaje = "fallback", 0
    for nombre, claves in INTENCIONES:
        p = len(tokens & claves)
        if p > puntaje:
            mejor, puntaje = nombre, p
    # Regla de prioridad: si el mensaje agradece y se despide ("Gracias, adiós"),
    # gana la despedida porque cierra la conversación.
    if mejor == "agradecimiento" and tokens & DESPEDIDA_FUERTE:
        mejor = "despedida"
    if mejor in ("tarifas", "ubicacion", "checkin_llegada") and extraer_fecha(texto) and \
            tokens & {"disponible", "libre", "hay", "fechas", "reservar"}:
        mejor = "disponibilidad"
    return mejor, puntaje

def responder_disponibilidad(texto: str) -> str:
    f = extraer_fecha(texto)
    if f is None:
        return ("¿Para qué fecha te interesa? Escríbela así: '14 de noviembre'. "
                f"Estancia mínima: {BASE_CONOCIMIENTO['tarifas']['minimo_noches']} noches.")
    if f in BASE_CONOCIMIENTO["fechas_ocupadas"]:
        return (f"Lo siento, la noche del {f.strftime('%d/%m/%Y')} ya está ocupada. "
                "¿Quieres revisar otra fecha o que Marina te proponga opciones?")
    return (f"¡Sí! La noche del {f.strftime('%d/%m/%Y')} está disponible (sujeto a confirmación "
            "de la anfitriona). Para apartar, escribe 'hablar con Marina'.")

def responder(intencion: str, texto: str) -> str:
    kb, ta, ci = BASE_CONOCIMIENTO, BASE_CONOCIMIENTO["tarifas"], BASE_CONOCIMIENTO["checkin"]
    respuestas = {
        "saludo": f"¡Hola! 👋 Bienvenido a {kb['nombre']}. Puedo ayudarte con tarifas, disponibilidad, ubicación, check-in o contactarte con Marina.",
        "tarifas": (f"Tarifas por noche: domingo a jueves ${ta['entre_semana']} MXN; viernes y sábado ${ta['fin_de_semana']} MXN. "
                    f"Incluye hasta 4 personas; persona extra ${ta['persona_extra']} MXN/noche (máx. {ta['capacidad']}). "
                    f"Limpieza ${ta['limpieza']} MXN por estancia. Mínimo {ta['minimo_noches']} noches."),
        "ubicacion": f"📍 {kb['ubicacion']}",
        "checkin_llegada": f"Check-in desde las {ci['hora_entrada']}. {ci['llegada']} Pasos: 1) caseta del fraccionamiento, 2) portón del estacionamiento, 3) cerradura electrónica. Pregúntame por cualquiera.",
        "acceso_fraccionamiento": f"🛂 {ci['fraccionamiento']}",
        "cerradura": f"🔐 {ci['cerradura']}",
        "porton_estacionamiento": f"🚗 {ci['porton']}",
        "checkout": f"Check-out a las {ci['hora_salida']}. {ci['salida']}",
        "servicios": f"Servicios incluidos: {kb['servicios']}",
        "reglas": f"Reglas de la casa: {kb['reglas']}",
        "contactar_anfitrion": f"Te comunico con {kb['anfitriona']}. 🙋‍♀️ Le dejé aviso y te escribirá en breve. Teléfono: {kb['telefono_anfitriona']}. [TRANSFERENCIA A HUMANO]",
        "agradecimiento": "¡Con gusto! ¿Algo más en lo que te ayude?",
        "despedida": "¡Hasta pronto! Que disfrutes tu estancia. 🌴",
    }
    if intencion == "disponibilidad":
        return responder_disponibilidad(texto)
    return respuestas[intencion]

class ChatbotHospedaje:
    """Mantiene un estado mínimo: fallbacks consecutivos para escalar a la anfitriona."""
    def __init__(self):
        self.fallbacks = 0

    def procesar(self, texto: str) -> tuple[str, str]:
        intencion, _ = identificar_intencion(texto)
        if intencion != "fallback":
            self.fallbacks = 0
            return intencion, responder(intencion, texto)
        r = chat_regex.respond(normalizar(texto))
        if r:
            return "regex_chat", r
        self.fallbacks += 1
        if self.fallbacks >= 2:
            self.fallbacks = 0
            return "fallback→contactar_anfitrion", ("No logro entender tu consulta. Para no hacerte esperar: "
                                                    + responder("contactar_anfitrion", texto))
        return "fallback", ("Perdón, no entendí. 🤔 Puedo ayudarte con: tarifas, disponibilidad, "
                            "ubicación, check-in (caseta, portón, cerradura), check-out o hablar con Marina.")

def conversar():
    """Bucle interactivo (Colab o terminal). Escribe 'salir' para terminar."""
    bot = ChatbotHospedaje()
    print("Bot: ¡Hola! Soy el asistente de Casa Almendro (datos de ejemplo). Escribe 'salir' para terminar.")
    while True:
        try:
            u = input("Tú: ")
        except EOFError:
            break
        if normalizar(u).strip() in ("salir", "exit", "quit"):
            print("Bot: ¡Hasta pronto!")
            break
        print("Bot:", bot.procesar(u)[1])

# ----------------------------------------------------------------------------
# 5. Pruebas funcionales: 12 conversaciones distintas
# ----------------------------------------------------------------------------
CONVERSACIONES = {
    1: ("Saludo y tarifas", ["Hola, buenas tardes", "¿Cuánto cuesta la noche?", "Gracias"]),
    2: ("Disponibilidad libre", ["Hola", "¿Está disponible el 14 de diciembre?", "Perfecto, quiero reservar, hablar con Marina"]),
    3: ("Disponibilidad ocupada", ["¿Tienen libre el 31 de octubre?", "¿y el 7 de noviembre está disponible?"]),
    4: ("Ubicación", ["¿Dónde están ubicados?", "¿Me pasas la dirección?"]),
    5: ("Llegada / check-in", ["¿A qué hora es el check-in?", "¿Qué instrucciones hay para la llegada?"]),
    6: ("Acceso al fraccionamiento", ["Ya estoy en la caseta del fraccionamiento, ¿qué le digo al guardia?"]),
    7: ("Cerradura electrónica", ["No puedo abrir la puerta", "¿Cuál es el código de la cerradura?"]),
    8: ("Portón del estacionamiento", ["¿Dónde dejo el carro?", "¿Cómo abro el portón?"]),
    9: ("Servicios y reglas", ["¿Tienen wifi?", "¿Aceptan mascotas?", "¿Se puede fumar?"]),
    10: ("Check-out y despedida", ["¿A qué hora es la salida?", "Gracias, adiós"]),
    11: ("Transferencia directa a la anfitriona", ["Tengo una emergencia, quiero hablar con la anfitriona"]),
    12: ("Mensajes no reconocidos → escalamiento", ["¿Eres un bot?", "asdfgh", "¿venden boletos para el concierto?"]),
}

def ejecutar_pruebas():
    import sys, platform
    print(f"Python {platform.python_version()} | NLTK {nltk.__version__}")
    print("Datos: DATOS DE EJEMPLO / FICTICIOS\n")
    total_turnos, sin_entender = 0, 0
    for n, (titulo, mensajes) in CONVERSACIONES.items():
        bot = ChatbotHospedaje()
        print("=" * 78)
        print(f"Conversación {n}: {titulo}")
        print("-" * 78)
        for m in mensajes:
            intencion, r = bot.procesar(m)
            total_turnos += 1
            sin_entender += intencion.startswith("fallback")
            print(f"Usuario  : {m}")
            print(f"Intención: {intencion}")
            print(f"Bot      : {r}\n")
    print("=" * 78)
    print(f"Resumen: {len(CONVERSACIONES)} conversaciones, {total_turnos} turnos, "
          f"{sin_entender} turnos en fallback.")

if __name__ == "__main__":
    import sys
    if "--chat" in sys.argv:
        conversar()
    else:
        ejecutar_pruebas()
