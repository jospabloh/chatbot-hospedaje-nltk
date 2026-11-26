# Flujo de coordinación entre el chatbot, el clasificador visual y el agente

El agente es **híbrido**: una capa reactiva (reglas rápidas de seguridad, sin modelos) y una capa deliberativa
(estado de la conversación del huésped y política de decisión) que decide qué servicio llamar.

## Flujo de decisión de `POST /agent`

```mermaid
flowchart TD
    A([Huésped: texto y/o foto + estado]) --> B[Flask: validar entradas]
    B -->|inválido| E1[/400, 413, 415, 422 con código y mensaje/]
    B --> C{Capa reactiva<br/>reglas de seguridad}
    C -->|urgencia| R1[ESCALAR_URGENTE<br/>911 + aviso a la anfitriona]
    C -->|dato sensible: tarjeta, CVV| R2[PROTEGER_DATOS]
    C -->|pide hablar con una persona| R3[ESCALAR a la anfitriona]
    C -->|política: late check-out / check-in anticipado| R3
    C -->|fuera de alcance: restaurantes, clima| R3
    C -->|sin regla| D[Capa deliberativa]
    D --> N[Chatbot NLP:<br/>idioma, intención, confianza]
    D --> V[Clasificador visual:<br/>clase de la foto, confianza]
    N --> P{Política de decisión}
    V --> P
    P -->|foto confiable y texto ambiguo<br/>o acceso: decide la foto| F[intención final]
    P -->|texto ambiguo sobre un acceso, sin foto| Q[PEDIR_FOTO]
    P -->|falla o intención de la anfitriona| H[ESCALAR]
    P -->|no entendió 2 veces| H
    P -->|no entendió 1 vez| L[ACLARAR]
    F --> K[Base de conocimiento<br/>datos reales de Cas'aguacate]
    K --> T{¿Tono molesto?}
    T -->|no| S1[RESPONDER]
    T -->|sí| S2[RESPONDER_Y_AVISAR]
    R1 & R2 & R3 & Q & H & L & S1 & S2 --> Z[/respuesta + decisión + traza + estado nuevo/]
```

## Intercambio de información entre componentes

```mermaid
sequenceDiagram
    participant U as Streamlit
    participant A as Flask /agent
    participant G as Agente híbrido
    participant N as Chatbot NLP
    participant V as Clasificador visual
    participant K as Base de conocimiento
    U->>A: texto, foto (opcional), estado de la conversación
    A->>A: validar (longitud, formato real de la imagen, estado)
    A->>G: texto limpio, imagen PIL, estado
    G->>G: capa reactiva (reglas)
    G->>N: texto, idioma detectado
    N-->>G: intención, confianza, top 3
    G->>V: imagen 160x160
    V-->>G: clase, confianza, top 3 (otro si < 0.60)
    G->>G: capa deliberativa: combina evidencia y estado
    G->>K: intención final, texto, idioma
    K-->>G: respuesta con datos reales
    G-->>A: acción, respuesta, avisar_anfitriona, traza, estado nuevo
    A-->>U: JSON (y registro + métricas, sin el texto del huésped)
```

## Qué recibe y devuelve cada componente

| Componente | Entrada | Salida |
|---|---|---|
| Capa reactiva | texto normalizado | regla activada o nada (y marca de tono molesto) |
| Chatbot NLP | texto, idioma | intención (14), confianza, top 3 |
| Clasificador visual | foto 160x160 | clase (cerradura, porton, caseta, controles, interior, otro), confianza, top 3 |
| Capa deliberativa | salidas anteriores + estado | acción, intención final, quién decidió, motivo |
| Base de conocimiento | intención final, texto, idioma | texto de respuesta (sin códigos de acceso ni claves) |

## Estado de la conversación

Viaja con cada petición (el cliente lo guarda y lo reenvía), así que la API no guarda nada del huésped:

| Campo | Uso |
|---|---|
| `idioma` | último idioma del huésped, desempata mensajes cortos |
| `turnos` | número de turnos |
| `fallos` | mensajes seguidos que el asistente no entendió (a 2, escala) |
| `pidio_foto` | ya se pidió una foto en este problema (no se pide dos veces) |
| `intencion_pendiente` | intención que se espera confirmar con la foto |
| `escalada` | ya se avisó a la anfitriona en esta conversación |

## Acciones

`RESPONDER`, `RESPONDER_Y_AVISAR`, `PEDIR_FOTO`, `ACLARAR`, `ESCALAR`, `ESCALAR_URGENTE`, `PROTEGER_DATOS`.
Las que marcan `avisar_anfitriona = true`: `RESPONDER_Y_AVISAR`, `ESCALAR`, `ESCALAR_URGENTE`.

Nota: en esta versión el aviso se marca, se registra (`evento: aviso_anfitriona`) y se cuenta en `/metrics`;
el canal que lo entrega a la anfitriona (correo o WhatsApp) es una mejora futura.
