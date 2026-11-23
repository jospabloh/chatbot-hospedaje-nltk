"""Unit tests of the input validation."""
import io

import pytest
from werkzeug.datastructures import FileStorage

from casaguacate import config
from casaguacate.validacion import ApiError, validate_image, validate_lang, validate_state, validate_text
from tests.helpers import make_image


def upload(data: bytes, name: str) -> FileStorage:
    return FileStorage(stream=io.BytesIO(data), filename=name)


def error_of(fn, *args):
    with pytest.raises(ApiError) as info:
        fn(*args)
    return info.value


def test_texto_valido_se_limpia():
    assert validate_text("  hola \x00mundo  ") == "hola mundo"


@pytest.mark.parametrize("value,code", [(None, "texto_requerido"), ("   ", "texto_vacio"), (5, "texto_invalido"),
                                        (["a"], "texto_invalido"), ("x" * (config.MAX_TEXT_CHARS + 1), "texto_demasiado_largo")])
def test_texto_invalido(value, code):
    err = error_of(validate_text, value)
    assert err.status == 400 and err.code == code


def test_texto_opcional():
    assert validate_text(None, required=False) == ""


def test_idioma():
    assert validate_lang("en") == "en" and validate_lang(None) is None and validate_lang("") is None
    assert error_of(validate_lang, "fr").code == "idioma_invalido"


def test_estado_desde_json_y_vacio():
    assert validate_state(None).turnos == 0
    assert validate_state("").turnos == 0
    assert validate_state('{"turnos": 3, "idioma": "en"}').idioma == "en"


@pytest.mark.parametrize("value", ["{no es json", '{"idioma": "fr"}', '{"turnos": "x"}', '{"hack": 1}', "[1, 2]", 7])
def test_estado_invalido(value):
    err = error_of(validate_state, value)
    assert err.status == 400 and err.code == "estado_invalido"


def test_imagen_valida_png_y_jpg():
    assert validate_image(upload(make_image("PNG"), "foto.png")).size == (120, 90)
    assert validate_image(upload(make_image("JPEG"), "FOTO.JPG")).mode == "RGB"


def test_extension_no_permitida():
    err = error_of(validate_image, upload(make_image("PNG"), "foto.gif"))
    assert err.status == 415 and err.code == "formato_no_permitido"
    assert error_of(validate_image, upload(make_image("PNG"), "sinextension")).status == 415


def test_contenido_que_no_es_la_extension():
    err = error_of(validate_image, upload(make_image("GIF"), "foto.png"))
    assert err.status == 415 and err.code == "formato_no_permitido"


def test_archivo_corrupto_o_texto_con_extension_de_imagen():
    assert error_of(validate_image, upload(b"esto no es una imagen", "foto.jpg")).code == "imagen_corrupta"
    assert error_of(validate_image, upload(make_image("PNG")[:40], "foto.png")).code == "imagen_corrupta"


def test_archivo_vacio_y_sin_nombre():
    assert error_of(validate_image, upload(b"", "foto.png")).code == "archivo_vacio"
    assert error_of(validate_image, upload(make_image("PNG"), "")).code == "archivo_sin_nombre"


def test_archivo_demasiado_grande():
    err = error_of(validate_image, upload(b"0" * (config.MAX_IMAGE_BYTES + 10), "foto.jpg"))
    assert err.status == 413 and err.code == "archivo_demasiado_grande"


def test_imagen_muy_pequena():
    err = error_of(validate_image, upload(make_image("PNG", size=(16, 16)), "foto.png"))
    assert err.status == 422 and err.code == "imagen_muy_pequena"
