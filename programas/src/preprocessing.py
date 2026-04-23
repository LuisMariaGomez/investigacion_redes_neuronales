"""Utilidades para construir el texto enriquecido y separar la etiqueta."""


oraciones_para_formatear = {
    "origeningreso": "el siniestro fue ingresado por",
    "codzonaSancor": "en la zona de",
    "nveh_valor": "el valor del auto es",
    "reciboDomicializado": "el recibo domiciliado tiene valor",
    "codProductor": "el codigo del productor es",
    "diasVigPolDenuncia": "los dias de vigencia de poliza hasta la denuncia son",
    "notifPolicial": "la notificacion policial tiene valor",
    "antAuto": "la antiguedad del auto es",
    "vigenciaCliente": "la vigencia del cliente es",
    "CoberturaContra": "la cobertura contra es",
    "tipoLugarOcurrencia": "el tipo de lugar de ocurrencia es",
    "diaSemanaOcu": "el dia de la semana del hecho fue",
    "horaOcu": "la hora del hecho fue",
    "inspeccion": "la inspeccion tiene valor",
    "inicioCobertura": "el inicio de cobertura tiene valor",
    "cantiSinCli": "la cantidad de siniestros del cliente es",
    "TextoDenuncia": "el texto de la denuncia es",
}


def normalizar_espacios_o_vacios(value):
    if value is None:
        return "missing"

    if isinstance(value, str):
        cleaned_value = value.strip()
        return cleaned_value if cleaned_value else "missing"

    try:
        if value != value:
            return "missing"
    except Exception:
        pass

    return str(value).strip()


def formatear_texto(df, target_column):
    df = df.copy()
    campos_del_xlsx = [column for column in df.columns if column != target_column]

    def reescribir_campo_como_texto(row):
        campo_formateado = []

        for column in campos_del_xlsx:
            texto_de_columna = oraciones_para_formatear.get(column, f"el valor de {column} es")
            valor_campo = normalizar_espacios_o_vacios(row[column])
            campo_formateado.append(f"{texto_de_columna} {valor_campo}")

        return ", ".join(campo_formateado).lower() + "."

    df["FormattedText"] = df.apply (reescribir_campo_como_texto, axis=1)
    return df


def separar_texto_etiquetas(df, text_column, target_column):
    X_text = df[text_column]
    y = df[target_column]
    return X_text, y
