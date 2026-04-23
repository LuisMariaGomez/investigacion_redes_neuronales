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


def _normalize_value(value):
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


def build_formatted_text(df, target_column):
    df = df.copy()
    feature_columns = [column for column in df.columns if column != target_column]

    def row_to_text(row):
        parts = []

        for column in feature_columns:
            label = oraciones_para_formatear.get(column, f"el valor de {column} es")
            value = _normalize_value(row[column])
            parts.append(f"{label} {value}")

        return ", ".join(parts).lower() + "."

    df["FormattedText"] = df.apply(row_to_text, axis=1)
    return df


def split_features(df, text_column, target_column):
    X_text = df[text_column]
    y = df[target_column]
    return X_text, y
