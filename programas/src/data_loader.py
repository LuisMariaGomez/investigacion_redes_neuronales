import pandas as pd

def load_data(path):
    df = pd.read_excel(path)
    
    df["TextoDenuncia"] = df["TextoDenuncia"].fillna("")
    df["CoberturaContra"] = df["CoberturaContra"].fillna("")

    return df