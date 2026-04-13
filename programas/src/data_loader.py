import pandas as pd

def load_data(path):
    df = pd.read_excel("./data/xlsx/Robo_Ruedas.xlsx")
    return df