import numpy as np
import pandas as pd

from src.data.preprocess import clean_data


def test_clean_data_logic():
    """
    Testa se a limpeza remove o ID, trata nulos em TotalCharges e mapeia o Churn.
    """
    data = {
        'customerID': ['001-A', '002-B'],
        'TotalCharges': ['100.5', ' '],
        'Churn': ['Yes', 'No'],
        'gender': ['Male', 'Female']
    }
    df = pd.DataFrame(data)

    df_cleaned = clean_data(df)

    # Verificações
    assert 'customerID' not in df_cleaned.columns
    assert df_cleaned['TotalCharges'].iloc[1] == 0
    assert df_cleaned['Churn'].iloc[0] == 1
    assert df_cleaned['Churn'].iloc[1] == 0
    assert df_cleaned['TotalCharges'].dtype == np.float64
