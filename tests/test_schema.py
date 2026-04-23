import pytest
import pandas as pd
from src.data.load_data import load_data
from pandera.errors import SchemaError

def test_load_data_schema_error(tmp_path):
    # Cria um CSV inválido (sem a coluna 'gender', por exemplo)
    d = {'customerID': ['123'], 'MonthlyCharges': [50.0]}
    df = pd.DataFrame(data=d)
    file_path = tmp_path / "invalid_data.csv"
    df.to_csv(file_path, index=False)
    
    # O load_data deve levantar um SchemaError do Pandera
    with pytest.raises(SchemaError):
        load_data(str(file_path))