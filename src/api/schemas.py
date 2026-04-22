from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    gender: str = Field(..., json_schema_extra={"example": "Male"})
    SeniorCitizen: int = Field(..., json_schema_extra={"example": 0})
    Partner: str = Field(..., json_schema_extra={"example": "Yes"})
    Dependents: str = Field(..., json_schema_extra={"example": "Yes"})
    tenure: int = Field(..., json_schema_extra={"example": 15})
    PhoneService: str = Field(..., json_schema_extra={"example": "Yes"})
    MultipleLines: str = Field(..., json_schema_extra={"example": "No phone service"})
    InternetService: str = Field(..., json_schema_extra={"example": "DSL"})
    OnlineSecurity: str = Field(..., json_schema_extra={"example": "No"})
    OnlineBackup: str = Field(..., json_schema_extra={"example": "Yes"})
    DeviceProtection: str = Field(..., json_schema_extra={"example": "No"})
    TechSupport: str = Field(..., json_schema_extra={"example": "Yes"})
    StreamingTV: str = Field(..., json_schema_extra={"example": "No"})
    StreamingMovies: str = Field(..., json_schema_extra={"example": "No"})
    Contract: str = Field(..., json_schema_extra={"example": "Month-to-month"})
    PaperlessBilling: str = Field(..., json_schema_extra={"example": "Yes"})
    PaymentMethod: str = Field(..., json_schema_extra={"example": "Electronic check"})
    MonthlyCharges: float = Field(..., json_schema_extra={"example": 70.0})
    TotalCharges: str = Field(..., json_schema_extra={"example": "1700"})

class PredictionOutput(BaseModel):
    churn_probability: float
    churn_prediction: int
    message: str
