from pydantic import BaseModel


class HeartData(BaseModel):
        Age: float
        Sex: str
        ChestPainType: str
        RestingBP: float
        Cholesterol: float
        FastingBS: int
        RestingECG: str
        MaxHR: float
        ExerciseAngina: str
        Oldpeak: float
        ST_Slope: str 