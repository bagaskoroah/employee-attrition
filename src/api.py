from utils import *
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from path import *

# load config
config = load_config(config_path=CONFIG_PATH)

# create API object
app = FastAPI()

# load model
best_model = deserialize_object(path=BEST_MODEL_DIR/config['PATH_BEST_MODEL'])

# define input data structure
class DataAPI(BaseModel):
    ''' Represents the user input data structure. '''
    Age: int
    BusinessTravel: str
    DailyRate: int
    Department: str
    DistanceFromHome: int
    Education: str
    EducationField: str
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int

# define handlers
@app.post("/predict")
def predict(data: DataAPI):
    # convert DataAPI to Pandas DataFrame
    data = pd.DataFrame([data.dict()])

    # predict data
    predict_proba_res = best_model.predict_proba(data)[:, 1][0]

    # apply best threshold
    y_pred = int(predict_proba_res >= config['BEST_THRESHOLD_POINT'])

    if y_pred == 0:
        res = 'Attrition: No'
    else:
        res = 'Attrition: Yes'

    return {
        "prediction": y_pred,
        "probability": float(predict_proba_res),
        "label": res
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8080)