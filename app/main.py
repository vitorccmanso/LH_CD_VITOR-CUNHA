import pandas as pd
from fastapi import FastAPI, Form, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.pipeline import PredictPipeline, CustomData

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
pipeline = PredictPipeline()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/manual_predict", response_class=HTMLResponse)
def render_manual_form(request: Request):
    return templates.TemplateResponse("manual_predict.html", {"request": request})

@app.post("/manual_predict", response_class=HTMLResponse)
def manual_predict(request: Request, bairro_group: str = Form(...),
                    bairro: str = Form(...),
                    latitude: float = Form(...),
                    longitude: float = Form(...),
                    room_type: str = Form(...),
                    minimo_noites: int = Form(...),
                    numero_de_reviews: int = Form(...),
                    reviews_por_mes: float = Form(...),
                    calculado_host_listings_count: int = Form(...)):
    custom_data = CustomData(bairro_group, bairro, latitude, longitude, room_type, minimo_noites, numero_de_reviews, reviews_por_mes, calculado_host_listings_count)
    data_df = custom_data.get_data_as_dataframe()
    prediction = pipeline.predict(data_df, manual=True)
    return templates.TemplateResponse("manual_predict.html", {"request": request, "predicted_class": prediction})

@app.get("/dataset_predict", response_class=HTMLResponse)
def render_dataset_form(request: Request):
    return templates.TemplateResponse("dataset_predict.html", {"request": request})

@app.post("/dataset_predict", response_class=HTMLResponse)
def predict_dataset(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        processed_df = pipeline.process_dataset(df)
        predictions = pipeline.predict(processed_df)
        return templates.TemplateResponse("dataset_predict.html", {"request": request, "predicted_classes": predictions})
    except ValueError as e:
        return templates.TemplateResponse("dataset_predict.html", {"request": request, "error_message": str(e)})