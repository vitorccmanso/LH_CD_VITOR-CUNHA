FROM python:3.11
WORKDIR /rent_pred
COPY /requirements.txt /rent_pred/requirements.txt
RUN pip install -r /rent_pred/requirements.txt
COPY /app /rent_pred/app
COPY /artifacts /rent_pred/app/artifacts

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]