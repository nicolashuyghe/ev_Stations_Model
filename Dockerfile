
FROM python:3.8.6-buster


COPY model.joblib /model.joblib
COPY model_2.joblib /model_2.joblib
COPY ev_Stations_Model /ev_Stations_Model
COPY api /api
COPY requirements.txt /requirements.txt


RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
