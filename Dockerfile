FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/services

COPY requirements-docker.txt /app/
RUN pip install --no-cache-dir -r /app/requirements-docker.txt

# Modell-súlyok előre letöltése build időben (a cwd, /app/services, innen
# tölti be futáskor is a YOLO('yolov8n.pt') hívás), hogy ne kelljen minden
# konténerindításkor újra letölteni / internet-függő legyen a futás.
# Ez a réteg csak a requirements-docker.txt-től függ, így src/ vagy
# services/ módosítása után a build nem tölti le újra a súlyokat.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

COPY src/ /app/src/
COPY services/ /app/services/
COPY models/ /app/models/
COPY data/team_colors.csv /app/data/team_colors.csv

ENV PYTHONUNBUFFERED=1
