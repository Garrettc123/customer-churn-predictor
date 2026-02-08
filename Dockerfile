FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train model on startup (or load pre-trained)
RUN python src/train_model.py

EXPOSE 8000

CMD ["python", "src/api.py"]
