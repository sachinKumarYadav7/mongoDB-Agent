FROM python:3.10-slim

WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Copy the rest of the app
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["streamlit", "run", "main.py"]
