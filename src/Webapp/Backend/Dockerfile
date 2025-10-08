# base image
FROM python:3.12-slim

# setting a working directory
WORKDIR /app

#copying requirements and installing dependeces
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python","backend.py"]