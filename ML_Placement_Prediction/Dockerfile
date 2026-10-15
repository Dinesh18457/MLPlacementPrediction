FROM python:3.11-slim
WORKDIR / app
COPY requirements.txt .
RUN pip install -no-cache-di -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "--blind", "0.0.0.0:5000","app:app"]