FROM python:3.8


RUN apt update 
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN pip install --upgrade pip

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN python3 -m pip install -r ./requirements.txt

EXPOSE 8501

COPY . .

ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]