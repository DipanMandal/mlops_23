FROM python:3.9.0
WORKDIR /digits

COPY . /digits/

RUN pip3 install -r /digits/requirements.txt

CMD ["sh", "-c", "python exp.py"]
