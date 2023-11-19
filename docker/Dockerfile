FROM python:3.9.0
WORKDIR /digits

COPY . /digits/

RUN pip3 install -r /digits/requirements.txt

ENV FLASK_APP=api/prediction

EXPOSE 80

CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]