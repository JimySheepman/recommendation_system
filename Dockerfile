FROM python:latest
RUN apt-get update
ADD . /opt/webapp/
WORKDIR /opt/webapp 
RUN pip install -r requirements.txt
ENV FLASK_APP=main.py 
EXPOSE 5000 
CMD ["flask", "run", "--host=0.0.0.0"] 