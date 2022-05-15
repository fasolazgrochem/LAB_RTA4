##init a base image
FROM python:3.9
#define the present working directory
WORKDIR /RTA_LAB4
# copy the cibtebts ubti the wirjubg dur
ADD . /RTA_LAB4
#run pip to install the dependencies of the flask app
RUN pip install -r requirements.txt
#define the command to start the container
CMD ["python","main.py","Modeldefaultdata.py","Modeluserdata.py"]