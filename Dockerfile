from tensorflow/tensorflow:1.14.0-py3
add . /server
workdir /server
env MODEL = '124M'
run pip3 install -r requirements.txt && server.py


