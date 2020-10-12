from tensorflow/tensorflow:1.14.0-py3
add . /server
workdir /server
env MODEL = '124M'
run pip install -r requirements.txt 
run python app.py
expose 8000

