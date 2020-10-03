import tensorflow as tf
import gpt_2_simple as gpt2
import os
from flask import Flask, jsonify

# model_type = os.environ('MODEL')
model_type = '124M'

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, model_name=model_type)

app = Flask(__name__)


def generateText(start=None, length=500):
    text = gpt2.generate(sess, model_name=model_type,
                         prefix=start, length=500)[0]
    return text


@app.route('/get/<text>')
async def response(text):
    res = generateText(text)
    return jsonify({'status': 200, 'text': res})
