'''

Flask implementation 

import tensorflow as tf
import gpt_2_simple as gpt2
import os
from flask import Flask, jsonify
import time
import uvicorn


# model_type = os.environ('MODEL')
model_type = '124M'
# gpt2.download_gpt2(model_name=model_type)   
sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, model_name=model_type)

app = Flask(__name__)


def generateText(text, length=500):
	print('generating .... ',text)
	global sess
	res = gpt2.generate(sess, model_name=model_type,length=100,prefix = "sample",return_as_list = True)
	return res[0]


@app.route('/get/<text>')
async def response(text):
	start_time = time.time()
	res = generateText(text)
	print(f'executed in {0} time', time.time() - start_time)
	return jsonify({'status': 200, 'text': "sample"})


if __name__ == '__main__':
	uvicorn.run(app, host='0.0.0.0', port = 8080)
	# print(generateText("sample"))

'''

from starlette.applications import Starlette
from starlette.responses import UJSONResponse
from starlette.responses import HTMLResponse
import gpt_2_simple as gpt2
import tensorflow as tf
import uvicorn
import os
import gc
import time


app = Starlette(debug=False)


if not os.path.isdir(os.path.join("models", '124M')):
	print(f"Downloading  model...")
	gpt2.download_gpt2(model_name="124M")   

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess,model_name = '124M')

response_header = {
    'Access-Control-Allow-Origin': '*'
}

generate_count = 0


@app.route('/random')
async def random_page(request):
	global sess
	params = request.query_params
	subject = params.get('sub',"Srujan")
	print('Generating text')
	start_time = time.time()
	text  = gpt2.generate(sess,model_name="124M",prefix = subject ,length = 500, return_as_list = True)[0]
	html_data = f""" 
				<h1> Random news about {subject} </h1>
				<br>
				<p> {text} </p>
				<h2> Time elapsed  : {time.time() - start_time } </h2<	
				"""
	page  = HTMLResponse(html_data)
	print('generated message',text)
	return page


@app.route('/', methods=['GET', 'POST', 'HEAD'])
async def homepage(request):
    global generate_count
    global sess

    if request.method == 'GET':
        params = request.query_params
    elif request.method == 'POST':
        params = await request.json()
    elif request.method == 'HEAD':
        return UJSONResponse({'text': ''},
                             headers=response_header)
    print('generating ')
    text = gpt2.generate(sess,
    					 model_name='124M',
                         length=int(params.get('length', 100)),
                         temperature=float(params.get('temperature', 0.7)),
                         top_k=int(params.get('top_k', 0)),
                         top_p=float(params.get('top_p', 0)),
                         prefix=params.get('prefix', '')[:500],
                         truncate=params.get('truncate', None),
                         include_prefix=str(params.get(
                             'include_prefix', True)).lower() == 'true',
                         return_as_list=True
                         )[0]

    generate_count += 1
    if generate_count == 8:
        tf.reset_default_graph()
        sess.close()
        sess = gpt2.start_tf_sess(threads=1)
        gpt2.load_gpt2(sess)
        generate_count = 0

    gc.collect()
    return UJSONResponse({'text': text},
                         headers=response_header)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
