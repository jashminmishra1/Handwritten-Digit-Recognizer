import os
import Train
import numpy as np
from flask import Flask, send_file, request, redirect

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/index')
def hello_world():
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

# post-data: image array of length 784 with elements [0.0 - 1.0]
@app.route('/index', methods=['POST'])
def digit():
    if 'file' not in request.files:
        return redirect('/index')

    file = request.files['file']

    if file.filename == '':
        return redirect('/index')
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(filename)
        result = Train.recognise_image(filename)
        return '''
        <!doctype html>
        <title>Result</title>
        <h1>Digits Recognised Are</h1>
        <p>'''+result+'''</p>
        '''


if __name__ == '__main__':
    app.run()
