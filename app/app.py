from flask import Flask, request, jsonify
import os
import subprocess
from werkzeug.utils import secure_filename


app = Flask(__name__)

@app.route('/')
def render_home():
    return "hello world"


if __name__ == '__main__':
    app.run(debug=True)
