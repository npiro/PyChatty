# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 17:06:06 2016

@author: piromast
"""

from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    return 'Index Page'

@app.route('/hello')
def hello():
    return 'Hello, World'