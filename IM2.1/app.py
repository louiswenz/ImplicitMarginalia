from flask import Flask, render_template, request
import openai
from bs4 import BeautifulSoup
import requests
from jinja2 import Environment
import string
import re

app = Flask(__name__, static_url_path='/static')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        paragraphs = text.split('\n\n')

        sentences_all = []
        for paragraph in paragraphs:
            sentences = []
            paragraph = re.split(r'(?<=[?!\.])\s', paragraph)
            for sentence in paragraph:
                if sentence.strip():
                    sentences.append(sentence.strip())
            sentences_all.append(sentences)
        print(sentences_all)

        # sentences = [paragraph.split('. ') for paragraph in paragraphs]
        return render_template('index.html', paragraphs=sentences_all)
    return render_template('index.html', paragraphs=[])


if __name__ == '__main__':
    app.run()
