from flask import Flask, render_template, request
import openai
from bs4 import BeautifulSoup
import requests
from jinja2 import Environment
import string
import re

app = Flask(__name__, static_url_path='/static')

# Set up OpenAI API key


def extract_sentences(text):
    pattern = r'("[^"]*"[.!?])'
    sentences = re.findall(pattern, text)
    return [sentence.strip() for sentence in sentences]


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form.get('text')
        paragraphs = text.split('\n\n')

        # sentences_all = []
        for paragraph in paragraphs:
            sentences = []
            paragraph = re.split(r'(?<=[?!\.])\s', paragraph)
            for sentence in paragraph:
                if sentence.strip():
                    sentences.append(sentence.strip())
        #     sentences_all.append(sentences)
        # print(sentences_all)

        return render_template('index.html', sentences=sentences)
    return render_template('index.html', sentences=[])


if __name__ == '__main__':
    app.run(debug=True)
