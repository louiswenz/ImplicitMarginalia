from flask import Flask, render_template, request, jsonify
import openai
from bs4 import BeautifulSoup
import requests
from jinja2 import Environment
import string
import re
import nltk.data

app = Flask(__name__, static_url_path='/static')
openai.api_key = ""


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        paragraphs = text.split('\r\n\r\n')

        sentences_all = split_sentence(paragraphs)

        # sentence_insights = []
        # for paragraph in sentences_all:
        #     for sentence in paragraph:
        #         citing_titles, citing_contexts = find_citing_articles(sentence)
        #         summaries = summarize(citing_contexts)
        #         sentence_insight = zip(citing_titles, summaries)
        #     sentence_insights.append(sentence_insight)

        return render_template('index.html', paragraphs=sentences_all)
    return render_template('index.html', paragraphs=[])


def split_sentence(paragraphs):
    sentences_all = []
    for paragraph in paragraphs:
        # Split text.
        sentences = nltk.tokenize.sent_tokenize(paragraph, language='english')
        sentences_all.append(sentences)
    return sentences_all


def strip_punctuation(input_string):
    # Remove leading and trailing whitespace and punctuation
    cleaned_string = input_string.strip(string.whitespace + string.punctuation)
    return cleaned_string


def find_citing_articles(citation_sentence):
    citation_sentence = strip_punctuation(citation_sentence)
    search_url = f"https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q={citation_sentence}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    articles_title = []
    articles_context = []

    results = soup.find_all('div', class_='gs_ri')

    for result in results:
        title_element = result.find('h3', class_='gs_rt')
        if title_element:
            title = title_element.text.strip()
            title = title.replace("[HTML]", "").strip()

            # Extracting the citation sentence and its context
            context_paragraph = result.find('div', class_='gs_rs').text
            context_paragraph = context_paragraph.replace("\n", "").strip()
            # Checking if the citation sentence is present in the context sentences
            if citation_sentence.lower() in context_paragraph.lower():
                articles_title.append(title)
                articles_context.append(context_paragraph)

    return articles_title, articles_context


def summarize(contexts):
    # Prepare the search query
    summaries = []
    for i in contexts:
        query = f"Summarize the following text: {i}"

        # Issue the search request
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that summarizes text. Be descriptive. Talk in first person"},
                {"role": "user", "content": query}
            ],
            max_tokens=100,
            stop=None,
            n=1,
            temperature=0.3
        )
        summary = response["choices"][0]["message"]["content"]
        summaries.append(summary)
    return summaries


@app.route('/process', methods=['POST'])
def process():
    # Get the input data from the request
    data = request.json
    input_text = data['original_text']

    articles_title, articles_context = find_citing_articles(input_text)
    print(articles_context)
    # Return the output as a JSON response
    return jsonify({'output_text': articles_context})


if __name__ == '__main__':
    app.run(debug=True)
