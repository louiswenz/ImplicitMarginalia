from flask import Flask, render_template, request, jsonify
import openai
from bs4 import BeautifulSoup
import requests
from jinja2 import Environment
import string
import re
import nltk.data
import json

# core api
api_key = "8mNcp3SVfrGqwZI9Oe0YivbxFPUhBMTk"  # core api key
api_endpoint = "https://api.core.ac.uk/v3/"

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

# new


def makeQ(target_sentence, field=''):
    return f"({target_sentence}) AND (_exists_:doi)"


def check_words_in_string(string, words):
    for word in words:
        if word in string:
            return True
    return False


def beautify_string(text):
    text = text.replace('-', '')
    text = re.sub(
        r'(?<=[a-z])(?=[A-Z0-9])|(?<=[A-Z0-9])(?=[A-Z][a-z])', ' ', text)

    return text


def query_api(url_fragment, query, limit=2):
    headers = {"Authorization": "Bearer "+api_key}
    query = {"q": query,  "limit": limit}
    response = requests.post(
        f"{api_endpoint}{url_fragment}", data=json.dumps(query), headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error code {response.status_code}, {response.content}")


def get_result(results):
    articles_title = []
    articles_fulltext = []
    if results == None:
        return articles_title, articles_fulltext
    for i in results['results']:
        # if len(i['fullText']) > 2500:  #filter some text
        articles_title.append(i['title'])
        articles_fulltext.append(i['fullText'])
    return articles_title, articles_fulltext


def find_sentence_contexts(text, target_sentence):
    # Split the text into sentences
    target_sentence = target_sentence.lower()
    sentences = nltk.tokenize.sent_tokenize(text.lower(), language='english')
    # Find the target sentence and its context
    noise = ['pdf', 'doi', 'copyright', 'https']
    contexts = []
    for i, sentence in enumerate(sentences):
        if check_words_in_string(sentence, noise):
            continue
        if target_sentence in sentence:
            prev_sentence = sentences[i - 1] if i > 0 else ""
            next_sentence = sentences[i + 1] if i < len(sentences) - 1 else ""
            if check_words_in_string(prev_sentence, noise) or check_words_in_string(next_sentence, noise):
                continue
            context = prev_sentence + " " + sentence + " " + next_sentence
            context = context.strip()
            context = beautify_string(context)
            contexts.append(context)
            prev_sentence, sentence, next_sentence = '', '', ''

    return '<br>'.join(contexts)


def find_articles(sentence, limit=2):
    sentence = strip_punctuation(sentence)
    results = query_api("search/works", makeQ(sentence), limit=limit)
    articles_title, articles_fulltext = get_result(results)
    articles_context = [find_sentence_contexts(
        text, sentence) for text in articles_fulltext]
    return articles_title, articles_context


@app.route('/process', methods=['POST'])
def process():
    # Get the input data from the request
    data = request.json
    input_text = data['original_text']
    input_text = input_text.replace('\n', '').strip()
    articles_title, articles_context = find_articles(input_text)

    # Return the output as a JSON response
    return jsonify({'output_text': articles_context, 'output_title': articles_title, })


if __name__ == '__main__':
    app.run(debug=True)
