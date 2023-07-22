from flask import Flask, render_template, request, jsonify
import openai
from bs4 import BeautifulSoup
import requests
from jinja2 import Environment
import string
import re
import nltk.data
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from cachetools import cached, TTLCache
from newsapi import NewsApiClient
import newspaper

# core api
api_key = "8mNcp3SVfrGqwZI9Oe0YivbxFPUhBMTk"  # core api key
api_endpoint = "https://api.core.ac.uk/v3/"

# news api
newsapi = NewsApiClient(api_key='fa9f684c11bf41e9917bed3fe109a308')

# openai
openai.api_key = ""

app = Flask(__name__, static_url_path='/static')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        paragraphs = text.split('\r\n\r\n')

        sentences_all = split_sentence(paragraphs)

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


def find_citing_articles(citation_sentence):  # not using

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


def summarize(contexts):  # not using
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


def makeQ(target_sentence, field='Sociology'):
    # (fullText:{target_sentence})
    # AND (fieldOfStudy:{field})
    return f"{target_sentence} AND (_exists_:doi) AND (fullText:{target_sentence})"


def check_noise_in_string(sentence, noise):
    sentence = sentence.lower()
    for n in noise:
        if n in sentence:
            return True
    return False


def beautify_string(text):
    text = text.replace('-', '')
    text = re.sub(
        r'(?<=[a-z])(?=[A-Z0-9])|(?<=[A-Z0-9])(?=[A-Z][a-z])', ' ', text)

    return text


def get_article_fromurl(url):
    # Initialize the Article object
    article = newspaper.Article(url)

    # Download and parse the article
    article.download()
    article.parse()

    # Return the article's main content
    return article.text


@cached(cache=TTLCache(maxsize=100, ttl=300))
def query_api(url_fragment, query, limit=2):
    headers = {"Authorization": "Bearer "+api_key}
    query = {"q": query,  "limit": limit}
    response = requests.post(
        f"{api_endpoint}{url_fragment}", data=json.dumps(query), headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error code {response.status_code}, {response.content}")


def getnewsapi(q, limit=10):
    # q = '"' + q + '"'
    response = newsapi.get_everything(q=q,
                                      language='en',
                                      sort_by='popularity',
                                      page=1,
                                      page_size=limit)
    if response['status'] != 'ok':
        print("Bad Request")

    return response


def core_get_results(results, target_sentence):
    articles_title = []
    articles_context = []
    for i in results['results']:
        context = find_sentence_contexts(i['fullText'], target_sentence)
        if context:
            articles_title.append(i['title'])
            context = find_most_opinionated_paragraph(context)
            context = beautify_string(context)
            articles_context.append(context)
    return articles_title, articles_context


def news_get_results(response, target_sentence):
    titles = []
    contexts = []
    for v in response['articles']:
        article = get_article_fromurl(v['url'])
        context = find_sentence_contexts(article, target_sentence)
        if context:
            titles.append(v['title'])
            context = find_most_opinionated_paragraph(context)
            context = beautify_string(context)
            contexts.append(context)
    return titles, contexts


def find_sentence_contexts(text, target_sentence):
    # Split the text into sentences
    target_sentence = target_sentence.lower()
    sentences = nltk.tokenize.sent_tokenize(text, language='english')
    # Find the target sentence and its context
    noise = ['pdf', 'doi', 'copyright', 'https',
             'all rights reserved', 'http://', '©']
    contexts = []
    for i, sentence in enumerate(sentences):
        if check_noise_in_string(sentence, noise):
            continue
        if target_sentence in sentence.lower():
            prev_sentence = sentences[i - 1] if i > 0 else ""
            next_sentence = sentences[i + 1] if i < len(sentences) - 1 else ""
            if check_noise_in_string(prev_sentence, noise) or check_noise_in_string(next_sentence, noise):
                prev_sentence, next_sentence = '', ''
                continue
            context = prev_sentence + " " + sentence + " " + next_sentence
            context = context.strip()
            context = beautify_string(context)
            contexts.append(context)
            prev_sentence, sentence, next_sentence = '', '', ''

    return '<br><br>'.join(contexts)


def find_most_opinionated_paragraph(text):
    # Split text into paragraphs
    paragraphs = text.split("<br><br>")

    # Initialize Sentiment Intensity Analyzer
    sia = SentimentIntensityAnalyzer()

    # Find the paragraph with the highest sentiment score
    max_sentiment_score = -1
    most_opinionated_paragraph = ""

    for paragraph in paragraphs:
        if paragraph.strip():
            sentiment_score = sia.polarity_scores(paragraph)['compound']
            if sentiment_score > max_sentiment_score:
                max_sentiment_score = sentiment_score
                most_opinionated_paragraph = paragraph

    return most_opinionated_paragraph


def find_articles(target_sentence, limit=5):
    target_sentence = strip_punctuation(target_sentence)
    core_results = query_api(
        "search/works", makeQ(target_sentence), limit=limit)
    news_response = getnewsapi(f'"{target_sentence}"', limit=limit)
    core_articles_title, core_articles_context = core_get_results(
        core_results, target_sentence)
    news_articles_title, news_articles_context = news_get_results(
        news_response, target_sentence)

    return core_articles_title+news_articles_title, core_articles_context+news_articles_context


@app.route('/process', methods=['POST'])
def process():
    # Get the input data from the request
    data = request.json
    # data cleaning
    input_text = data['original_text']
    input_text = input_text.replace('\n', '').strip()
    articles_title, articles_context = find_articles(input_text)

    # Return the output as a JSON response
    return jsonify({'output_text': articles_context, 'output_title': articles_title})


if __name__ == '__main__':
    app.run(debug=True, port=5002)
