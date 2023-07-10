import requests
from bs4 import BeautifulSoup
import json
import openai
import re
import nltk.data

openai.api_key = ''  # open ai key (do not upload to github)

api_key = "8mNcp3SVfrGqwZI9Oe0YivbxFPUhBMTk"  # core api key
api_endpoint = "https://api.core.ac.uk/v3/"


def check_words_in_string(string, words):
    for word in words:
        if word in string:
            return True
    return False


def find_sentence_contexts(text, target_sentence):
    # Split the text into sentences
    target_sentence = target_sentence.lower()
    # sentences = re.split(
    #     r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text.lower())
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
            contexts.append(context)
            prev_sentence, sentence, next_sentence = '', '', ''

    return '\n'.join(contexts)


def pretty_json(json_object):
    print(json.dumps(json_object, indent=2))


def query_api(url_fragment, query, limit=10):
    headers = {"Authorization": "Bearer "+api_key}
    query = {"q": query,  "limit": limit}
    response = requests.post(
        f"{api_endpoint}{url_fragment}", data=json.dumps(query), headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error code {response.status_code}, {response.content}")


def get_result(results):
    results = results
    articles = []
    for i in results['results']:
        # if len(i['fullText']) > 2500:
        article = {
            'title': i['title'],
            'text': i['fullText']
        }
        articles.append(article)
    return articles


def summarize(contexts):
    # Prepare the search query
    summaries = []
    for i in contexts:
        query = f"Summarize the text in no more than 4 sentences: {i}"

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


def makeQ(target_sentence, field=''):
    return f"({target_sentence}) AND (_exists_:doi)"


sentence = 'I have a dream'
results = query_api("search/works", makeQ(sentence), limit=2)
full_text = get_result(results)[1]['text']
contexts = find_sentence_contexts(full_text, sentence)
print(contexts)
# summary = summarize(results)
# print(summary)
