import requests
from bs4 import BeautifulSoup
import json
import openai
import re
import nltk.data
from spellchecker import SpellChecker

# open ai key (do not upload to github)
openai.api_key = ''

api_key = "8mNcp3SVfrGqwZI9Oe0YivbxFPUhBMTk"  # core api key
api_endpoint = "https://api.core.ac.uk/v3/"


def check_words_in_string(string, words):
    for word in words:
        if word in string:
            return True
    return False


def beautify_string(text):
    words = nltk.word_tokenize(text)

    # Separate sticky words by adding white spaces
    beautified_words = []
    for word in words:
        if len(beautified_words) > 0 and beautified_words[-1][-1].isalpha() and word[0].isalpha():
            beautified_words.append(' ')
        beautified_words.append(word)

    # Join the beautified words with appropriate white spaces
    beautified_text = ''.join(beautified_words)

    return beautified_text


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
            # context = beautify_string(context)
            contexts.append(context)
            prev_sentence, sentence, next_sentence = '', '', ''

    return ' --- '.join(contexts)


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
    for i in results['results']:
        # if len(i['fullText']) > 2500:
        # article = {
        #     'title': i['title'],
        #     'text': i['fullText']
        # }
        articles_title.append(i['title'])
        articles_fulltext.append(i['fullText'])
    return articles_title, articles_fulltext


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


def davinci03(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200,
        temperature=0.3,
        n=1,
        stop='.'
    )
    return response.choices[0].text.strip()


def makeQ(target_sentence, field=''):
    # AND (_exists_:doi)
    # return f"fullText:{target_sentence} AND (_exists_:doi)"
    return f"{target_sentence} AND (_exists_:doi)"


def find_articles(sentence, limit=20):
    results = query_api("search/works", makeQ(sentence), limit=limit)
    articles_title, articles_fulltext = get_result(results)
    # articles_context = [find_sentence_contexts(
    #     text, sentence) for text in articles_fulltext]
    articles_context = []
    for i, v in enumerate(articles_fulltext):
        context = find_sentence_contexts(v, sentence)
        if context:
            articles_context.append(context)
        else:
            articles_context.append('')
            articles_title[i] = ''

    articles_context = [string for string in articles_context if string != '']
    articles_title = [string for string in articles_title if string != '']

    return articles_title, articles_context


# sentence = 'I have a dream'
# results = query_api("search/works", makeQ(sentence), limit=2)
# articles_title, articles_fulltext = get_result(results)
# contexts = [find_sentence_contexts(x, sentence) for x in articles_fulltext]
# print(contexts)

# summary = davinci03(contexts[0])
# print(summary)

sentence = 'I have a dream'
articles_title, articles_context = find_articles(sentence)
print()
