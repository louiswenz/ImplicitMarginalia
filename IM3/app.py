from flask import Flask, render_template, request, jsonify
import openai
import requests
import string
import nltk.data
import nltk
from cachetools import cached, TTLCache
from newsapi import NewsApiClient
import time

# core api
api_key = "8mNcp3SVfrGqwZI9Oe0YivbxFPUhBMTk"  # core api key
api_endpoint = "https://api.core.ac.uk/v3/"

# news api
newsapi = NewsApiClient(api_key='fa9f684c11bf41e9917bed3fe109a308')

# openai
openai.api_key = "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"

# bing web search
subscription_key = "fb1daf6e834947edba318a368a24b620"
assert subscription_key
bing_search_url = "https://api.bing.microsoft.com/v7.0/search"

app = Flask(__name__, static_url_path='/static')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        global input_text
        input_text = request.form['text']  # input text
        global Category
        # selected category(field of study)
        Category = request.form['category']

        sentences = nltk.tokenize.sent_tokenize(input_text, language='english')
        grouped_text = [sentences[i:i+10]
                        for i in range(0, len(sentences), 10)]

        return render_template('index.html', paragraphs=grouped_text)
    return render_template('index.html', paragraphs=[])


def strip_punctuation(input_string):
    # Remove leading and trailing whitespace and punctuation
    cleaned_string = input_string.strip(string.whitespace + string.punctuation)
    return cleaned_string

# new


@cached(cache=TTLCache(maxsize=100, ttl=300))
def bing_web_search(search_term, limit):
    search_term = "'" + search_term + "'"
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": search_term, "textDecorations": False,
              "textFormat": "HTML", "count": limit, 'responseFilter': ['Webpages']}
    response = requests.get(bing_search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    search_results = search_results["webPages"]["value"]
    names, snippets = [], []
    for i in search_results:
        if (('act' in i["name"].lower()) and ('scene' in i["name"].lower())) or (('google translate' in i["name"].lower())):
            continue
        names.append((i["name"]))
        snippets.append(i["snippet"])
    return names, snippets


def filter_texts_by_field_of_study(texts, titles, field_of_study):
    # Prepare the prompt to instruct GPT-3 for text filtering
    prompt = f"You will only find for comments about King Lear by Shakespeare. Filter the following texts that is in the field for '{field_of_study}':\n\n"
    for text, title in zip(texts, titles):
        prompt += f"Title: {title}\n"
        prompt += f"Text: {text}\n\n"

    # Make API call to GPT-3
    response = openai.Completion.create(
        engine="text-davinci-002",  # GPT-3 engine
        prompt=prompt,
        max_tokens=1000,  # Maximum number of tokens in the response
        temperature=0.1,  # Controls the randomness of the response
    )

    # Extract the filtered texts and titles from the response
    filtered_texts_and_titles = response['choices'][0]['text'].split("\n\n")
    filtered_texts_and_titles = [x.strip()
                                 for x in filtered_texts_and_titles if x.strip()]
    if len(filtered_texts_and_titles) == 0:
        print("No related contexts.")
        return ["No Reults"], ["No Reults"]
    # Separate the filtered texts and titles into separate lists
    filtered_texts = []
    filtered_titles = []
    for i in filtered_texts_and_titles:
        info = i.split('\n')
        filtered_titles.append(info[0].replace("Title: ", ""))
        filtered_texts.append(info[1].replace("Text: ", ""))

    return filtered_titles, filtered_texts


def get_filtered_contexts(sentence_ls, cat, eachLimit=8):
    generated_content_ls = []
    generated_content_title_ls = []

    for sentence in sentence_ls:
        names, snippets = bing_web_search(sentence, eachLimit)
        generated_content_ls.extend(snippets)
        generated_content_title_ls.extend(names)

    if (cat == 'None'):
        return generated_content_title_ls[:10], generated_content_ls[:10]

    filtered_titles, filtered_content = filter_texts_by_field_of_study(
        generated_content_ls, generated_content_title_ls, cat)

    return filtered_titles, filtered_content


@app.route('/process', methods=['POST'])
def process():
    # Get the input data from the request
    start_time = time.time()
    data = request.json
    sentence_ls = data['original_text']

    sentence_ls = [sent.replace('\n', '').strip() for sent in sentence_ls]
    # articles_title, articles_context = find_articles(input_text, limit=3)

    articles_title, articles_context = get_filtered_contexts(
        sentence_ls, Category)
    end_time = time.time()
    print(f"Elapsed time: {end_time-start_time} seconds")
    # Return the output as a JSON response
    return jsonify({'output_text': articles_context, 'output_title': articles_title, 'category': Category})


if __name__ == '__main__':
    app.run(debug=True, port=5002)
