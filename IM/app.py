from flask import Flask, render_template, request
import openai
from bs4 import BeautifulSoup
import requests
from jinja2 import Environment
import string

app = Flask(__name__, static_url_path='/static')

# Set up OpenAI API key
# Replace with your OpenAI API key
openai.api_key = ""

# Home page


@app.route('/')
def home():
    return render_template('index.html')


def zip_lists(list1, list2):
    return zip(list1, list2)


env = Environment()
env.filters['zip_lists'] = zip_lists


@app.route('/citing_articles', methods=['POST'])
def citing_articles():
    # Get input article title from form submission
    find_quote = request.form['find_quote']

    # Perform a search to find citing articles
    citing_titles, citing_contexts = find_citing_articles(find_quote)
    summaries = summarize(citing_contexts)

    mylist = zip(citing_titles, summaries)

    if citing_titles:
        return render_template('citing_articles.html', input_title=input_article_title, articles=mylist)
    else:
        return render_template('no_citing_articles.html')


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
                {"role": "system", "content": "You are an assistant that summarizes text. Be descriptive"},
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


def strip_punctuation(text):
    translator = str.maketrans("", "", string.punctuation)
    stripped_text = text.translate(translator)
    return stripped_text


if __name__ == '__main__':
    app.run(debug=True)
