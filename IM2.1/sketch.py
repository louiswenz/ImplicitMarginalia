import scholarly
import requests
from bs4 import BeautifulSoup
import openai
import nltk.data

# Set up OpenAI API key
# Replace with your OpenAI API key
openai.api_key = "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"


def find_citing_articles1(citation_sentence):
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


_sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')


def split_sentence(text):
    # Split text.
    sentences = _sent_detector.tokenize(text)
    # Find each sentence's offset.
    needle = 0
    triples = []
    for sent in sentences:
        start = text.find(sent, needle)
        end = start + len(sent) - 1
        needle += len(sent)
        triples.append(sent)
    # Return results
    return sentences


# Example usage
text = "Some essay samples below are by students who chose to write about a challenge, while other examples may be helpful if you’re looking to write about yourself more generally. And yes, a few of these essays did help these students get accepted into the Ivy League, (I’m not telling you which!) though these are all great essays regardless of where (or if) students were admitted to their top choice school."
print(nltk.tokenize.sent_tokenize(text, language='english'))
