import urllib
import os
from xml.etree import ElementTree as ET
import openai
from scipy import spatial
import pandas as pd
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

if not os.path.exists('data'):
    os.mkdir('data')

openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI()
embedding_model = "text-embedding-ada-002"
gpt_model = "gpt-3.5-turbo"

def fetch_papers():

    """Fetches papers from the arXiv API and returns them as a list of strings."""

    url = 'http://export.arxiv.org/api/query?search_query=ti:llama&start=0&max_results=70'

    response = urllib.request.urlopen(url)

    data = response.read().decode('utf-8')

    root = ET.fromstring(data)



    papers_list = []

    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):

        title = entry.find('{http://www.w3.org/2005/Atom}title').text

        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text

        paper_info = f"Title: {title}\nSummary: {summary}\n"

        papers_list.append(paper_info)



    return papers_list

def create_embedding(papers_list):
    embeddings = []
    for paper in papers_list:
        response = client.embeddings.create(model=embedding_model, input=[paper])
        #print(type(response.data[0]))
        vectors = response.data[0].embedding
        #print(len(vectors))
        embeddings.append(vectors)
    df = pd.DataFrame({"text":papers_list,"embedding":embeddings})
    df.to_csv('data/embeddings.csv')
    return df

def get_question_embedding(question):
    res = client.embeddings.create(
        input=[question],
        model=embedding_model
    )
    query_embedding = [record.embedding for record in res.data]
    return query_embedding[0]

def get_context_index(query, df):
    relatedness = df["embedding"].apply(lambda x:1-spatial.distance.cosine(query,x))
    relatedness.sort_values(ascending=False, inplace=True)
    return relatedness.index[:10]


def prepare_gpt_context(question, df):
    query = get_question_embedding(question)
    indices = get_context_index(query, df)
    context = question
    for idx in indices:
    # Create a single context string from the chunks, the query and the question
        context += '\n\n' + df["text"][idx]
    context += '\n\n' + question
    return context


def generate_response(context):
    response = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "user", "content": context}
            ]
    )
    return response.choices[0].message.content#.text.strip()

def chat_with_openai(question,df):
    #query = get_query_embedding(question)
    context = prepare_gpt_context(question, df)
    response = generate_response(context)
    return response



