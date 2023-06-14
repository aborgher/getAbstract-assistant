import streamlit as st
import openai
import tiktoken
from sentence_transformers import SentenceTransformer
import gc
from pymilvus import Collection, connections
import time


openai_key = st.secrets["openai_key"]
milvus_uri = st.secrets["milvus_uri"]
milvus_user = st.secrets["milvus_user"]
milvus_pwd = st.secrets["milvus_pwd"]

openai.api_key = openai_key
GPTMODEL = 'gpt-3.5-turbo'
MAX_TOKEN_LIMIT = 4096 if GPTMODEL == 'gpt-3.5-turbo' else 8192
MAX_OUTPUT_TOKEN = 250
token_counter = tiktoken.encoding_for_model(GPTMODEL)


def num_tokens_from_string(text):
    return len(token_counter.encode(text))


def search_collection(query, collection_name, model, expr="", topk=100):
    start_time = time.time()
    collection = Collection(collection_name)
    search_vec = model.encode(query)
    search_params = {"metric_type": "IP",  "params": {"level": 2}}
    results = collection.search(
        [search_vec],
        anns_field="text_vector",
        param=search_params,
        limit=topk,
        expr=expr,
        output_fields=["text", "summaryid"]
    )[0]
    min_distance = 0.25
    parsed_results = [r for r in results if r.distance >= min_distance]
    end_time = time.time()
    st.write(f"time spend to query the vector database: {end_time - start_time:.2} seconds")
    return [(r.entity.get('summaryid'), r.entity.get('text')) for r in parsed_results]


def get_system_text(context, query):
    return f"""You are given the following extracted parts of a document and a question. Provide an answer based only on the context provided.
If you can't find the answer in the context below, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not related to the context, or the context is empty, politely respond that you are tuned to only answer questions that are related to the context.
Answer in the same language of the question provided by the user. Answer in Markdown.
Context: {context}

Question: {query}"""


def get_context(sources, query):
    context = ""
    ids = []
    for i, s in sources:
        additional_context = f"\n{s}"
        if num_tokens_from_string(get_system_text(context + additional_context, query)) > (MAX_TOKEN_LIMIT - MAX_OUTPUT_TOKEN):
            return context, set(ids)
        else:
            context += additional_context
            ids.append(i)
    return context, set(ids)


def get_answer(context, query):
    start_time = time.time()
    response = openai.ChatCompletion.create(
        model=GPTMODEL,
        messages=[
            {"role": "system", "content": get_system_text(context, query)}
        ],
        max_tokens=MAX_OUTPUT_TOKEN,
        temperature=0.2
    )
    end_time = time.time()
    st.write(f"GPT time to give us a full answer: {end_time - start_time:.2} seconds")
    return response["choices"][0]["message"]["content"]


def create_answer(sources, query):
    context, ids = get_context(sources, query)
    return get_answer(context, query), ids


@st.cache_resource
def connect_to_milvus():
    connections.connect("default", uri=milvus_uri, user=milvus_user, password=milvus_pwd, secure=True)


@st.cache_resource
def get_all_summaries_available():
    collection = Collection("summary_text")
    res = collection.query(
      expr='(summaryid >= 0) and (language == "en") and (publication_date > 2017)',
      output_fields=["summaryid", "title"]
    )
    return {r["summaryid"]: r["title"] for r in res}


def get_text_summaryid(summaryid):
    collection = Collection("summary_text")
    res = collection.query(
        expr=f'summaryid in [{summaryid}]',
        output_fields=["text"],
        consistency_level="Strong"
    )
    return res[0]["text"]


st.title('getAbstract assistant')
st.caption('Please remember to close this page in your browser when you have finished to question the book.')

if 'model' not in st.session_state:
    st.session_state['model'] = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

connect_to_milvus()
if 'titles' not in st.session_state:
    st.session_state['titles'] = get_all_summaries_available()

query = st.text_input("Enter a standalone question here (the bot is not aware of the previous questions asked, describe your context as much as possible).")
mode = st.selectbox("Control the sources.", ["getabstract library", "fixed summary"])
if mode == "fixed summary":
    summary_title = st.selectbox("choose the summaryid you want to use as source", st.session_state['titles'].values())

if query:
    start_time = time.time()
    if mode == 'fixed summary':
        summary_id = {v: k for k, v in st.session_state['titles'].items()}[summary_title]
        sources = [(summary_id, get_text_summaryid(summary_id))]
    else:
        sources = search_collection(query, "extracted_takeaway", st.session_state['model'], f'language in ["en"]')
    answer, ids = create_answer(sources, query)
    st.markdown(answer, unsafe_allow_html=True)
    summaries_used = "using these summaries as context:  \n"
    for ix in ids:
        summaries_used += f"[{st.session_state['titles'][ix]}](https://www.getabstract.com/en/summary/test/{ix})  \n"
    st.markdown(summaries_used)
    end_time = time.time()
    st.write(f"Overall time passed: {end_time - start_time:.2} seconds")
    del answer, sources, summaries_used, ids
    gc.collect()
