import streamlit as st
import openai
import tiktoken
from sentence_transformers import SentenceTransformer
import gc
from pymilvus import Collection, connections
import requests

openai_key = st.secrets["openai_key"]
milvus_uri = st.secrets["milvus_uri"]
milvus_user = st.secrets["milvus_user"]
milvus_pwd = st.secrets["milvus_pwd"]
ask_miso_qa_url = st.secrets["ask_miso_qa_url"]

openai.api_key = openai_key
GPTMODEL = 'gpt-3.5-turbo'
MAX_TOKEN_LIMIT = 4096 if GPTMODEL == 'gpt-3.5-turbo' else 8192
MAX_OUTPUT_TOKEN = 350
MAX_NUM_SUMMARIES = 4
token_counter = tiktoken.encoding_for_model(GPTMODEL)
use_ask_miso_qa = False


def get_run_gpt_prompt(prompt, in_text, model='gpt-4'):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt}
            , {"role": "user", "content": in_text}
        ],
        max_tokens=min(MAX_TOKEN_LIMIT - num_tokens_from_string(prompt + "\n" + in_text) - 150, MAX_OUTPUT_TOKEN),
        temperature=0.1
    )
    return response["choices"][0]["message"]["content"]


def num_tokens_from_string(text):
    return len(token_counter.encode(text))


def search_collection(query, collection_name, model, expr="", topk=100, ids_to_ignore={}):
    collection = Collection(collection_name)
    search_vec = model.encode(query)
    search_params = {"metric_type": "IP", "params": {"level": 2}}
    results = collection.search(
        [search_vec],
        anns_field="text_vector",
        param=search_params,
        limit=topk,
        expr=expr,
        output_fields=["text", "summaryid"]
    )[0]
    min_distance = 0.4
    parsed_results = [r for r in results if
                      (r.distance >= min_distance) & (r.entity.get('summaryid') not in ids_to_ignore)]
    return [(r.entity.get('summaryid'), r.entity.get('text')) for r in parsed_results[0:MAX_NUM_SUMMARIES]]


def get_sources_from_ask_miso_qa(question):
    d = {
        "version": "v1.3",
        "q": question,
        "min_probability": 0.3,
        "rows": 10
    }
    x = requests.post(ask_miso_qa_url, json=d)
    xj = x.json()
    return [(a['product_id'], a['answer']['text']) for a in xj['data']['answers']] if xj['message'] == 'success' else []


def get_prompt(sources):
    exps = "\n\n".join([f'[{i + 1}]:\n{s}' for i, s in enumerate([s[1] for s in sources])])
    return f"""Based on the sources below, answer to the user question. For each phrase in the answer, provide a reference to the sources using the [].
If you can't find the answer in the sources or if the question is not related to the sources, politely respond that you can't answer that.
Answer in the same language of the question provided by the user.

Sources:
{exps}
"""


def limit_sources(sources, query):
    i = 0
    for i, _ in enumerate(sources):
        if num_tokens_from_string(get_prompt(sources[0: i]) + f"\n{query}") > (MAX_TOKEN_LIMIT - MAX_OUTPUT_TOKEN):
            break
    return sources[0: i]


def create_answer(sources, query):
    source_sel = limit_sources(sources, query)
    return get_run_gpt_prompt(get_prompt(source_sel), query, GPTMODEL), [s[0] for s in source_sel]


def create_follow_up_questions(in_text):
    prompt = "You are an AI assitant which given an input question produces 5 distinct follow up questions related to that text. The follow up questions should be different from the input question. Provide the questions as a bullet list using - as marker."
    out_text = get_run_gpt_prompt(prompt, in_text, GPTMODEL)
    return out_text.split('\n')


@st.cache_resource
def connect_to_milvus():
    connections.connect("default", uri=milvus_uri, user=milvus_user, password=milvus_pwd, secure=True)


@st.cache_resource
def get_summaries_in_milvus():
    collection = Collection("extracted_takeaway")
    res = collection.query(
        expr='summaryid >= 0',
        output_fields=["summaryid"]
    )
    return {r["summaryid"] for r in res}


@st.cache_resource
def get_all_summaries_available():
    collection = Collection("summary_text")
    res = collection.query(
        expr='(summaryid >= 0) and (language == "en") and (publication_date >= 2017)',
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
# st.caption('Please remember to close this page in your browser when you have finished to question the book.')

if 'model' not in st.session_state:
    st.session_state['model'] = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

connect_to_milvus()
if 'titles' not in st.session_state:
    st.session_state['titles'] = get_all_summaries_available()
    if 'restricted_ids' not in st.session_state:
        st.session_state['restricted_ids'] = {i for i in get_summaries_in_milvus() if
                                              i not in st.session_state['titles'].keys()}

query = st.text_input("Enter a question")
# "Enter a standalone question (the bot is not aware of the previous questions asked)."

if query:
    if use_ask_miso_qa:
        sources = get_sources_from_ask_miso_qa(query)
    else:
        sources = search_collection(query, "extracted_takeaway", st.session_state['model'], f'language in ["en"]',
                                    100, st.session_state['restricted_ids'])
    answer, ids = create_answer(sources, query)
    st.markdown(answer, unsafe_allow_html=True)
    summaries_used = "My reply is based on the following:  \n"
    for i, ix in enumerate(ids):
        title = f"summaryid={ix}" if ix not in st.session_state['titles'].keys() else st.session_state['titles'][ix]
        link = f"https://www.getabstract.com/en/summary/test/{ix}"
        summaries_used += f"[{title}]({link})[{i+1}]  \n"
    st.markdown(summaries_used)

    follow_up_questions = "  \nFollow up questions:  \n"
    fw_qs = create_follow_up_questions(query)
    for fq in fw_qs:
        follow_up_questions += f"*{fq}*  \n"
    st.markdown(follow_up_questions, unsafe_allow_html=True)
    del answer, sources, summaries_used, ids, fw_qs, follow_up_questions
    gc.collect()
