import boto3
import json
import streamlit as st

from botocore.exceptions import ClientError
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_community.llms import OpenAI

def get_secret():
    secret_name = 'aws-managed-openai-secret'
    region_name = 'ap-southeast-1'

    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise e

    secret = json.loads(get_secret_value_response['SecretString'])
    return secret['OPENAI_API_KEY']

def prediction_pipeline(text):
    text_splitter=CharacterTextSplitter(separator='\n',
                                        chunk_size=1000,
                                        chunk_overlap=20)
    text_chunks=text_splitter.split_text(text)
    print(len(text_chunks))

    llm = OpenAI(openai_api_key=get_secret())

    docs = [Document(page_content=t) for t in text_chunks]
    chain=load_summarize_chain(llm=llm, chain_type='map_reduce', verbose=True)
    summary = chain.run(docs)

    return summary

user_input = st.text_area('Enter text to summarize')
button = st.button('Generate Summary')
if user_input and button:
    summary = prediction_pipeline(user_input)
    st.write('Summary:\n', summary)