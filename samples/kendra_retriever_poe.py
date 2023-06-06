from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
import poe
from aws_langchain.kendra_index_retriever import KendraIndexRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import time


class PoeLLM(LLM):
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        client = connect_poe()
        model_dic = {
            'claude': 'a2',
            'chatgpt': 'chinchilla',
            'sage': 'capybara',
            'claude+': 'a2_2',
            'gpt4': 'beaver',
            'claude-instant-100k': 'a2_100k'
        }
        model_name = os.environ["POE_MODEL"]
        prompt_length = len(prompt)
        chunk_lst = [
            chunk["text_new"] for chunk in client.send_message(model_dic[model_name], prompt, with_chat_break=True)
            ]
        # print(chunk_lst)
        response = ''.join(chunk_lst).strip()

        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        model_name = os.environ["POE_MODEL"]
        return {"name_of_poe_model": model_name}
    
    @property
    def _llm_type(self) -> str:
        return "custom"


def connect_poe():
    token = os.environ["POE_TOKEN"]
    client = poe.Client(token)
    return client


def disconnect_poe():
    model_dic = {
        'claude': 'a2',
        'chatgpt': 'chinchilla',
        'sage': 'capybara',
        'claude+': 'a2_2',
        'gpt4': 'beaver',
        'claude-instant-100k': 'a2_100k'
    }
    model_name = os.environ["POE_MODEL"]
    client = connect_poe()
    try:
        client.send_chat_break(model_dic[model_name])
    except Exception as e:
        pass


def build_chain():
    region = os.environ["AWS_REGION"]
    kendra_index_id = os.environ["KENDRA_INDEX_ID"]

    llm = PoeLLM()
        
    retriever = KendraIndexRetriever(kendraindex=kendra_index_id, 
        awsregion=region, 
        return_source_documents=True)

    prompt_template = """
    下面是一段人与 AI 的友好对话。 
    AI 很健谈，并根据其上下文提供了许多具体细节。
    如果 AI 不知道问题的答案，它会如实说出不知道。
    说明：请根据 {context} 中的内容，用中文为 {question} 提供详细的答案。
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    return RetrievalQA.from_chain_type(
        llm, 
        chain_type="stuff", 
        retriever=retriever, 
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )

def run_chain(chain, prompt: str, history=[]):
    result = chain(prompt)
    # To make it compatible with chat samples
    return {
        "answer": result['result'],
        "source_documents": result['source_documents']
    }

if __name__ == "__main__":
    chain = build_chain()
    #result = run_chain(chain, "What's SageMaker?")
    result = run_chain(chain, "什么是人车管控方案？")
    print(result['answer'])
    if 'source_documents' in result:
        print('Sources:')
        for d in result['source_documents']:
            print(d.metadata['source'])
    disconnect_poe()

