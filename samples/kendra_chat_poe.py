from typing import Any, List, Mapping, Optional
from aws_langchain.kendra_index_retriever import KendraIndexRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
import poe
import sys
import json
import os
import time

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

MAX_HISTORY_LENGTH = 5


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


    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, qa_prompt=PROMPT, return_source_documents=True)
    return qa

def run_chain(chain, prompt: str, history=[]):
    return chain({"question": prompt, "chat_history": history})


if __name__ == "__main__":
    chat_history = []
    qa = build_chain()
    print(bcolors.OKBLUE + "Hello! How can I help you?" + bcolors.ENDC)
    print(bcolors.OKCYAN + "Ask a question, start a New search: or CTRL-D to exit." + bcolors.ENDC)
    print(">", end=" ", flush=True)
    for query in sys.stdin:
        if (query.strip().lower().startswith("new search:")):
            query = query.strip().lower().replace("new search:","")
            chat_history = []
        elif (len(chat_history) == MAX_HISTORY_LENGTH):
            chat_history.pop(0)
        result = run_chain(qa, query, chat_history)
        chat_history.append((query, result["answer"]))
        print(bcolors.OKGREEN + result['answer'] + bcolors.ENDC)
        if 'source_documents' in result:
            print(bcolors.OKGREEN + 'Sources:')
            for d in result['source_documents']:
                print(d.metadata['source'])
        print(bcolors.ENDC)
        print(bcolors.OKCYAN + "Ask a question, start a New search: or CTRL-D to exit." + bcolors.ENDC)
        print(">", end=" ", flush=True)
    print(bcolors.OKBLUE + "Bye" + bcolors.ENDC)
    disconnect_poe()
