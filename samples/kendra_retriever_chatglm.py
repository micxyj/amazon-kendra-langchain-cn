from aws_langchain.kendra_index_retriever import KendraIndexRetriever
from langchain.chains import RetrievalQA
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import ContentHandlerBase
import json
import os


def build_chain():
    region = os.environ["AWS_REGION"]
    kendra_index_id = os.environ["KENDRA_INDEX_ID"]
    endpoint_name = os.environ["CHATGLM_ENDPOINT"]

    class ContentHandler(ContentHandlerBase):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
            #input_str = json.dumps({"text_inputs": prompt, "parameters": model_kwargs})
            input_str = json.dumps({"ask": prompt})
            #print(input_str)
            return input_str.encode('utf-8')
        
        def transform_output(self, output: bytes) -> str:
            #print(output)
            #response_json = json.loads(output.read().decode("utf-8"))
            response_json = json.loads(output.read())
            #print(response_json)
            #return response_json[0]["generated_texts"]
            return response_json["answer"]

    content_handler = ContentHandler()

    llm=SagemakerEndpoint(
            endpoint_name=endpoint_name, 
            region_name=region, 
            model_kwargs={"temperature":1e-10, "max_length": 500},
            content_handler=content_handler
        )

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
    qa = RetrievalQA.from_chain_type(
        llm, 
        chain_type="stuff", 
        retriever=retriever, 
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )
    return qa

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
