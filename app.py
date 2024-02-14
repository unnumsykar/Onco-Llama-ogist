from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template = """ You're an ethical and knowledgeable assistant specializing in oncology. Provide helpful and
accurate information on oncology-related topics. Prioritize safety, respect, and honesty in your responses. Clarify if a 
question is unclear or outside the scope of oncology,and refrain from sharing false information. Your goal is to 
positively contribute to understanding and managing oncological issues.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful Answer:
"""

def set_custom_prompt():
    """  
    Prompt template for QA retrieval for each vectors
    
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def load_llm():
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q2_K.bin",
        #model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type = "llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm 

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever = db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents = True,
        chain_type_kwargs = {'prompt': prompt}
    )
    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs = {'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH,embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    
    return qa 

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query':query})
    return response


#chainlit
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the answer engine...")
    await msg.send()
    msg.content = "Hi, Welcome to Onco-Llama-logist !"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached=True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["sources_documents"]
    
    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += f"\n There may not be sufficient evidence to answer exactly."
    
    #cl.user_session.set("chain",chain)   
        
    await cl.Message(content=answer).send()        
    