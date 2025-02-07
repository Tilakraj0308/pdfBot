from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import RetrievalQA
import chainlit as cl

def get_db(db_dir = 'db', model_name='sentence-transformers/all-mpnet-base-v2'):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.load_local(db_dir, embeddings)
    return db

def get_llm(model_name, model_type, temperature, max_new_token):
    llm = CTransformers(model=model_name,
                        model_type=model_type,
                        temperature = temperature,
                        max_new_token=max_new_token)
    return llm

def get_prompt():
    prompt = ChatPromptTemplate(input_variables=['question', 'context'], output_parser=None, partial_variables={}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['question', 'context'], output_parser=None, partial_variables={}, template="[INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.<</SYS>> \nQuestion: {question} \nContext: {context} \nAnswer: [/INST]", template_format='f-string', validate_template=True), additional_kwargs={})])
    return prompt

def get_llm_chain(model_name, model_type, temperature, max_new_token, db):
    llm = get_llm(model_name, model_type, temperature, max_new_token)
    prompt = get_prompt()
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain




@cl.on_chat_start
async def start():
    db=get_db()
    chain = get_llm_chain(model_name='llama-2-7b-chat.Q5_K_M.gguf', model_type='llama', temperature=0.01, max_new_token=1000, db=db)
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to pdf Bot. Ask questions related to the pdf"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = False
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]
    await cl.Message(content=answer).send()

    # if sources:
    #     answer += f"\nSources:" + str(sources)
    #     await cl.Message(content=sources).send()
    # else:
    #     # answer += "\nNo sources found"
    #     await cl.Message(content="Source not found!!").send()
