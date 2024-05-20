import os
import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import KonlpyTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_transformers import LongContextReorder
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import requests
from io import BytesIO
import tempfile

st.set_page_config(page_title="KT 수행 변호사 봇", page_icon="🐧")
st.title("🐧KT 수행 변호사 봇🐧")

os.environ["OPENAI_API_KEY"] = "e"
embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "store" not in st.session_state:
    st.session_state["store"] = dict()

if "key_paragraphs_dict" not in st.session_state:
    st.session_state["key_paragraphs_dict"] = {}
# 법령 PDF 파일 경로 상수
LAW_PDF_URL = "https://drive.google.com/uc?export=download&id=1c_0RmkN8zSu_aQeaG70hfqjSgwlDRd1h"
# 사례 PDF 파일 경로 상수 
CASE_PDF_URL = "https://drive.google.com/uc?export=download&id=1ee6LynCvDLPEp_8dPoyz3Wp6YzoMg_h-"

@st.cache_data
def load_pdf_from_url(url):
    response = requests.get(url)
    return BytesIO(response.content)

@st.cache_data
def save_pdf_to_tempfile(pdf_bytes):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.write(pdf_bytes.read())
    temp_file.close()
    return temp_file.name

@st.cache_data
def load_docs(url):
    pdf_bytes = load_pdf_from_url(url)
    temp_file_path = save_pdf_to_tempfile(pdf_bytes)
    loader = PyMuPDFLoader(temp_file_path)
    docs = loader.load()
    os.remove(temp_file_path)
    # 문서에 파일명 추가
    for doc in docs:
        doc.metadata['source'] = os.path.basename(url)
    return docs

@st.cache_data
def load_law_docs():
    return load_docs(LAW_PDF_URL)

@st.cache_data
def load_case_docs():
    return load_docs(CASE_PDF_URL)

law_docs = load_law_docs()
case_docs = load_case_docs()

@st.cache_data
def law_split_docs(_docs):
    text_splitter = CharacterTextSplitter(
        separator="\n\n\n",
        chunk_size=1300,
        chunk_overlap=0,
    )
    return text_splitter.split_documents(_docs)

@st.cache_data
def case_split_docs(_docs):
    text_splitter = KonlpyTextSplitter(
        chunk_size=1200,
        chunk_overlap=0,
    )
    return text_splitter.split_documents(_docs)

law_splits = law_split_docs(law_docs)
case_splits = case_split_docs(case_docs)

@st.cache_resource
def load_law_vectordb(_splits):
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    law_vectordb = FAISS.from_documents(documents=_splits, embedding=embedding)
    return law_vectordb

@st.cache_resource
def load_case_vectordb(_splits):
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    case_vectordb = FAISS.from_documents(documents=_splits, embedding=embedding)
    return case_vectordb

law_vectordb = load_law_vectordb(law_splits)
case_vectordb = load_case_vectordb(case_splits)

context_reorder = LongContextReorder()

with st.sidebar:
    session_id = st.text_input("Session ID", value="penguin")
    clear_btn = st.button("대화기록 초기화")
    if clear_btn:
        st.session_state["messages"] = []
        st.session_state["store"] = dict()
        st.experimental_rerun()

def print_messages():
    if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
        for chat_message in st.session_state["messages"]:
            st.chat_message(chat_message.role).write(chat_message.content)

print_messages()

def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    if session_ids not in st.session_state["store"]:
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]

law_faiss_retriever = law_vectordb.as_retriever(search_kwargs={"k": 2})
case_faiss_retriever = case_vectordb.as_retriever(search_kwargs={"k": 2})

law_bm25_retriever = BM25Retriever.from_documents(law_docs)
law_bm25_retriever.k = 2

case_bm25_retriever = BM25Retriever.from_documents(case_docs)
case_bm25_retriever.k = 2

law_ensemble_retriever = EnsembleRetriever(
    retrievers=[law_bm25_retriever, law_faiss_retriever], weights=[0.5, 0.5]
)
case_ensemble_retriever = EnsembleRetriever(
    retrievers=[case_bm25_retriever, case_faiss_retriever], weights=[0.5, 0.5]
)

if user_input := st.chat_input("메세지를 입력해 주세요. "):
    st.chat_message("user").write(f"{user_input}")
    user_chat_message = ChatMessage(role="user", content=user_input)

    st.session_state["messages"].append(user_chat_message)

    session_history = get_session_history(session_id)
    session_history.add_message(user_chat_message)

    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
    law_docs = law_ensemble_retriever.invoke(user_input)
    law_reordered_docs = context_reorder.transform_documents(law_docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system", "너는 KT의 매우 엄격하고 단호한 법률 변호사입니다. 받는 질문에 대해서 단 하나라도 법에 위반될 가능성이 있다면 불가능하다고 해야 해. 먼저, 답변의 핵심 내용을 2문장으로 요약해줘. 그리고 다음 문단에 '[구체적 설명]\n'하고 법적 근거에 대해서 설명해줘",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "나는 대기업에서 일하는 KT의 프로젝트 관리자야."+"{question}")
        ]
    )

    chain = prompt | llm

    chain_with_memory = (
        RunnableWithMessageHistory(
            chain,
            lambda _: session_history,
            context="context",
            input_message_key="question",
            history_messages_key="history",
        )
    )

    def generate_response(_user_input, _session_id):
        response = chain_with_memory.invoke(
            {
                "question": _user_input,
            },
            config={"configurable": {"session_id": _session_id}},
        )
        return response

    response = generate_response(user_input, session_id)
    msg = response.content

    with st.chat_message("assistant"):
        st.write(msg)
        assistant_chat_message = ChatMessage(role="assistant", content=msg)
        st.session_state["messages"].append(assistant_chat_message)
        session_history.add_message(assistant_chat_message)

# ...

answer_count = 1
for i in range(len(st.session_state["messages"])):
    if i % 2 == 1:  # 답변 메시지만 표시
        with st.expander(f"[답변 {answer_count}] 참고 법령"):
            law_answer = st.session_state["messages"][i].content

            if not isinstance(law_answer, str):
                law_answer = str(law_answer)

            law_docs_with_scores = law_vectordb.similarity_search_with_score(law_answer, k=5)
            law_keyword_docs = law_ensemble_retriever.invoke(st.session_state["messages"][i-1].content, k=5)

            law_final_docs = []
            for law_doc, law_score in law_docs_with_scores:
                law_keyword_score = 0
                for law_keyword_doc in law_keyword_docs:
                    if law_doc.metadata['source'] == law_keyword_doc.metadata['source'] and law_doc.metadata['page'] == law_keyword_doc.metadata['page']:
                        law_keyword_score = 1
                        break
                law_final_score = law_score + law_keyword_score
                law_final_docs.append((law_doc, law_final_score))

            law_final_docs = sorted(law_final_docs, key=lambda x: x[1], reverse=True)

            for law_doc, law_score in law_final_docs:
                if law_doc.metadata['source']=="uc?export=download&id=1c_0RmkN8zSu_aQeaG70hfqjSgwlDRd1h":
                    law_pdf_name = "부정청탁금지법"  # 원래 PDF 파일명 사용
                else:
                    law_pdf_name=law_doc.metadata['source']
                st.markdown(f"**출처**: {law_pdf_name}, **페이지**: {law_doc.metadata['page']}, **점수**: {law_score:.2f}")
                st.markdown(f"<div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-bottom: 10px;'>{law_doc.page_content}</div>", unsafe_allow_html=True)

        with st.expander(f"[답변 {answer_count}] 유사 사례"):
            case_answer = st.session_state["messages"][i].content

            if not isinstance(case_answer, str):
                case_answer = str(case_answer)

            case_docs_with_scores = case_vectordb.similarity_search_with_score(case_answer, k=5)
            case_keyword_docs = case_ensemble_retriever.invoke(st.session_state["messages"][i-1].content, k=5)

            case_final_docs = []
            for case_doc, case_score in case_docs_with_scores:
                case_keyword_score = 0
                for case_keyword_doc in case_keyword_docs:
                    if case_doc.metadata['source'] == case_keyword_doc.metadata['source'] and case_doc.metadata['page'] == case_keyword_doc.metadata['page']:
                        case_keyword_score = 1
                        break
                case_final_score = case_score + case_keyword_score
                case_final_docs.append((case_doc, case_final_score))

            case_final_docs = sorted(case_final_docs, key=lambda x: x[1], reverse=True)

            for case_doc, case_score in case_final_docs:
                case_pdf_name = case_doc.metadata['source']  # 원래 PDF 파일명 사용
                if case_doc.metadata['source']=="uc?export=download&id=1ee6LynCvDLPEp_8dPoyz3Wp6YzoMg_h-":
                    case_pdf_name = "부정청탁금지법 해설집"  # 원래 PDF 파일명 사용
                else:
                    case_pdf_name=case_doc.metadata['source']
                st.markdown(f"**출처**: {case_pdf_name}, **페이지**: {case_doc.metadata['page']}, **점수**: {case_score:.2f}")
                st.markdown(f"<div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-bottom: 10px; white-space: pre-wrap;'>{case_doc.page_content}</div>", unsafe_allow_html=True)

        answer_count += 1
