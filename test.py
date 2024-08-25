import os
import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_communityimport streamlit as st
import requests
import tempfile
import os
import csv
import pandas as pd
import re
from datetime import datetime, timezone, timedelta
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from io import BytesIO
from kiwipiepy import Kiwi
from typing import TypedDict, Dict, List
from langgraph.graph import END, StateGraph
from IPython.display import Image, display
from langchain_core.runnables import RunnableConfig
from langchain_upstage import UpstageGroundednessCheck
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.schema import Document
from langchain_community.document_transformers import LongContextReorder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from sklearn.metrics.pairwise import cosine_similarity


# 사용자 인증 정보
users = {
    "admin": "admin",
    "test" : "test",
    "10128722":"10128722", #이광준 부장님
    "10154372": "10154372", # 김도완 사원님
    "10154502": "10154502", # 김나현 사원님
    "10154455": "10154455", #이령현 사원님
    "10030124": "10030124", # 김근우 팀장님
    "10050490": "10050490", # 이동원 부장님
    "10053105" : "10053105", # 윤재호 부장님
    "10054788" : "10054788", # 이민아 차장님
    "10073609" : "10073609", # 양수경 차장님
}

# 법령 및 사례 PDF 파일 URL
LAW_PDF_URLS = {
    "청탁금지법": "https://drive.google.com/uc?export=download&id=1ZxmCf7dOEd8Y8pYp9ojRgXDhxsitGF44",
    "중대재해처벌법": "https://drive.google.com/uc?export=download&id=1AvA8fwxChGkNcE4O34R088F5sZz1Sbwz",
    "산업안전보건법": "https://drive.google.com/uc?export=download&id=1uO_uTf1xIpa87MRkuUQnVilCw1GPIjeQ",
    "하도급법": "https://drive.google.com/uc?export=download&id=1RNeYXHY1zENKXF9J6J_G3YgPboJt02lg",
    "상생협력법": "https://drive.google.com/uc?export=download&id=183pgpXkYbtmacFcdnUQdc5uvvpZNE6Im",
    "공정거래법": "https://drive.google.com/uc?export=download&id=11SWqG4p7WNY4Gb4pJdkl8dElMYszU6Pz",
    "정보통신공사업법": "https://drive.google.com/uc?export=download&id=1qRCnYXa6Vcp3VOh4vajOzaKpeHsotH-f",
    "국가계약법": "https://drive.google.com/uc?export=download&id=1YYTe7UXkCkf0coGZ_0zl7Cz5YCUf3W4S",
    "소프트웨어진흥법": "https://drive.google.com/uc?export=download&id=1spVRGsFELrvy7Cs4vJdplAMpiYuqcxTq",
}

CASE_PDF_URLS = {
    "청탁금지법":"https://drive.google.com/uc?export=download&id=1C3L--keFrMPNuJfQBRv8-AhKJvXiKkIm",
    "중대재해처벌법":"https://drive.google.com/uc?export=download&id=1NE8o8XWJxfXb2yCVan66ZeexzsFTV81F",
    "산업안전보건법":"https://drive.google.com/uc?export=download&id=1s4szmiDuCUvf8KaN7AbmCwQSVMYj6X5Q",
    "하도급법":"https://drive.google.com/uc?export=download&id=1FlP338Gz42-w2aXGC7WcTx38rJ-Zp0M3",
    "상생협력법":"https://drive.google.com/uc?export=download&id=1At3yWegX8fTqebWCeCY5wPmKyIioHQJw",
    "공정거래법":"https://drive.google.com/uc?export=download&id=1x3PG4zug0-ALHcLyDeJOLdb4FwvFP8TB",
    "정보통신공사업법":"",
    "국가계약법":"https://drive.google.com/uc?export=download&id=1zm61TCNuZFW6cdjAkrzT-PJ6RA4rNpla",
    "소프트웨어진흥법":"",
}

FOR_SHOW_LAW_PDF_URLS = {
    "청탁금지법": "https://drive.google.com/uc?export=download&id=1Eqx5xE8ewtoFkWG3GDQuivURXRmE2hIZ",
    "중대재해처벌법": "https://drive.google.com/uc?export=download&id=1h8C4GAUOHsqB5uP1MxrKLgovubxLUFsJ",
    "산업안전보건법": "https://drive.google.com/uc?export=download&id=1r2WOFNEzPy0pnANAZ486H8MFbeN4YMhD",
    "하도급법": "https://drive.google.com/uc?export=download&id=1-u6DlMQVQ7qe1DsJoHLisn1EbXuGEFHo",
    "상생협력법": "https://drive.google.com/uc?export=download&id=1iI1GG6Ob2o-rbZ1qfjPOZ0s8RZplJxy9",
    "공정거래법": "https://drive.google.com/uc?export=download&id=1k7WUJX8geVZb-0mH8sWZG5KUMsLLnCoQ",
    "정보통신공사업법": "https://drive.google.com/uc?export=download&id=1qRCnYXa6Vcp3VOh4vajOzaKpeHsotH-f",
    "국가계약법": "https://drive.google.com/uc?export=download&id=1YYTe7UXkCkf0coGZ_0zl7Cz5YCUf3W4S",
    "소프트웨어진흥법": "https://drive.google.com/uc?export=download&id=1spVRGsFELrvy7Cs4vJdplAMpiYuqcxTq",
}

# 비밀번호 확인 함수
def check_password():
    def password_entered():
        if (
            st.session_state["username"] in users
            and st.session_state["password"] == users[st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            st.session_state["logged_in_user"] = st.session_state["username"]
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        
        # 공지사항 추가
        st.markdown("""
        <div style="border:1px solid #e0e0e0; padding:10px; margin-top:20px; border-radius:5px;">
            <h3 style="color:#1E90FF;">📋 사용방법</h3>
            <ol>
                <li>로그인하기 (ID/PW는 본인 사번입니다.)</li>
                <li>셋팅하기 (왼쪽 sidebar)
                    <ul>
                        <li>질문 유형을 선택합니다.</li>
                        <li>검토받고 싶은 법률을 선택합니다.</li>
                    </ul>
                </li>
                <li>챗봇에 메시지를 입력하여 질문을 던집니다.(❗만약, 실제 법률과 관련된 질문을 넣었는데 법과 관련이 없다라는 답변을 받게 된다면, "기존 질문 + ~~~ 법률 검토를 해주세요."라는 식으로 수정하여 다시 질문을 해보세요.)</li>
                <li>답변에 대한 피드백을 남깁니다. (필수는 아니나, 따로 피드백을 주고 싶은 부분이 있다면 남겨주시면 되십니다.)</li>
                <li>최종적으로 이용을 한 후기를, 왼쪽 sidebar하단에 '전반적인 사용후기'에 적어주세요.</li>
                <li>로그아웃합니다.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="border:1px solid #e0e0e0; padding:10px; margin-top:20px; border-radius:5px;">
            <h3 style="color:#1E90FF;">💡 Tip</h3>
            <ol>
                <li>질문 유형에 따라 답변의 형식이 다르게 나옵니다. 질문의 상황에 맞게 유형을 선택해야 더욱 정확한 답변을 얻으실 수 있을 것입니다.</li>
                <li>법률 선택은 최소 1개 선택을 해야하며, 총 9개의 법령을 선택할 수 있습니다.
                    <ul>
                        <li>법령 + 시행령 + 사례 추가 : 청탁금지법, 하도급법(0820), 상생협력법(0821), 산업안전보건법(0822), 공정거래법(0823), 중대재해처벌법(0824) </li>
                        <li>법령 + 사례 추가: 국가계약법</li>
                        <li>법령 only: 정보통신공사업법, 소프트웨어진흥법</li>
                    </ul>
                        * 활용 데이터가 많을 수록 정확도가 높을 가능성이 있습니다.
                </li>
                <li>질문은 할 때마다 비용이 듭니다. 신중하게 질문을 해주시면 감사하겠습니다.</li>
                <li>대화 내역은 이전 5번의 대화까지 기억합니다. 정확도 향상을 위해 질문 주제가 바뀐다면, '새로운 대화 주제' 버튼을 클릭해주세요. </li>
                <li>하나의 법령으로 답을 못한 부분도 여러 법령들의 검토를 받으면 답을 할 수도 있습니다.</li>
                <li>비용과 보안 이슈로, 법과 관련된 질문이 아니면 답을 하지 않도록 필터링해놓았습니다. </li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        st.error("😕 알 수 없는 사용자이거나 비밀번호가 틀립니다.")
        return False
    else:
        return True
    
def main():
    # Streamlit 페이지 설정
    st.set_page_config(page_title="AI 수행 컴플라이언스 봇", page_icon="🐧")
    st.title("🦁AI 수행 컴플라이언스 봇🦁")

    # CSS 스타일 정의
    st.markdown("""
    <style>
        .stButton > button {
            margin: 0px;
            padding: 0px 10px;
            height: 30px;
        }
        .feedback-buttons {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 5px;
        }
        .feedback-container {
            display: flex;
            justify-content: flex-end;
            align-items: center;
        }
        .feedback-message {
            margin-right: 10px;
        }
        .button-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .logout-button {
            margin-left: 10px;
        }
        .logout-button .stButton > button {
            background-color: #f63366;
            color: white;
            border: none;
            padding: 0.25rem 0.75rem;
            font-size: 0.8rem;
            line-height: 1.6;
            border-radius: 0.25rem;
            margin-top: -5px;  /* 버튼을 약간 위로 올립니다 */
        }
        .stButton > button {
            height: 2.2rem;
        }
        .button-container .element-container {
            margin-bottom: 0 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    if check_password():

        if st.session_state.get("logged_in_user") == "admin":
            st.header("관리자 섹션")
            
            if st.button("전반적인 사용후기 확인"):
                if os.path.exists('feedback.csv'):
                    feedback_df = pd.read_csv('feedback.csv')
                    st.dataframe(feedback_df)
                else:
                    st.info("아직 제출된 전반적인 사용후기가 없습니다.")
            
            if st.button("챗봇 답변 피드백 확인"):
                if os.path.exists('chatbot_feedback.csv'):
                    chatbot_feedback_df = pd.read_csv('chatbot_feedback.csv')
                    st.dataframe(chatbot_feedback_df)
                else:
                    st.info("아직 제출된 챗봇 답변 피드백이 없습니다.")
        else:    

            os.environ["OPENAI_API_KEY"] = "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"
            os.environ["UPSTAGE_API_KEY"] = "up_rWvsR4s5dHERp6Y9D77KFp2UmEV5F"
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
            os.environ["LANGCHAIN_PROJECT"] = "Complionss"
            os.environ["LANGCHAIN_API_KEY"] = "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"

            upstage_ground_checker = UpstageGroundednessCheck()

            # Kiwi 초기화
            kiwi = Kiwi()

            # CSS를 사용하여 스타일 정의
            st.markdown("""
            <style>
                .notice-box {
                    border: 2px solid #4CAF50;
                    border-radius: 10px;
                    padding: 15px;
                    margin-top: 20px;
                    margin-bottom: 30px;
                    background-color: #f1f8e9;
                }
                .notice-title {
                    color: #4CAF50;
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 10px;
                }
                .notice-content {
                    font-size: 14px;
                    color: #333;
                }
                .notice-footer {
                    font-size: 12px;
                    color: #666;
                    margin-top: 10px;
                }
            </style>
            """, unsafe_allow_html=True)

            # 공지사항 추가
            st.markdown("""
            <div class="notice-box">
                <div class="notice-title">📢 공지사항</div>
                <div class="notice-content">
                    안녕하세요! AI 수행 컴플라이언스 봇입니다🦁<br><br>
                    프로젝트 수행 관련 법률*에 대해 빠른 답변과 관련 법률 항목과 연관 사례들을 제공합니다. <br>
                    답변은 참고용이며, 법적 효력이 없음을 고지드립니다. <br>
                </div>
                <div class="notice-footer">
                    *청탁금지법, 공정거래법, 중대재해처벌법, 산업안전보건법, 하도급법, 상생협력법, 정보통신공사업법, 국가계약법, 소프트웨어진흥법
                </div>
            </div>
            """, unsafe_allow_html=True)

            session_id = st.session_state.get("logged_in_user", "default_user")

            # 사용자별 상태 초기화 및 동기화
            if "user_states" not in st.session_state:
                st.session_state["user_states"] = {}

            if session_id not in st.session_state["user_states"]:
                st.session_state["user_states"][session_id] = {
                    "messages": [],
                    "store": {},
                    "law_references": [],
                    "similar_cases": [],
                    "relevance_results": [],
                    "selected_messages": []
                }

            user_state = st.session_state["user_states"][session_id]

            # 전체 세션 상태와 사용자별 상태 동기화
            if "messages" not in st.session_state:
                st.session_state.messages = user_state["messages"]
            else:
                user_state["messages"] = st.session_state.messages

            # 사이드바 설정
            with st.sidebar:
                st.header(f"접속자: {session_id}")
                
                clear_btn = st.button("새로운 대화 주제", key="clear_button")
            
                model_name = st.selectbox("언어 모델 선택", ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"], key="model_select")
                bm25_weight = st.slider("키워드 검색 비중 [키워드 검색 + 의미 검색 = 1.00]", 0.0, 1.0, 0.5, 0.1, key="bm25_slider")
                faiss_weight = 1 - bm25_weight

                question_type = st.selectbox("질문 유형", ["법 저촉 여부 상황 판단", "단순 질의응답", "금액 계산", "그 외", "<자동 분류>"], key="question_type_select")        
                   
                # 법률 선택
                st.header("법률 선택")
                selected_laws = []
                for law in LAW_PDF_URLS.keys():
                    if st.checkbox(law, key=f"checkbox_{law}"):
                        selected_laws.append(law)
                
                if not selected_laws:
                    st.warning("최소 1개 이상의 법률을 선택해주세요.")
            
                # 피드백 섹션
                st.header("피드백")
                feedback = st.text_area("전반적인 사용후기를 입력해주세요:", key="feedback_text")
                
                # 피드백 제출 버튼과 로그아웃 버튼을 같은 줄에 배치
                st.markdown('<div class="button-container">', unsafe_allow_html=True)
                col1, col2 = st.columns([2, 1])
                with col1:
                    submit_btn = st.button("피드백 제출", key="feedback_submit")
                with col2:
                    logout_btn = st.button("로그아웃", key="logout_button")
                st.markdown('</div>', unsafe_allow_html=True)
                st.write("🐧 저작자: @AI컴플라이언스봇 TF")

            
            # 피드백 제출 로직
            if submit_btn:
                if feedback:
                    kst = timezone(timedelta(hours=9))
                    timestamp = datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S")
                    feedback_data = pd.DataFrame({
                        'User': [st.session_state.get("logged_in_user", "Unknown")],
                        'Feedback': [feedback], 
                        'Timestamp': [timestamp]
                    })
                    feedback_data.to_csv('feedback.csv', mode='a', header=not os.path.exists('feedback.csv'), index=False)
                    st.sidebar.success("피드백이 제출되었습니다. 감사합니다!")
                else:   
                    st.sidebar.warning("피드백을 입력해주세요.")
            
            if clear_btn:
                user_state["messages"] = []
                user_state["store"] = {}
                user_state["law_references"] = []
                user_state["similar_cases"] = []
                user_state["relevance_results"] = []
                st.session_state.messages = []  # 전체 세션 상태도 초기화
                st.experimental_rerun()
            
            if logout_btn:
                # 로그아웃 처리
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.experimental_rerun()

            # 임베딩 모델 설정
            embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

            if "messages" not in st.session_state:
                st.session_state.messages = []
            if "feedback_data" not in st.session_state:
                st.session_state.feedback_data = []
            if "store" not in st.session_state:
                st.session_state["store"] = {}
            if "relevance_results" not in st.session_state:
                st.session_state["relevance_results"] = []
            if "law_references" not in st.session_state:
                st.session_state["law_references"] = []
            if "similar_cases" not in st.session_state:
                st.session_state["similar_cases"] = []
            if "selected_messages" not in st.session_state:
                st.session_state["selected_messages"] = []
            if "user_states" not in st.session_state:
                st.session_state["user_states"] = {}
            if "feedback_states" not in st.session_state:
                st.session_state.feedback_states = {}
            if "question_type_select" not in st.session_state:
                st.session_state.question_type_select = "법 저촉 여부 상황 판단"

            if session_id not in st.session_state["user_states"]:
                st.session_state["user_states"][session_id] = {
                    "messages": [],
                    "store": {},
                    "relevance_results": [],
                    "law_references": [],
                    "similar_cases": [],
                    "selected_messages": []
                }

            # 사용자별 상태에 접근
            user_state = st.session_state["user_states"][session_id]

            # PDF 로딩 및 처리 함수들
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
            def load_docs(url, law_name):
                pdf_bytes = load_pdf_from_url(url)
                temp_file_path = save_pdf_to_tempfile(pdf_bytes)
                loader = PyMuPDFLoader(temp_file_path)
                docs = loader.load()
                
                for doc in docs:
                    doc.metadata['source'] = law_name  # URL 대신 법률 이름 저장
                
                return docs

            @st.cache_data
            def load_law_docs():
                global LAW_PDF_URLS
                law_docs = []
                for law_name, url in LAW_PDF_URLS.items():
                    docs = load_docs(url, law_name)
                    for doc in docs:
                        doc.metadata['law_name'] = law_name
                    law_docs.extend(docs)
                return law_docs

            @st.cache_data
            def load_case_docs():
                global CASE_PDF_URLS
                case_docs = []
                for law_name, url in CASE_PDF_URLS.items():
                    if url:  # URL이 비어있지 않은 경우에만 문서 로드
                        docs = load_docs(url, law_name)
                        for doc in docs:
                            doc.metadata['law_name'] = law_name
                        case_docs.extend(docs)
                    else:
                        # URL이 비어있는 경우 빈 문서 생성
                        empty_doc = Document(page_content="사례 정보가 없습니다.", metadata={'law_name': law_name, 'source': law_name, 'page': 0})
                        case_docs.append(empty_doc)
                return case_docs
            
            
            @st.cache_data
            def load_for_show_law_docs():
                for_show_law_docs = []
                for for_show_law_name, url in FOR_SHOW_LAW_PDF_URLS.items():
                    docs = load_docs(url, for_show_law_name)
                    for doc in docs:
                        doc.metadata['for_show_law_name'] = for_show_law_name
                    for_show_law_docs.extend(docs)
                return for_show_law_docs

            def create_selected_law_vectordbs(selected_laws):
                selected_law_vectordbs = {law: law_vectordbs[law] for law in selected_laws}
                return selected_law_vectordbs
            
            def create_selected_for_show_law_vectordbs(selected_for_show_laws):
                selected_for_show_law_vectordbs = {for_show_law: for_show_law_vectordbs[for_show_law] for for_show_law in selected_for_show_laws}
                return selected_for_show_law_vectordbs

            # 법령별 리트리버 설정
            def create_law_retrievers(selected_law_vectordbs):
                law_retrievers = {}
                embeddings = OpenAIEmbeddings()
                reordering = LongContextReorder()
                
                for law_name, vectordb in selected_law_vectordbs.items():
                    faiss_retriever = vectordb.as_retriever(search_kwargs={"k": 5})
                    bm25_retriever = BM25Retriever.from_documents([doc for doc in law_docs if doc.metadata['law_name'] == law_name])
                    bm25_retriever.k = 5
                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever, faiss_retriever], weights=[bm25_weight, faiss_weight]
                    )
                    
                    def retrieve_and_rerank(query, retriever=ensemble_retriever, embed=embeddings, reorder=reordering):
                        docs = retriever.get_relevant_documents(query)
                        query_embedding = embed.embed_query(query)
                        doc_embeddings = embed.embed_documents([doc.page_content for doc in docs])
                        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
                        reranked_docs = [doc for _, doc in sorted(zip(similarities, docs), key=lambda x: x[0], reverse=True)]
                        return reorder.transform_documents(reranked_docs[:3])  # top 3 문서만 반환하고 LongContextReorder 적용
                    
                    law_retrievers[law_name] = retrieve_and_rerank
                
                return law_retrievers


            # 문서 분할
            @st.cache_data
            def law_split_docs(_docs):
                text_splitter = CharacterTextSplitter(
                    separator="\n\n\n\n",
                    chunk_size=5500,
                    chunk_overlap=0,
                )
                return text_splitter.split_documents(_docs)
        

            @st.cache_data
            def case_split_docs(_docs):
                text_splitter = CharacterTextSplitter(
                    separator="\n\n\n\n",
                    chunk_size=5500,
                    chunk_overlap=0,
                )
                return text_splitter.split_documents(_docs)
            
            @st.cache_data
            def show_law_split_docs(_docs):
                text_splitter = CharacterTextSplitter(
                    separator="\n\n\n\n",
                    chunk_size=4000,
                    chunk_overlap=0,
                )
                return text_splitter.split_documents(_docs)

            # 벡터 데이터베이스 로딩
            @st.cache_resource
            def load_law_vectordbs(_splits):
                embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
                law_vectordbs = {}
                for law_name in LAW_PDF_URLS.keys():
                    law_splits = [split for split in _splits if split.metadata['law_name'] == law_name]
                    law_vectordbs[law_name] = FAISS.from_documents(documents=law_splits, embedding=embedding)
                return law_vectordbs

            @st.cache_resource
            def load_case_vectordbs(_splits):
                embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
                case_vectordbs = {}
                for law_name in CASE_PDF_URLS.keys():
                    case_splits = [split for split in _splits if split.metadata['law_name'] == law_name]
                    if case_splits:  # 문서가 있는 경우에만 vectorDB 생성
                        case_vectordbs[law_name] = FAISS.from_documents(documents=case_splits, embedding=embedding)
                    else:
                        st.warning(f"{law_name}에 대한 사례 문서가 없습니다.")
                        # 빈 vectorDB 생성 (선택적)
                        case_vectordbs[law_name] = FAISS.from_texts(["빈 문서"], embedding=embedding)
                return case_vectordbs

            @st.cache_resource
            def load_for_show_law_vectordbs(_splits):
                embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
                for_show_law_vectordbs = {}
                for for_show_law_name in FOR_SHOW_LAW_PDF_URLS.keys():
                    for_show_law_splits = [split for split in _splits if split.metadata['for_show_law_name'] == for_show_law_name]
                    for_show_law_vectordbs[for_show_law_name] = FAISS.from_documents(documents=for_show_law_splits, embedding=embedding)
                return for_show_law_vectordbs
            
            @st.cache_data
            def load_and_split_case_docs():
                case_docs = load_case_docs()
                if not case_docs:
                    st.warning("사례 문서를 로드할 수 없습니다.")
                    return []
                return case_split_docs(case_docs)

            # 사례별 리트리버 설정
            def create_case_retrievers(selected_case_vectordbs):
                case_retrievers = {}
                embeddings = OpenAIEmbeddings()
                reordering = LongContextReorder()
                
                for law_name, vectordb in selected_case_vectordbs.items():
                    if vectordb.docstore._dict:  # vectordb에 문서가 있는 경우
                        faiss_retriever = vectordb.as_retriever(search_kwargs={"k": 5})
                        bm25_retriever = BM25Retriever.from_documents([doc for doc in case_docs if doc.metadata['law_name'] == law_name])
                        bm25_retriever.k = 5
                        ensemble_retriever = EnsembleRetriever(
                            retrievers=[bm25_retriever, faiss_retriever], weights=[bm25_weight, faiss_weight]
                        )
                        
                        def retrieve_and_rerank(query, retriever=ensemble_retriever, embed=embeddings, reorder=reordering):
                            docs = retriever.get_relevant_documents(query)
                            query_embedding = embed.embed_query(query)
                            doc_embeddings = embed.embed_documents([doc.page_content for doc in docs])
                            similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
                            reranked_docs = [doc for _, doc in sorted(zip(similarities, docs), key=lambda x: x[0], reverse=True)]
                            return reorder.transform_documents(reranked_docs[:3])  # top 3 문서만 반환하고 LongContextReorder 적용
                        
                        case_retrievers[law_name] = retrieve_and_rerank
                    else:  # vectordb가 비어있는 경우
                        case_retrievers[law_name] = lambda x: []  # 빈 리스트를 반환하는 함수
                return case_retrievers

            def create_selected_case_vectordbs(selected_laws):
                selected_case_vectordbs = {law: case_vectordbs[law] for law in selected_laws}
                return selected_case_vectordbs

            # 문서 로딩
            law_docs = load_law_docs()
            law_splits = law_split_docs(law_docs)
            law_vectordbs = load_law_vectordbs(law_splits)

            case_docs = load_case_docs()
            case_splits = case_split_docs(case_docs)
            case_vectordbs = load_case_vectordbs(case_splits)

            case_splits = load_and_split_case_docs()
            if case_splits:
                case_vectordbs = load_case_vectordbs(case_splits)
            else:
                st.error("사례 문서를 처리할 수 없습니다. 프로그램을 계속 실행할 수 없습니다.")
                st.stop()

            for_show_law_docs = load_for_show_law_docs()
            for_show_law_splits = show_law_split_docs(for_show_law_docs)
            for_show_law_vectordbs = load_for_show_law_vectordbs(for_show_law_splits)

            # 세션 히스토리 관리
            def get_session_history(session_id: str) -> ChatMessageHistory:
                if "store" not in st.session_state:
                    st.session_state["store"] = {}
                if session_id not in st.session_state["store"]:
                    st.session_state["store"][session_id] = ChatMessageHistory()
                return st.session_state["store"][session_id]

            # 텍스트 전처리
            def preprocess_text(text):
                result = kiwi.analyze(text)
                keywords = [token.form for token in result[0][0] if token.tag.startswith('N') or token.tag.startswith('V') or token.tag.startswith('MA')]
                return ' '.join(keywords)

            # GraphState 정의
            class GraphState(TypedDict):
                question: str
                law_context: str
                case_context: str
                response: str
                relevance: str
                attempts: int
                law_name: str
                law_references: List[Dict]
                similar_cases: List[Dict]
                question_type: str 

            memory = ConversationBufferMemory(return_messages=True)

            # 법령별 특화 프롬프트 정의
            LAW_SPECIFIC_PROMPTS = {
                "청탁금지법": "당신은 꼼꼼하게 법적 사실을 확인하고, 쉽게 단정하지 않는 대기업 KT의 청탁금지법 전문 사내 변호사입니다. 이 법의 주요 목적인 공직자의 공정한 직무수행과 공공기관의 신뢰성 제고에 중점을 두고 답변해주세요.",
                "중대재해처벌법": "당신은 꼼꼼하게 법적 사실을 확인하고, 쉽게 단정하지 않는 대기업 KT의 중대재해처벌법 전문 사내 변호사입니다. 사업주와 경영책임자의 안전 및 보건 확보의무와 위반 시 처벌에 초점을 맞춰 답변해주세요. ",
                "산업안전보건법": "당신은 꼼꼼하게 법적 사실을 확인하고, 쉽게 단정하지 않는 대기업 KT의 산업안전보건법 전문 사내 변호사입니다. 근로자의 안전과 보건을 유지·증진하기 위한 사업주의 의무와 근로자의 권리에 중점을 두고 답변해주세요.",
                "하도급법": "당신은 꼼꼼하게 법적 사실을 확인하고, 쉽게 단정하지 않는 대기업 KT의 하도급법 전문 사내 변호사입니다. 원사업자와 수급사업자 간의 공정한 거래관계 확립에 초점을 맞춰 답변해주세요.",
                "상생협력법": "당신은 꼼꼼하게 법적 사실을 확인하고, 쉽게 단정하지 않는 대기업 KT의 상생협력법 전문 사내 변호사입니다. 대기업과 중소기업 간의 상생협력 관계 구축과 동반성장에 중점을 두고 답변해주세요.",
                "공정거래법": "당신은 꼼꼼하게 법적 사실을 확인하고, 쉽게 단정하지 않는 대기업 KT의 공정거래법 전문 사내 변호사입니다. 시장에서의 자유롭고 공정한 경쟁을 촉진하고, 독과점 및 불공정 거래 행위의 규제를 통해 소비자의 이익을 보호하는 데 중점을 두고 답변해주세요.",
                "정보통신공사업법": "당신은 꼼꼼하게 법적 사실을 확인하고, 쉽게 단정하지 않는 대기업 KT의 정보통신공사업법 전문 사내 변호사입니다. 정보통신공사의 적정한 시공과 공사 품질의 확보, 기술자의 자격 요건 및 준수사항에 중점을 두고 답변해주세요.",
                "국가계약법": "당신은 꼼꼼하게 법적 사실을 확인하고, 쉽게 단정하지 않는 대기업 KT의 국가계약법 전문 사내 변호사입니다. 국가와 공공기관의 계약 체결 시 공정성과 투명성을 보장하고, 계약 절차와 이행에서의 법적 준수 사항에 중점을 두고 답변해주세요.",
                "소프트웨어진흥법": "당신은 꼼꼼하게 법적 사실을 확인하고, 쉽게 단정하지 않는 대기업 KT의 소프트웨어진흥법 전문 사내 변호사입니다. 소프트웨어 산업의 발전과 공정한 시장 환경 조성, 그리고 소프트웨어의 품질과 안전성 확보에 중점을 두고 답변해주세요."
            }

            def select_persona_prompt(question_type):
                if question_type == "법 저촉 여부 상황 판단":
                    return """당신은 꼼꼼하게 법적 사실을 확인하고, 쉽게 단정하지 않는 대기업 KT의 법률 전문 사내 변호사입니다. 당신의 역할은 주어진 상황이 법적으로 허용 가능한지에 대해 판단해야 합니다. 그리고 사전적으로 위반 소지가 있는 사항을 안내하고, 질문하는 직원들이 법을 지키는 선에서 담당 업무를 수행할 수 있도록 답변해주는 것입니다. 당신의 응답은 <제공된 문서>에 기반해야 합니다."""

                elif question_type == "단순 질의응답":
                    return """당신은 꼼꼼하게 법적 사실을 확인하고, 쉽게 단정하지 않는 대기업 KT의 법률 전문 사내 변호사입니다. 주어진 법률 관련 질문에 대해 간단명료하게 답변해야 합니다. 다음 지침을 따라 답변해 주세요:

                    1. 답변 근거: <제공된 문서>에 기반하여 답변하세요.
                    2. 질문 이해: 주어진 질문의 핵심을 정확히 파악하세요.
                    3. 관련 법령 확인: 질문과 관련된 법령을 명시하세요.
                    4. 명확한 답변: 질문에 대해 명확하고 간결하게 답변하세요.
                    5. 추가 설명: 필요한 경우 간단한 부연 설명을 제공하세요.
                    6. 한계 명시: 답변의 한계나 예외 사항이 있다면 언급하세요.

                    답변은 법률 전문가가 아닌 사람도 이해할 수 있도록 쉽게 설명해 주세요. 당신의 응답은 <제공된 문서>에 기반해야 합니다.
                    """

                elif question_type == "금액 계산":
                    return """당신은 꼼꼼하게 법적 사실을 확인하고, 쉽게 단정하지 않는 대기업 KT의 법률 전문 사내 변호사입니다. 주어진 상황에 대해 법적으로 정해진 금액을 계산해야 합니다. 다음 지침을 따라 답변해 주세요:

                    1. 답변 근거: <제공된 문서>에 기반하여 답변하세요.
                    2. 상황 분석: 주어진 상황을 정확히 파악하세요.
                    3. 관련 법령 확인: 금액 계산과 관련된 법령을 명시하세요.
                    4. 계산 과정 설명: 금액 계산 과정을 단계별로 명확히 설명하세요.
                    5. 결과 제시: 최종 계산된 금액을 명확히 제시하세요.
                    6. 주의사항 언급: 계산 결과에 영향을 줄 수 있는 요소나 예외 사항을 설명하세요.

                    답변 시 사용된 공식이나 기준을 명확히 제시하세요. 당신의 응답은 <제공된 문서>에 기반해야 합니다."""

                else:  # "그 외"
                    return """당신은 대기업 KT의 법률 전문 사내 변호사입니다. 주어진 질문에 대해 법률적 관점에서 최선의 답변을 제공해야 합니다. 다음 지침을 따라 답변해 주세요:

                    1. 답변 근거: <제공된 문서>에 기반하여 답변하세요.
                    2. 질문 분석: 주어진 질문의 본질을 파악하세요.
                    3. 관련 법령 검토: 질문과 관련될 수 있는 법령을 검토하세요.
                    4. 종합적 답변: 법률적 관점에서 종합적인 답변을 제공하세요.
                    5. 한계 명시: 답변의 한계나 추가 검토가 필요한 사항을 언급하세요.
                    6. 조언 제공: 필요하다면 법률적 조언이나 주의사항을 제시하세요.

                    당신의 응답은 <제공된 문서>에 기반해야 합니다. 답변 시 확실하지 않은 부분은 명확히 언급하고, 필요하다면 추가적인 법률 자문을 권고하세요."""

            def select_task_prompt(question_type):

                if question_type == "법 저촉 여부 상황 판단":
                    return """
                    [Task 1: 단계별 지침]
                    1. 질문에서 주체와 객체를 식별하고, 그들의 관계(예: 상급자-하급자, 대기업-중소기업, 공공기관-민간기업 등)를 파악합니다.
                    2. 식별된 관계에 따라 적용되는 법적 기준이 다를 수 있음을 고려합니다.
                    3. 질문이 해당 법률과 관련이 있는지 확인합니다. 관련이 없다면, '이 질문은 해당 법과 관련성이 낮은 것으로 판단되어 답변할 수 없습니다.'라고 답변을 거부합니다.
                    4. 관련이 있다면, 질문을 분석하여 법적 위반 가능성을 식별합니다.
                    5. 해당 행동이 법적으로 문제가 되지 않을 수 있는 관점과 문제가 될 수 있는 관점을 모두 고려합니다.
                    6. 각 관점에 대해 관련 법령과 함께 근거를 제시합니다.
                    7. 두 관점을 종합하여 균형 잡힌 결론을 도출합니다.
                    8. <제공된 문서>에서 특정 법률, 규정 또는 조항을 참조하십시오.

                    [Task 2: 출력 형식]
                    응답은 다음 주요 부분으로 구성되어야 합니다:
                    1. 관계 분석: '[관계 분석]'으로 시작하는 단락으로, 식별된 주체와 객체의 관계를 설명합니다.
                    2. 법적으로 문제되지 않을 수 있는 관점: '[문제되지 않을 수 있는 관점]'으로 시작하는 단락으로, 해당 행동이 법적으로 허용될 수 있는 이유와 근거를 설명합니다.
                    3. 법적으로 문제될 수 있는 관점: '[문제될 수 있는 관점]'으로 시작하는 단락으로, 해당 행동이 법적으로 문제될 수 있는 이유와 근거를 설명합니다.
                    4. 결론: '[결론]'으로 시작하는 단락과 두 관점을 종합한 균형 잡힌 결론을 제시합니다.
                    5. 권고사항: '[권고사항]'으로 시작하는 단락으로, 법적 리스크를 최소화하면서 업무를 수행할 수 있는 방안을 제시합니다.

                    [Task 3: 품질 보증]
                    응답이 다음을 보장하도록 합니다:
                    1. 질문이 해당 법률과 관련이 없다면, '이 질문은 해당 법과 관련성이 낮은 것으로 판단되어 답변할 수 없습니다.'라고 답변을 거부합니다.
                    2. 관련이 있는 경우, 문제되지 않을 수 있는 관점과 문제될 수 있는 관점을 균형있게 제시합니다.
                    3. 제공된 문서에서 정확한 법적 참조를 제공합니다.
                    4. 변호사의 페르소나와 일치하는 꼼꼼하게 법적 사실을 확인하고, 쉽게 단정하지 않는 어조를 유지하되, 조언과 권고를 포함합니다.

                    [Reflection]
                    각 응답이 법적 준수를 엄격히 따르고 명확하고 정확한 법적 참조를 제공하는지 확인합니다. 응답이 질문에서 제기된 모든 잠재적 법적 문제를 충분히 다루고 있는지 고려합니다.

                    [Feedback]
                    응답의 명확성과 유용성에 대한 피드백을 요청합니다. 법적 참조가 도움이 되었는지, 설명이 충분히 상세했는지를 사용자가 알려줄 것을 요청합니다.

                    [Constraints]
                    1. 응답은 <제공된 문서>에만 기반해야 합니다.
                    2. 법적으로 문제가 될 가능성이 있는 상황이라면 이에 대해 명확히 인지시켜주고 가능한 대안을 제시하며, 필요하다면 사내 변호사에게 상담을 권장해주세요.
                    3. 결론은 간결하게 제시되어야 합니다.
                    4. <제공된 문서>에서 답을 할 수 없는 질문 또는 해당 법률과 관련이 없는 질문에 대해서는 '이 질문은 해당 법과 관련성이 낮은 것으로 판단되어 답변할 수 없습니다.'라고 답을 해야 합니다.
                    5. 질문을 새로 생성하면 안됩니다.

                    [Context]
                    사용자는 대기업 KT의 프로젝트 관리자이며, 프로젝트 관리, 계약 및 규정 준수와 관련된 법적 질문을 다루고 있을 가능성이 큽니다.
                    """
                
                elif question_type == "질의응답":
                    return """
                    [Task 1: 단계별 지침]
                    1. 질문의 주요 키워드와 핵심 내용을 파악합니다.
                    2. <제공된 문서>에서 질문과 관련된 정보를 찾습니다.
                    3. 관련 정보가 없다면, '이 질문에 대한 정보는 제공된 문서에서 찾을 수 없으므로 답변할 수 없습니다.'라고 답변합니다.
                    4. 관련 정보가 있다면, 해당 정보를 바탕으로 답변을 구성합니다.
                    5. 필요한 경우, 추가적인 설명이나 예시를 제공합니다.
                    6. 답변의 출처를 명확히 제시합니다.

                    [Task 2: 출력 형식]
                    응답은 다음 주요 부분으로 구성되어야 합니다:
                    1. 답변: '[답변]'으로 시작하는 단락으로, 질문에 대한 직접적인 답변을 제공합니다.
                    2. 설명: '[설명]'으로 시작하는 단락으로, 필요한 경우 추가적인 설명이나 예시를 제공합니다.
                    3. 출처: '[출처]'로 시작하는 단락으로, 답변의 근거가 되는 <제공된 문서>의 해당 부분을 명시합니다.

                    [Task 3: 품질 보증]
                    응답이 다음을 보장하도록 합니다:
                    1. 질문에 대한 정보가 <제공된 문서>에 없다면, 그 사실을 명확히 밝힙니다.
                    2. 답변은 정확하고 간결하며, 질문의 핵심을 다룹니다.
                    3. 추가 설명이나 예시는 이해를 돕는 데 필요한 경우에만 제공합니다.
                    4. 모든 정보의 출처를 명확히 제시합니다.

                    [Reflection]
                    각 응답이 질문에 충실히 답하고 있는지, 필요한 정보를 모두 포함하고 있는지 확인합니다. 답변이 명확하고 이해하기 쉬운지 고려합니다.

                    [Feedback]
                    응답의 명확성과 유용성에 대한 피드백을 요청합니다. 제공된 정보가 충분했는지, 추가 설명이 필요한지 사용자가 알려줄 것을 요청합니다.

                    [Constraints]
                    1. 응답은 <제공된 문서>에만 기반해야 합니다.
                    2. 추측이나 개인적인 의견을 포함하지 않습니다.
                    3. <제공된 문서>에서 답을 할 수 없는 질문에 대해서는 '이 질문에 대한 정보는 제공된 문서에서 찾을 수 없으므로 답변할 수 없습니다.'라고 답해야 합니다.
                    4. 질문을 새로 생성하면 안됩니다.

                    [Context]
                    사용자는 대기업 KT의 프로젝트 관리자이며, 프로젝트 관리, 계약 및 규정 준수와 관련된 일반적인 질문을 할 가능성이 큽니다.
                    """
                
                elif question_type == "금액 계산":
                    return """
                    [Task 1: 단계별 지침]
                    1. 질문에서 계산에 필요한 모든 정보와 변수를 식별합니다.
                    2. <제공된 문서>에서 계산에 필요한 추가 정보나 규정을 찾습니다.
                    3. 필요한 정보가 부족하다면, '계산에 필요한 일부 정보가 부족하여 답변할 수 없습니다.'라고 명시합니다.
                    4. 모든 정보가 있다면, 단계별로 계산 과정을 수행합니다.
                    5. 계산 결과를 명확하게 제시합니다.
                    6. 필요한 경우, 계산 결과에 대한 추가 설명이나 해석을 제공합니다.

                    [Task 2: 출력 형식]
                    응답은 다음 주요 부분으로 구성되어야 합니다:
                    1. 입력 정보: '[입력 정보]'로 시작하는 단락으로, 계산에 사용된 모든 변수와 값을 나열합니다.
                    2. 계산 과정: '[계산 과정]'으로 시작하는 단락으로, 단계별 계산 과정을 상세히 설명합니다.
                    3. 계산 결과: '[계산 결과]'로 시작하는 단락으로, 최종 계산 결과를 명확히 제시합니다.
                    4. 해석: '[해석]'으로 시작하는 단락으로, 필요한 경우 계산 결과에 대한 추가 설명이나 해석을 제공합니다.
                    5. 참고 사항: '[참고 사항]'으로 시작하는 단락으로, 계산에 적용된 규정이나 예외 사항 등을 명시합니다.

                    [Task 3: 품질 보증]
                    응답이 다음을 보장하도록 합니다:
                    1. 모든 계산이 정확하고 <제공된 문서>의 규정을 준수합니다.
                    2. 계산 과정이 명확하고 단계별로 설명되어 있습니다.
                    3. 최종 결과가 명확하게 제시되어 있습니다.
                    4. 필요한 경우, 결과에 대한 해석이나 추가 설명이 포함되어 있습니다.

                    [Reflection]
                    각 응답이 계산의 정확성을 보장하는지, 모든 필요한 정보를 포함하고 있는지 확인합니다. 계산 과정과 결과가 이해하기 쉽게 설명되어 있는지 고려합니다.

                    [Feedback]
                    응답의 명확성과 유용성에 대한 피드백을 요청합니다. 계산 과정이 이해하기 쉬웠는지, 결과 해석이 도움이 되었는지 사용자가 알려줄 것을 요청합니다.

                    [Constraints]
                    1. 모든 계산은 <제공된 문서>의 규정과 정보에 기반해야 합니다.
                    2. 추측이나 가정을 포함하지 않습니다. 정보가 부족할 경우 이를 명시합니다.
                    3. 계산에 필요한 정보가 부족할 경우 '계산에 필요한 일부 정보가 부족합니다.'라고 답해야 합니다.
                    4. 질문을 새로 생성하면 안됩니다.

                    [Context]
                    사용자는 대기업 KT의 프로젝트 관리자이며, 프로젝트 비용, 계약금액, 위약금 등과 관련된 금액 계산 질문을 할 가능성이 큽니다.
                    """
                
                else:  # "그 외"
                    return """
                    [Task 1: 단계별 지침]
                    1. 질문의 주요 주제와 요점을 파악합니다.
                    2. <제공된 문서>에서 질문과 관련된 정보를 찾습니다.
                    3. 관련 정보가 없다면, '이 질문에 대한 정보는 제공된 문서에서 찾을 수 없으므로 답변할 수 없습니다.'라고 답변합니다.
                    4. 관련 정보가 있다면, 해당 정보를 바탕으로 포괄적이고 정보가 풍부한 답변을 구성합니다.
                    5. 필요한 경우, 추가적인 설명, 예시, 또는 관련 정보를 제공합니다.
                    6. 답변의 출처를 명확히 제시합니다.
                    7. 질문이 여러 측면을 다루고 있다면, 각 측면에 대해 체계적으로 답변합니다.

                    [Task 2: 출력 형식]
                    응답은 다음 주요 부분으로 구성되어야 합니다:
                    1. 요약: '[요약]'으로 시작하는 단락으로, 질문에 대한 간략한 답변을 제공합니다.
                    2. 상세 설명: '[상세 설명]'으로 시작하는 단락으로, 질문의 각 측면에 대해 자세히 설명합니다.
                    3. 예시 또는 추가 정보: '[예시/추가 정보]'로 시작하는 단락으로, 필요한 경우 구체적인 예시나 관련 정보를 제공합니다.
                    4. 관련 규정: '[관련 규정]'으로 시작하는 단락으로, 해당되는 경우 관련 법규나 회사 정책을 언급합니다.
                    5. 결론 또는 권고사항: '[결론/권고사항]'으로 시작하는 단락으로, 필요한 경우 종합적인 결론이나 권고사항을 제시합니다.
                    6. 출처: '[출처]'로 시작하는 단락으로, 답변의 근거가 되는 <제공된 문서>의 해당 부분을 명시합니다.

                    [Task 3: 품질 보증]
                    응답이 다음을 보장하도록 합니다:
                    1. 질문의 모든 측면을 포괄적으로 다룹니다.
                    2. 정보가 정확하고 <제공된 문서>에 기반합니다.
                    3. 설명이 명확하고 논리적으로 구성되어 있습니다.
                    4. 필요한 경우 구체적인 예시나 추가 정보를 제공합니다.
                    5. 관련 규정이나 정책을 적절히 언급합니다.

                    [Reflection]
                    각 응답이 질문의 모든 측면을 충분히 다루고 있는지, 제공된 정보가 유용하고 관련성 있는지 확인합니다. 답변이 명확하고 이해하기 쉬운지 고려합니다.

                    [Feedback]
                    응답의 포괄성, 명확성, 유용성에 대한 피드백을 요청합니다. 추가 설명이나 정보가 필요한지 사용자가 알려줄 것을 요청합니다.

                    [Constraints]
                    1. 응답은 <제공된 문서>에만 기반해야 합니다.
                    2. 추측이나 개인적인 의견을 포함하지 않습니다.
                    3. <제공된 문서>에서 답을 할 수 없는 질문에 대해서는 '이 질문에 대한 정보는 제공된 문서에서 찾을 수 없으므로 답변할 수 없습니다.'라고 답해야 합니다.
                    4. 질문을 새로 생성하면 안됩니다.
                    5. 답변은 가능한 한 포괄적이고 정보가 풍부해야 하지만, 불필요하게 길어서는 안 됩니다.

                    [Context]
                    사용자는 대기업 KT의 프로젝트 관리자이며, 프로젝트 관리, 계약, 규정 준수, 기업 정책 등 다양한 주제에 대한 질문을 할 수 있습니다.
                    """

            # llm_answer_response 함수 수정
            def llm_answer_response(state: GraphState) -> GraphState:
                question = state["question"]
                law_context = state["law_context"]
                case_context = state["case_context"]
                law_name = state["law_name"]
                question_type = state["question_type"]  # 여기서 question_type을 가져옵니다
                
                context = f"법률 정보:\n{law_context}\n\n사례 정보:\n{case_context}\n\n"
                
                llm = ChatOpenAI(model_name=model_name, temperature=0)
                
                law_specific_prompt = LAW_SPECIFIC_PROMPTS.get(law_name, "전문 사내 변호사입니다.")
                
                # 질문 유형에 따른 프롬프트 선택
                type_specific_prompt = select_persona_prompt(question_type)
                task_specific_prompt = select_task_prompt(question_type)
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "이 프롬프트의 목적은 대기업 KT의 프로젝트 관리자로부터 오는 법적 질문에 대해서 법적 준수를 보장하는 답변을 제공하는 것입니다. " + 
                    f" AI는 꼼꼼하게 법적 사실을 확인하고, 법적으로 문제가 되지 않을 가능성과 법을 위반할 가능성이 있는 모든 행동에 대해 명확히 안내하고 <제공된 문서>를 기반으로 상세한 설명을 제공해야 합니다. <제공된 문서>: " + "{context}"),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "[Persona]" + f"{law_specific_prompt} {type_specific_prompt}" +
                    "[Input] AI는 다음 형식의 질문을 받게 됩니다: " + "{question}" + f"{task_specific_prompt}")
                ])

                chain = prompt | llm

                chain_with_memory = RunnableWithMessageHistory(
                    chain,
                    lambda _: get_session_history(session_id),
                    input_messages_key="question",
                    history_messages_key="history",
                )

                response = chain_with_memory.invoke(
                    {
                        "question": question,
                        "context": context,
                    },
                    config={"configurable": {"session_id": session_id}},
                )
                
                memory.chat_memory.add_user_message(question)
                memory.chat_memory.add_ai_message(response.content)
                
                state["response"] = response.content
                return state

            def select_final_prompt(question_type):
                if question_type == "법 저촉 여부 상황 판단":
                    return """
                    당신은 여러 법률 전문가의 의견을 종합하여 최종 답변을 제시하는 역할입니다. 제공된 법령별 검토 결과를 바탕으로 종합적이고 정확한 답변을 제공해야 합니다. 답변 시 다음 구조와 지침을 따르세요:

                    ### 결론
                    - 모든 법령의 검토 결과를 종합한 질문에 대한 최종 답변
                    - 문제 될 수 있는 관점에서 종합 결론
                    - 문제되지 않을 수 있는 관점에서 종합 결론
                    - 주의해야 할 점

                    ### 법령별 검토 결과
                    (관련된 각 법률에 대해 다음 구조로 작성)
                    [숫자]. [법률 이름] 검토
                    (해당 법률 검토시 문제될 수 있는 관점과 문제되지 않을 수 있는 관점을 요약하여 제시)

                    (위 구조를 관련된 모든 법률에 대해 반복)

                    ### 권고사항
                    (모든 법률에 대해 문제될 수 있는 관점과 문제되지 않을 수 있는 관점을 고려하여 종합적인 권고사항 제시)

                    ### 주의사항
                    (법적 해석의 한계, 추가 법률 자문의 필요성 등 언급)

                    ### 관련 법령
                    (분석에 사용된 모든 법령 조항을 정확히 나열. 법명과 조항은 1세트로 계속 같이 나와야 함. (예: 참고 법령: 산업안전보건법 제26조, 산업안전보건법 시행규칙 제27조, 산업안전보건기준에 관한 규칙 제28조, 중대재해처벌법 제8조, 상생협력법 제20조의2, 청탁금지법 시행령 제26조, 청탁금지법 시행령 별표2))

                    답변 작성 시 다음 사항을 준수하세요:
                    1. 제공된 법령별 검토 결과만을 사용하여 답변을 작성하세요.
                    2. 각 법률의 관점을 균형있게 고려하여 종합적인 답변을 제공하세요.
                    3. 법령 간 충돌이 있는 경우, 이를 명시하고 가장 적절한 해석을 제시하세요.
                    4. 확실하지 않은 부분에 대해서는 명확히 언급하세요.
                    5. 모든 법령 조항을 정확히 명시하세요. 법명과 조항은 1세트로 계속 같이 명시하세요. (예: 참고 법령: 산업안전보건법 제26조, 산업안전보건법 시행규칙 제27조, 산업안전보건기준에 관한 규칙 제28조, 중대재해처벌법 제8조, 상생협력법 제20조의2, 청탁금지법 시행령 제26조, 청탁금지법 시행령 별표2)
                    6. 질문과 관련이 없는 법률은 분석에서 제외하세요.
                    7. 각 법률 분석은 간결하면서도 충분한 정보를 포함하도록 하세요.
                    """
                
                elif question_type == "단순 질의응답":
                    return """
                    당신은 여러 법률 전문가의 의견을 종합하여 최종 답변을 제시하는 역할입니다. 제공된 법령별 검토 결과들을 바탕으로 종합적이고 정확한 답변을 제공해야 합니다. 답변 시 다음 구조와 지침을 따르세요:

                    ### 요약 답변
                    - 질문에 대한 간략하고 직접적인 답변
                    - 핵심 포인트 나열 (2-3개)

                    ### 법령별 검토 결과
                    (관련된 각 법률에 대해 다음 구조로 작성)
                    [숫자]. [법률 이름] 검토

                    (위 구조를 관련된 모든 법률에 대해 반복)

                    ### 주의사항
                    - 답변의 한계 또는 예외 상황 언급
                    - 추가 확인이 필요한 사항 안내

                    ### 관련 법령
                    - 답변에 사용된 모든 정보 소스를 정확히 나열
                    (예: 참고 법령: 산업안전보건법 제26조, 산업안전보건법 시행규칙 제27조, 산업안전보건기준에 관한 규칙 제28조, 중대재해처벌법 제8조, 상생협력법 제20조의2, 청탁금지법 시행령 제26조, 청탁금지법 시행령 별표2)

                    답변 작성 시 다음 사항을 준수하세요:
                    1. 제공된 법령별 검토 결과의 내용만을 사용하여 답변을 작성하세요.
                    2. 질문의 모든 측면을 균형있게 다루어 종합적인 답변을 제공하세요.
                    3. 정보 간 불일치가 있는 경우, 이를 명시하고 가장 신뢰할 수 있는 정보를 제시하세요.
                    4. 확실하지 않은 부분에 대해서는 명확히 언급하세요.
                    5. 모든 법령 조항을 정확히 명시하세요. 법명과 조항은 1세트로 계속 같이 명시하세요. (예: 참고 법령: 산업안전보건법 제26조, 산업안전보건법 시행규칙 제27조, 산업안전보건기준에 관한 규칙 제28조, 중대재해처벌법 제8조, 상생협력법 제20조의2, 청탁금지법 시행령 제26조, 청탁금지법 시행령 별표2)
                    6. 질문과 관련이 없는 정보는 답변에서 제외하세요.
                    7. 각 설명은 간결하면서도 충분한 정보를 포함하도록 하세요.
                    8. 전문 용어를 사용할 경우, 필요에 따라 간단한 설명을 추가하세요.
                    9. 답변은 객관적이고 중립적인 톤을 유지하세요.
                    10. 필요한 경우, 추가 문의나 전문가 상담을 권장하세요.
                    """
                
                elif question_type == "금액 계산":
                    return """
                    당신은 여러 법률 전문가의 의견을 종합하여 정확한 금액 계산 결과를 제시하는 역할입니다. 제공된 법령별 검토 결과를 바탕으로 상세하고 정확한 계산 결과를 제공해야 합니다. 답변 시 다음 구조와 지침을 따르세요:

                    ### 계산 결과 요약
                    - 최종 계산된 금액
                    - 계산 결과에 대한 간략한 설명 (1-2문장)

                    ### 입력 정보
                    - 계산에 사용된 모든 변수와 값을 나열
                    - 각 변수의 출처 또는 근거 명시

                    ### 계산 과정
                    (각 단계별로 다음 구조로 작성)
                    [숫자]. [계산 단계 설명]
                    - 사용된 공식 또는 규칙
                    - 세부 계산 과정
                    - 중간 결과값

                    (위 구조를 모든 주요 계산 단계에 대해 반복)

                    ### 법령별 검토 결과
                    (관련된 각 법률에 대해 다음 구조로 작성)
                    [숫자]. [법률 이름] 검토

                    (위 구조를 관련된 모든 법률에 대해 반복)

                    ### 결과 해석
                    - 계산 결과의 의미 설명
                    - 결과가 미치는 영향이나 중요성 언급

                    ### 주의사항
                    - 계산 결과 적용 시 고려해야 할 제한사항
                    - 예외 상황이나 변동 가능성 언급

                    ### 추가 고려사항
                    - 계산에 영향을 줄 수 있는 기타 요소
                    - 필요한 경우 대안적 계산 방법 제시

                    ### 관련 법령
                    - 답변에 사용된 모든 법령과 사례를 정확히 나열. 법명과 조항은 1세트로 계속 같이 명시 (예: 참고 법령: 산업안전보건법 제26조, 산업안전보건법 시행규칙 제27조, 산업안전보건기준에 관한 규칙 제28조, 중대재해처벌법 제8조, 상생협력법 제20조의2, 청탁금지법 시행령 제26조, 청탁금지법 시행령 별표2)

                    답변 작성 시 다음 사항을 준수하세요:
                    1. 제공된 법령별 검토 결과의 내용과 관련 규정만을 사용하여 계산을 수행하세요.
                    2. 모든 계산 단계를 명확하고 상세하게 설명하세요.
                    3. 사용된 모든 변수와 값의 출처를 명확히 밝히세요.
                    4. 계산 과정에서 가정이나 추정이 필요한 경우, 이를 명시하고 그 근거를 제시하세요.
                    5. 최종 결과뿐만 아니라 중간 계산 결과도 제시하세요.
                    6. 계산 결과의 의미와 영향을 설명하세요.
                    7. 계산 결과의 한계나 주의사항을 명확히 언급하세요.
                    8. 필요한 경우, 추가 검토나 전문가 확인을 권장하세요.
                    9. 모든 금액은 원 단위까지 정확히 계산하고, 필요에 따라 반올림 여부를 명시하세요.
                    10. 복잡한 계산의 경우, 단계별로 나누어 설명하세요.
                    """
                
                else:  # "그 외"
                    return """
                    당신은 여러 법률 전문가의 의견을 종합하여 다양한 주제에 대한 질문에 종합적이고 정확한 답변을 제공하는 역할입니다. 제공된 법령별 검토 결과의 내용을 바탕으로 포괄적이고 유용한 정보를 제공해야 합니다. 답변 시 다음 구조와 지침을 따르세요:

                    ### 핵심 답변
                    - 질문의 핵심에 대한 간결하고 직접적인 답변
                    - 주요 포인트 요약 (2-3개)

                    ### 법령별 검토 결과
                    (관련된 각 법률에 대해 다음 구조로 작성)
                    [숫자]. [법률 이름] 검토

                    (위 구조를 관련된 모든 법률에 대해 반복)

                    ### 주의사항 및 제한점
                    - 제공된 정보의 한계 또는 예외 상황 언급
                    - 추가 확인이 필요한 사항 안내

                    ### 추천 사항 또는 다음 단계
                    - 질문과 관련하여 권장되는 행동이나 절차
                    - 추가 정보를 얻을 수 있는 방법 제안

                    ### 관련 법령
                    - 답변에 사용된 모든 법령과 사례를 정확히 나열. 법명과 조항은 1세트로 계속 같이 명시 (예: 참고 법령: 산업안전보건법 제26조, 산업안전보건법 시행규칙 제27조, 산업안전보건기준에 관한 규칙 제28조, 중대재해처벌법 제8조, 상생협력법 제20조의2, 청탁금지법 시행령 제26조, 청탁금지법 시행령 별표2)

                    답변 작성 시 다음 사항을 준수하세요:
                    1. 제공된 법령별 검토 결과의 내용만을 사용하여 답변을 작성하세요.
                    2. 질문의 모든 측면을 균형있게 다루어 종합적인 답변을 제공하세요.
                    3. 정보 간 불일치가 있는 경우, 이를 명시하고 가장 신뢰할 수 있는 정보를 제시하세요.
                    4. 확실하지 않은 부분에 대해서는 명확히 언급하세요.
                    5. 모든 참고 자료와 출처를 정확히 명시하세요.
                    6. 질문과 관련이 없는 정보는 답변에서 제외하세요.
                    7. 각 설명은 간결하면서도 충분한 정보를 포함하도록 하세요.
                    8. 전문 용어를 사용할 경우, 필요에 따라 간단한 설명을 추가하세요.
                    9. 답변은 객관적이고 중립적인 톤을 유지하세요.
                    10. 필요한 경우, 추가 문의나 전문가 상담을 권장하세요.
                    11. 질문의 성격에 따라 답변 구조를 유연하게 조정하세요.
                    """

            def generate_final_answer(question: str, law_answers: List[GraphState]) -> GraphState:
                # "답변할 수 없습니다"가 포함되지 않은 유효한 답변만 필터링
                valid_answers = [ans for ans in law_answers if "답변할 수 없습니다" not in ans['response'] and ans['law_name'] in ans['response']]
                
                if not valid_answers:
                    return GraphState(
                        question=question,
                        law_context="",
                        case_context="",
                        response="모든 법률에 대해 유효한 답변을 생성할 수 없으므로 답변할 수 없습니다.",
                        relevance="",
                        attempts=0,
                        law_name="종합",
                        law_answers=law_answers
                    )
                
                combined_answers = "\n\n".join([f"법률: {ans['law_name']}\n답변: {ans['response']}" for ans in valid_answers])
                combined_context = "\n\n".join([f"법률: {ans['law_name']}\n법령 정보: {ans['law_context']}\n사례 정보: {ans['case_context']}" for ans in valid_answers])
                
                llm = ChatOpenAI(model_name=model_name, temperature=0)

                prompt = ChatPromptTemplate.from_messages([
                    ("system", select_final_prompt(st.session_state.question_type_select)),
                    ("human", "질문: {question}\n\n각 법령별 검토 결과:\n{combined_answers}\n\n법령 및 사례 정보:\n{combined_context}")
                ])
                
                chain = prompt | llm
                final_response = chain.invoke({
                    "question": question,
                    "combined_answers": combined_answers,
                    "combined_context": combined_context
                })
                
                mentioned_laws = list(set(re.findall(r'((?:\w+법|(?:\w+기준에 관한 규칙))(?:\s+시행(?:령|규칙))?)\s+((?:제\d+조(?:의\d+)?(?:\s*제\d+항)?(?:\s*제\d+호)?(?:\s*[가-힣]목)?)|(?:별표\s*\d+))', final_response.content)))
                print(f"추출된 법령 목록 (중복 제거): {mentioned_laws}")

                final_state = GraphState(
                    question=question,
                    law_context=combined_context,
                    case_context="",
                    response=final_response.content,
                    relevance="",
                    attempts=0,
                    law_name="종합",
                    law_answers=law_answers,
                    mentioned_laws=mentioned_laws,  # 언급된 법령 조항 추가
                    law_references=[],  # 초기화
                    similar_cases=[]  # 초기화
                )
                
                return final_state

            def check_relevance(state: GraphState) -> GraphState:
                result_input = {"context": state["question"] + state["law_context"] + state["case_context"], "answer": state["response"]}
                relevance_result = upstage_ground_checker.invoke(result_input)
                state["relevance"] = relevance_result
                return state

            def rephrase_question(state: GraphState) -> GraphState:
                llm = ChatOpenAI(model_name=model_name, temperature=0.7)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "당신은 법률 질문을 더 명확하고 구체적으로 만드는 전문가입니다. 주어진 질문을 다시 작성하여 법적 맥락에서 더 정확한 답변을 얻을 수 있도록 해주세요."),
                    ("human", "원래 질문: {question}\n\n이 질문을 법적 맥락에서 더 명확하고 구체적으로 다시 작성해주세요.")
                ])
                
                chain = prompt | llm
                response = chain.invoke({"question": state["question"]})
                
                new_state = state.copy()
                new_state["question"] = response.content
                new_state["attempts"] += 1
                return new_state
            
            def should_continue(state: GraphState) -> str:
                if state["relevance"] == "grounded":
                    return "end"
                elif state["relevance"] == "notGrounded" or state["attempts"] >= 3:
                    return "no_answer"
                elif state["relevance"] == "notSure":
                    return "rephrase"
                else:
                    return "rephrase"
                
            def display_feedback_buttons(i, question, answer, selected_laws):
                user_id = st.session_state.get("logged_in_user", "Unknown")
                feedback_key = f"{user_id}_{question}_{answer}"
                
                if "feedback_states" not in st.session_state:
                    st.session_state.feedback_states = {}
                
                current_feedback = st.session_state.feedback_states.get(feedback_key)
            
                col1, col2 = st.columns([9, 1])
                with col2:
                    st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
                    st.markdown('<div class="feedback-buttons">', unsafe_allow_html=True)
                    like, dislike = st.columns(2)
                    with like:
                        if st.button("👍", key=f"like_{i}", help="좋아요"):
                            new_feedback = "좋아요" if current_feedback != "좋아요" else None
                            st.session_state.feedback_states[feedback_key] = new_feedback
                            st.session_state[f"show_text_feedback_{i}"] = True
                            st.rerun()
                    with dislike:
                        if st.button("👎", key=f"dislike_{i}", help="싫어요"):
                            if current_feedback != "싫어요":
                                st.session_state.feedback_states[feedback_key] = "싫어요"
                                st.session_state[f"show_text_feedback_{i}"] = True
                            else:
                                st.session_state.feedback_states[feedback_key] = None
                                st.session_state[f"show_text_feedback_{i}"] = False
                            st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col1:
                    if current_feedback:
                        st.markdown(f'<div class="feedback-message">{current_feedback} 피드백이 제출되었습니다. 감사합니다.</div>', unsafe_allow_html=True)
                    
                    if st.session_state.get(f"show_text_feedback_{i}", False):
                        feedback_prompt = "[좋았던 점이나 개선이 필요한 점을 자유롭게 입력해주세요.]" if current_feedback == "좋아요" else "아쉬웠던 의견을 말씀해주시면 반영하여 개선하겠습니다."
                        text_feedback = st.text_area(feedback_prompt, key=f"text_feedback_{i}")
                        if st.button("피드백 제출", key=f"submit_text_feedback_{i}"):
                            save_feedback(user_id, question, answer, current_feedback, selected_laws, st.session_state.question_type_select, text_feedback)
                            st.session_state[f"show_text_feedback_{i}"] = False
                            st.success("피드백이 제출되었습니다. 감사합니다.")
                            st.rerun()
                
            def save_feedback(user_id, question, answer, feedback_type, selected_laws, question_type, text_feedback=None):
                filename = 'chatbot_feedback.csv'
                if feedback_type is None:
                    # 피드백 삭제 로직
                    if os.path.exists(filename):
                        df = pd.read_csv(filename)
                        df = df[(df['User ID'] != user_id) | (df['Question'] != question) | (df['Answer'] != answer)]
                        df.to_csv(filename, index=False, encoding='utf-8-sig')
                else:
                    # 기존 피드백 저장 또는 업데이트 로직
                    kst = timezone(timedelta(hours=9))
                    timestamp = datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S")
                    selected_laws_str = ", ".join(selected_laws)
                    new_feedback = pd.DataFrame([[user_id, timestamp, question, answer, feedback_type, selected_laws_str, question_type, text_feedback]], 
                                                columns=['User ID', 'Timestamp', 'Question', 'Answer', 'Feedback', 'Selected Laws', 'Question Type', 'Text Feedback'])
                    if os.path.exists(filename):
                        df = pd.read_csv(filename)
                        df = df[(df['User ID'] != user_id) | (df['Question'] != question) | (df['Answer'] != answer)]
                        df = pd.concat([df, new_feedback], ignore_index=True)
                    else:
                        df = new_feedback
                    df.to_csv(filename, index=False, encoding='utf-8-sig')

            def filter_question(question, question_type):
                llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

                if question_type == "<자동 분류>":
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", """당신은 법률 전문가입니다. 주어진 질문을 분석하여 다음 작업을 수행해야 합니다:
                        1. 질문이 청탁금지법, 공정거래법, 중대재해처벌법, 산업안전보건법, 하도급법, 상생협력법, 정보통신공사업법, 국가계약법, 소프트웨어진흥법 중 하나 이상과 관련이 있는지 판단하세요.
                        2. 질문의 유형을 다음 중 하나로 분류하세요:
                        a) 법 저촉 여부 상황 판단
                        b) 단순 질의응답
                        c) 금액 계산
                        d) 그 외
                        3. 질문이 법과 관련이 있고 유형이 정확히 분류되었다면 '예'를, 그렇지 않다면 '아니오'를 답하세요.

                        출력 형식:
                        관련성: [예/아니오]
                        질문 유형: [a/b/c/d]
                        최종 판단: [예/아니오]"""),
                        ("human", "다음 질문을 분석해주세요: {question}")])
                    chain = prompt | llm
                    response = chain.invoke({"question": question})
                    
                    # 응답 파싱
                    lines = response.content.split('\n')
                    relevance = lines[0].split(': ')[1].strip().lower()
                    detected_type = lines[1].split(': ')[1].strip()
                    final_judgment = lines[2].split(': ')[1].strip().lower()
                    
                    # 질문 유형 매핑
                    type_mapping = {
                        'a': "법 저촉 여부 상황 판단",
                        'b': "단순 질의응답",
                        'c': "금액 계산",
                        'd': "그 외"
                    }
                    
                    # 자동 모드일 경우 감지된 유형으로 설정
                    if question_type == "<자동 분류>":
                        question_type = type_mapping.get(detected_type, "그 외")
                    
                    # 최종 판단 결과 반환
                    if final_judgment == '예':
                        print(final_judgment)
                        print(question_type)
                        question_type_select = question_type
                        return final_judgment, question_type_select
                    
                    else: 
                        st.warning("법과 관련된 질문이 아니거나, 모든 법률에 대해 유효한 답변을 생성할 수 없으므로 답변할 수 없습니다.")

                else:        
                    if question_type == "법 저촉 여부 상황 판단":
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", "당신은 법률 전문가입니다. 주어진 질문이 특정 9가지 법(청탁금지법, 공정거래법, 중대재해처벌법, 산업안전보건법, 하도급법, 상생협력법, 정보통신공사업법, 국가계약법, 소프트웨어진흥법)에 관련된 법 저촉 여부를 판단하는 질문인지 확인해야 합니다."),
                            ("human", "다음 질문이 청탁금지법, 공정거래법, 중대재해처벌법, 산업안전보건법, 하도급법, 상생협력법, 정보통신공사업법, 국가계약법, 소프트웨어진흥법과 관련하여 법 저촉 여부를 판단해야 하는 질문인가요? 맞다면 '예', 아니면 '아니오'라고 답해주세요. 질문: {question}")
                        ])
                    elif question_type == "단순 질의응답":
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", "당신은 법률 전문가입니다. 주어진 질문이 특정 9가지 법(청탁금지법, 공정거래법, 중대재해처벌법, 산업안전보건법, 하도급법, 상생협력법, 정보통신공사업법, 국가계약법, 소프트웨어진흥법)에 관련된 일반적인 질문인지 확인해야 합니다."),
                            ("human", "다음 질문이 청탁금지법, 공정거래법, 중대재해처벌법, 산업안전보건법, 하도급법, 상생협력법, 정보통신공사업법, 국가계약법, 소프트웨어진흥법과 관련된 일반적인 질문인가요? 맞다면 '예', 아니면 '아니오'라고 답해주세요. 질문: {question}")
                        ])
                    elif question_type == "금액 계산":
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", "당신은 법률 전문가입니다. 주어진 질문이 특정 9가지 법(청탁금지법, 공정거래법, 중대재해처벌법, 산업안전보건법, 하도급법, 상생협력법, 정보통신공사업법, 국가계약법, 소프트웨어진흥법)에 관련된 금액 계산 질문인지 확인해야 합니다."),
                            ("human", "다음 질문이 청탁금지법, 공정거래법, 중대재해처벌법, 산업안전보건법, 하도급법, 상생협력법, 정보통신공사업법, 국가계약법, 소프트웨어진흥법과 관련된 금액 계산 질문인가요? 맞다면 '예', 아니면 '아니오'라고 답해주세요. 질문: {question}")
                        ])
                    else:  # "그 외"
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", "당신은 법률 전문가입니다. 주어진 질문이 특정 9가지 법(청탁금지법, 공정거래법, 중대재해처벌법, 산업안전보건법, 하도급법, 상생협력법, 정보통신공사업법, 국가계약법, 소프트웨어진흥법)과 관련이 있는지 확인해야 합니다."),
                            ("human", "다음 질문이 청탁금지법, 공정거래법, 중대재해처벌법, 산업안전보건법, 하도급법, 상생협력법, 정보통신공사업법, 국가계약법, 소프트웨어진흥법과 관련이 있나요? 맞다면 '예', 아니면 '아니오'라고 답해주세요. 질문: {question}")
                        ])
                    
                    chain = prompt | llm
                    response = chain.invoke({"question": question})
                    
                    return response.content.strip().lower().startswith("예")

            def exact_match_search(vectordb, search_term, k=1):
                docs = vectordb.docstore._dict.values()
                matched_docs = []
                for doc in docs:
                    content_lines = doc.page_content.split('\n')
                    if content_lines:
                        first_line = content_lines[0].strip()
                        # 시행령, 시행규칙, 별표, 규칙 등을 정확히 매칭
                        law_name, article = search_term.rsplit(' ', 1)
                        if law_name in first_line and '시행령' not in first_line and '시행규칙' not in first_line:
                            article_pattern = re.escape(article).replace(r'\\', '\\').replace(r'\d+', r'\d+')
                            if re.search(article_pattern, first_line):
                                matched_docs.append((doc, 0))
                            elif article.split()[0] in first_line:
                                matched_docs.append((doc, 1))
                        else:
                            if search_term.replace(" ", "") in first_line.replace(" ", ""):
                                matched_docs.append((doc, 2))
                
                # 우선순위에 따라 정렬하고 상위 k개 반환
                return [doc for doc, _ in sorted(matched_docs, key=lambda x: x[1])[:k]]
    

            # 법령 이름과 조항 번호로 정렬
            def sort_key(doc):
                content_lines = doc.page_content.split('\n')
                if content_lines:
                    first_line = content_lines[0].strip()
                    match = re.search(law_pattern, first_line)
                    if match:
                        law_name, article = match.groups()
                        article_num = re.search(r'\d+', article)
                        return (law_name, int(article_num.group()) if article_num else 0)
                return (doc.metadata['source'], 0)
      
            # LangGraph 워크플로우 정의
            workflow = StateGraph(GraphState)

            workflow.add_node("generate_answer", llm_answer_response)
            workflow.add_node("check_relevance", check_relevance)
            workflow.add_node("rephrase_question", rephrase_question)
            workflow.set_entry_point("generate_answer")
            workflow.add_edge("generate_answer", "check_relevance")
            workflow.add_conditional_edges(
                "check_relevance",
                should_continue,
                {
                    "end": END,
                    "no_answer": END,
                    "rephrase": "rephrase_question"
                }
            )
            workflow.add_edge("rephrase_question", "generate_answer")

            app = workflow.compile()

            # 대화 내역 표시
            displayed_messages = set()  # 이미 표시된 메시지를 추적하기 위한 집합
            
            for i, message in enumerate(user_state["messages"]):
                message_key = (message["role"], message["content"])
                
                if message_key not in displayed_messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                        
                        if message["role"] == "assistant":
                            if "law_references" in message and message["law_references"]:
                                with st.expander("참고 법령", expanded=False):
                                    for law_ref in message["law_references"]:
                                        st.markdown(f"**출처**: {law_ref['source']}")
                                        st.markdown(f"<div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-bottom: 10px; white-space: pre-wrap;'>{law_ref['content']}</div>", unsafe_allow_html=True)

                            if "similar_cases" in message and message["similar_cases"]:
                                with st.expander("유사 사례", expanded=False):
                                    for case in message["similar_cases"]:
                                        st.markdown(f"**출처**: {case['source']}, **페이지**: {case['page']}, **점수**: {case['score']:.2f}")
                                        st.markdown(f"<div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-bottom: 10px; white-space: pre-wrap;'>{case['content']}</div>", unsafe_allow_html=True)

                            if i//2 < len(user_state["relevance_results"]):
                                st.write(user_state["relevance_results"][i//2])
                            if i > 0:
                                display_feedback_buttons(i, user_state["messages"][i-1]["content"], message["content"], selected_laws)
                    
                    displayed_messages.add(message_key)

            if check_password():
                case_final_docs = []
                # Streamlit 인터페이스
                if user_input := st.chat_input("메세지를 입력해 주세요. "):
                    if not selected_laws:
                        st.warning("답변을 생성하기 전에 최소 1개 이상의 법률을 선택해주세요.")
                    elif filter_question(user_input, st.session_state.question_type_select):
                        user_message = {"role": "user", "content": user_input}
                        user_state["messages"].append(user_message)
                        st.session_state.messages.append(user_message)

                        st.chat_message("user").write(f"{user_input}")
                        
                        session_history = get_session_history(session_id)
                        session_history.add_user_message(user_input)
                        memory.chat_memory.add_user_message(user_input)

                        # 선택된 법령에 대한 vectorDB 생성
                        selected_law_vectordbs = create_selected_law_vectordbs(selected_laws)
                        selected_case_vectordbs = create_selected_case_vectordbs(selected_laws)
                        selected_for_show_law_vectordbs = create_selected_for_show_law_vectordbs(selected_laws)
                        
                        # 법령 및 사례별 리트리버 생성
                        law_retrievers = create_law_retrievers(selected_law_vectordbs)
                        case_retrievers = create_case_retrievers(selected_case_vectordbs)
                        
                        law_answers = []
                        for law in selected_laws:

                            law_context = law_retrievers[law](user_input)
                            case_context = case_retrievers[law](user_input)

                            initial_state = GraphState(
                                question=user_input,
                                law_context="\n".join([doc.page_content for doc in law_context]),
                                case_context="\n".join([doc.page_content for doc in case_context]),
                                response="",
                                relevance="",
                                attempts=0,
                                law_name=law,
                                law_references=[],
                                similar_cases=[],
                                question_type=st.session_state.question_type_select 
                            )

                            result = app.invoke(initial_state)
                            law_answers.append(result)

                        final_answer = generate_final_answer(user_input, law_answers)

                        if final_answer["response"] == "모든 법률에 대해 유효한 답변을 생성할 수 없으므로 답변할 수 없습니다.":
                            with st.chat_message("assistant"):
                                st.write(final_answer["response"])
                                
                                # AI 응답 추가
                                ai_message = {"role": "assistant", "content": final_answer["response"]}
                                user_state["messages"].append({"role": "assistant", "content": final_answer["response"]})
                                st.session_state.messages.append(ai_message)
                                
                                session_history.add_ai_message(final_answer["response"])
                                memory.chat_memory.add_ai_message(final_answer["response"])
                                
                                display_feedback_buttons(len(st.session_state.messages)-1, user_input, final_answer["response"], selected_laws)
                        else:
                            with st.chat_message("assistant"):
                                st.write(final_answer["response"])
                                # 참고 법령 처리 및 저장 부분
                                with st.expander("참고 법령", expanded=False):
                                    all_law_docs = []
                                    first_law_doc = None
                                    law_pattern = r'((?:\w+법|(?:\w+기준에 관한 규칙))(?:\s+시행(?:령|규칙))?)\s+((?:제\d+조(?:의\d+)?(?:\s*제\d+항)?(?:\s*제\d+호)?(?:\s*[가-힣]목)?)|(?:별표\s*\d+))'
                                    mentioned_laws = final_answer.get("mentioned_laws", [])
                                    
                                    # 모든 법령 처리
                                    for i, mentioned_law in enumerate(mentioned_laws):
                                        if isinstance(mentioned_law, tuple):
                                            full_law_name, law_reference = mentioned_law
                                        else:
                                            match = re.search(law_pattern, mentioned_law)
                                            if match:
                                                full_law_name, law_reference = match.groups()
                                            else:
                                                continue
                                        law_name = full_law_name.split()[0] if "시행령" in full_law_name or "시행규칙" in full_law_name else full_law_name
                                        if any(law_name.startswith(selected_law) for selected_law in selected_laws):
                                            search_term = f"{full_law_name} {law_reference}"
                                            law_docs = exact_match_search(selected_for_show_law_vectordbs[law_name], search_term, k=1)
                                            if not law_docs:
                                                broader_search_term = f"{full_law_name} {law_reference.split()[0]}"  # 조항의 첫 부분만 사용
                                                law_docs = exact_match_search(selected_for_show_law_vectordbs[law_name], broader_search_term, k=1)
                                            
                                            if i == 0 and law_docs:  # 첫 번째로 언급된 법령
                                                first_law_doc = law_docs[0]
                                            else:
                                                all_law_docs.extend(law_docs)

                                    # 중복 제거 (원래 순서 유지)
                                    unique_law_docs = []
                                    seen = set()
                                    if first_law_doc:
                                        unique_law_docs.append(first_law_doc)
                                        seen.add((first_law_doc.page_content, first_law_doc.metadata['source'], first_law_doc.metadata['page']))
                                    
                                    for doc in all_law_docs:
                                        key = (doc.page_content, doc.metadata['source'], doc.metadata['page'])
                                        if key not in seen:
                                            seen.add(key)
                                            unique_law_docs.append(doc)

                                    unique_law_docs.sort(key=sort_key)
                                    
                                    law_refs = []
                                    for law_doc in unique_law_docs:
                                        law_name = law_doc.metadata['source']
                                        st.markdown(f"**출처**: {law_name}")
                                        content = law_doc.page_content.replace("[[", "\n<hr>")
                                        
                                        lines = content.split('\n')
                                        if lines:
                                            first_line = lines[0].strip()
                                            match = re.match(law_pattern, first_line)
                                            if match:
                                                law_name, article = match.groups()
                                                formatted_first_line = f"{law_name} {article.replace(' ', '')}"
                                                formatted_content = f"<strong>{formatted_first_line}</strong><br><br>" + '\n'.join(lines[1:])
                                            else:
                                                formatted_content = content
                                        else:
                                            formatted_content = content
                                    
                                        st.markdown(f"<div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-bottom: 10px; white-space: pre-wrap;'>{formatted_content}</div>", unsafe_allow_html=True)                                        
                                        law_refs.append({
                                            'source': law_name,
                                            'page': law_doc.metadata['page'],
                                            'content': content
                                        })

                                    final_answer["law_references"] = law_refs

                                # 유사 사례 처리 및 저장
                                with st.expander("유사 사례", expanded=False):
                                    combined_query = " ".join([m.content for m in memory.chat_memory.messages])
                                    preprocessed_query = preprocess_text(combined_query)
                                    
                                    if not isinstance(preprocessed_query, str):
                                        preprocessed_query = str(preprocessed_query)
                                
                                    similar_cases = []
                                    all_case_docs_with_scores = []
                                    all_case_keyword_docs = []
                                    
                                    for law in selected_laws:
                                        if any(ans['law_name'] in ans['response'] and "답변할 수 없습니다" not in ans['response'] for ans in final_answer["law_answers"] if ans['law_name'] == law):
                                            case_retriever = case_retrievers[law]
                                            if callable(case_retriever):  # 빈 문서에 대한 처리
                                                case_docs_with_scores = []
                                                case_keyword_docs = []
                                            else:
                                                case_docs_with_scores = selected_case_vectordbs[law].similarity_search_with_score(preprocessed_query, k=3)
                                                case_keyword_docs = case_retriever.invoke(preprocessed_query, k=3)
                                            
                                            all_case_docs_with_scores.extend(case_docs_with_scores)
                                            all_case_keyword_docs.extend(case_keyword_docs)
                                    
                                    case_final_docs = []
                                    for case_doc, case_score in all_case_docs_with_scores:
                                        case_keyword_score = 0
                                        for case_keyword_doc in all_case_keyword_docs:
                                            if case_doc.metadata['source'] == case_keyword_doc.metadata['source'] and case_doc.metadata['page'] == case_keyword_doc.metadata['page']:
                                                case_keyword_score = 1
                                                break
                                        case_final_score = case_score + case_keyword_score
                                        case_final_docs.append((case_doc, case_final_score))
                                    
                                    case_final_docs = sorted(case_final_docs, key=lambda x: x[1], reverse=True)
                                    case_final_docs = [(doc, score) for doc, score in case_final_docs if score >= 1]
                                    case_final_docs = case_final_docs[:10]

                                    for case_doc, case_score in case_final_docs:
                                        case_pdf_name = case_doc.metadata['law_name']
                                        st.markdown(f"**출처**: {case_pdf_name}, **페이지**: {case_doc.metadata['page']}, **점수**: {case_score:.2f}")
                                        content = case_doc.page_content.replace("[[", "\n<hr>")
                                        st.markdown(f"<div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-bottom: 10px; white-space: pre-wrap;'>{content}</div>", unsafe_allow_html=True)
                                        
                                        similar_cases.append({
                                            'source': case_pdf_name,
                                            'page': case_doc.metadata['page'],
                                            'score': case_score,
                                            'content': content
                                        })

                                    final_answer["similar_cases"] = similar_cases

                                # AI 응답 추가 (참고법령과 유사사례 포함)
                                ai_message = {
                                    "role": "assistant", 
                                    "content": final_answer["response"],
                                    "law_references": final_answer.get("law_references", []),
                                    "similar_cases": final_answer.get("similar_cases", [])
                                }
                                user_state["messages"].append(ai_message)
                                st.session_state.messages.append(ai_message)
                                
                                session_history.add_ai_message(final_answer["response"])
                                memory.chat_memory.add_ai_message(final_answer["response"])
                                
                                display_feedback_buttons(len(st.session_state.messages)-1, user_input, final_answer["response"], selected_laws)

                            st.write("---")
                            st.write("아래는 가장 최근 질문에 대한 선택하신 법령별 관련성 검토 결과입니다.")
                            # 법령별 검토 결과를 가장 마지막에 표시
                            for law_answer in final_answer["law_answers"]:
                                with st.expander(f"{law_answer['law_name']} 검토 결과", expanded=False):
                                    if law_answer['law_name'] not in law_answer['response'] or "답변할 수 없습니다" in law_answer['response']:
                                        st.write(f"이 질문은 {law_answer['law_name']}과 관련성이 낮은 것으로 판단되어 답변할 수 없습니다.")
                                    else:
                                        st.write(law_answer["response"])
                                        if law_answer['relevance'] == "grounded":
                                            st.write("관련성 검사: 근거 문서에 기반한 답변입니다.")
                                        elif law_answer['relevance'] == "notGrounded":
                                            st.write("관련성 검사: 이 답변은 근거 문서들과의 연관성이 낮게 측정된 답변입니다. 참고하시기 바랍니다.")
                                        else:
                                            st.write(f"관련성 검사: 더 좋은 의견을 제공해드리기 위해, 질문을 재생성하고 답변을 제공하였습니다. 참고하시기 바랍니다.")
                    else:
                        st.warning("법과 관련된 질문이 아니거나, 모든 법률에 대해 유효한 답변을 생성할 수 없으므로 답변할 수 없습니다.")

                if "feedback_message" in st.session_state:
                    st.success(st.session_state["feedback_message"])
                    del st.session_state["feedback_message"]
                    
if __name__ == "__main__":
    main()
