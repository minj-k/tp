import streamlit as st
import os
import asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- 비동기 이벤트 루프 설정 ---
# Streamlit 환경에서 asyncio를 원활하게 사용하기 위함
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# --- 페이지 설정 ---
st.set_page_config(
    page_title="� 예산관리 챗봇",
    page_icon="🤖",
    layout="wide",
)

# --- 제목 ---
st.title("💰 예산관리 챗봇")
st.write("내부 규정 문서(PDF) 기반 질의응답 시스템입니다.")

# --- API 키 설정 ---
# Streamlit의 secrets 관리 기능을 사용
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Google API 키를 .streamlit/secrets.toml 파일에 설정해주세요!")
    st.info("secrets.toml 파일 예시:\nGOOGLE_API_KEY = \"YOUR_API_KEY_HERE\"")
    st.stop()

# --- 상수 정의 ---
DATA_FOLDER = "data"
FAISS_INDEX_PATH = "faiss_index_combined"

# --- 핵심 기능 함수 ---

@st.cache_resource(show_spinner="지식 베이스(PDF)를 구축하고 있습니다...")
def build_or_load_vector_store(data_folder, index_path):
    """
    지정된 폴더의 PDF를 읽어 FAISS 인덱스를 생성하거나, 이미 존재하면 로드합니다.
    """
    # 1. 인덱스가 이미 존재하는지 확인
    if os.path.exists(index_path):
        st.info("기존 지식 베이스를 로드합니다.")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        vector_store = FAISS.load_local(
            folder_path=index_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store

    # 2. 인덱스가 없으면 새로 생성
    st.info("새로운 지식 베이스를 생성합니다.")
    all_documents = []
    pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
    
    if not pdf_files:
        st.error(f"'{data_folder}' 폴더에 PDF 파일이 없습니다. PDF 파일을 추가해주세요.")
        st.stop()

    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(data_folder, pdf_file))
        documents = loader.load()
        all_documents.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vector_store = FAISS.from_documents(texts, embeddings)
    vector_store.save_local(index_path)
    
    return vector_store

def create_conversational_rag_chain(vector_store):
    """
    VectorStore를 기반으로 대화형 RAG 체인을 생성합니다.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key, temperature=0.1)
    
    # Retriever 설정: 검색 결과 개수를 4개로 제한하여 API 요청 크기를 조절
    retriever = vector_store.as_retriever(search_kwargs={'k': 4})

    # 1. 질문 재구성 프롬프트 및 체인
    # 대화 기록을 바탕으로 후속 질문을 독립적인 질문으로 재구성합니다.
    contextualize_q_system_prompt = """주어진 대화 기록과 최근 사용자 질문을 바탕으로, 대화 기록을 참조할 필요가 없는 독립적인 질문으로 바꾸세요. 질문에 대한 답변은 하지 말고, 필요한 경우 질문을 재구성만 하세요."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. 답변 생성 프롬프트 및 체인
    # **전체 대화 기록(chat_history)을 제외하여 API 요청을 최소화합니다.**
    qa_system_prompt = """당신은 제공된 문서를 기반으로 질문에 답변하는 전문 AI 어시스턴트입니다.

    **중요 지침:**
    1.  **내용 기반 답변:** 아래에 제공된 <context> 내용만을 사용하여 사용자의 질문에 답변해야 합니다. <context>내에 정확하게 일치하지는 않지만 비슷한 내용이 있다면 최대한 사실에 기반하여 정답에 근접하게 대답을 하세요.
    2.  **표(Table) 데이터 활용:** <context>에 표 형식의 데이터가 있다면, 그 정보를 우선적으로 활용하여 질문에 답해야 합니다.
    3.  **답변 형식:** 답변은 최대한 상세하고 명확하게, 완전한 문장으로 작성해주세요.
    4.  **정보 부재 시:** 만약 <context> 안에 질문에 대한 답변을 찾을 수 없다면, "제공된 문서에서는 해당 정보를 찾을 수 없습니다."라고만 답변해주세요.

    <context>
    {context}
    </context>
    """
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"), # MessagesPlaceholder를 제거하고 사용자의 마지막 입력만 사용
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 3. 두 체인 결합
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# --- 메인 로직 ---

# 1. Vector Store 생성 또는 로드
vector_store = build_or_load_vector_store(DATA_FOLDER, FAISS_INDEX_PATH)

# 2. 대화형 RAG 체인 생성
conversational_rag_chain = create_conversational_rag_chain(vector_store)

# 3. 채팅 기록 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="안녕하세요! 내부 규정 문서에 대해 궁금한 점을 질문해주세요.")]

# 4. 채팅 기록 표시
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# 5. 사용자 입력 처리
if user_query := st.chat_input("질문을 입력하세요..."):
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        with st.spinner("답변을 생성하는 중입니다..."):
            # 응답을 받기 전에 사용자 질문을 기록에 추가
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            
            response = conversational_rag_chain.invoke(
                {"input": user_query, "chat_history": st.session_state.chat_history}
            )
            answer = response.get("answer", "답변을 생성하지 못했습니다.")
            st.write(answer)
            
            # 응답 받은 후, AI 답변을 기록에 추가
            st.session_state.chat_history.append(AIMessage(content=answer))
