import streamlit as st
import asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# --- 비동기 이벤트 루프 설정 ---
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# --- 페이지 설정 ---
st.set_page_config(
    page_title="💰 예산관리 챗봇",
    page_icon="🤖",
    layout="wide",
)

# --- 제목 ---
st.title("💰 예산관리 챗봇")
st.write("내부 규정 문서 기반 질의응답 시스템입니다.")

# --- API 키 설정 ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Google API 키를 .streamlit/secrets.toml에 설정해주세요!")
    st.stop()

# --- FAISS 인덱스 로드 및 체인 생성 함수 ---
@st.cache_resource(show_spinner="지식 베이스를 로딩하는 중입니다...")
def load_retrieval_chain(index_path):
    """지정된 경로의 FAISS 인덱스를 로드하고 Retrieval 체인을 생성합니다."""
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vector_store = FAISS.load_local(
        folder_path=index_path, 
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key, temperature=0.1)
    
    prompt = ChatPromptTemplate.from_template("""
    당신은 제공된 문서를 기반으로 질문에 답변하는 전문 AI 어시스턴트입니다.

    **중요 지침:**
    1.  **내용 기반 답변:** 아래에 제공된 <context> 내용만을 사용하여 사용자의 질문에 답변해야 합니다. 추측하거나 외부 지식을 사용하지 마세요.
    2.  **표(Table) 데이터 활용:** <context>에 '표 정보:'로 시작하는 내용이 있다면, 이는 표에서 추출된 정보입니다. 이 정보를 우선적으로 활용하여 질문에 답해야 합니다.
    3.  **답변 형식:** 답변은 최대한 상세하고 명확하게, 완전한 문장으로 작성해주세요.
    4.  **정보 부재 시:** 만약 <context> 안에 질문에 대한 답변을 찾을 수 없다면, "제공된 문서에서는 해당 정보를 찾을 수 없습니다."라고만 답변해주세요.

    <context>
    {context}
    </context>

    [사용자 질문]
    {input}

    [답변]
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# --- 메인 로직 ---
FAISS_INDEX_PATH = "faiss_index_combined"
retrieval_chain = load_retrieval_chain(FAISS_INDEX_PATH)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="안녕하세요! 내부 규정에 대해 궁금한 점을 질문해주세요.")]

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

if user_query := st.chat_input("질문을 입력하세요..."):
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        with st.spinner("답변을 생성하는 중입니다..."):
            response = retrieval_chain.invoke({"input": user_query})
            answer = response.get("answer", "답변을 생성하지 못했습니다.")
            st.write(answer)
            st.session_state.chat_history.append(AIMessage(content=answer))
