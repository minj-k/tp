import streamlit as st
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
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
st.write("통합된 지식 베이스를 기반으로 질문에 답변합니다.")

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
    
    # 1. 임베딩 모델 준비
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=google_api_key
    )
    
    # 2. FAISS 인덱스 로드
    vector_store = FAISS.load_local(
        folder_path=index_path, 
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    # 3. LLM 및 프롬프트 준비
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key, temperature=0.3)
    
    prompt = ChatPromptTemplate.from_template("""
     당신은 제공된 문서를 기반으로 질문에 답변하는 전문 AI 어시스턴트입니다.
    오직 아래에 제공된 <context> 내용만을 사용하여 사용자의 질문에 답변해야 합니다.
    답변은 최대한 상세하고 명확하게 작성해주세요.
    만약 <context> 안에 질문에 대한 답변이 없다면, 추측하거나 외부 지식을 사용하지 말고 "제공된 문서에서는 해당 정보를 찾을 수 없습니다."라고만 답변해주세요.


    <context>
    {context}
    </context>

    Question: {input}
    """)
    
    # 4. Retrieval 체인 생성
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# --- 메인 로직 ---

# 항상 단일 인덱스를 로드
FAISS_INDEX_PATH = "faiss_index_combined"
retrieval_chain = load_retrieval_chain(FAISS_INDEX_PATH)

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="안녕하세요! 궁금한 점을 질문하시면 답변해 드릴게요."),
    ]

# 대화 기록 표시
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# 사용자 입력 처리
if user_query := st.chat_input("질문을 입력하세요..."):
    # 대화 기록에 사용자 질문 추가 및 화면에 표시
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    # AI 응답 생성 및 표시
    with st.chat_message("AI"):
        with st.spinner("답변을 생성하는 중입니다..."):
            response = retrieval_chain.invoke({
                "chat_history": st.session_state.chat_history,
                "input": user_query
            })
            
            if "answer" in response:
                st.write(response["answer"])
                st.session_state.chat_history.append(AIMessage(content=response["answer"]))
            else:
                st.error("답변을 생성하지 못했습니다.")
