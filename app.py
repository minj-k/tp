# app.py

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import os

# --- 페이지 설정 ---
st.set_page_config(
    page_title="💰 예산관리 챗봇",
    page_icon="🤖",
    layout="wide",
)

# --- 제목 ---
st.title("💰 예산관리 챗봇")
st.write("질문 내용에 맞춰 가장 적합한 지식 베이스를 스스로 찾아 답변합니다.")

# --- API 키 설정 ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Google API 키를 .streamlit/secrets.toml에 설정해주세요!")
    st.stop()

# --- 지식 베이스 설명 정의 ---
# 각 FAISS 인덱스 폴더가 어떤 내용을 다루는지 LLM에게 알려주기 위한 설명입니다.
# 폴더 이름은 실제 프로젝트의 폴더 이름과 정확히 일치해야 합니다.
KNOWLEDGE_BASE_DESCRIPTIONS = {
    "faiss_index_ict": "정보통신기술(ICT)과 관련된 최신 기술, 트렌드, 용어에 대한 정보를 담고 있습니다.",
    "faiss_index_law": "법률, 규제, 판례 등 법과 관련된 전문적인 내용을 다룹니다.",
    "faiss_index_qa": "일반적인 질문과 답변(Q&A) 형식의 매뉴얼입니다.",
    "faiss_index_tp": "테크노파크(TP)의 규정 자료를 포함합니다."
}

# --- 핵심 기능 함수 ---

def get_best_knowledge_base(user_query):
    """사용자 질문에 가장 적합한 지식 베이스 폴더 이름을 결정하는 라우터 함수."""
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key, temperature=0)
    
    # 지식 베이스 설명을 프롬프트에 포함
    descriptions_text = "\n".join([f"- {name}: {desc}" for name, desc in KNOWLEDGE_BASE_DESCRIPTIONS.items()])
    
    prompt = ChatPromptTemplate.from_template(f"""
    당신은 사용자의 질문을 분석하여 가장 적합한 지식 베이스를 추천하는 전문가입니다.
    아래 지식 베이스 목록과 설명을 참고하여, 사용자의 질문에 가장 적합한 지식 베이스의 이름(폴더명) 단 하나만 정확히 출력해주세요.
    다른 설명이나 문장은 절대 추가하지 마세요.

    [지식 베이스 목록]
    {descriptions_text}

    [사용자 질문]
    {{question}}

    [가장 적합한 지식 베이스 이름]
    """)
    
    routing_chain = prompt | llm
    
    # .content를 통해 결과 문자열만 추출하고, 공백 제거
    result = routing_chain.invoke({"question": user_query}).content.strip()
    
    # 유효한 폴더 이름인지 확인
    if result in KNOWLEDGE_BASE_DESCRIPTIONS:
        return result
    else:
        # LLM이 예상 외의 답변을 할 경우, 기본값으로 fallback (예: 'faiss_index_qa')
        return "faiss_index_qa" 

@st.cache_resource(show_spinner="지식 베이스를 로딩하는 중입니다...")
def load_retrieval_chain(index_name):
    """선택된 이름의 FAISS 인덱스를 로드하고 Retrieval 체인을 생성합니다."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vector_store = FAISS.load_local(
        folder_path=index_name, 
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key, temperature=0.3)
    
    prompt = ChatPromptTemplate.from_template("""
    주어진 내용을 바탕으로 질문에 답변해주세요.
    내용에 없는 정보는 답변할 수 없다고 솔직하게 말해주세요.

    <context>
    {context}
    </context>

    Question: {input}
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# --- 세션 상태 초기화 ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="안녕하세요! 궁금한 점을 질문하시면 관련 지식 베이스를 찾아 답변해 드릴게요."),
    ]

# --- 대화 기록 표시 ---
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# --- 사용자 입력 처리 ---
if user_query := st.chat_input("질문을 입력하세요..."):
    # 대화 기록에 사용자 질문 추가 및 화면에 표시
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    # AI 응답 생성 및 표시
    with st.chat_message("AI"):
        with st.spinner("질문을 분석하고 적합한 지식 베이스를 찾는 중..."):
            # 1단계: 질문에 가장 적합한 지식 베이스 선택 (라우팅)
            selected_index = get_best_knowledge_base(user_query)
            st.info(f"'{KNOWLEDGE_BASE_DESCRIPTIONS[selected_index]}' 지식 베이스를 사용하여 답변을 찾고 있습니다.")

        with st.spinner(f"'{selected_index}'에서 답변을 생성하는 중입니다..."):
            # 2단계: 선택된 지식 베이스로 RAG 체인 로드 및 답변 생성
            retrieval_chain = load_retrieval_chain(selected_index)
            response = retrieval_chain.invoke({
                "chat_history": st.session_state.chat_history,
                "input": user_query
            })
            
            if "answer" in response:
                st.write(response["answer"])
                st.session_state.chat_history.append(AIMessage(content=response["answer"]))
            else:
                st.error("답변을 생성하지 못했습니다.")
