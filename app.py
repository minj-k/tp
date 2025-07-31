import streamlit as st
from langchain_openai import ChatOpenAI

# --- 페이지 설정 ---
# st.set_page_config()는 항상 가장 먼저 실행되어야 합니다.
st.set_page_config(
    page_title="☃️ 예산관리 챗봇",
    page_icon="🤖",
    layout="centered",
)

# --- 제목 ---
st.title("☃️ 예산관리 챗봇")
st.write("안녕하세요! 무엇이든 물어보세요.")

# --- API 키 설정 및 LLM 초기화 ---
# Streamlit의 secrets 기능을 사용하여 API 키를 안전하게 관리합니다.
try:
    llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model_name="gpt-4o-mini")
except Exception:
    st.error("OpenAI API 키를 설정해주세요! (.streamlit/secrets.toml)")
    st.stop()


# --- 세션 상태 초기화 ---
# 'messages'가 세션 상태에 없으면 빈 리스트로 초기화합니다.
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 이전 대화 기록 표시 ---
# 사용자가 새 입력을 하기 전에도 항상 이전 대화 내용을 표시합니다.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 사용자 입력 처리 ---
if prompt := st.chat_input("메시지를 입력하세요."):
    # 1. 사용자 메시지를 세션 상태와 화면에 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. AI 응답 생성 및 표시
    with st.chat_message("assistant"):
        # LangChain의 stream 기능을 사용하여 실시간으로 답변을 생성합니다.
        stream = llm.stream(st.session_state.messages)
        # st.write_stream을 통해 스트리밍 응답을 화면에 표시합니다.
        response = st.write_stream(stream)
    
    # 3. AI 응답을 세션 상태에 추가
    st.session_state.messages.append({"role": "assistant", "content": response})
