# app.py 의 전체 내용입니다. 이 코드로 교체해주세요.

import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 초기 설정 (미리 만들어진 벡터DB 로드) ---
@st.cache_resource
def load_rag_chain():
    try:
        os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]
    except Exception as e:
        st.error("Streamlit Secrets에 GOOGLE_API_KEY가 설정되지 않았습니다.")
        st.stop()

    # 미리 만들어둔 faiss_index 폴더를 로드합니다.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
    st.sidebar.success("임베딩 DB 로드 완료!", icon="✅")

    # RAG 체인 구성 (이전과 동일)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
    prompt_template_str = """
    당신은 제공된 문맥(context)을 바탕으로 사용자의 질문에 정확하고 근거 있는 답변을 하는 AI 어시스턴트입니다.
    답변은 반드시 주어진 문맥에 있는 정보만을 사용해야 합니다. 문맥에 없는 내용은 답변하지 말고, 정보가 없다고 솔직하게 말해주세요.

    [Context]
    {context}

    [Question]
    {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(prompt_template_str)
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- Streamlit UI 구성 및 실행 ---
try:
    rag_chain = load_rag_chain()
    st.set_page_config(page_title="규정 질의응답 챗봇", page_icon="📚")
    st.title("📚 규정 질의응답 챗봇 (미리 학습된 버전)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("질문을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("답변을 생성 중입니다..."):
            response = rag_chain.invoke(prompt)
            with st.chat_message("assistant"):
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
    st.info("API 키가 올바르게 설정되었는지, faiss_index 폴더가 제대로 업로드되었는지 확인해주세요.")