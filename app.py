import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document # Document 타입을 명시하기 위해 추가

# --- 검색된 문서들을 하나의 문자열로 합쳐주는 함수 ---
def format_docs(docs: list[Document]) -> str:
    """검색된 Document 객체 리스트를 하나의 문자열로 결합합니다."""
    return "\n\n".join(doc.page_content for doc in docs)
# ---------------------------------------------------


# --- 초기 설정 (미리 만들어진 벡터DB 로드) ---
@st.cache_resource
def load_rag_chain():
    try:
        os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]
    except Exception as e:
        st.error("Streamlit Secrets에 GOOGLE_API_KEY가 설정되지 않았습니다.")
        st.stop()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
    st.sidebar.success("임베딩 DB 로드 완료!", icon="✅")

    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
    prompt_template_str = """
    당신은 전북테크노파크의 규정과 재정 지침을 분석하여 명확한 결론을 내리는 전문 분석가 AI입니다.
    아래의 프로세스에 따라 사용자의 질문에 답변해주세요.

    [프로세스]
    1. 사용자의 질문 의도를 명확히 파악합니다.
    2. 제공된 [Context]에서 질문과 관련된 모든 규정과 정보를 종합적으로 검토합니다.
    3. 검토한 내용을 바탕으로 사용자의 질문에 대해 "예, 가능합니다." 또는 "아니요, 불가능합니다." 와 같이 **명확하고 두괄식인 결론을 먼저 제시**합니다. 애매하거나 모호하게 답변하지 마세요.
    4. 왜 그런 결론이 나왔는지, **핵심적인 이유와 근거 규정을 2~3개의 항목으로 나누어 간결하게 설명**합니다. 각 항목은 번호를 붙여주세요.
    5. 마지막으로, 전체 내용을 한두 문장으로 요약하여 최종적으로 답변을 마무리합니다.

    [Context]
    {context}

    [Question]
    {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(prompt_template_str)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)

    # --- RAG 체인 구성 (수정된 부분) ---
    rag_chain = (
        # retriever의 검색 결과를 format_docs 함수로 넘겨 문자열로 변환
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    # ------------------------------------

    return rag_chain

# --- Streamlit UI 구성 및 실행 ---
try:
    rag_chain = load_rag_chain()
    st.set_page_config(page_title="규정 질의응답 챗봇", page_icon="📚")
    st.title("📚 예산 질의응답 챗봇")

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
