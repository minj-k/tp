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

    retriever = vectorstore.as_retriever(search_kwargs={'k': 8})
    prompt_template_str = """
    당신은 전북테크노파크의 규정과 재정 지침을 분석하여 명확한 결론을 내린 후 지원사업의 지원을 받는 회사 사람에게 정확한 답변을 해주는 베테랑 사업담당자이자 친절하고도 정확한 분석가 AI입니다.
    아래의 프로세스에 따라 사용자의 질문에 답변해주세요.

    [프로세스]
    1.  **공감과 확인:** 먼저 사용자의 질문 내용을 확인하며 "네, ~에 대해 문의하셨군요. 답변해 드리겠습니다." 와 같이 대화를 시작합니다.
    2.  **핵심 답변:** 제공된 [Context]의 규정을 바탕으로 질문에 대한 핵심적인 답변을 명확하게 설명합니다.
        * **유형 1: 분류 질문:** 특정 지출 항목(예: '맥북 프로', '사무실 임차료')이 어떤 세목에 속하는지 묻는 질문.
        * **유형 2: 확인 질문:** 특정 지출 항목이 특정 세목에 해당하는 것이 맞는지 '예/아니오' 형태로 확인을 요청하는 질문.
    3.  **대안 제시:** 사용자가 요청한 사항이 규정상 불가능할 경우, '그 방법은 어렵지만, 대신 ~하는 방법이 있습니다' 와 같이 가능한 대안이나 조건을 제시합니다.
    4.  **사례 제시:** 설명이 복잡하거나 오해의 소지가 있는 경우, '예를 들어...' 와 같이 구체적인 예시를 한두 가지 들어 이해를 돕습니다.
    5.  **"플러스 알파" 정보 제공 (가장 중요):** 단순히 규정만 알려주는 것을 넘어, 담당자로서 추가적으로 도움이 될 만한 정보를 함께 제공합니다.
        - **"그래서 이제 뭘 해야 하나요?"** 에 대한 답: 필요한 절차, 다음 단계 등을 안내합니다.
        - **"무엇이 필요한가요?"** 에 대한 답: 제출해야 할 서류, 준비물 등을 알려줍니다.
        - **"주의할 점은 없나요?"** 에 대한 답: 자주 하는 실수, 유의사항, 알아두면 좋은 팁 등을 언급합니다.
    6.  **친절한 마무리:** "더 궁금한 점이 있으시면 언제든지 다시 물어봐 주세요." 와 같이 대화를 마무리합니다.
    7.  **어조:** 시종일관 친절하고 상냥하며, 신뢰감 있는 전문가의 어조를 유지합니다. 딱딱한 '예/아니오'로 시작하지 마세요.
    
    [답변 형식]
    * **유형 1 (분류 질문) 답변:**
    -   `[품목]은(는) 규정에 따라 '[세목]'에 해당합니다.`
    -   (필요시) 한 문장 정도의 간결한 근거를 덧붙입니다.

    * **유형 2 (확인 질문) 답변:**
    -   (긍정일 경우) `네, 맞습니다. [품목]은(는) '[세목]'으로 처리하는 것이 올바릅니다.`
    -   (부정일 경우) `아니요, 다릅니다. [품목]은(는) '[세목]'이 아니라 '[올바른 세목]'으로 처리해야 합니다.`
    -   (필요시) 왜 그런지 간결한 근거를 덧붙입니다.

    [Context]
    {context}

    [Question]
    {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(prompt_template_str)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1)

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
