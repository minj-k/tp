import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever

# --- 검색된 문서들을 하나의 문자열로 합쳐주는 함수 ---
def format_docs(docs: list[Document]) -> str:
    """검색된 Document 객체 리스트를 하나의 문자열로 결합합니다."""
    formatted_docs = []
    for doc in docs:
        source_info = doc.metadata.get('source', '알 수 없는 출처')
        
        # [최종 확정] 출처 구분 로직 수정
        # 'faiss_index_law'를 먼저 확인하여, 'faiss_index'에 포함되는 경우를 명확히 분리합니다.
        if "faiss_index_law" in source_info:
            source_name = "관련 법규 (회계/세법)"
        elif "faiss_index" in source_info:
            source_name = "전북테크노파크 규정"
        else:
            source_name = "기타 자료"
            
        formatted_docs.append(f"--- [참고 자료: {source_name}] ---\n{doc.page_content}")
    return "\n\n".join(formatted_docs)
# ---------------------------------------------------


# --- 초기 설정 (최종 확정된 2개 벡터DB 로드 및 RAG 체인 구성) ---
@st.cache_resource
def load_rag_chain():
    try:
        os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]
    except Exception as e:
        st.error("Streamlit Secrets에 GOOGLE_API_KEY가 설정되지 않았습니다.")
        st.stop()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # --- [최종 확정] 'faiss_index'와 'faiss_index_law'를 사용하도록 변경 ---
    try:
        # 'faiss_index' 폴더를 로드 (규정집)
        vectorstore_reg = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
        # 'faiss_index_law' 폴더를 로드 (법규)
        vectorstore_law = FAISS.load_local("./faiss_index_law", embeddings, allow_dangerous_deserialization=True)
        st.sidebar.success("모든 임베딩 DB 로드 완료!", icon="✅")
    except Exception as e:
        st.error(f"FAISS 인덱스 로드 중 오류가 발생했습니다: {e}")
        # 안내 문구 수정
        st.info("faiss_index와 faiss_index_law 폴더가 모두 존재하는지 확인해주세요.")
        st.stop()

    # --- [최종 확정] 2개의 Retriever 생성 (변수명 명확화) ---
    # 규정집 Retriever
    retriever_reg = vectorstore_reg.as_retriever(search_kwargs={'k': 5})
    # 법규 Retriever
    retriever_law = vectorstore_law.as_retriever(search_kwargs={'k': 3})

    # --- Ensemble Retriever 구성 (변경 없음) ---
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_reg, retriever_law],
        weights=[0.6, 0.4]
    )
    # ------------------------------------

    # --- 프롬프트 템플릿 (변경 없음) ---
    prompt_template_str = """
    당신은 전북테크노파크의 규정과 재정 지침은 물론, 관련 법규(회계/세법)까지 깊이 이해하고 있는 최고의 사업 담당자이자 명쾌한 분석가 AI입니다.
    아래의 프로세스에 따라 지원사업을 수행하는 회사 담당자의 질문에 답변해주세요.
    
    [가장 중요한 원칙]
        - 당신은 이미 모든 규정집과 법규를 숙지한 최고의 전문가입니다.
        - 절대로 '규정집을 직접 찾아보세요', '원문을 확인하세요', '담당자에게 문의하세요' 와 같이 사용자에게 책임을 넘기는 답변을 해서는 안 됩니다.
        - 당신은 사용자를 대신하여 DB에서 정보를 찾고, 그것을 분석하고, 핵심 내용을 요약하여 직접적인 답변을 제공해야 할 의무가 있습니다.
        - 만약 DB에서 관련된 정보를 찾지 못했다면, "현재 제가 가진 정보로는 해당 부분에 대한 명확한 답변을 드리기 어렵습니다." 와 같이 솔직하게 한계를 인정해야 합니다.
        
    [프로세스]
    1.  **공감과 확인:** 먼저 사용자의 질문 내용을 확인하며 "네, ~에 대해 문의하셨군요. 답변해 드리겠습니다." 와 같이 대화를 시작합니다.
    2.  **핵심 답변:** 제공된 [Context]의 '전북테크노파크 규정'과 '관련 법규(회계/세법)' 정보를 종합하여 질문에 대한 핵심적인 답변을 명확하게 설명합니다.
        * **우선순위:** 가장 먼저 '전북테크노파크 규정'을 기준으로 판단해야 합니다.
        * **심화 답변:** 만약 규정만으로 답하기 어렵거나 해석이 필요한 복잡한 사안이라면, '관련 법규(회계/세법)' 정보를 적극적으로 활용하여 더 깊이 있고 전문적인 답변을 제공합니다.
    3.  **대안 제시:** 사용자가 요청한 사항이 규정상 불가능할 경우, '그 방법은 어렵지만, 대신 ~하는 방법이 있습니다' 와 같이 가능한 대안이나 조건을 제시합니다.
    4.  **사례 제시:** 설명이 복잡하거나 오해의 소지가 있는 경우, '예를 들어...' 와 같이 구체적인 예시를 한두 가지 들어 이해를 돕습니다.
    5.  **"플러스 알파" 정보 제공 (가장 중요):** 단순히 규정만 알려주는 것을 넘어, 담당자로서 추가적으로 도움이 될 만한 정보를 함께 제공합니다.
        -   **"그래서 이제 뭘 해야 하나요?"** 에 대한 답: 필요한 절차, 다음 단계 등을 안내합니다.
        -   **"무엇이 필요한가요?"** 에 대한 답: 제출해야 할 서류, 준비물 등을 알려줍니다.
        -   **"주의할 점은 없나요?"** 에 대한 답: 자주 하는 실수, 유의사항, 알아두면 좋은 팁 등을 언급합니다.
    6.  **친절한 마무리:** "더 궁금한 점이 있으시면 언제든지 다시 물어봐 주세요." 와 같이 대화를 마무리합니다.
    7.  **어조:** 시종일관 친절하고 상냥하며, 신뢰감 있는 전문가의 어조를 유지합니다. 딱딱한 '예/아니오'로 시작하지 마세요.
    
    [답변 형식]
    * **유형 1 (분류 질문) 답변:**
    -   `[품목]은(는) 전북테크노파크 규정에 따라 '[세목]'에 해당합니다.`
    -   (필요시) 관련 법규(회계/세법)상 추가 고려사항이 있다면 함께 설명합니다.

    * **유형 2 (확인 질문) 답변:**
    -   (긍정일 경우) `네, 맞습니다. [품목]은(는) '[세목]'으로 처리하는 것이 올바릅니다.`
    -   (부정일 경우) `아니요, 다릅니다. [품목]은(는) '[세목]'이 아니라 '[올바른 세목]'으로 처리해야 합니다.`
    -   (필요시) 왜 그런지 규정과 관련 법규(회계/세법)를 종합하여 간결한 근거를 덧붙입니다.
    * **유형 3 (일반 설명 질문) 답변:**
    -   규정과 관련 법규(회계/세법)를 종합하여, 사용자가 궁금해할 만한 추가 정보(절차, 주의사항 등)를 포함하여 상세하고 친절하게 설명합니다.

    [Context]
    {context}

    [Question]
    {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(prompt_template_str)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1)

    rag_chain = (
        {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# --- Streamlit UI 구성 및 실행 ---
try:
    rag_chain = load_rag_chain()
    st.set_page_config(page_title="규정 질의응답 챗봇", page_icon="📚")
    st.title("📚 예산 및 규정 질의응답 챗봇 (심화 버전)")

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
    # [최종 확정] 오류 안내 문구
    st.info("API 키가 올바르게 설정되었는지, 그리고 faiss_index와 faiss_index_law 폴더가 제대로 업로드되었는지 확인해주세요.")
