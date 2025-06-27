import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

# --- 1. 초기 리소스 로드 (3개의 DB, LLM) ---
@st.cache_resource
def load_resources():
    try:
        os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        st.error("Streamlit Secrets에 GOOGLE_API_KEY가 설정되지 않았습니다.")
        st.stop()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # 3개의 분리된 벡터 스토어를 모두 로드합니다.
    ict_retriever = FAISS.load_local("./faiss_index_ict", embeddings, allow_dangerous_deserialization=True).as_retriever(search_kwargs={'k': 5})
    tp_retriever = FAISS.load_local("./faiss_index_tp", embeddings, allow_dangerous_deserialization=True).as_retriever(search_kwargs={'k': 5})
    law_retriever = FAISS.load_local("./faiss_index_law", embeddings, allow_dangerous_deserialization=True).as_retriever(search_kwargs={'k': 5})
    
    fast_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.05)
    
    return ict_retriever, tp_retriever, law_retriever, fast_llm, smart_llm

# --- 2. 체인 및 프롬프트 정의 ---
def setup_chains(fast_llm, smart_llm):
    # 체인 1: 질문 재구성 체인 (꼬리 질문 처리용)
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "이전 대화 내용과 사용자의 최근 질문이 주어집니다. 대화 내용을 참고해야만 이해할 수 있는 질문이라면, 대화 내용 없이도 이해할 수 있는 독립적인 질문으로 재구성해주세요. 질문에 대한 답변은 절대 하지 말고, 오직 질문을 재구성하는 역할만 수행하세요."),
            ("user", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    rewrite_chain = rewrite_prompt | fast_llm | StrOutputParser()

    # 체인 2: 최종 답변 생성 체인 (계층적 우선순위 지시 포함)
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """당신은 전북테크노파크의 규정과 상위 법률에 모두 통달한 최고의 AI 전문가입니다.
사용자의 질문에 답변할 때, 아래의 [정보 활용 규칙]을 반드시 준수해야 합니다.

[정보 활용 규칙]
1.  **최우선 순위:** [ICT 기금 지침]에 관련 내용이 있는지 먼저 확인하고, 내용이 있다면 반드시 해당 지침을 근거로 답변해야 합니다.
2.  **2차 순위:** [ICT 기금 지침]에 명확한 내용이 없을 경우, [전북 TP 규정]을 참고하여 답변합니다.
3.  **참고용:** [상위 법률(세법, 회계기준)]은 용어의 정의를 명확히 하거나, 다른 두 규정에 내용이 없을 때 일반적인 원칙을 설명하기 위해서만 참고합니다.
4.  답변 시, 어떤 규정을 근거로 답변하는지 명시해주면 신뢰도가 높아집니다. (예: "ICT 기금 지침 제O조에 따라...")

---
[ICT 기금 지침]:
{ict_context}

[전북 TP 규정]:
{tp_context}

[상위 법률(세법, 회계기준)]:
{law_context}
"""),
            ("user", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    final_chain = final_prompt | smart_llm | StrOutputParser()
    
    return rewrite_chain, final_chain

# --- 3. 메인 로직 실행 함수 ---
def get_response(user_input, chat_history, retrievers, chains):
    rewrite_chain, final_chain = chains
    ict_retriever, tp_retriever, law_retriever = retrievers

    # 1. 질문 재구성
    rewritten_question = rewrite_chain.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    
    # 2. 3개의 DB에서 병렬적으로 관련 문서 검색
    ict_docs = ict_retriever.invoke(rewritten_question)
    tp_docs = tp_retriever.invoke(rewritten_question)
    law_docs = law_retriever.invoke(rewritten_question)
    
    # 3. 모든 정보를 종합하여 최종 답변 생성
    final_answer = final_chain.invoke({
        "ict_context": "\n".join([doc.page_content for doc in ict_docs]),
        "tp_context": "\n".join([doc.page_content for doc in tp_docs]),
        "law_context": "\n".join([doc.page_content for doc in law_docs]),
        "chat_history": chat_history,
        "input": user_input
    })
    
    return final_answer

# --- Streamlit UI 설정 ---
st.set_page_config(page_title="최종 규정 질의응답 챗봇", page_icon="🏛️")
st.title("🏛️ 최종 규정 질의응답 챗봇")
st.info("ICT지침 > TP규정 > 상위법 순서로 답변하며, 이전 대화를 기억합니다.")

try:
    ict_retriever, tp_retriever, law_retriever, fast_llm, smart_llm = load_resources()
    rewrite_chain, final_chain = setup_chains(fast_llm, smart_llm)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"): st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"): st.markdown(message.content)

    if prompt := st.chat_input("규정에 대해 질문해주세요..."):
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        with st.chat_message("user"): st.markdown(prompt)

        with st.spinner("답변을 생각하고 있습니다..."):
            answer = get_response(prompt, st.session_state.chat_history, (ict_retriever, tp_retriever, law_retriever), (rewrite_chain, final_chain))
            
            st.session_state.chat_history.append(AIMessage(content=answer))
            with st.chat_message("assistant"): st.markdown(answer)

except Exception as e:
    st.error(f"오류가 발생했습니다. 잠시 후 다시 시도해주세요.\n\n오류 상세: {e}")
