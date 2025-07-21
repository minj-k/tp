import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

# --- 1. 초기 리소스 로드 (4개의 DB, LLM) ---
@st.cache_resource
def load_resources():
    """
    API 키 설정, 4개의 벡터 DB 로드, LLM 모델을 초기화합니다.
    """
    try:
        os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        st.error("Streamlit Secrets에 GOOGLE_API_KEY가 설정되지 않았습니다. Manage app > Settings > Secrets에 키를 추가해주세요.")
        st.stop()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # 4개의 분리된 벡터 스토어를 모두 로드합니다.
    qa_retriever = FAISS.load_local("./faiss_index_qa", embeddings, allow_dangerous_deserialization=True).as_retriever(search_kwargs={'k': 1})
    ict_retriever = FAISS.load_local("./faiss_index_ict", embeddings, allow_dangerous_deserialization=True).as_retriever(search_kwargs={'k': 5})
    tp_retriever = FAISS.load_local("./faiss_index_tp", embeddings, allow_dangerous_deserialization=True).as_retriever(search_kwargs={'k': 5})
    law_retriever = FAISS.load_local("./faiss_index_law", embeddings, allow_dangerous_deserialization=True).as_retriever(search_kwargs={'k': 5})
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    
    return qa_retriever, ict_retriever, tp_retriever, law_retriever, llm

# --- 2. 체인 및 프롬프트 정의 ---
def setup_chains(llm):
    """
    하나의 LLM을 사용하여 3개의 핵심 체인을 정의합니다.
    """
    # 체인 1: Q&A 게이트키퍼 체인 (Q&A DB에 답이 있는지 판단)
    qa_gate_prompt = ChatPromptTemplate.from_template(
        "당신은 질문과 가장 유사한 Q&A 문서를 보고, 이 Q&A가 사용자의 질문에 대한 직접적이고 완전한 답변이 되는지 판단하는 전문가입니다. 답변은 'yes' 또는 'no'로만 해주세요.\n\n"
        "[미리 준비된 Q&A]:\n{qa_context}\n\n[사용자 질문]:\n{input}\n\n[Q&A가 질문에 대한 완벽한 답변이 됩니까? (yes/no)]:"
    )
    qa_gate_chain = qa_gate_prompt | llm | StrOutputParser()

    # 체인 2: 질문 재구성 및 검색어 최적화 체인
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """당신은 사용자의 질문을 분석하여, RAG 시스템이 관련 문서를 가장 잘 찾을 수 있도록 질문을 재구성하는 검색 전문가입니다.
[규칙]
1. **주제 일관성 확인:** 사용자의 [최근 질문]이 [이전 대화 기록]과 같은 주제에 대한 후속 질문인지 먼저 판단합니다.
2. **후속 질문 처리:** 만약 후속 질문이라면, 대화 기록을 참고하여 맥락이 완전한 독립적인 질문으로 만드세요.
3. **새로운 주제 처리:** 만약 [최근 질문]이 새로운 주제라면, **이전 대화 기록을 무시**하고 [최근 질문] 내용만 사용하여 검색어를 만드세요.
4. **검색어 최적화:** 위 과정을 통해 만들어진 질문을, 규정집에 있을 법한 일반적이고 공식적인 키워드를 포함한 문장으로 최종 변환합니다.
결과는 답변이 아닌, 오직 검색에 최적화된 질문 하나여야 합니다."""),
            ("user", "[이전 대화 기록]:\n{chat_history}"),
            ("human", "[최근 질문]:\n{input}"),
        ]
    )
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    # 체인 3: 최종 답변 생성 체인 (계층적 우선순위 지시 포함)
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """당신은 전북테크노파크의 규정과 상위 법률에 모두 통달한 최고의 AI 전문가입니다.
사용자의 질문에 답변할 때, 아래의 [정보 활용 규칙]을 반드시 준수해야 합니다.

[정보 활용 규칙]
1.  **최우선 순위:** [ICT 기금 지침]에 관련 내용이 있는지 먼저 확인하고, 내용이 있다면 반드시 해당 지침을 근거로 답변해야 합니다.
2.  **2차 순위:** [ICT 기금 지침]에 명확한 내용이 없을 경우, [전북 TP 규정]을 참고하여 답변합니다.
3.  **참고용:** [상위 법률(세법, 회계기준)]은 용어의 정의를 명확히 하거나, 다른 두 규정에 내용이 없을 때 일반적인 원칙을 설명하기 위해서만 참고합니다.
4.  답변 시, 어떤 규정을 근거로 답변하는지 명시해주면 신뢰도가 높아집니다.

---
[ICT 기금 지침]: {ict_context}
[전북 TP 규정]: {tp_context}
[상위 법률(세법, 회계기준)]: {law_context}
"""),
            ("user", "[이전 대화 기록]:\n{chat_history}"),
            ("human", "[최근 질문]:\n{input}"),
        ]
    )
    final_chain = final_prompt | llm | StrOutputParser()
    
    return qa_gate_chain, rewrite_chain, final_chain

# --- 3. 메인 로직 실행 함수 ---
def get_response(user_input, chat_history, retrievers, chains):
    qa_retriever, ict_retriever, tp_retriever, law_retriever = retrievers
    qa_gate_chain, rewrite_chain, final_chain = chains

    # 1. Q&A DB에서 먼저 검색
    qa_docs = qa_retriever.invoke(user_input)
    
    if qa_docs:
        # 2. Q&A가 질문에 대한 답이 되는지 LLM이 판단
        is_answer_in_qa = qa_gate_chain.invoke({"qa_context": "\n".join([doc.page_content for doc in qa_docs]), "input": user_input})
        if "yes" in is_answer_in_qa.lower():
            # 3. 답이 맞으면, Q&A의 답변 부분을 바로 반환하고 종료
            answer_part = qa_docs[0].page_content.split('A:')[1].strip()
            return f"✅ [모범 답안]:\n\n{answer_part}"

    # 4. Q&A에 답이 없으면, 기존의 계층적 RAG 파이프라인 실행
    rewritten_question = rewrite_chain.invoke({"input": user_input, "chat_history": chat_history})
    ict_docs = ict_retriever.invoke(rewritten_question)
    tp_docs = tp_retriever.invoke(rewritten_question)
    law_docs = law_retriever.invoke(rewritten_question)
    
    # 5. 모든 정보를 종합하여 최종 답변 생성
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
st.info("FAQ > ICT지침 > TP규정 > 상위법 순서로 답변하며, 이전 대화를 기억합니다.")

try:
    # 리소스 로드 및 체인 설정
    qa_retriever, ict_retriever, tp_retriever, law_retriever, llm = load_resources()
    qa_gate_chain, rewrite_chain, final_chain = setup_chains(llm)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"): st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"): st.markdown(message.content)

    if prompt := st.chat_input("규정에 대해 질문해주세요..."):
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("답변을 분석하고 있습니다..."):
            answer = get_response(prompt, st.session_state.chat_history, (qa_retriever, ict_retriever, tp_retriever, law_retriever), (qa_gate_chain, rewrite_chain, final_chain))
            
            st.session_state.chat_history.append(AIMessage(content=answer))
            with st.chat_message("assistant"):
                st.markdown(answer)

except Exception as e:
    st.error(f"오류가 발생했습니다. 잠시 후 다시 시도해주세요.\n\n오류 상세: {e}")
