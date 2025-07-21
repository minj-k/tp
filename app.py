import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

# --- 1. 초기 리소스 로드 (리트리버 4개, LLM 1개 반환) ---
@st.cache_resource
def load_resources():
    try:
        os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        st.error("Streamlit Secrets에 GOOGLE_API_KEY가 설정되지 않았습니다.")
        st.stop()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    qa_retriever = FAISS.load_local("./faiss_index_qa", embeddings, allow_dangerous_deserialization=True).as_retriever(search_kwargs={'k': 1})
    ict_retriever = FAISS.load_local("./faiss_index_ict", embeddings, allow_dangerous_deserialization=True).as_retriever(search_kwargs={'k': 5})
    tp_retriever = FAISS.load_local("./faiss_index_tp", embeddings, allow_dangerous_deserialization=True).as_retriever(search_kwargs={'k': 5})
    law_retriever = FAISS.load_local("./faiss_index_law", embeddings, allow_dangerous_deserialization=True).as_retriever(search_kwargs={'k': 5})
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    
    # 총 5개의 변수를 반환합니다.
    return qa_retriever, ict_retriever, tp_retriever, law_retriever, llm

# --- 2. 체인 및 프롬프트 정의 ---
def setup_chains(llm):
    # (이 부분은 이전과 동일합니다)
    qa_gate_prompt = ChatPromptTemplate.from_template(
        "당신은 질문과 가장 유사한 Q&A 문서를 보고, 이 Q&A가 사용자의 질문에 대한 직접적이고 완전한 답변이 되는지 판단하는 전문가입니다. 답변은 'yes' 또는 'no'로만 해주세요.\n\n"
        "[미리 준비된 Q&A]:\n{qa_context}\n\n[사용자 질문]:\n{input}\n\n[Q&A가 질문에 대한 완벽한 답변이 됩니까? (yes/no)]:"
    )
    qa_gate_chain = qa_gate_prompt | llm | StrOutputParser()

    rewrite_prompt = ChatPromptTemplate.from_messages(...) # 이전 최종 프롬프트와 동일
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    final_prompt = ChatPromptTemplate.from_messages(...) # 이전 최종 프롬프트와 동일
    final_chain = final_prompt | llm | StrOutputParser()
    
    return qa_gate_chain, rewrite_chain, final_chain

# --- 3. 메인 로직 실행 함수 (수정된 부분) ---
# 함수가 튜플 대신 각각의 구성요소를 직접 받도록 변경합니다.
def get_response(user_input, chat_history, qa_retriever, ict_retriever, tp_retriever, law_retriever, qa_gate_chain, rewrite_chain, final_chain):
    # Q&A DB에서 먼저 검색
    qa_docs = qa_retriever.invoke(user_input)
    
    if qa_docs:
        # Q&A가 질문에 대한 답이 되는지 LLM이 판단
        is_answer_in_qa = qa_gate_chain.invoke({"qa_context": "\n".join([doc.page_content for doc in qa_docs]), "input": user_input})
        if "yes" in is_answer_in_qa.lower():
            # 답이 맞으면, Q&A의 답변 부분을 바로 반환하고 종료
            answer_part = qa_docs[0].page_content.split('A:')[1].strip()
            return f"✅ [모범 답안]:\n\n{answer_part}"

    # Q&A에 답이 없으면, 계층적 RAG 파이프라인 실행
    rewritten_question = rewrite_chain.invoke({"input": user_input, "chat_history": chat_history})
    ict_docs = ict_retriever.invoke(rewritten_question)
    tp_docs = tp_retriever.invoke(rewritten_question)
    law_docs = law_retriever.invoke(rewritten_question)
    
    # 모든 정보를 종합하여 최종 답변 생성
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
    # 5개의 구성요소를 각각의 변수로 받습니다.
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
            # 수정된 부분: get_response 함수에 모든 구성요소를 개별적으로 전달합니다.
            answer = get_response(
                prompt,
                st.session_state.chat_history,
                qa_retriever,
                ict_retriever,
                tp_retriever,
                law_retriever,
                qa_gate_chain,
                rewrite_chain,
                final_chain
            )
            
            st.session_state.chat_history.append(AIMessage(content=answer))
            with st.chat_message("assistant"):
                st.markdown(answer)

except Exception as e:
    st.error(f"오류가 발생했습니다. 잠시 후 다시 시도해주세요.\n\n오류 상세: {e}")
