import streamlit as st
import os

st.set_page_config(page_title="파일 경로 진단")
st.title("📂 Streamlit Cloud 파일 경로 진단")

st.write("이 앱은 배포된 저장소의 파일 구조를 확인하기 위한 진단용 페이지입니다.")
st.warning("이 화면을 캡처하여 AI 어시스턴트에게 보여주세요.")

# --- 1. 현재 작업 디렉토리 확인 ---
try:
    cwd = os.getcwd()
    st.subheader("1. 현재 스크립트의 실행 위치 (CWD):")
    st.code(cwd, language='bash')
except Exception as e:
    st.error(f"현재 작업 디렉토리를 얻는 중 오류 발생: {e}")
    st.stop()

# --- 2. 현재 위치의 폴더 및 파일 목록 ---
st.subheader(f"2. 현재 위치 '{cwd}' 의 전체 파일 및 폴더 목록:")
try:
    root_contents = os.listdir(cwd)
    if root_contents:
        st.write(root_contents)
    else:
        st.write("폴더가 비어있습니다.")
except Exception as e:
    st.error(f"폴더 내용을 읽는 중 오류 발생: {e}")

# --- 3. 'faiss_index_qa' 폴더 존재 여부 정밀 확인 ---
st.subheader("3. 'faiss_index_qa' 폴더를 찾을 수 있는지 확인:")
qa_path_relative = "faiss_index_qa"
qa_path_absolute = os.path.join(cwd, qa_path_relative)

st.write(f"찾으려는 절대 경로: `{qa_path_absolute}`")

if os.path.exists(qa_path_absolute):
    st.success(f"✅ 성공: '{qa_path_absolute}' 경로에서 폴더를 찾았습니다.")
    try:
        qa_contents = os.listdir(qa_path_absolute)
        st.write(f"'faiss_index_qa' 폴더 내용: `{qa_contents}`")
    except Exception as e:
        st.error(f"'faiss_index_qa' 폴더 내용을 읽는 중 오류 발생: {e}")
else:
    st.error(f"❌ 실패: '{qa_path_absolute}' 경로에 폴더가 존재하지 않습니다.")
    st.write("GitHub 저장소의 최상위 위치에 'faiss_index_qa' 폴더가 있는지 다시 확인해주세요.")
