import streamlit as st
from PIL import Image
import os
import subprocess
import sys

# --- 페이지 설정 ---
st.set_page_config(page_title="AI 가상 피팅", layout="wide")
st.title("👕 AI 가상 피팅 (CP-VTON+)")
st.write("사람 이미지와 옷 이미지를 업로드하면, 옷을 입은 모습을 생성합니다.")

# --- 전역 변수 및 경로 설정 ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULT_DIR = os.path.join(PROJECT_DIR, "results", "vton_gen", "test", "try-on")
TEST_IMAGE_DIR = os.path.join(DATA_DIR, "test", "image")
TEST_CLOTH_DIR = os.path.join(DATA_DIR, "test", "cloth")

# --- 필요한 디렉토리 생성 ---
os.makedirs(TEST_IMAGE_DIR, exist_ok=True)
os.makedirs(TEST_CLOTH_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

def run_virtual_tryon(person_image_name, cloth_image_name):
    """CP-VTON+의 test.py 스크립트를 실행하는 함수"""
    
    # 현재 가상환경의 python 실행 파일 경로를 직접 가져오기
    python_executable = sys.executable
    
    # 1. 테스트 페어 파일 생성
    test_pairs_path = os.path.join(DATA_DIR, "test_pairs.txt")
    with open(test_pairs_path, "w") as f:
        f.write(f"{person_image_name} {cloth_image_name}")

    # 2. test.py 스크립트 실행 (절대 경로로 python 호출)
    command = [
        python_executable, "test.py",
        "--name", "vton_gen",
        "--datamode", "test",
        "--data_list", "test_pairs.txt",
        "--gpu_ids", "0"  # GPU가 있으면 0, 없으면 -1
    ]
    
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        log_container = st.empty()
        logs = ""
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logs += output.strip() + "\n"
                log_container.code(logs, language='bash')
        
        stderr = process.communicate()[1]
        if process.returncode != 0:
            st.error("가상 피팅 중 오류가 발생했습니다.")
            st.code(stderr, language='bash')
            return None

    except Exception as e:
        st.error(f"스크립트 실행 중 예외 발생: {e}")
        return None

    # 3. 결과 이미지 경로 반환
    result_image_path = os.path.join(RESULT_DIR, person_image_name)
    if os.path.exists(result_image_path):
        return result_image_path
    else:
        st.error("결과 이미지를 찾을 수 없습니다. 모델 실행에 실패했을 수 있습니다.")
        return None


# --- Streamlit UI 구성 ---
col1, col2 = st.columns(2)

with col1:
    st.header("👤 사람 이미지")
    person_file = st.file_uploader("상반신이 잘 보이는 사진을 올려주세요.", type=["jpg", "jpeg", "png"], key="person")
    if person_file:
        person_image = Image.open(person_file)
        person_image = person_image.resize((768, 1024))
        person_image_name = person_file.name
        person_save_path = os.path.join(TEST_IMAGE_DIR, person_image_name)
        person_image.save(person_save_path)
        st.image(person_image, caption="업로드된 사람 이미지")

with col2:
    st.header("👚 옷 이미지")
    cloth_file = st.file_uploader("배경이 없는 옷 사진을 올려주세요.", type=["jpg", "jpeg", "png"], key="cloth")
    if cloth_file:
        cloth_image = Image.open(cloth_file)
        cloth_image = cloth_image.resize((768, 1024))
        cloth_image_name = cloth_file.name
        cloth_save_path = os.path.join(TEST_CLOTH_DIR, cloth_image_name)
        cloth_image.save(cloth_save_path)
        st.image(cloth_image, caption="업로드된 옷 이미지")

if 'person_file' in locals() and 'cloth_file' in locals() and person_file and cloth_file:
    if st.button("🚀 가상 피팅 시작하기", use_container_width=True):
        with st.spinner("AI가 옷을 입히고 있습니다... (최대 몇 분 소요)"):
            result_path = run_virtual_tryon(person_image_name, cloth_image_name)
        
        if result_path:
            st.success("🎉 가상 피팅 완료!")
            final_image = Image.open(result_path)
            st.image(final_image, caption="가상 피팅 결과", use_container_width=True)
        else:
            st.error("결과 생성에 실패했습니다. 터미널 로그를 확인해주세요.")