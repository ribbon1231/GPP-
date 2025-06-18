import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
import numpy as np
from rembg import remove

# --- 1. 모델 로딩 (최초 한 번만 실행) ---
@st.cache_resource
def load_model():
    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = keypointrcnn_resnet50_fpn(weights=weights, progress=False)
    model.eval()
    return model, weights.transforms()

model, transforms = load_model()

# --- 2. 핵심 로직 함수들 ---
def get_person_pose(image_pil):
    """사람 이미지(PIL)를 받아 포즈 키포인트를 추출하는 함수"""
    image_tensor = transforms(image_pil)
    with torch.no_grad():
        output = model([image_tensor])
    
    # 가장 확률 높은 사람의 키포인트, 점수 추출
    scores = output[0]['scores'].detach().numpy()
    if len(scores) == 0:
        return None, None
        
    best_person_idx = np.argmax(scores)
    keypoints = output[0]['keypoints'][best_person_idx].detach().numpy()
    keypoints_scores = output[0]['keypoints_scores'][best_person_idx].detach().numpy()
    
    return keypoints, keypoints_scores

def overlay_clothing(person_img, clothing_img, keypoints):
    """사람 이미지 위에 옷 이미지를 합성하는 함수"""
    # 키포인트 인덱스: 5(왼쪽어깨), 6(오른쪽어깨), 11(왼쪽엉덩이), 12(오른쪽엉덩이)
    left_shoulder = keypoints[5][:2]
    right_shoulder = keypoints[6][:2]
    left_hip = keypoints[11][:2]
    right_hip = keypoints[12][:2]

    # 1. 옷의 너비 계산: 어깨 너비를 기준으로
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    clothing_width = int(shoulder_width * 1.9) # 이 숫자를 조절하면 옷의 '너비'가 바뀝니다.

    # 2. 상체 길이 계산
    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2
    body_height = np.linalg.norm(shoulder_center - hip_center)
    
    # 몸 높이가 0이 되는 것을 방지
    if body_height == 0:
        body_height = shoulder_width * 1.2 

    # 상체 길이(body_height)를 기준으로 옷의 높이를 계산합니다.
    clothing_height = int(body_height * 1.5) # 이 숫자를 조절하면 옷의 '높이'가 바뀝니다.

    # 3. 옷의 위치(중심점) 계산
    center_x = int(shoulder_center[0])
    center_y = int(shoulder_center[1] + body_height * 0.5) 

    # 4. 옷의 각도 계산
    angle = np.degrees(np.arctan2(right_shoulder[1] - left_shoulder[1], right_shoulder[0] - left_shoulder[0]))

    # 5. 옷 이미지 변환
    # 배경 제거
    clothing_nobg = remove(clothing_img)
    
    # 크기 조절 (너비와 높이를 각각 독립적으로 적용)
    resized_clothing = clothing_nobg.resize((clothing_width, clothing_height))
    
    # 각도 조절 (뒤집힘 문제 해결)
    rotated_clothing = resized_clothing.rotate(-angle + 180, expand=True, resample=Image.BICUBIC)

    # 6. 원본 이미지에 옷 합성
    result_img = person_img.copy()
    
    # 옷을 붙여넣을 위치 계산
    paste_x = center_x - rotated_clothing.width // 2
    paste_y = center_y - rotated_clothing.height // 2
    
    # RGBA 이미지의 투명도를 마스크로 사용하여 붙여넣기
    result_img.paste(rotated_clothing, (paste_x, paste_y), rotated_clothing)

    return result_img

# --- 3. Streamlit UI 구성 ---
st.set_page_config(layout="wide")
st.title("👕 AI 가상 피팅 (Virtual Try-On) 데모")
st.write("---")
st.info("사람의 상반신이 잘 보이는 사진과, 배경이 있는 옷 사진을 각각 업로드 해보세요!")

col1, col2 = st.columns(2)

with col1:
    st.header("👤 사람 사진 업로드")
    person_file = st.file_uploader("사람 사진을 선택하세요.", type=['jpg', 'jpeg', 'png'])

with col2:
    st.header("👚 옷 사진 업로드")
    clothing_file = st.file_uploader("옷 사진을 선택하세요.", type=['jpg', 'jpeg', 'png'])


if person_file and clothing_file:
    person_img = Image.open(person_file).convert("RGB")
    clothing_img = Image.open(clothing_file).convert("RGBA") # RGBA로 열어야 투명도 처리 가능

    st.write("---")
    st.header("▶️ 원본 이미지")
    c1, c2 = st.columns(2)
    with c1:
        st.image(person_img, caption="사람 원본", use_column_width=True)
    with c2:
        st.image(clothing_img, caption="옷 원본", use_column_width=True)
    
    if st.button("가상 피팅 시작!", use_container_width=True):
        with st.spinner('AI가 사람의 포즈를 분석하고 옷을 입히는 중입니다...'):
            # 1. 포즈 추정
            keypoints, scores = get_person_pose(person_img)
            
            if keypoints is None:
                st.error("사진에서 사람을 찾지 못했습니다. 다른 사진을 이용해주세요.")
            else:
                # 2. 이미지 합성
                final_image = overlay_clothing(person_img, clothing_img, keypoints)
                
                # 3. 결과 표시
                st.write("---")
                st.header("✅ 합성 결과")
                st.image(final_image, caption="AI 가상 피팅 결과", use_column_width=True)