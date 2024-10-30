import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 임베딩 모델 로드
encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 포트폴리오 관련 질문과 답변 데이터
questions = [
    "포트폴리오나 프로젝트의 주제가 뭔가요?",
    "모델은 어떤 것을 사용했나요?",
    "프로젝트나 포트폴리오의 기간은 어떻게 되나요?",
    "데이터는 무엇을 이용했나요?",
    "프로젝트나 포트폴리오를 하는 데 어려움은 없었나요?"
]

answers = [
    "딥러닝을 활용하여 무인 점포의 절도 범죄를 예방할 수 있는 모델을 개발하는 것입니다.",
    "RNN과 CNN을 사용하였습니다.",
    "약 한달 동안 진행되었습니다.",
    "AI Hub에서 '실내(편의점, 매장) 사람 이상행동 데이터'를 사용하였습니다.",
    "모델의 정확도를 높이기 위해 영상과 라벨링 데이터를 함께 학습시키는 것에 어려움이 있었습니다."
]

# 질문 임베딩과 답변 데이터프레임 생성
question_embeddings = encoder.encode(questions)
df = pd.DataFrame({'question': questions, '챗봇': answers, 'embedding': list(question_embeddings)})

# 대화 이력을 저장하기 위한 Streamlit 상태 설정
if 'history' not in st.session_state:
    st.session_state.history = []

# 챗봇 함수 정의
def get_response(user_input):
    # 사용자 입력 임베딩
    embedding = encoder.encode(user_input)
    
    # 유사도 계산하여 가장 유사한 응답 찾기
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    # 대화 이력에 추가
    st.session_state.history.append({"user": user_input, "bot": answer['챗봇']})

# Streamlit 인터페이스
st.title("포트폴리오에 대해 질문하세요 :)")

# 이미지 표시
st.image("ice.png", caption="Welcome to the Restaurant Chatbot", use_column_width=True)

st.write("포트폴리오에 대해 궁금한 것을 물어보세요. 예: 어떤 주제로 진행했나요?")

user_input = st.text_input("user", "")

if st.button("Submit"):
    if user_input:
        get_response(user_input)
        user_input = ""  # 입력 초기화

# 대화 이력 표시
for message in st.session_state.history:
    st.write(f"**사용자**: {message['user']}")
    st.write(f"**챗봇**: {message['bot']}")
