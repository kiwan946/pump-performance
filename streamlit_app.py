#!/usr/bin/env python3
# streamlit_app.py
# 펌프 성능 예측 및 시각화 웹 애플리케이션 (Streamlit)

import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# 페이지 설정
st.set_page_config(page_title="펌프 성능 웹앱", layout="wide")

@st.cache_data(show_spinner=False)
def load_sheet_names(uploaded_file):
    """업로드된 엑셀 파일의 시트 목록 반환"""
    xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
    return xls.sheet_names

@st.cache_data(show_spinner=False)
def load_dataframe(uploaded_file, sheet_name):
    """선택한 시트의 데이터를 DataFrame으로 로드"""
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine="openpyxl")
    return df


def main():
    st.title("🚀 펌프 성능 예측 및 분석 웹앱")

    # 1) 엑셀 파일 업로드
    uploaded_file = st.file_uploader("🔄 엑셀 파일 업로드 (.xls, .xlsx, .xlsm)", type=["xls", "xlsx", "xlsm"])
    if not uploaded_file:
        st.info("먼저 파일 업로더에서 엑셀 파일을 선택해주세요.")
        return

    # 2) 시트 선택
    sheets = load_sheet_names(uploaded_file)
    sheet = st.selectbox("📄 분석할 시트 선택", sheets)

    # 3) 데이터 로드
    df = load_dataframe(uploaded_file, sheet)
    st.success(f"`{sheet}` 시트 로드 완료! ({len(df)}개 행)")

    # 4) 컬럼명 매핑: 영문/한글 컬럼 대응
    col_model = next((c for c in ['Model', '모델명'] if c in df.columns), None)
    col_Q     = next((c for c in ['Q', '유량'] if c in df.columns), None)
    col_H     = next((c for c in ['H', '양정', '토출양정', '전양정', '토출양정&전양정'] if c in df.columns), None)
    col_P     = next((c for c in ['Power', 'Power(kW)', '축동력', '동력', '기준 동력'] if c in df.columns), None)

    if not all([col_model, col_Q, col_H, col_P]):
        missing = [name for name, col in [('모델명/Model', col_model), ('유량/Q', col_Q), ('양정/H', col_H), ('동력/Power', col_P)] if col is None]
        st.error(f"다음 필수 컬럼이 없습니다: {', '.join(missing)}")
        return

    # 5) 모델 선택 및 데이터 분리
    models = df[col_model].dropna().unique()
    model = st.selectbox("🔧 모델 선택", models)
    sub_df = df[df[col_model] == model].dropna(subset=[col_Q, col_H, col_P]).sort_values(col_Q)

    # 6) 보간 입력값
    st.sidebar.header("🔧 보간 입력값")
    Q_min, Q_max = float(sub_df[col_Q].min()), float(sub_df[col_Q].max())
    Q_input = st.sidebar.number_input("유량 입력", min_value=Q_min, max_value=Q_max, value=(Q_min + Q_max)/2.0)
    kind = st.sidebar.selectbox("보간 방식", ['linear', 'quadratic', 'cubic'])

    # 7) 보간 함수
    f_H = interp1d(sub_df[col_Q], sub_df[col_H], kind=kind, fill_value="extrapolate")
    f_P = interp1d(sub_df[col_Q], sub_df[col_P], kind=kind, fill_value="extrapolate")

    # 8) 예측 결과 표시
    H_pred = float(f_H(Q_input))
    P_pred = float(f_P(Q_input))
    col1, col2 = st.columns(2)
    col1.metric(label="예측 양정 (m)", value=f"{H_pred:.2f}")
    col2.metric(label="예측 동력 (kW)", value=f"{P_pred:.2f}")

    # 9) Q-H 시각화
    st.subheader("📈 Q-H 곡선")
    fig1, ax1 = plt.subplots()
    ax1.scatter(sub_df[col_Q], sub_df[col_H], label='실측', marker='o')
    Q_line = np.linspace(Q_min, Q_max, 200)
    ax1.plot(Q_line, f_H(Q_line), '-', label='보간곡선')
    ax1.axvline(Q_input, linestyle='--', alpha=0.7)
    ax1.axhline(H_pred, linestyle='--', alpha=0.7)
    ax1.set_xlabel(f'{col_Q}')
    ax1.set_ylabel(f'{col_H}')
    ax1.legend()
    st.pyplot(fig1)

    # 10) Q-동력 시각화
    st.subheader("📊 Q-동력 곡선")
    fig2, ax2 = plt.subplots()
    ax2.scatter(sub_df[col_Q], sub_df[col_P], label='실측', marker='o')
    ax2.plot(Q_line, f_P(Q_line), '-', label='보간곡선')
    ax2.axvline(Q_input, linestyle='--', alpha=0.7)
    ax2.axhline(P_pred, linestyle='--', alpha=0.7)
    ax2.set_xlabel(f'{col_Q}')
    ax2.set_ylabel(f'{col_P}')
    ax2.legend()
    st.pyplot(fig2)

    # 11) 원본 데이터 테이블
    with st.expander("📋 원본 데이터 확인"):
        st.dataframe(sub_df[[col_Q, col_H, col_P]].reset_index(drop=True))

if __name__ == "__main__":
    main()
