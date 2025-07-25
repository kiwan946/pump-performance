import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np

# 페이지 기본 설정
st.set_page_config(page_title="Dooch XRL(F) 성능 곡선 뷰어 v5.4", layout="wide")
st.title("📊 Dooch XRL(F) 성능 곡선 뷰어 v5.4")

# --- 유틸리티 함수들 ---

SERIES_ORDER = [
    "XRF3", "XRF5", "XRF10", "XRF15", "XRF20", "XRF32",
    "XRF45", "XRF64", "XRF95", "XRF125", "XRF155", "XRF185",
    "XRF215", "XRF255"
]

def get_best_match_column(df, names):
    """키워드를 기반으로 DataFrame에서 최적의 컬럼 이름을 찾아 기본값으로 제안합니다."""
    for n in names:
        for col in df.columns:
            if n in col:
                return col
    return None

def calculate_efficiency_user_formula(df, q_col, h_col, k_col):
    """사용자 지정 공식을 바탕으로 펌프 효율을 계산합니다."""
    if not all(col in df.columns for col in [q_col, h_col, k_col] if col):
        return df
    df_copy = df.copy()
    hydraulic_power = 0.163 * df_copy[q_col] * df_copy[h_col]
    shaft_power = df_copy[k_col]
    efficiency = np.where(shaft_power > 0, (hydraulic_power / shaft_power) * 100, 0)
    df_copy['Efficiency'] = efficiency
    return df_copy

def load_sheet(name):
    """Excel 시트를 로드하고 기본적인 전처리를 수행합니다."""
    try:
        df = pd.read_excel(uploaded_file, sheet_name=name)
    except Exception:
        return None, pd.DataFrame()
    
    mcol = get_best_match_column(df, ["모델명", "모델", "Model"])
    if not mcol:
        return None, pd.DataFrame()
    df['Series'] = df[mcol].astype(str).str.extract(r"(XRF\d+)")
    df['Series'] = pd.Categorical(df['Series'], categories=SERIES_ORDER, ordered=True)
    df = df.sort_values('Series')
    return mcol, df

# --- 분석 및 시각화 함수들 (이전과 동일) ---
def analyze_operating_point(df, models, target_q, target_h, m_col, q_col, h_col, k_col):
    if target_q <= 0 or target_h <= 0: return pd.DataFrame()
    results = []
    for model in models:
        model_df = df[df[m_col] == model].sort_values(q_col)
        if len(model_df) < 2 or not (model_df[q_col].min() <= target_q <= model_df[q
