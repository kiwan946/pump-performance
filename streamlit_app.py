import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np

# 페이지 기본 설정
st.set_page_config(page_title="Dooch XRL(F) 성능 곡선 뷰어 v3.7", layout="wide")
st.title("📊 Dooch XRL(F) 성능 곡선 뷰어 v3.7")

# --- 유틸리티 함수들 ---

# 시리즈 순서 정의
SERIES_ORDER = [
    "XRF3", "XRF5", "XRF10", "XRF15", "XRF20", "XRF32",
    "XRF45", "XRF64", "XRF95", "XRF125", "XRF155", "XRF185",
    "XRF215", "XRF255"
]

def get_best_match_column(df, names):
    for n in names:
        for col in df.columns:
            if n in col:
                return col
    return None

# ★★★ 수정된 부분: 사용자 지정 공식으로 효율 계산 ★★★
def calculate_efficiency_user_formula(df, q_col, h_col, k_col):
    """사용자 지정 공식을 바탕으로 펌프 효율을 계산합니다."""
    # 필수 컬럼이 없으면 원본 반환
    if not all([q_col, h_col, k_col, q_col in df.columns, h_col in df.columns, k_col in df.columns]):
        return df
    
    df_copy = df.copy()
    
    # 수동력 계산 (Hydraulic Power)
    # 0.163 계수는 유량(Q)이 m³/min, 양정(H)이 m, 동력(P)이 kW일 때 통용됩니다.
    hydraulic_power = 0.163 * df_copy[q_col] * df_copy[h_col]
    
    # 축동력 (Shaft Power)
    shaft_power = df_copy[k_col]
    
    # 효율 계산 및 0으로 나누기 오류 방지
    efficiency = np.where(
        shaft_power > 0,
        (hydraulic_power / shaft_power) * 100,
        0
    )
    df_copy['Efficiency'] = efficiency
    return df_copy

def load_sheet(name):
    """Excel 시트를 로드하고 기본적인 전처리를 수행합니다."""
    try:
        df = pd.read_excel(uploaded_file, sheet_name=name)
    except Exception:
        return None, None, None, None, pd.DataFrame()

    mcol = get_best_match_column(df, ["모델명", "모델", "Model"])
    qcol = get_best_match_column(df, ["토출량", "유량"])
    # 두 종류의 양정 컬럼을 모두 찾습니다.
    hcol_discharge = get_best_match_column(df, ["토출양정"])
    hcol_total = get_best_match_column(df, ["전양정"])
    kcol = get_best_match_column(df, ["축동력"])

    # '토출양정'과 '전양정' 둘 다 없으면 진행 불가
    if not mcol or not qcol or not (hcol_discharge or hcol_total):
        return None, None, None, None, pd.DataFrame()

    # 숫자 데이터 정제
    cols_to_check = [qcol, kcol, hcol_discharge, hcol_total]
    for col in cols_to_check:
        if col and col in df.columns:
            df = df.dropna(subset=[col])
            df = df[pd.to_numeric(df[col], errors='coerce').notna()]
            df[col] = pd.to_numeric(df[col])

    # 시리즈 컬럼 생성
    df['Series'] = df[mcol].astype(str).str.extract(r"(XRF\d+)")
    df['Series'] = pd.Categorical(df['Series'], categories=SERIES_ORDER, ordered=True)
    df = df.sort_values('Series')
    
    # hcol_discharge와 hcol_total 이름을 반환하여 나중에 선택할 수 있도록 함
    return mcol, qcol, hcol_discharge, hcol_total, kcol, df


# (이하 다른 함수들은 이전 버전과 동일)
def analyze_operating_point(df, models, target_q, target_h, m_col, q_col, h_col, k_col):
    if target_q <= 0 or target_h <= 0: return pd.DataFrame()
    results = []
    for model in models:
        model_df = df[df[m_col] == model].sort_values(q_col)
        if len(model_df) < 2 or not (model_df[q_col].min() <= target_q <= model_df[q_col].max()): continue
        interp_h = np.interp(target_q, model_df[q_col], model_df[h_col])
        if interp_h >= target_h:
            interp_kw = np.interp(target_q, model_df[q_col], model_df[k_col]) if k_col and k_col in model_df.columns else np.nan
            interp_eff = np.interp(target_q, model_df[q_col], model_df['Efficiency']) if 'Efficiency' in model_df.columns else np.nan
            results.append({"모델명": model, "요구 유량": target_q, "요구 양정": target_h, "예상 양정": f"{interp_h:.2f}", "예상 동력(kW)": f"{interp_kw:.2f}", "예상 효율(%)": f"{interp_eff:.2f}", "선정 가능": "✅"})
    return pd.DataFrame(results)

def analyze_fire_pump_point(df, models, target_q, target_h, m_col, q_col, h_col, k_col):
    if target_q <= 0 or target_h <= 0: return pd.DataFrame()
    results = []
    for model in models:
        model_df = df[df[m_col] == model].sort_values(q_col)
        if len(model_df) < 2: continue
        interp_h_rated = np.interp(target_q, model_df[q_col], model_df[h_col], left=np.nan, right=np.nan)
        if np.isnan(interp_h_rated) or interp_h_rated < target_h: continue
        h_churn = model_df.iloc[0][h_col]
        cond1_ok = h_churn <= (1.40 * target_h)
        q_over
