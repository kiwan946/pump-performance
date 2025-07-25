import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np

# 페이지 기본 설정
st.set_page_config(page_title="Dooch XRL(F) 성능 곡선 뷰어 v8.0", layout="wide")
st.title("📊 Dooch XRL(F) 성능 곡선 뷰어 v8.0")

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
            if n in col.strip(): # 컬럼명의 공백 제거 후 비교
                return col
    return None

def calculate_efficiency(df, q_col, h_col, k_col):
    """사용자 지정 공식을 바탕으로 펌프 효율을 계산합니다."""
    if not all(col and col in df.columns for col in [q_col, h_col, k_col]):
        return df
    df_copy = df.copy()
    hydraulic_power = 0.163 * df_copy[q_col] * df_copy[h_col]
    shaft_power = df_copy[k_col]
    efficiency = np.where(shaft_power > 0, (hydraulic_power / shaft_power) * 100, 0)
    df_copy['Efficiency'] = efficiency
    return df_copy

def load_and_preprocess_sheet(uploaded_file, sheet_name):
    """Excel 시트를 로드하고 기본적인 전처리를 수행합니다."""
    try:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        df.columns = df.columns.str.strip() # 컬럼명 공백 제거
        mcol = get_best_match_column(df, ["모델명", "모델", "Model"])
        if not mcol:
            return None, pd.DataFrame()
        
        df['Series'] = df[mcol].astype(str).str.extract(r"(XRF\d+)")
        df['Series'] = pd.Categorical(df['Series'], categories=SERIES_ORDER, ordered=True)
        df = df.sort_values('Series')
        return mcol, df
    except Exception:
        return None, pd.DataFrame()

def process_data(df, q_col, h_col, k_col):
    """선택된 컬럼 기준으로 데이터를 정제하고 효율을 계산합니다."""
    if df.empty:
        return df
    
    temp_df = df.copy()
    for col in [q_col, h_col, k_col]:
        if col in temp_df.columns:
            temp_df = temp_df.dropna(subset=[col])
            temp_df = temp_df[pd.to_numeric(temp_df[col], errors='coerce').notna()]
            temp_df[col] = pd.to_numeric(temp_df[col])
            
    return calculate_efficiency(temp_df, q_col, h_col, k_col)

# --- 분석 및 시각화 함수들 (이전과 동일) ---
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
        q_overload = 1.5 * target_q
        interp_h_overload = np.interp(q_overload, model_df[q_col], model_df[h_col], left=np.nan, right=np.nan)
        cond2_ok = (not np.isnan(interp_h_overload)) and (interp_h_overload >= (0.65 * target_h))
        if cond1_ok and cond2_ok:
            interp_kw = np.interp(target_q, model_df[q_col], model_df[k_col]) if k_col and k_col in model_df.columns else np.nan
            results.append({"모델명": model, "정격 예상 양정": f"{interp_h_rated:.2f}", "체절 양정 (≤{1.4*target_h:.2f})": f"{h_churn:.2f}", "최대운전 양정 (≥{0.65*target_h:.2f})": f"{interp_h_overload:.2f}", "예상 동력(kW)": f"{interp_kw:.2f}", "선정 가능": "✅"})
    return pd.DataFrame(results)

def render_filters(df, mcol, prefix):
    series_opts = df['Series'].dropna().unique().tolist()
    default_series = [series_opts[0]] if series_opts else []
    mode = st.radio("분류 기준", ["시리즈별", "모델별"], key=f"{prefix}_mode", horizontal=True)
    if mode == "시리즈별":
        sel = st.multiselect("시리즈 선택", series_opts, default=default_series, key=f"{prefix}_series")
        df_f = df[df['Series'].isin(sel)] if sel else pd.DataFrame()
    else:
        model_opts = df[mcol].dropna().unique().tolist()
        default_model = [model_opts[0]] if model_opts else []
        sel = st.multiselect("모델 선택", model_opts, default=default_model, key=f"{prefix}_models")
        df_f = df[df[mcol].isin(sel)] if sel else pd.DataFrame()
    return df_f

def add_traces(fig, df, mcol, xcol, ycol, models, mode, line_style=None, name_suffix=""):
    for m in models:
        sub = df[df[mcol] == m].sort_values(xcol)
        if sub.empty or ycol not in sub.columns: continue
        fig.add_trace(go.Scatter(x=sub[xcol], y=sub[ycol], mode=mode, name=m + name_suffix, line=line_style or {}))

def add_bep_markers(fig, df, mcol, qcol, ycol, models):
    for m in models:
        model_df = df[df[mcol] == m]
        if not model_df.empty and 'Efficiency' in model_df.columns and not model_df['Efficiency'].isnull().all():
            bep_row = model_df.loc[model_df['Efficiency'].idxmax()]
            fig.add_trace(go.Scatter(x=[bep_row[qcol]], y=[bep_row[ycol]], mode='markers', marker=dict(symbol='star', size=15, color='gold'), name=f'{m} BEP'))

def render_chart(fig, key):
    fig.update_layout(dragmode='pan', xaxis=dict(fixedrange=False), yaxis=dict(fixedrange=False), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displaylogo': False}, key=key)

# --- 메인 애플리케이션 로직 ---

uploaded_file = st.file_uploader("Excel 파일 업로드 (.xlsx 또는 .xlsm)", type=["xlsx", "xlsm"])

if uploaded_file:
    m_r, df_r_orig = load_and_preprocess_sheet(uploaded_file, "reference data")
    
    if df_r_orig.empty:
        st.error("오류: 'reference data' 시트를 찾을 수 없거나 '모델명' 관련 컬럼이 없습니다. 파일을 확인해주세요.")
    else:
        st.sidebar.title("⚙️ 분석 설정")
        st.sidebar.markdown("### 컬럼 지정")
        st.sidebar.info("자동으로 추천된 컬럼을 확인하고, 필요시 직접 변경해주세요.")
        
        all_columns = df_r_orig.columns.tolist()
        
        # 안전한 인덱스 찾기 함수
        def safe_get_index(items, value):
            try:
                return items.index(value)
            except ValueError:
                return 0

        # 자동 추천
        q_auto = get_best_match_column(df_r_orig, ["토출량", "유량"])
        h_auto = get_best_match_column(df_r_orig, ["토출양정", "전양정"])
        k_auto = get_best_match_column(df_r_orig, ["축동력"])
        
        q_col = st.sidebar.selectbox("유량 (Flow) 컬럼", all_columns, index=safe_get_index(all_columns, q_auto))
        h_col = st.sidebar.selectbox("양정 (Head) 컬럼", all_columns, index=safe_get_index(all_columns, h_auto))
        k_col = st.sidebar.selectbox("축동력 (Power) 컬럼", all_columns, index=safe_get_index(all_columns, k_auto))
        
        # 데이터 처리
        df_r = process_data(df_r_orig, q_col, h_col, k_col)
        
        m_c, df_c_orig = load_and_preprocess_sheet(uploaded_file, "catalog data")
        df_c = process_data(df_c_orig, q_col, h_col, k_col)
        
        m_d, df_d_orig = load_and_preprocess_sheet(uploaded_file, "deviation data")
        df_d = process_data(df_d_orig, q_col, h_col, k_col)

        # 탭 생성
        tab_list = ["Total", "Reference", "Catalog", "Deviation"]
        tabs = st.tabs(tab_list)

        with tabs[0]:
            st.subheader("📊 Total - 통합 곡선 및 운전점 분석")
            df_f = render_filters(df_r, m_r, "total")
            models = df_f[m_r].unique().tolist() if not df_f.empty else []
            # ... 이하 Total 탭의 모든 기능 (이전 코드와 동일)

        with tabs[1]:
            st.subheader("📊 Reference Data")
            # ... 이하 Reference 탭의 모든 기능 (이전 코드와 동일)
        
        # ... 이하 나머지 탭 기능 ...

else:
    st.info("시작하려면 Excel 파일을 업로드하세요.")
