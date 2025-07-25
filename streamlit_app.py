import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np

# 페이지 기본 설정
st.set_page_config(page_title="Dooch XRL(F) 성능 곡선 뷰어 v5.0", layout="wide")
st.title("📊 Dooch XRL(F) 성능 곡선 뷰어 v5.0")

# --- 유틸리티 함수들 ---

SERIES_ORDER = [
    "XRF3", "XRF5", "XRF10", "XRF15", "XRF20", "XRF32",
    "XRF45", "XRF64", "XRF95", "XRF125", "XRF155", "XRF185",
    "XRF215", "XRF255"
]

def get_best_match_column(df, names, current_selection=None):
    """키워드를 기반으로 DataFrame에서 최적의 컬럼 이름을 찾습니다."""
    # 사용자가 이미 선택한 값이 유효하면 그대로 사용
    if current_selection and current_selection in df.columns:
        return current_selection
    # 자동 매칭
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
        return None, None, None, None, None, pd.DataFrame()
    mcol = get_best_match_column(df, ["모델명", "모델", "Model"])
    qcol = get_best_match_column(df, ["토출량", "유량"])
    hcol_discharge = get_best_match_column(df, ["토출양정"])
    hcol_total = get_best_match_column(df, ["전양정"])
    kcol = get_best_match_column(df, ["축동력"])
    if not mcol or not qcol or not (hcol_discharge or hcol_total):
        return None, None, None, None, None, pd.DataFrame()
    cols_to_check = [qcol, kcol, hcol_discharge, hcol_total]
    for col in cols_to_check:
        if col and col in df.columns:
            df = df.dropna(subset=[col])
            df = df[pd.to_numeric(df[col], errors='coerce').notna()]
            df[col] = pd.to_numeric(df[col])
    df['Series'] = df[mcol].astype(str).str.extract(r"(XRF\d+)")
    df['Series'] = pd.Categorical(df['Series'], categories=SERIES_ORDER, ordered=True)
    df = df.sort_values('Series')
    return mcol, qcol, hcol_discharge, hcol_total, kcol, df

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
    m_r, q_r_auto, h_r_d_auto, h_r_t_auto, k_r_auto, df_r_orig = load_sheet("reference data")
    m_c, q_c_auto, h_c_d_auto, h_c_t_auto, k_c_auto, df_c_orig = load_sheet("catalog data")
    m_d, q_d_auto, h_d_d_auto, h_d_t_auto, k_d_auto, df_d_orig = load_sheet("deviation data")

    # ★★★ 추가된 부분: 수동 컬럼 지정 기능 ★★★
    st.sidebar.title("⚙️ 분석 설정")
    st.sidebar.markdown("### 수동 컬럼 지정")
    
    if df_r_orig.empty:
        st.error("오류: 'reference data' 시트를 찾을 수 없거나 데이터가 비어있습니다. 파일을 확인해주세요.")
    else:
        all_columns = df_r_orig.columns.tolist()
        
        # 자동 감지된 컬럼을 기본값으로 사용
        q_col_selected = st.sidebar.selectbox("유량(Flow) 컬럼 선택", all_columns, index=all_columns.index(q_r_auto) if q_r_auto in all_columns else 0)
        
        # 양정 컬럼 옵션 제공
        head_options = [h for h in [h_r_d_auto, h_r_t_auto] if h and h in all_columns]
        h_col_selected = st.sidebar.selectbox("양정(Head) 컬럼 선택", all_columns, index=all_columns.index(head_options[0]) if head_options else 0)
        
        k_col_selected = st.sidebar.selectbox("축동력(Power) 컬럼 선택", all_columns, index=all_columns.index(k_r_auto) if k_r_auto in all_columns else 0)

        st.sidebar.info(f"유량: **{q_col_selected}** / 양정: **{h_col_selected}** / 축동력: **{k_col_selected}** 기준으로 분석합니다.")

        # 선택된 컬럼 기준으로 효율 계산
        df_r = calculate_efficiency_user_formula(df_r_orig.copy(), q_col_selected, h_col_selected, k_col_selected)
        df_c = calculate_efficiency_user_formula(df_c_orig.copy(), q_c_auto, h_col_selected, k_c_auto) # 다른 시트도 동일 기준 적용
        df_d = calculate_efficiency_user_formula(df_d_orig.copy(), q_d_auto, h_col_selected, k_d_auto)

        tab_list = ["Total", "Reference", "Catalog", "Deviation"]
        tabs = st.tabs(tab_list)

        with tabs[0]:
            st.subheader("📊 Total - 통합 곡선 및 운전점 분석")
            df_f = render_filters(df_r, m_r, "total")
            models = df_f[m_r].unique().tolist() if not df_f.empty else []

            with st.expander("운전점 분석 (Operating Point Analysis)", expanded=True):
                analysis_mode = st.radio("분석 모드", ["기계", "소방"], key="analysis_mode", horizontal=True)
                op_col1, op_col2 = st.columns(2)
                with op_col1:
                    target_q = st.number_input("목표 유량 (Q)", value=0.0, format="%.2f")
                with op_col2:
                    target_h = st.number_input("목표 양정 (H)", value=0.0, format="%.2f")
                if analysis_mode == "소방":
                    st.info("소방 펌프 성능 기준 3점을 자동으로 분석합니다.")
                if st.button("운전점 분석 실행"):
                    if not models:
                        st.warning("먼저 분석할 시리즈나 모델을 선택해주세요.")
                    else:
                        with st.spinner("선택된 모델들을 분석 중입니다..."):
                            # 분석 시 사용자가 선택한 컬럼 전달
                            if analysis_mode == "소방":
                                op_results_df = analyze_fire_pump_point(df_r, models, target_q, target_h, m_r, q_col_selected, h_col_selected, k_col_selected)
                            else:
                                op_results_df = analyze_operating_point(df_r, models, target_q, target_h, m_r, q_col_selected, h_col_selected, k_col_selected)
                            if not op_results_df.empty:
                                st.success(f"총 {len(op_results_df)}개의 모델이 요구 성능을 만족합니다.")
                                st.dataframe(op_results_df, use_container_width=True)
                            else:
                                st.info("요구 성능을 만족하는 모델을 찾지 못했습니다.")

            st.markdown("---")
            ref_show = st.checkbox("Reference 표시", value=True)
            cat_show = st.checkbox("Catalog 표시")
            dev_show = st.checkbox("Deviation 표시")

            st.markdown(f"#### Q-H (유량-{h_col_selected})")
            fig_h = go.Figure()
            if ref_show and not df_f.empty: add_traces(fig_h, df_f, m_r, q_col_selected, h_col_selected, models, 'lines+markers'); add_bep_markers(fig_h, df_f, m_r, q_col_selected, h_col_selected, models)
            # (이하 다른 시트 그래프 로직도 선택된 컬럼 사용하도록 수정 가능)
            render_chart(fig_h, "total_qh")
            
            # (Q-kW, Q-Eff 차트 로직 생략)

        # (개별 탭 로직 생략)

else:
    st.info("시작하려면 Excel 파일을 업로드하세요.")
