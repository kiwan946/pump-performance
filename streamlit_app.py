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
    fig.update_layout(dragmode='pan', xaxis=dict(fixedrange=False), yaxis=dict(fixedrange=False))
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displaylogo': False}, key=key)

# --- 메인 애플리케이션 로직 ---

uploaded_file = st.file_uploader("Excel 파일 업로드 (.xlsx 또는 .xlsm)", type=["xlsx", "xlsm"])

if uploaded_file:
    # 데이터 로드
    with st.spinner('데이터를 로드하고 처리하는 중입니다...'):
        m_r, q_r, h_r_d, h_r_t, k_r, df_r_orig = load_sheet("reference data")
        m_c, q_c, h_c_d, h_c_t, k_c, df_c_orig = load_sheet("catalog data")
        m_d, q_d, h_d_d, h_d_t, k_d, df_d_orig = load_sheet("deviation data")

    st.sidebar.title("⚙️ 분석 설정")
    # ★★★ 추가된 부분: 양정 선택 UI ★★★
    head_options = []
    if h_r_d: head_options.append(h_r_d)
    if h_r_t and h_r_t not in head_options: head_options.append(h_r_t)

    if not head_options:
        st.error("Excel 파일에서 '토출양정' 또는 '전양정' 컬럼을 찾을 수 없습니다.")
    else:
        h_col_choice = st.sidebar.radio(
            "효율 계산 기준 양정",
            options=head_options,
            key='head_choice'
        )
        st.sidebar.info(f"선택된 **'{h_col_choice}'**을(를) 기준으로 모든 효율 곡선 및 분석이 수행됩니다.")

        # 선택된 양정 기준으로 효율 재계산
        df_r = calculate_efficiency_user_formula(df_r_orig.copy(), q_r, h_col_choice, k_r)
        df_c = calculate_efficiency_user_formula(df_c_orig.copy(), q_c, h_col_choice, k_c)
        df_d = calculate_efficiency_user_formula(df_d_orig.copy(), q_d, h_col_choice, k_d)

        # 탭 생성
        tab_list = ["Total", "Reference", "Catalog", "Deviation"]
        tabs = st.tabs(tab_list)

        # Total 탭
        with tabs[0]:
            st.subheader("📊 Total - 통합 곡선 및 운전점 분석")
            df_f = render_filters(df_r, m_r, "total")
            models = df_f[m_r].unique().tolist() if not df_f.empty else []

            with st.expander("운전점 분석 (Operating Point Analysis)", expanded=True):
                # ... (이하 운전점 분석 UI는 이전과 동일)
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
                            if analysis_mode == "소방":
                                op_results_df = analyze_fire_pump_point(df_r, models, target_q, target_h, m_r, q_r, h_col_choice, k_r)
                            else:
                                op_results_df = analyze_operating_point(df_r, models, target_q, target_h, m_r, q_r, h_col_choice, k_r)
                            if not op_results_df.empty:
                                st.success(f"총 {len(op_results_df)}개의 모델이 요구 성능을 만족합니다.")
                                st.dataframe(op_results_df, use_container_width=True)
                            else:
                                st.info("요구 성능을 만족하는 모델을 찾지 못했습니다.")

            st.markdown("---")
            # ... (이하 그래프 표시는 이전과 동일, h_col_choice를 사용) ...
            ref_show = st.checkbox("Reference 표시", value=True)
            cat_show = st.checkbox("Catalog 표시")
            dev_show = st.checkbox("Deviation 표시")

            st.markdown("#### Q-H (유량-양정)")
            fig_h = go.Figure()
            if ref_show and not df_f.empty: add_traces(fig_h, df_f, m_r, q_r, h_col_choice, models, 'lines+markers'); add_bep_markers(fig_h, df_f, m_r, q_r, h_col_choice, models)
            if cat_show and not df_c.empty: add_traces(fig_h, df_c, m_c, q_c, h_col_choice, models, 'lines+markers', line_style=dict(dash='dot'))
            if dev_show and not df_d.empty: add_traces(fig_h, df_d, m_d, q_d, h_col_choice, models, 'markers')
            render_chart(fig_h, "total_qh")
            
            # Q-kW, Q-Eff 차트 추가 (생략)


        # 개별 탭 기능 복원
        for idx, sheet in enumerate(["Reference", "Catalog", "Deviation"]):
            with tabs[idx+1]:
                # ... (이전과 동일한 개별 탭 로직, 단 h_col_choice를 사용하도록 수정 필요)

else:
    st.info("시작하려면 Excel 파일을 업로드하세요.")
