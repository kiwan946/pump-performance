import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np

# 페이지 기본 설정
st.set_page_config(page_title="Dooch XRL(F) 성능 곡선 뷰어 v4.0", layout="wide")
st.title("📊 Dooch XRL(F) 성능 곡선 뷰어 v4.0")

# --- 유틸리티 함수들 ---

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

def calculate_efficiency_user_formula(df, q_col, h_col, k_col):
    if not all(col in df.columns for col in [q_col, h_col, k_col] if col):
        return df
    df_copy = df.copy()
    hydraulic_power = 0.163 * df_copy[q_col] * df_copy[h_col]
    shaft_power = df_copy[k_col]
    efficiency = np.where(shaft_power > 0, (hydraulic_power / shaft_power) * 100, 0)
    df_copy['Efficiency'] = efficiency
    return df_copy

def load_sheet(name):
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

# --- 분석 및 시각화 함수들 ---

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
    m_r, q_r, h_r_d, h_r_t, k_r, df_r_orig = load_sheet("reference data")
    m_c, q_c, h_c_d, h_c_t, k_c, df_c_orig = load_sheet("catalog data")
    m_d, q_d, h_d_d, h_d_t, k_d, df_d_orig = load_sheet("deviation data")

    st.sidebar.title("⚙️ 분석 설정")
    head_options = []
    if h_r_d: head_options.append(h_r_d)
    if h_r_t and h_r_t not in head_options: head_options.append(h_r_t)

    if not head_options:
        st.error("오류: 'reference data' 시트에서 '토출양정' 또는 '전양정' 컬럼을 찾을 수 없습니다.")
    else:
        h_col_choice = st.sidebar.radio("효율 계산 기준 양정 (Total 탭 적용)", options=head_options, key='head_choice')
        st.sidebar.info(f"**'{h_col_choice}'** 기준으로 'Total' 탭의 효율 및 분석이 수행됩니다.")

        df_r = calculate_efficiency_user_formula(df_r_orig.copy(), q_r, h_col_choice, k_r)
        df_c = calculate_efficiency_user_formula(df_c_orig.copy(), q_c, h_col_choice, k_c)
        df_d = calculate_efficiency_user_formula(df_d_orig.copy(), q_d, h_col_choice, k_d)

        tab_list = ["Total", "Reference", "Catalog", "Deviation"]
        tabs = st.tabs(tab_list)

        with tabs[0]:
            st.subheader("📊 Total - 통합 곡선 및 운전점 분석")
            if df_r.empty:
                st.warning("'reference data' 시트가 비어있거나 필수 컬럼이 부족합니다.")
            else:
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
                ref_show = st.checkbox("Reference 표시", value=True)
                cat_show = st.checkbox("Catalog 표시")
                dev_show = st.checkbox("Deviation 표시")

                # ★★★ 수정된 부분: Total 탭 차트 레이아웃 변경 ★★★
                st.markdown(f"#### Q-H (유량-{h_col_choice})")
                fig_h = go.Figure()
                if ref_show and not df_f.empty: add_traces(fig_h, df_f, m_r, q_r, h_col_choice, models, 'lines+markers'); add_bep_markers(fig_h, df_f, m_r, q_r, h_col_choice, models)
                if cat_show and not df_c.empty: add_traces(fig_h, df_c, m_c, q_c, h_col_choice, models, 'lines+markers', line_style=dict(dash='dot'))
                if dev_show and not df_d.empty: add_traces(fig_h, df_d, m_d, q_d, h_col_choice, models, 'markers')
                if target_q > 0 and target_h > 0:
                    fig_h.add_trace(go.Scatter(x=[target_q], y=[target_h], mode='markers', marker=dict(symbol='cross', size=15, color='magenta'), name='정격 운전점'))
                    if analysis_mode == "소방":
                        churn_h_limit = 1.4 * target_h
                        fig_h.add_trace(go.Scatter(x=[0], y=[churn_h_limit], mode='markers', marker=dict(symbol='x', size=12, color='red'), name=f'체절점 상한'))
                        overload_q = 1.5 * target_q
                        overload_h_limit = 0.65 * target_h
                        fig_h.add_trace(go.Scatter(x=[overload_q], y=[overload_h_limit], mode='markers', marker=dict(symbol='diamond-open', size=12, color='blue'), name=f'최대점 하한'))
                render_chart(fig_h, "total_qh")

                st.markdown("#### Q-kW (유량-축동력)")
                fig_k = go.Figure()
                if ref_show and not df_f.empty: add_traces(fig_k, df_f, m_r, q_r, k_r, models, 'lines+markers')
                if cat_show and not df_c.empty: add_traces(fig_k, df_c, m_c, q_c, k_c, models, 'lines+markers', line_style=dict(dash='dot'))
                if dev_show and not df_d.empty: add_traces(fig_k, df_d, m_d, q_d, k_d, models, 'markers')
                render_chart(fig_k, "total_qk")
                
                st.markdown("#### Q-Efficiency (유량-효율)")
                fig_e = go.Figure()
                if ref_show and not df_f.empty: add_traces(fig_e, df_f, m_r, q_r, 'Efficiency', models, 'lines+markers')
                if cat_show and not df_c.empty: add_traces(fig_e, df_c, m_c, q_c, 'Efficiency', models, 'lines+markers', line_style=dict(dash='dot'))
                if dev_show and not df_d.empty: add_traces(fig_e, df_d, m_d, q_d, 'Efficiency', models, 'markers')
                render_chart(fig_e, "total_qe")
        
        # ★★★ 수정된 부분: 개별 탭 기능 전체 복원 및 수정 ★★★
        for idx, sheet_name in enumerate(["Reference", "Catalog", "Deviation"]):
            with tabs[idx+1]:
                st.subheader(f"📊 {sheet_name} Data")
                
                # 원본 데이터프레임과 컬럼명 변수들 할당
                if sheet_name == "Reference": df_orig, mcol, qcol, hcol_d_sheet, hcol_t_sheet, kcol = df_r_orig, m_r, q_r, h_r_d, h_r_t, k_r
                elif sheet_name == "Catalog": df_orig, mcol, qcol, hcol_d_sheet, hcol_t_sheet, kcol = df_c_orig, m_c, q_c, h_c_d, h_c_t, k_c
                else: df_orig, mcol, qcol, hcol_d_sheet, hcol_t_sheet, kcol = df_d_orig, m_d, q_d, h_d_d, h_d_t, k_d

                if df_orig.empty:
                    st.info(f"'{sheet_name.lower()} data' 시트의 데이터가 없거나 로드에 실패했습니다.")
                    continue
                
                # 각 탭의 양정 기준을 자체적으로 결정
                h_col_for_tab = hcol_d_sheet if hcol_d_sheet else hcol_t_sheet
                if not h_col_for_tab:
                    st.warning("이 시트에는 양정 데이터가 없어 Q-H 및 효율 곡선을 표시할 수 없습니다.")
                    continue
                    
                # 선택된 양정 기준으로 효율 계산
                df_tab = calculate_efficiency_user_formula(df_orig.copy(), qcol, h_col_for_tab, kcol)

                df_f_tab = render_filters(df_tab, mcol, sheet_name)
                models_tab = df_f_tab[mcol].unique().tolist() if not df_f_tab.empty else []

                if not models_tab:
                    st.info("차트를 보려면 모델을 선택해주세요.")
                    continue
                
                mode, style = ('markers', None) if sheet_name == "Deviation" else ('lines+markers', dict(dash='dot') if sheet_name == "Catalog" else None)

                st.markdown(f"#### Q-H ({h_col_for_tab})")
                fig1 = go.Figure()
                add_traces(fig1, df_f_tab, mcol, qcol, h_col_for_tab, models_tab, mode, line_style=style)
                # 개별 탭에서는 BEP 마커 삭제
                render_chart(fig1, key=f"{sheet_name}_qh")
                
                if kcol:
                    st.markdown("#### Q-kW (축동력)")
                    fig2 = go.Figure()
                    add_traces(fig2, df_f_tab, mcol, qcol, kcol, models_tab, mode, line_style=style)
                    render_chart(fig2, key=f"{sheet_name}_qk")

                if 'Efficiency' in df_f_tab.columns:
                    st.markdown("#### Q-Efficiency (효율)")
                    fig3 = go.Figure()
                    add_traces(fig3, df_f_tab, mcol, qcol, 'Efficiency', models_tab, mode, line_style=style)
                    # 개별 탭에서는 BEP 마커 삭제
                    fig3.update_layout(yaxis_title="효율 (%)", yaxis=dict(range=[0, 100]))
                    render_chart(fig3, key=f"{sheet_name}_qe")

                st.markdown("#### 데이터 확인")
                st.dataframe(df_f_tab, use_container_width=True)

else:
    st.info("시작하려면 Excel 파일을 업로드하세요.")
