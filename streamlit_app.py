import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import math

# 페이지 기본 설정
st.set_page_config(page_title="Dooch XRL(F) 성능 곡선 뷰어 v18.0", layout="wide")
st.title("📊 Dooch XRL(F) 성능 곡선 뷰어 v18.0")

# --- 유틸리티 함수들 ---
SERIES_ORDER = ["XRF3", "XRF5", "XRF10", "XRF15", "XRF20", "XRF32", "XRF45", "XRF64", "XRF95", "XRF125", "XRF155", "XRF185", "XRF215", "XRF255"]

def get_best_match_column(df, names):
    if df is None or df.empty: return None
    for n in names:
        for col in df.columns:
            if n in str(col).strip():
                return col
    return None

def calculate_efficiency(df, q_col, h_col, k_col):
    if not all(col and col in df.columns for col in [q_col, h_col, k_col]): return df
    df_copy = df.copy()
    hydraulic_power = 0.163 * df_copy[q_col] * df_copy[h_col]
    shaft_power = df_copy[k_col]
    df_copy['Efficiency'] = np.where(shaft_power > 0, (hydraulic_power / shaft_power) * 100, 0)
    return df_copy

def load_sheet(uploaded_file, sheet_name):
    try:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        df.columns = df.columns.str.strip()
        mcol = get_best_match_column(df, ["모델명", "모델", "Model"])
        if not mcol: return None, pd.DataFrame()
        if 'Series' in df.columns: df = df.drop(columns=['Series'])
        df['Series'] = df[mcol].astype(str).str.extract(r"(XRF\d+)")
        df['Series'] = pd.Categorical(df['Series'], categories=SERIES_ORDER, ordered=True)
        df = df.sort_values('Series')
        return mcol, df
    except Exception:
        return None, pd.DataFrame()

def process_data(df, q_col, h_col, k_col):
    if df.empty: return df
    temp_df = df.copy()
    for col in [q_col, h_col, k_col]:
        if col in temp_df.columns and col is not None:
            temp_df = temp_df.dropna(subset=[col])
            temp_df = temp_df[pd.to_numeric(temp_df[col], errors='coerce').notna()]
            temp_df[col] = pd.to_numeric(temp_df[col])
    return calculate_efficiency(temp_df, q_col, h_col, k_col)

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
    if df is None or df.empty or mcol is None or 'Series' not in df.columns:
        st.warning("필터링할 데이터가 없습니다.")
        return pd.DataFrame()
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
        fig.add_trace(go.Scatter(x=sub[xcol], y=sub[ycol], mode=mode, name=m + name_suffix, line=line_style or {}, opacity=1.0))

def add_bep_markers(fig, df, mcol, qcol, ycol, models):
    for m in models:
        model_df = df[df[mcol] == m]
        if not model_df.empty and 'Efficiency' in model_df.columns and not model_df['Efficiency'].isnull().all():
            bep_row = model_df.loc[model_df['Efficiency'].idxmax()]
            fig.add_trace(go.Scatter(x=[bep_row[qcol]], y=[bep_row[ycol]], mode='markers', marker=dict(symbol='star', size=15, color='gold'), name=f'{m} BEP'))

def add_guide_lines(fig, h_line, v_line):
    if h_line is not None and h_line > 0:
        fig.add_shape(type="line", x0=0, x1=1, xref="paper", y0=h_line, y1=h_line, yref="y", line=dict(color="gray", dash="dash"))
    if v_line is not None and v_line > 0:
        fig.add_shape(type="line", x0=v_line, x1=v_line, xref="x", y0=0, y1=1, yref="paper", line=dict(color="gray", dash="dash"))

def render_chart(fig, key):
    fig.update_layout(dragmode='pan', xaxis=dict(fixedrange=False), yaxis=dict(fixedrange=False), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displaylogo': False}, key=key)

# =========================
# [Validation] 통계 검증 유틸
# =========================
# t_{0.975, df} 근사표 (df<=30), 그 외 1.96 사용
TCRIT_975 = {
    1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,
    11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,18:2.101,19:2.093,20:2.086,
    21:2.080,22:2.074,23:2.069,24:2.064,25:2.060,26:2.056,27:2.052,28:2.048,29:2.045,30:2.042
}

def tcrit_975(n):
    if n is None or n < 2: return float('inf')  # 표본 1개면 CI 불가
    df = n - 1
    if df <= 0: return float('inf')
    if df in TCRIT_975: return TCRIT_975[df]
    return 1.96  # df>30 근사

def linear_interp_extrap(x, y, x_new):
    """x,y는 1D, x_new에 대해 선형 보간/양끝 외삽."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]; y = y[mask]
    if len(x) < 2:
        return np.full_like(x_new, np.nan, dtype=float)
    order = np.argsort(x)
    x = x[order]; y = y[order]
    y_new = np.interp(x_new, x, y)
    # 왼쪽 외삽
    if np.any(x_new < x[0]) and x[1] != x[0]:
        slope_left = (y[1]-y[0])/(x[1]-x[0])
        m = x_new < x[0]
        y_new[m] = y[0] + slope_left*(x_new[m]-x[0])
    # 오른쪽 외삽
    if np.any(x_new > x[-1]) and x[-1] != x[-2]:
        slope_right = (y[-1]-y[-2])/(x[-1]-x[-2])
        m = x_new > x[-1]
        y_new[m] = y[-1] + slope_right*(x_new[m]-x[-1])
    return y_new

def make_q_grid_from_reference(model_df, q_col, num_points=10, include_zero=True):
    sub = model_df.dropna(subset=[q_col]).sort_values(q_col)
    if sub.empty: return np.array([])
    q_min = float(sub[q_col].min())
    q_max = float(sub[q_col].max())
    start = 0.0 if include_zero else q_min
    if q_max <= start:  # 방어
        start = q_min
    return np.linspace(start, q_max, num_points)

def interpolate_curve(df, q_col, y_col, q_grid):
    if df is None or df.empty or q_col not in df.columns or y_col not in df.columns:
        return np.full_like(q_grid, np.nan, dtype=float)
    return linear_interp_extrap(df[q_col].values, df[y_col].values, q_grid)

def compute_stats_at_grid(sample_matrix):
    """
    sample_matrix: shape (n_tests, n_q). NaN 허용.
    return: dict(mean, std, n) 각 shape (n_q,)
    """
    mean = np.nanmean(sample_matrix, axis=0)
    std = np.nanstd(sample_matrix, axis=0, ddof=1)  # 표본표준편차
    n = np.sum(~np.isnan(sample_matrix), axis=0)
    # n<2 인 곳의 std는 NaN 처리
    std[(n < 2)] = np.nan
    return {"mean": mean, "std": std, "n": n}

def summarize_validation_per_model(ref_q, ref_y, stats_dict, ci_alpha=0.05, coverage_z=1.96):
    mean, std, n = stats_dict["mean"], stats_dict["std"], stats_dict["n"]
    # 95% CI 반경
    tcrit = np.array([tcrit_975(int(ni)) if not np.isnan(ni) else np.nan for ni in n])
    half_ci = tcrit * (std / np.sqrt(n, where=(n>0), out=np.ones_like(n)))
    ci_low = mean - half_ci
    ci_high = mean + half_ci
    # 95% 개별 범위(≈±1.96σ)
    band_low = mean - coverage_z*std
    band_high = mean + coverage_z*std
    # 통과 플래그
    in_ci = (ref_y >= ci_low) & (ref_y <= ci_high)
    in_band = (ref_y >= band_low) & (ref_y <= band_high)
    # 유효 비교가 가능한 지점만(표본 n>=2) 집계
    valid_mask = (~np.isnan(ci_low)) & (~np.isnan(ci_high)) & (~np.isnan(ref_y))
    ci_pass_rate = float(np.mean(in_ci[valid_mask]))*100 if np.any(valid_mask) else 0.0
    band_pass_rate = float(np.mean(in_band[valid_mask]))*100 if np.any(valid_mask) else 0.0

    return {
        "ci_low": ci_low, "ci_high": ci_high,
        "band_low": band_low, "band_high": band_high,
        "mean": mean, "std": std, "n": n,
        "in_ci": in_ci, "in_band": in_band,
        "ci_pass_rate": ci_pass_rate, "band_pass_rate": band_pass_rate,
        "valid_mask": valid_mask
    }

def plot_validation_band(q_grid, ref_y, stats, model_name, y_label, key):
    mean = stats["mean"]; ci_low = stats["ci_low"]; ci_high = stats["ci_high"]
    band_low = stats["band_low"]; band_high = stats["band_high"]

    fig = go.Figure()
    # 평균선
    fig.add_trace(go.Scatter(x=q_grid, y=mean, mode='lines', name='Deviation 평균', line=dict(width=2)))
    # 95% CI 밴드
    fig.add_trace(go.Scatter(x=np.concatenate([q_grid, q_grid[::-1]]),
                             y=np.concatenate([ci_high, ci_low[::-1]]),
                             fill='toself', name='95% CI(평균)', opacity=0.25))
    # ±1.96σ 밴드
    fig.add_trace(go.Scatter(x=np.concatenate([q_grid, q_grid[::-1]]),
                             y=np.concatenate([band_high, band_low[::-1]]),
                             fill='toself', name='±1.96σ(개별)', opacity=0.15))
    # Reference
    fig.add_trace(go.Scatter(x=q_grid, y=ref_y, mode='lines+markers', name='Reference', line=dict(width=3)))
    fig.update_layout(xaxis_title="유량(Q)", yaxis_title=y_label, title=f"{model_name} - Validation 밴드")
    render_chart(fig, key)

# --- 메인 애플리케이션 로직 ---

uploaded_file = st.file_uploader("Excel 파일 업로드 (.xlsx 또는 .xlsm)", type=["xlsx", "xlsm"])

if uploaded_file:
    # 1. 시트별 원본 데이터 로드
    m_r, df_r_orig = load_sheet(uploaded_file, "reference data")
    m_c, df_c_orig = load_sheet(uploaded_file, "catalog data")
    m_d, df_d_orig = load_sheet(uploaded_file, "deviation data")
    
    if df_r_orig.empty:
        st.error("오류: 'reference data' 시트를 찾을 수 없거나 '모델명' 관련 컬럼이 없습니다. 파일을 확인해주세요.")
    else:
        # 2. 사이드바 컬럼 선택 UI (Total 탭 및 분석용)
        st.sidebar.title("⚙️ 분석 설정")
        st.sidebar.markdown("### Total 탭 & 운전점 분석 컬럼 지정")
        
        all_columns = df_r_orig.columns.tolist()
        def safe_get_index(items, value, default=0):
            try: return items.index(value)
            except (ValueError, TypeError): return default

        q_auto_r = get_best_match_column(df_r_orig, ["토출량", "유량", "Flow", "Q"])
        h_auto_r = get_best_match_column(df_r_orig, ["토출양정", "전양정", "Head", "H"])
        k_auto_r = get_best_match_column(df_r_orig, ["축동력", "kW", "Power"])
        
        q_col_total = st.sidebar.selectbox("유량 (Flow) 컬럼", all_columns, index=safe_get_index(all_columns, q_auto_r))
        h_col_total = st.sidebar.selectbox("양정 (Head) 컬럼", all_columns, index=safe_get_index(all_columns, h_auto_r))
        k_col_total = st.sidebar.selectbox("축동력 (Power) 컬럼", all_columns, index=safe_get_index(all_columns, k_auto_r))
        
        # ★ 각 시트별 컬럼 자동 감지
        q_c, h_c, k_c = (get_best_match_column(df_c_orig, ["토출량", "유량", "Flow", "Q"]),
                         get_best_match_column(df_c_orig, ["토출양정", "전양정", "Head", "H"]),
                         get_best_match_column(df_c_orig, ["축동력", "kW", "Power"]))
        q_d, h_d, k_d = (get_best_match_column(df_d_orig, ["토출량", "유량", "Flow", "Q"]),
                         get_best_match_column(df_d_orig, ["토출양정", "전양정", "Head", "H"]),
                         get_best_match_column(df_d_orig, ["축동력", "kW", "Power"]))
        # [Validation] 시험번호 컬럼 자동 탐지
        test_col = get_best_match_column(df_d_orig, ["시험번호", "시험 No", "시험", "Test No", "TestNo", "Test ID", "TestID", "ID"])

        # 3. 각 데이터 정제 및 효율 계산
        df_r = process_data(df_r_orig, q_col_total, h_col_total, k_col_total)
        df_c = process_data(df_c_orig, q_c, h_c, k_c)
        df_d = process_data(df_d_orig, q_d, h_d, k_d)
        
        # 4. 탭 생성 및 화면 표시 (+ Validation)
        tab_list = ["Total", "Reference", "Catalog", "Deviation", "Validation"]
        tabs = st.tabs(tab_list)

        # =========================
        # Total 탭
        # =========================
        with tabs[0]:
            st.subheader("📊 Total - 통합 곡선 및 운전점 분석")
            df_f = render_filters(df_r, m_r, "total")
            models = df_f[m_r].unique().tolist() if m_r and not df_f.empty else []

            with st.expander("운전점 분석 (Operating Point Analysis)"):
                analysis_mode = st.radio("분석 모드", ["기계", "소방"], key="analysis_mode", horizontal=True)
                op_col1, op_col2 = st.columns(2)
                with op_col1: target_q = st.number_input("목표 유량 (Q)", value=0.0, format="%.2f")
                with op_col2: target_h = st.number_input("목표 양정 (H)", value=0.0, format="%.2f")
                if analysis_mode == "소방": st.info("소방 펌프 성능 기준 3점을 자동으로 분석합니다.")
                if st.button("운전점 분석 실행"):
                    if not models: st.warning("먼저 분석할 시리즈나 모델을 선택해주세요.")
                    else:
                        with st.spinner("선택된 모델들을 분석 중입니다..."):
                            if analysis_mode == "소방": op_results_df = analyze_fire_pump_point(df_r, models, target_q, target_h, m_r, q_col_total, h_col_total, k_col_total)
                            else: op_results_df = analyze_operating_point(df_r, models, target_q, target_h, m_r, q_col_total, h_col_total, k_col_total)
                            if not op_results_df.empty: st.success(f"총 {len(op_results_df)}개의 모델이 요구 성능을 만족합니다."); st.dataframe(op_results_df, use_container_width=True)
                            else: st.info("요구 성능을 만족하는 모델을 찾지 못했습니다.")

            with st.expander("차트 보조선 추가"):
                g_col1, g_col2, g_col3 = st.columns(3)
                with g_col1: h_guide_h, v_guide_h = st.number_input("Q-H 수평선", value=0.0), st.number_input("Q-H 수직선", value=0.0)
                with g_col2: h_guide_k, v_guide_k = st.number_input("Q-kW 수평선", value=0.0), st.number_input("Q-kW 수직선", value=0.0)
                with g_col3: h_guide_e, v_guide_e = st.number_input("Q-Eff 수평선", value=0.0), st.number_input("Q-Eff 수직선", value=0.0)

            st.markdown("---")
            ref_show = st.checkbox("Reference 표시", value=True)
            cat_show = st.checkbox("Catalog 표시")
            dev_show = st.checkbox("Deviation 표시")

            st.markdown(f"#### Q-H (유량-{h_col_total})")
            fig_h = go.Figure()
            if ref_show and not df_f.empty: add_traces(fig_h, df_f, m_r, q_col_total, h_col_total, models, 'lines+markers'); add_bep_markers(fig_h, df_f, m_r, q_col_total, h_col_total, models)
            if cat_show and not df_c.empty: add_traces(fig_h, df_c, m_c, q_c, h_c, models, 'lines+markers', line_style=dict(dash='dot'))
            if dev_show and not df_d.empty: add_traces(fig_h, df_d, m_d, q_d, h_d, models, 'markers')
            if 'target_q' in locals() and target_q > 0 and target_h > 0:
                fig_h.add_trace(go.Scatter(x=[target_q], y=[target_h], mode='markers', marker=dict(symbol='cross', size=15, color='magenta'), name='정격 운전점'))
                if analysis_mode == "소방":
                    churn_h_limit = 1.4 * target_h; fig_h.add_trace(go.Scatter(x=[0], y=[churn_h_limit], mode='markers', marker=dict(symbol='x', size=12, color='red'), name=f'체절점 상한'))
                    overload_q, overload_h_limit = 1.5 * target_q, 0.65 * target_h; fig_h.add_trace(go.Scatter(x=[overload_q], y=[overload_h_limit], mode='markers', marker=dict(symbol='diamond-open', size=12, color='blue'), name=f'최대점 하한'))
            add_guide_lines(fig_h, h_guide_h, v_guide_h)
            render_chart(fig_h, "total_qh")

            st.markdown("#### Q-kW (유량-축동력)")
            fig_k = go.Figure()
            if ref_show and not df_f.empty: add_traces(fig_k, df_f, m_r, q_col_total, k_col_total, models, 'lines+markers')
            if cat_show and not df_c.empty: add_traces(fig_k, df_c, m_c, q_c, k_c, models, 'lines+markers', line_style=dict(dash='dot'))
            if dev_show and not df_d.empty: add_traces(fig_k, df_d, m_d, q_d, k_d, models, 'markers')
            add_guide_lines(fig_k, h_guide_k, v_guide_k)
            render_chart(fig_k, "total_qk")
            
            st.markdown("#### Q-Efficiency (유량-효율)")
            fig_e = go.Figure()
            if ref_show and not df_f.empty: add_traces(fig_e, df_f, m_r, q_col_total, 'Efficiency', models, 'lines+markers'); add_bep_markers(fig_e, df_f, m_r, q_col_total, 'Efficiency', models)
            if cat_show and not df_c.empty: add_traces(fig_e, df_c, m_c, q_c, 'Efficiency', models, 'lines+markers', line_style=dict(dash='dot'))
            if dev_show and not df_d.empty: add_traces(fig_e, df_d, m_d, q_d, 'Efficiency', models, 'markers')
            add_guide_lines(fig_e, h_guide_e, v_guide_e)
            render_chart(fig_e, "total_qe")

        # =========================
        # Reference / Catalog / Deviation 탭
        # =========================
        for idx, sheet_name in enumerate(["Reference", "Catalog", "Deviation"]):
            with tabs[idx+1]:
                st.subheader(f"📊 {sheet_name} Data")
                
                df, mcol, df_orig = (df_r, m_r, df_r_orig) if sheet_name == "Reference" else \
                                  (df_c, m_c, df_c_orig) if sheet_name == "Catalog" else \
                                  (df_d, m_d, df_d_orig)
                
                if df.empty:
                    st.info(f"'{sheet_name.lower()}' 시트의 데이터가 없거나 처리할 수 없습니다.")
                    continue

                q_col_tab = get_best_match_column(df_orig, ["토출량", "유량", "Flow", "Q"])
                h_col_tab = get_best_match_column(df_orig, ["토출양정", "전양정", "Head", "H"])
                k_col_tab = get_best_match_column(df_orig, ["축동력", "kW", "Power"])

                df_f_tab = render_filters(df, mcol, sheet_name)
                models_tab = df_f_tab[mcol].unique().tolist() if not df_f_tab.empty else []

                if not models_tab:
                    st.info("차트를 보려면 모델을 선택해주세요.")
                    continue
                
                mode, style = ('markers', None) if sheet_name == "Deviation" else ('lines+markers', dict(dash='dot') if sheet_name == "Catalog" else None)
                
                if h_col_tab: 
                    st.markdown(f"#### Q-H ({h_col_tab})")
                    fig1 = go.Figure()
                    add_traces(fig1, df_f_tab, mcol, q_col_tab, h_col_tab, models_tab, mode, line_style=style)
                    render_chart(fig1, key=f"{sheet_name}_qh")
                if k_col_tab in df_f_tab.columns: 
                    st.markdown("#### Q-kW (축동력)")
                    fig2 = go.Figure()
                    add_traces(fig2, df_f_tab, mcol, q_col_tab, k_col_tab, models_tab, mode, line_style=style)
                    render_chart(fig2, key=f"{sheet_name}_qk")
                if 'Efficiency' in df_f_tab.columns: 
                    st.markdown("#### Q-Efficiency (효율)")
                    fig3 = go.Figure()
                    add_traces(fig3, df_f_tab, mcol, q_col_tab, 'Efficiency', models_tab, mode, line_style=style)
                    fig3.update_layout(yaxis_title="효율 (%)", yaxis=dict(range=[0, 100]))
                    render_chart(fig3, key=f"{sheet_name}_qe")
                st.markdown("#### 데이터 확인")
                st.dataframe(df_f_tab, use_container_width=True)

        # =========================
        # [Validation] 검증 탭
        # =========================
        with tabs[4]:
            st.subheader("✅ Validation - Reference 표준성능 통계 검증 (95%)")

            if df_r.empty or df_d.empty or m_r is None or m_d is None:
                st.info("Reference / Deviation 데이터가 필요합니다.")
            else:
                st.markdown("**검증 지표 선택 및 파라미터**")
                metric = st.selectbox("지표", ["양정(H)", "축동력(kW)", "효율(%)"], index=0)
                # 지표별 사용 컬럼 결정
                if metric == "양정(H)":
                    y_r_col, y_d_col, y_label = h_col_total, h_d, "양정 (H)"
                elif metric == "축동력(kW)":
                    y_r_col, y_d_col, y_label = k_col_total, k_d, "축동력 (kW)"
                else:
                    y_r_col, y_d_col, y_label = 'Efficiency', 'Efficiency', "효율 (%)"

                num_points = st.slider("검증 유량 포인트 개수", min_value=6, max_value=20, value=10, step=1)
                pass_threshold = st.slider("모델 판정 임계 통과율(%)", min_value=50, max_value=100, value=80, step=5)

                df_f_val = render_filters(df_r, m_r, "validation")
                models_val = df_f_val[m_r].unique().tolist() if not df_f_val.empty else []

                if not models_val:
                    st.info("검증할 시리즈/모델을 선택해주세요.")
                else:
                    if y_r_col not in df_r.columns:
                        st.warning(f"Reference 데이터에 '{y_r_col}' 컬럼이 없습니다.")
                    elif y_d_col not in df_d.columns:
                        st.warning(f"Deviation 데이터에 '{y_d_col}' 컬럼이 없습니다.")
                    elif test_col is None:
                        st.warning("Deviation 데이터에서 시험번호를 식별할 수 있는 컬럼(예: '시험번호', 'Test No')을 찾지 못했습니다.")
                    else:
                        # 모델별 요약
                        summary_rows = []
                        model_detail_buffers = {}

                        with st.spinner("검증 계산 중..."):
                            for model in models_val:
                                ref_sub = df_r[df_r[m_r] == model]
                                if ref_sub.empty: 
                                    continue
                                # 유량 격자 생성 (0 포함)
                                q_grid = make_q_grid_from_reference(ref_sub, q_col_total, num_points=num_points, include_zero=True)
                                if q_grid.size == 0:
                                    continue
                                # Reference y 보간
                                ref_y = interpolate_curve(ref_sub[[q_col_total, y_r_col]], q_col_total, y_r_col, q_grid)

                                # Deviation: 동일 모델의 시험번호별 곡선 보간
                                dev_sub = df_d[df_d[m_d] == model]
                                if dev_sub.empty or test_col not in dev_sub.columns:
                                    continue
                                samples = []
                                for tid, g in dev_sub.groupby(test_col):
                                    # 최소 2점 이상일 때만 유효
                                    g2 = g.dropna(subset=[q_d, y_d_col]).sort_values(q_d)
                                    if len(g2) < 2: 
                                        continue
                                    samples.append(interpolate_curve(g2[[q_d, y_d_col]], q_d, y_d_col, q_grid))
                                if len(samples) == 0:
                                    continue
                                sample_mat = np.vstack(samples)  # (n_tests, n_q)

                                stats = compute_stats_at_grid(sample_mat)
                                result = summarize_validation_per_model(q_grid, ref_y, stats)

                                ci_rate = result["ci_pass_rate"]
                                band_rate = result["band_pass_rate"]
                                verdict = "적합" if ci_rate >= pass_threshold else "부적합"
                                summary_rows.append({
                                    "모델명": model,
                                    "시험개수(n)": int(sample_mat.shape[0]),
                                    "CI 통과율(%)": round(ci_rate, 1),
                                    "±1.96σ 통과율(%)": round(band_rate, 1),
                                    "최종 판정": verdict
                                })
                                # 상세 차트를 위해 저장
                                model_detail_buffers[model] = (q_grid, ref_y, stats, y_label)

                        if len(summary_rows) == 0:
                            st.info("검증 가능한 데이터가 없습니다. (시험번호별 데이터 부족 또는 컬럼 불일치)")
                        else:
                            st.markdown("### 모델별 검증 요약")
                            st.dataframe(pd.DataFrame(summary_rows).sort_values(["최종 판정","CI 통과율(%)"], ascending=[True, False]), use_container_width=True)

                            # 상세 보기
                            st.markdown("---")
                            sel_model = st.selectbox("상세 차트 모델 선택", list(model_detail_buffers.keys()))
                            if sel_model:
                                q_grid, ref_y, stats, y_label = model_detail_buffers[sel_model]
                                plot_validation_band(q_grid, ref_y, stats, sel_model, y_label, key=f"val_band_{sel_model}")

else:
    st.info("시작하려면 Excel 파일을 업로드하세요.")

