import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np

# 페이지 기본 설정
st.set_page_config(page_title="Dooch XRL(F) 성능 곡선 뷰어 v3.5", layout="wide")
st.title("📊 Dooch XRL(F) 성능 곡선 뷰어 v3.5")

# --- 유틸리티 함수들 ---

# 시리즈 순서 정의
SERIES_ORDER = [
    "XRF3", "XRF5", "XRF10", "XRF15", "XRF20", "XRF32",
    "XRF45", "XRF64", "XRF95", "XRF125", "XRF155", "XRF185",
    "XRF215", "XRF255"
]

def get_best_match_column(df, names):
    """키워드를 기반으로 DataFrame에서 최적의 컬럼 이름을 찾습니다."""
    for n in names:
        for col in df.columns:
            if n in col:
                return col
    return None

def calculate_efficiency(df, q_col, h_col, k_col, q_unit='L/min'):
    """유량, 양정, 동력 데이터를 바탕으로 펌프 효율을 계산합니다."""
    if not all([q_col, h_col, k_col, k_col in df.columns]):
        return df
    df_copy = df.copy()
    # 유량 단위를 m^3/s로 통일
    if q_unit == 'L/min':
        q_m3_s = df_copy[q_col] / 60000
    elif q_unit == 'm3/h':
        q_m3_s = df_copy[q_col] / 3600
    else:  # 기본값 m3/s
        q_m3_s = df_copy[q_col]

    rho, g = 1000, 9.81  # 물 밀도, 중력 가속도
    power_kw = df_copy[k_col]
    # 0으로 나누기 오류 방지
    efficiency = np.where(
        power_kw > 0,
        (rho * g * q_m3_s * df_copy[h_col]) / (power_kw * 1000) * 100,
        0
    )
    df_copy['Efficiency'] = efficiency
    return df_copy

def load_sheet(name):
    """지정된 시트 이름으로 Excel 파일을 로드하고 전처리합니다."""
    try:
        df = pd.read_excel(uploaded_file, sheet_name=name)
    except Exception as e:
        st.error(f"'{name}' 시트를 읽는 중 에러 발생: {e}")
        return None, None, None, None, pd.DataFrame()

    mcol = get_best_match_column(df, ["모델명", "모델", "Model"])
    qcol = get_best_match_column(df, ["토출량", "유량"])
    hcol = get_best_match_column(df, ["토출양정", "전양정"])
    kcol = get_best_match_column(df, ["축동력"])

    if not mcol or not qcol or not hcol:
        return None, None, None, None, pd.DataFrame()

    cols_to_check = [qcol, hcol]
    if kcol:
        cols_to_check.append(kcol)

    for col in cols_to_check:
        df = df.dropna(subset=[col])
        df = df[pd.to_numeric(df[col], errors='coerce').notna()]
        df[col] = pd.to_numeric(df[col])

    df['Series'] = df[mcol].astype(str).str.extract(r"(XRF\d+)")
    df['Series'] = pd.Categorical(df['Series'], categories=SERIES_ORDER, ordered=True)
    df = df.sort_values('Series')

    df = calculate_efficiency(df, qcol, hcol, kcol, q_unit='L/min')
    return mcol, qcol, hcol, kcol, df

# --- 분석 함수들 ---

def analyze_operating_point(df, models, target_q, target_h, m_col, q_col, h_col, k_col):
    """기계 모드: 단일 운전점을 만족하는 모델을 분석합니다."""
    if target_q <= 0 or target_h <= 0: return pd.DataFrame()
    results = []
    for model in models:
        model_df = df[df[m_col] == model].sort_values(q_col)
        if len(model_df) < 2 or not (model_df[q_col].min() <= target_q <= model_df[q_col].max()): continue

        interp_h = np.interp(target_q, model_df[q_col], model_df[h_col])
        if interp_h >= target_h:
            interp_kw = np.interp(target_q, model_df[q_col], model_df[k_col]) if k_col and k_col in model_df.columns else np.nan
            interp_eff = np.interp(target_q, model_df[q_col], model_df['Efficiency']) if 'Efficiency' in model_df.columns else np.nan
            results.append({
                "모델명": model, "요구 유량": target_q, "요구 양정": target_h, "예상 양정": f"{interp_h:.2f}",
                "예상 동력(kW)": f"{interp_kw:.2f}", "예상 효율(%)": f"{interp_eff:.2f}", "선정 가능": "✅"
            })
    return pd.DataFrame(results)

def analyze_fire_pump_point(df, models, target_q, target_h, m_col, q_col, h_col, k_col):
    """소방 모드: 사용자가 입력한 목표 양정을 기준으로 3가지 조건을 모두 만족하는 모델을 분석합니다."""
    if target_q <= 0 or target_h <= 0: return pd.DataFrame()
    results = []
    for model in models:
        model_df = df[df[m_col] == model].sort_values(q_col)
        if len(model_df) < 2: continue

        # 1. 정격 운전점 (Rated Point) 확인: 펌프 성능이 목표 양정 이상인가?
        interp_h_rated = np.interp(target_q, model_df[q_col], model_df[h_col], left=np.nan, right=np.nan)
        if np.isnan(interp_h_rated) or interp_h_rated < target_h: continue

        # 2. 체절 운전점 (Churn/Shut-off) 확인: 체절 양정이 목표 양정의 140% 이하인가?
        h_churn = model_df.iloc[0][h_col]
        cond1_ok = h_churn <= (1.40 * target_h)

        # 3. 최대 운전점 (Overload) 확인: 150% 유량에서 양정이 목표 양정의 65% 이상인가?
        q_overload = 1.5 * target_q
        interp_h_overload = np.interp(q_overload, model_df[q_col], model_df[h_col], left=np.nan, right=np.nan)
        cond2_ok = (not np.isnan(interp_h_overload)) and (interp_h_overload >= (0.65 * target_h))

        if cond1_ok and cond2_ok:
            interp_kw = np.interp(target_q, model_df[q_col], model_df[k_col]) if k_col and k_col in model_df.columns else np.nan
            results.append({
                "모델명": model, "정격 예상 양정": f"{interp_h_rated:.2f}",
                "체절 양정 (≤{1.4*target_h:.2f})": f"{h_churn:.2f}",
                "최대운전 양정 (≥{0.65*target_h:.2f})": f"{interp_h_overload:.2f}",
                "예상 동력(kW)": f"{interp_kw:.2f}", "선정 가능": "✅"
            })
    return pd.DataFrame(results)

# --- UI 및 시각화 함수들 ---
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
    # 데이터 로딩
    with st.spinner('데이터를 로드하고 처리하는 중입니다...'):
        m_r, q_r, h_r, k_r, df_r = load_sheet("reference data")
        m_c, q_c, h_c, k_c, df_c = load_sheet("catalog data")
        m_d, q_d, h_d, k_d, df_d = load_sheet("deviation data")

    # 탭 생성
    tab_list = ["Total", "Reference", "Catalog", "Deviation"]
    tabs = st.tabs(tab_list)

    # Total 탭
    with tabs[0]:
        st.subheader("📊 Total - 통합 곡선 및 운전점 분석")

        if df_r.empty:
            st.error("'reference data' 시트를 처리할 수 없습니다. 파일 내용을 확인해주세요.")
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
                    st.info("소방 펌프 성능 기준 3점을 자동으로 분석합니다: 정격, 체절(140% 이하), 최대(150% 유량, 65% 이상)")

                if st.button("운전점 분석 실행"):
                    if not models:
                        st.warning("먼저 분석할 시리즈나 모델을 선택해주세요.")
                    else:
                        with st.spinner("선택된 모델들을 분석 중입니다..."):
                            if analysis_mode == "소방":
                                op_results_df = analyze_fire_pump_point(df_r, models, target_q, target_h, m_r, q_r, h_r, k_r)
                            else: # 기계 모드
                                op_results_df = analyze_operating_point(df_r, models, target_q, target_h, m_r, q_r, h_r, k_r)

                            if not op_results_df.empty:
                                st.success(f"총 {len(op_results_df)}개의 모델이 요구 성능을 만족합니다.")
                                st.dataframe(op_results_df, use_container_width=True)
                            else:
                                st.info("요구 성능을 만족하는 모델을 찾지 못했습니다.")

            st.markdown("---")
            ref_show = st.checkbox("Reference 표시", value=True)
            cat_show = st.checkbox("Catalog 표시")
            dev_show = st.checkbox("Deviation 표시")

            # Q-H 곡선
            st.markdown("#### Q-H (토출량-토출양정)")
            fig_h = go.Figure()
            if ref_show and not df_f.empty: add_traces(fig_h, df_f, m_r, q_r, h_r, models, 'lines+markers'); add_bep_markers(fig_h, df_f, m_r, q_r, h_r, models)
            if cat_show and not df_c.empty: add_traces(fig_h, df_c, m_c, q_c, h_c, models, 'lines+markers', line_style=dict(dash='dot'))
            if dev_show and not df_d.empty: add_traces(fig_h, df_d, m_d, q_d, h_d, models, 'markers')
            
            # ★★★ 수정된 부분: 운전점 마커 추가 로직 ★★★
            if target_q > 0 and target_h > 0:
                # 1. 주 운전점 (Rated Point)
                fig_h.add_trace(go.Scatter(
                    x=[target_q], y=[target_h], mode='markers',
                    marker=dict(symbol='cross', size=15, color='magenta'), name='정격 운전점'
                ))

                # 2. 소방 모드일 경우 추가 지점 표시
                if analysis_mode == "소방":
                    # 체절 운전점 상한선 (Churn Point Limit)
                    churn_h_limit = 1.4 * target_h
                    fig_h.add_trace(go.Scatter(
                        x=[0], y=[churn_h_limit], mode='markers',
                        marker=dict(symbol='x', size=12, color='red'), name=f'체절점 상한 (H≤{churn_h_limit:.2f})'
                    ))

                    # 최대 운전점 하한선 (Overload Point Limit)
                    overload_q = 1.5 * target_q
                    overload_h_limit = 0.65 * target_h
                    fig_h.add_trace(go.Scatter(
                        x=[overload_q], y=[overload_h_limit], mode='markers',
                        marker=dict(symbol='diamond-open', size=12, color='blue'), name=f'최대점 하한 (H≥{overload_h_limit:.2f})'
                    ))

            render_chart(fig_h, key="total_qh")

            # Q-kW 곡선
            st.markdown("#### Q-kW (토출량-축동력)")
            fig_k = go.Figure()
            if ref_show and not df_f.empty: add_traces(fig_k, df_f, m_r, q_r, k_r, models, 'lines+markers')
            if cat_show and not df_c.empty: add_traces(fig_k, df_c, m_c, q_c, k_c, models, 'lines+markers', line_style=dict(dash='dot'))
            if dev_show and not df_d.empty: add_traces(fig_k, df_d, m_d, q_d, k_d, models, 'markers')
            render_chart(fig_k, key="total_qk")

    # 나머지 개별 탭들 (이하 로직은 기능 확인을 위해 단순화)
    for idx, sheet in enumerate(["reference data", "catalog data", "deviation data"]):
        with tabs[idx+1]:
            st.subheader(sheet.title())
            # 필요한 경우 각 탭의 내용 구현

else:
    st.info("시작하려면 Excel 파일을 업로드하세요.")
