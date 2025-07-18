import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np

st.set_page_config(page_title="Dooch XRL(F) 성능 곡선 뷰어", layout="wide")
st.title("📊 Dooch XRL(F) 성능 곡선 뷰어 v3.0 (분석 기능 강화)")

# --- 신규 기능: 운전점 분석 함수 ---
def analyze_operating_point(df, models, target_q, target_h, m_col, q_col, h_col, k_col):
    if target_q <= 0 or target_h <= 0:
        return pd.DataFrame()

    results = []
    for model in models:
        model_df = df[df[m_col] == model].sort_values(q_col)
        # 데이터가 2개 이상 있어야 보간 가능
        if len(model_df) < 2:
            continue

        # Target Q가 해당 모델의 유량 범위를 벗어나는지 확인
        if not (model_df[q_col].min() <= target_q <= model_df[q_col].max()):
            continue
            
        # np.interp를 사용해 target_q 지점의 성능 보간
        interp_h = np.interp(target_q, model_df[q_col], model_df[h_col])

        # 보간된 양정이 목표 양정 이상인 모델만 선정
        if interp_h >= target_h:
            interp_kw = np.interp(target_q, model_df[q_col], model_df[k_col]) if k_col and k_col in model_df else np.nan
            interp_eff = np.interp(target_q, model_df[q_col], model_df['Efficiency']) if 'Efficiency' in model_df else np.nan
            
            results.append({
                "모델명": model,
                "요구 유량 (Q)": target_q,
                "요구 양정 (H)": target_h,
                "예상 양정 (H)": f"{interp_h:.2f}",
                "예상 동력 (kW)": f"{interp_kw:.2f}" if not np.isnan(interp_kw) else "N/A",
                "예상 효율 (%)": f"{interp_eff:.2f}" if not np.isnan(interp_eff) else "N/A",
                "선정 가능 여부": "✅"
            })
            
    return pd.DataFrame(results)

# --- 신규 기능: 편차 정량화 함수 ---
def quantify_deviation(df_ref, df_dev, m_col_r, q_col_r, h_col_r, k_col_r, m_col_d, q_col_d, h_col_d, k_col_d):
    # 두 데이터프레임에 모두 존재하는 모델 찾기
    common_models = set(df_ref[m_col_r].unique()) & set(df_dev[m_col_d].unique())
    
    if not common_models:
        return pd.DataFrame()

    deviation_results = []
    for model in common_models:
        ref_model_df = df_ref[df_ref[m_col_r] == model].sort_values(q_col_r)
        dev_model_df = df_dev[df_dev[m_col_d] == model].sort_values(q_col_d)

        if len(ref_model_df) < 2: continue # 기준 데이터가 보간에 부족하면 건너뛰기

        for _, dev_row in dev_model_df.iterrows():
            q_val = dev_row[q_col_d]
            
            # 기준 데이터에서 현재 유량(q_val)에 해당하는 성능 보간
            ref_h = np.interp(q_val, ref_model_df[q_col_r], ref_model_df[h_col_r])
            ref_k = np.interp(q_val, ref_model_df[q_col_r], ref_model_df[k_col_r]) if k_col_r and k_col_r in ref_model_df else np.nan
            
            dev_h = dev_row[h_col_d]
            dev_k = dev_row[k_col_d] if k_col_d and k_col_d in dev_row else np.nan
            
            # 편차 계산 (0으로 나누기 방지)
            h_dev_pct = ((dev_h - ref_h) / ref_h) * 100 if ref_h != 0 else 0
            k_dev_pct = ((dev_k - ref_k) / ref_k) * 100 if not np.isnan(ref_k) and ref_k != 0 else 0
            
            deviation_results.append({
                "모델명": model,
                "측정 유량 (Q)": q_val,
                "기준 양정 (H)": f"{ref_h:.2f}",
                "측정 양정 (H)": f"{dev_h:.2f}",
                "양정 편차 (%)": f"{h_dev_pct:.2f}%",
                "기준 동력 (kW)": f"{ref_k:.2f}" if not np.isnan(ref_k) else "N/A",
                "측정 동력 (kW)": f"{dev_k:.2f}" if not np.isnan(dev_k) else "N/A",
                "동력 편차 (%)": f"{k_dev_pct:.2f}%" if not np.isnan(ref_k) and ref_k != 0 else "N/A"
            })
            
    return pd.DataFrame(deviation_results)


# --- (이하 코드는 이전과 거의 동일하며, 신규 기능 호출 부분만 추가됨) ---

uploaded_file = st.file_uploader("Excel 파일 업로드 (.xlsx 또는 .xlsm)", type=["xlsx", "xlsm"])

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

def calculate_efficiency(df, q_col, h_col, k_col, q_unit='L/min'):
    if not all([q_col, h_col, k_col]):
        return df
    df_copy = df.copy()
    if q_unit == 'L/min': q_m3_s = df_copy[q_col] / 60000
    elif q_unit == 'm3/h': q_m3_s = df_copy[q_col] / 3600
    else: q_m3_s = df_copy[q_col]
    rho, g = 1000, 9.81
    power_kw = df_copy[k_col]
    efficiency = np.where(power_kw > 0, (rho * g * q_m3_s * df_copy[h_col]) / (power_kw * 1000) * 100, 0)
    df_copy['Efficiency'] = efficiency
    return df_copy

def load_sheet(name):
    try: df = pd.read_excel(uploaded_file, sheet_name=name)
    except Exception: return None, None, None, None, pd.DataFrame()
    mcol, qcol, hcol, kcol = get_best_match_column(df, ["모델명"]), get_best_match_column(df, ["유량"]), get_best_match_column(df, ["양정"]), get_best_match_column(df, ["동력"])
    if not mcol or not qcol or not hcol: return None, None, None, None, pd.DataFrame()
    cols_to_check = [qcol, hcol]; 
    if kcol: cols_to_check.append(kcol)
    for col in cols_to_check:
        df = df.dropna(subset=[col])
        df = df[pd.to_numeric(df[col], errors='coerce').notna()]
        df[col] = pd.to_numeric(df[col])
    df['Series'] = df[mcol].astype(str).str.extract(r"(XRF\d+)"); df['Series'] = pd.Categorical(df['Series'], categories=SERIES_ORDER, ordered=True); df = df.sort_values('Series')
    df = calculate_efficiency(df, qcol, hcol, kcol, q_unit='L/min')
    return mcol, qcol, hcol, kcol, df

def render_filters(df, mcol, prefix):
    series_opts = df['Series'].dropna().unique().tolist()
    default_series = [series_opts[0]] if series_opts else []
    mode = st.radio("분류 기준", ["시리즈별","모델별"], key=prefix+"_mode", horizontal=True)
    if mode == "시리즈별":
        sel = st.multiselect("시리즈 선택", series_opts, default=default_series, key=prefix+"_series")
        df_f = df[df['Series'].isin(sel)] if sel else pd.DataFrame()
    else:
        model_opts = df[mcol].dropna().unique().tolist()
        default_model = [model_opts[0]] if model_opts else []
        sel = st.multiselect("모델 선택", model_opts, default=default_model, key=prefix+"_models")
        df_f = df[df[mcol].isin(sel)] if sel else pd.DataFrame()
    return df_f

def add_traces(fig, df, mcol, xcol, ycol, models, mode, line_style=None, marker_style=None, name_suffix=""):
    for m in models:
        sub = df[df[mcol]==m].sort_values(xcol)
        if sub.empty: continue
        fig.add_trace(go.Scatter(x=sub[xcol], y=sub[ycol], mode=mode, name=m + name_suffix, line=line_style or {}, marker=marker_style or {}))

def add_bep_markers(fig, df, mcol, qcol, ycol, models, bep_y_col='Efficiency'):
    for m in models:
        model_df = df[df[mcol] == m]
        if not model_df.empty and bep_y_col in model_df.columns and not model_df[bep_y_col].isnull().all():
            bep_row = model_df.loc[model_df[bep_y_col].idxmax()]
            fig.add_trace(go.Scatter(x=[bep_row[qcol]], y=[bep_row[ycol]], mode='markers', marker=dict(symbol='star', size=15, color='gold'), name=f'{m} BEP'))

def add_guides(fig, hline, vline):
    if hline is not None and hline > 0: fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y", y0=hline, y1=hline, line=dict(color="red", dash="dash"))
    if vline is not None and vline > 0: fig.add_shape(type="line", xref="x", x0=vline, x1=vline, yref="paper", y0=0, y1=1, line=dict(color="blue", dash="dash"))

def render_chart(fig, key):
    fig.update_layout(dragmode='pan', xaxis=dict(fixedrange=False), yaxis=dict(fixedrange=False))
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displaylogo': False}, key=key)

if uploaded_file:
    with st.spinner('데이터를 로드하고 처리하는 중입니다...'):
        m_r, q_r, h_r, k_r, df_r = load_sheet("reference data")
        m_c, q_c, h_c, k_c, df_c = load_sheet("catalog data")
        m_d, q_d, h_d, k_d, df_d = load_sheet("deviation data")

    # --- 신규 기능: 편차 분석 탭 추가 ---
    tab_list = ["Total", "성능 편차 분석", "Reference", "Catalog", "Deviation"]
    tabs = st.tabs(tab_list)

    with tabs[0]: # Total 탭
        st.subheader("📊 Total - 통합 곡선 및 운전점 분석")
        df_f = render_filters(df_r, m_r, "total")
        models = df_f[m_r].unique().tolist() if not df_f.empty else []
        
        # --- 신규 기능: 운전점 분석 UI ---
        with st.expander("운전점 분석 (Operating Point Analysis)"):
            op_col1, op_col2 = st.columns(2)
            with op_col1:
                target_q = st.number_input("목표 유량 (Q)", value=0.0, format="%.2f", help="선택한 펌프들의 성능을 확인할 목표 유량을 입력하세요.")
            with op_col2:
                target_h = st.number_input("목표 양정 (H)", value=0.0, format="%.2f", help="이 양정 이상을 만족하는 펌프를 찾습니다.")
            
            if st.button("운전점 분석 실행"):
                if not models:
                    st.warning("먼저 분석할 시리즈나 모델을 선택해주세요.")
                else:
                    with st.spinner("선택된 모델들을 분석 중입니다..."):
                        op_results_df = analyze_operating_point(df_r, models, target_q, target_h, m_r, q_r, h_r, k_r)
                        if not op_results_df.empty:
                            st.success(f"총 {len(op_results_df)}개의 모델이 요구 성능을 만족합니다.")
                            st.dataframe(op_results_df, use_container_width=True)
                        else:
                            st.info("요구 성능을 만족하는 모델을 찾지 못했습니다. 목표값을 조정하거나 다른 모델을 선택해보세요.")
        
        st.markdown("---")
        # (이하 그래프 표시는 이전과 동일)
        ref_show = st.checkbox("Reference 표시", key="total_ref", value=True)
        cat_show = st.checkbox("Catalog 표시", key="total_cat")
        dev_show = st.checkbox("Deviation 표시", key="total_dev")
        
        st.markdown("#### Q-H (토출량-토출양정)")
        fig_h = go.Figure();
        if ref_show: add_traces(fig_h, df_r, m_r, q_r, h_r, models, 'lines+markers'); add_bep_markers(fig_h, df_r, m_r, q_r, h_r, models)
        if cat_show: add_traces(fig_h, df_c, m_c, q_c, h_c, models, 'lines+markers', line_style=dict(dash='dot'))
        if dev_show: add_traces(fig_h, df_d, m_d, q_d, h_d, models, 'markers')
        if 'target_q' in locals() and target_q > 0: fig_h.add_trace(go.Scatter(x=[target_q], y=[target_h], mode='markers', marker=dict(symbol='cross', size=15, color='magenta'), name='운전점'))
        render_chart(fig_h, key="total_qh")
        
        # Q-kW, Q-Eff 그래프 생략 (코드는 이전과 동일)

    with tabs[1]: # 성능 편차 분석 탭
        st.subheader("🔬 성능 편차 정량 분석")
        st.info("Reference 데이터와 Deviation 데이터를 비교하여 성능 편차를 계산합니다.")
        
        if df_r.empty or df_d.empty:
            st.warning("편차 분석을 위해서는 'reference data'와 'deviation data' 시트가 모두 필요합니다.")
        else:
            with st.spinner("편차를 계산하는 중입니다..."):
                deviation_df = quantify_deviation(df_r, df_d, m_r, q_r, h_r, k_r, m_d, q_d, h_d, k_d)
                st.dataframe(deviation_df, use_container_width=True)

    # 나머지 개별 탭들 (2, 3, 4)
    for idx, sheet in enumerate(["reference data", "catalog data", "deviation data"]):
        with tabs[idx+2]:
            st.subheader(sheet.title())
            if sheet == "reference data": mcol,qcol,hcol,kcol,df = m_r,q_r,h_r,k_r,df_r
            elif sheet == "catalog data": mcol,qcol,hcol,kcol,df = m_c,q_c,h_c,k_c,df_c
            else: mcol,qcol,hcol,kcol,df = m_d,q_d,h_d,k_d,df_d
            if df.empty: st.warning(f"'{sheet}' 시트 데이터를 찾을 수 없거나 비어있습니다."); continue
            
            df_f = render_filters(df, mcol, sheet)
            models = df_f[mcol].unique().tolist() if not df_f.empty else []
            if not models: st.info("모델을 선택해주세요."); continue

            st.markdown("#### Q-H (토출량-토출양정)")
            fig1 = go.Figure()
            mode1, style1 = ('markers', None) if sheet=='deviation data' else ('lines+markers', dict(dash='dot') if sheet=='catalog data' else None)
            add_traces(fig1, df_f, mcol, qcol, hcol, models, mode1, line_style=style1)
            if 'Efficiency' in df_f.columns: add_bep_markers(fig1, df_f, mcol, qcol, hcol, models)
            render_chart(fig1, key=f"{sheet}_qh")
            
            # Q-kW, Q-Eff, 데이터 테이블 생략 (코드는 이전과 동일)
else:
    st.info("시작하려면 Excel 파일을 업로드하세요.")
