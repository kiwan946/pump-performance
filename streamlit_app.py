import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np

# 페이지 기본 설정
st.set_page_config(page_title="Dooch XRL(F) 성능 곡선 뷰어 v15.0", layout="wide")
st.title("📊 Dooch XRL(F) 성능 곡선 뷰어 v15.0")

# --- 유틸리티 함수들 ---
SERIES_ORDER = ["XRF3", "XRF5", "XRF10", "XRF15", "XRF20", "XRF32", "XRF45", "XRF64", "XRF95", "XRF125", "XRF155", "XRF185", "XRF215", "XRF255"]

def get_best_match_column(df, names):
    if df is None or df.empty: return None
    for n in names:
        for col in df.columns:
            if n in col.strip():
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
        if col in temp_df.columns:
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

        q_auto = get_best_match_column(df_r_orig, ["토출량", "유량"])
        h_auto = get_best_match_column(df_r_orig, ["토출양정", "전양정"])
        k_auto = get_best_match_column(df_r_orig, ["축동력"])
        
        q_col = st.sidebar.selectbox("유량 (Flow) 컬럼", all_columns, index=safe_get_index(all_columns, q_auto))
        h_col = st.sidebar.selectbox("양정 (Head) 컬럼", all_columns, index=safe_get_index(all_columns, h_auto))
        k_col = st.sidebar.selectbox("축동력 (Power) 컬럼", all_columns, index=safe_get_index(all_columns, k_auto))
        
        # 3. 선택된 컬럼으로 데이터 정제 및 효율 계산
        df_r = process_data(df_r_orig, q_col, h_col, k_col)
        df_c = process_data(df_c_orig, q_col, h_col, k_col)
        df_d = process_data(df_d_orig, q_col, h_col, k_col)
        
        # 4. 탭 생성 및 화면 표시
        tab_list = ["Total", "Reference", "Catalog", "Deviation"]
        tabs = st.tabs(tab_list)

        with tabs[0]:
            st.subheader("📊 Total - 통합 곡선 및 운전점 분석")
            df_f = render_filters(df_r, m_r, "total")
            models = df_f[m_r].unique().tolist() if m_r and not df_f.empty else []

            with st.expander("운전점 분석 (Operating Point Analysis)", expanded=True):
                # ... (운전점 분석 UI 로직은 이전과 동일)
                pass
            
            st.markdown("---")
            # ... (그래프 표시 로직은 이전과 동일)
            pass

        # ★★★ 수정된 부분: 개별 탭 로직 ★★★
        for idx, sheet_name in enumerate(["Reference", "Catalog", "Deviation"]):
            with tabs[idx+1]:
                st.subheader(f"📊 {sheet_name} Data")
                
                df_orig, mcol = (df_r_orig, m_r) if sheet_name == "Reference" else \
                                  (df_c_orig, m_c) if sheet_name == "Catalog" else \
                                  (df_d_orig, m_d)
                
                if df_orig.empty:
                    st.info(f"'{sheet_name.lower()}' 시트의 데이터가 없습니다.")
                    continue

                # 각 탭에 맞는 컬럼을 독립적으로 자동 감지
                q_col_tab = get_best_match_column(df_orig, ["토출량", "유량"])
                h_col_tab = get_best_match_column(df_orig, ["토출양정", "전양정"])
                k_col_tab = get_best_match_column(df_orig, ["축동력"])
                
                # 독립적으로 감지된 컬럼으로 데이터 처리
                df_tab = process_data(df_orig, q_col_tab, h_col_tab, k_col_tab)
                
                if df_tab.empty:
                    st.warning("데이터 처리 후 표시할 내용이 없습니다.")
                    continue

                df_f_tab = render_filters(df_tab, mcol, sheet_name)
                models_tab = df_f_tab[mcol].unique().tolist() if not df_f_tab.empty else []

                if not models_tab:
                    st.info("차트를 보려면 모델을 선택해주세요.")
                    continue
                
                mode, style = ('markers', None) if sheet_name == "Deviation" else ('lines+markers', dict(dash='dot') if sheet_name == "Catalog" else None)
                
                st.markdown(f"#### Q-H ({h_col_tab})"); fig1 = go.Figure(); add_traces(fig1, df_f_tab, mcol, q_col_tab, h_col_tab, models_tab, mode, line_style=style); render_chart(fig1, key=f"{sheet_name}_qh")
                if k_col_tab in df_f_tab.columns: st.markdown("#### Q-kW (축동력)"); fig2 = go.Figure(); add_traces(fig2, df_f_tab, mcol, q_col_tab, k_col_tab, models_tab, mode, line_style=style); render_chart(fig2, key=f"{sheet_name}_qk")
                if 'Efficiency' in df_f_tab.columns: st.markdown("#### Q-Efficiency (효율)"); fig3 = go.Figure(); add_traces(fig3, df_f_tab, mcol, q_col_tab, 'Efficiency', models_tab, mode, line_style=style); fig3.update_layout(yaxis_title="효율 (%)", yaxis=dict(range=[0, 100])); render_chart(fig3, key=f"{sheet_name}_qe")
                st.markdown("#### 데이터 확인"); st.dataframe(df_f_tab, use_container_width=True)
else:
    st.info("시작하려면 Excel 파일을 업로드하세요.")
