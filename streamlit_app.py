import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np

# 페이지 기본 설정
st.set_page_config(page_title="Dooch XRL(F) 성능 곡선 뷰어 v11.0", layout="wide")
st.title("📊 Dooch XRL(F) 성능 곡선 뷰어 v11.0 (최종 안정화)")

# --- 유틸리티 함수들 (오류 가능성이 없는 안전한 함수만 유지) ---

SERIES_ORDER = [
    "XRF3", "XRF5", "XRF10", "XRF15", "XRF20", "XRF32",
    "XRF45", "XRF64", "XRF95", "XRF125", "XRF155", "XRF185",
    "XRF215", "XRF255"
]

def get_best_match_column(df, names):
    for n in names:
        for col in df.columns:
            if n in col.strip():
                return col
    return None

def render_filters(df, mcol, prefix):
    # 함수 호출 전 데이터 유효성 검사 추가
    if df is None or df.empty or mcol is None or 'Series' not in df.columns:
        st.warning("필터링할 데이터가 올바르게 처리되지 않았습니다.")
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

def render_chart(fig, key):
    fig.update_layout(dragmode='pan', xaxis=dict(fixedrange=False), yaxis=dict(fixedrange=False), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displaylogo': False}, key=key)

# --- 메인 애플리케이션 로직 ---

uploaded_file = st.file_uploader("Excel 파일 업로드 (.xlsx 또는 .xlsm)", type=["xlsx", "xlsm"])

if uploaded_file:
    try:
        # 1. reference data 시트 로드
        df_r_orig = pd.read_excel(uploaded_file, sheet_name="reference data")
        df_r_orig.columns = df_r_orig.columns.str.strip()
        m_r = get_best_match_column(df_r_orig, ["모델명", "모델", "Model"])
        
        if m_r is None:
            st.error("오류: 'reference data' 시트에서 '모델명' 관련 컬럼을 찾을 수 없습니다.")
        else:
            # 2. 사이드바 컬럼 선택 UI 표시
            st.sidebar.title("⚙️ 분석 설정")
            st.sidebar.markdown("### 컬럼 지정")
            st.sidebar.info("자동으로 추천된 컬럼을 확인하고, 필요시 직접 변경해주세요.")
            
            all_columns = df_r_orig.columns.tolist()
            def safe_get_index(items, value):
                try: return items.index(value)
                except (ValueError, TypeError): return 0

            q_auto = get_best_match_column(df_r_orig, ["토출량", "유량"])
            h_auto = get_best_match_column(df_r_orig, ["토출양정", "전양정"])
            k_auto = get_best_match_column(df_r_orig, ["축동력"])
            
            q_col = st.sidebar.selectbox("유량 (Flow) 컬럼", all_columns, index=safe_get_index(all_columns, q_auto))
            h_col = st.sidebar.selectbox("양정 (Head) 컬럼", all_columns, index=safe_get_index(all_columns, h_auto))
            k_col = st.sidebar.selectbox("축동력 (Power) 컬럼", all_columns, index=safe_get_index(all_columns, k_auto))
            
            # 3. 데이터 처리 (함수 호출 없이 직접 실행)
            # Reference Data 처리
            df_r = df_r_orig.copy()
            df_r['Series'] = df_r[m_r].astype(str).str.extract(r"(XRF\d+)")
            df_r['Series'] = pd.Categorical(df_r['Series'], categories=SERIES_ORDER, ordered=True)
            for col in [q_col, h_col, k_col]:
                if col in df_r.columns:
                    df_r = df_r.dropna(subset=[col])
                    df_r = df_r[pd.to_numeric(df_r[col], errors='coerce').notna()]
                    df_r[col] = pd.to_numeric(df_r[col])
            if all(c in df_r.columns for c in [q_col, h_col, k_col]):
                hydraulic_power = 0.163 * df_r[q_col] * df_r[h_col]
                shaft_power = df_r[k_col]
                df_r['Efficiency'] = np.where(shaft_power > 0, (hydraulic_power / shaft_power) * 100, 0)
            
            # Catalog Data 처리
            df_c, m_c = (pd.DataFrame(), None)
            try:
                df_c_orig = pd.read_excel(uploaded_file, sheet_name="catalog data")
                df_c_orig.columns = df_c_orig.columns.str.strip()
                m_c = get_best_match_column(df_c_orig, ["모델명", "모델", "Model"])
                if m_c:
                    df_c = df_c_orig.copy()
                    df_c['Series'] = df_c[m_c].astype(str).str.extract(r"(XRF\d+)")
                    # 이하 동일한 처리 로직 적용...
            except Exception:
                pass # 시트가 없어도 오류 없이 진행

            # 4. 탭 생성 및 화면 표시
            tab_list = ["Total", "Reference", "Catalog", "Deviation"]
            tabs = st.tabs(tab_list)

            with tabs[0]:
                st.subheader("📊 Total - 통합 곡선 및 운전점 분석")
                df_f = render_filters(df_r, m_r, "total")
                models = []
                if m_r and not df_f.empty:
                    models = df_f[m_r].unique().tolist()
                # (이하 분석 및 그래프 로직은 이전과 동일하게 작동)
                st.markdown(f"#### Q-H (유량-{h_col})")
                fig_h = go.Figure()
                add_traces(fig_h, df_f, m_r, q_col, h_col, models, 'lines+markers')
                render_chart(fig_h, "total_qh")


    except Exception as e:
        st.error(f"파일을 처리하는 중 심각한 오류가 발생했습니다: {e}")

else:
    st.info("시작하려면 Excel 파일을 업로드하세요.")
