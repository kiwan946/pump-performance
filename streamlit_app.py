import streamlit as st
import pandas as pd

# 페이지 기본 설정
st.set_page_config(page_title="디버깅 버전", layout="wide")
st.title("🐞 오류 해결을 위한 디버깅 버전")
st.info("이 버전은 파일 로드 및 컬럼 선택 기능의 안정성을 테스트하기 위한 것입니다.")

# --- 유틸리티 함수 ---
def get_best_match_column(df, names):
    """키워드를 기반으로 DataFrame에서 컬럼 이름을 찾아 제안합니다."""
    for n in names:
        for col in df.columns:
            if n in col.strip(): # 공백 제거 후 비교
                return col
    return None

def load_sheet_for_debug(uploaded_file, name):
    """지정된 시트 이름으로 Excel 파일을 로드합니다."""
    try:
        df = pd.read_excel(uploaded_file, sheet_name=name)
        # 컬럼 이름의 양쪽 공백을 모두 제거하여 잠재적인 오류 방지
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"'{name}' 시트를 읽는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

# --- 메인 애플리케이션 로직 ---

uploaded_file = st.file_uploader("Excel 파일 업로드 (.xlsx 또는 .xlsm)", type=["xlsx", "xlsm"])

if uploaded_file:
    # 1. 'reference data' 시트를 로드합니다.
    df_r_orig = load_sheet_for_debug(uploaded_file, "reference data")

    # 2. 시트 로드 성공 여부를 확인합니다.
    if df_r_orig.empty:
        st.error("'reference data' 시트를 찾을 수 없거나 비어 있습니다. 파일의 시트 이름을 확인해주세요.")
    else:
        # 3. 시트 로드 성공 시, 사이드바에 컬럼 선택 메뉴를 표시합니다.
        st.sidebar.title("⚙️ 컬럼 지정")
        st.success("파일이 로드되었습니다. 아래에서 분석에 사용할 컬럼을 선택하세요.")

        all_columns = df_r_orig.columns.tolist()

        # 자동으로 컬럼을 추천합니다.
        q_auto = get_best_match_column(df_r_orig, ["토출량", "유량"])
        h_auto = get_best_match_column(df_r_orig, ["토출양정", "전양정"])
        k_auto = get_best_match_column(df_r_orig, ["축동력"])

        # 추천된 컬럼을 기본값으로 설정합니다.
        q_index = all_columns.index(q_auto) if q_auto in all_columns else 0
        h_index = all_columns.index(h_auto) if h_auto in all_columns else 0
        k_index = all_columns.index(k_auto) if k_auto in all_columns else 0

        # 사용자가 직접 컬럼을 선택합니다.
        q_col = st.sidebar.selectbox("유량 (Flow) 컬럼", all_columns, index=q_index)
        h_col = st.sidebar.selectbox("양정 (Head) 컬럼", all_columns, index=h_index)
        k_col = st.sidebar.selectbox("축동력 (Power) 컬럼", all_columns, index=k_index)

        st.subheader("선택된 컬럼 및 데이터 확인")
        st.write(f"선택된 유량 컬럼: **{q_col}**")
        st.write(f"선택된 양정 컬럼: **{h_col}**")
        st.write(f"선택된 축동력 컬럼: **{k_col}**")

        st.markdown("---")
        
        # 선택된 컬럼의 데이터를 화면에 표시합니다.
        if all([q_col, h_col, k_col]):
            st.write("선택된 컬럼의 데이터 샘플 (상위 10개):")
            try:
                # 숫자 형식으로 변환 시도
                df_to_show = df_r_orig[[q_col, h_col, k_col]].copy()
                for col in [q_col, h_col, k_col]:
                    df_to_show[col] = pd.to_numeric(df_to_show[col], errors='coerce')
                
                st.dataframe(df_to_show.head(10))
                st.success("데이터 표시 성공! 이 버전에서는 오류가 발생하지 않습니다.")
            except Exception as e:
                st.error(f"선택된 컬럼으로 데이터를 표시하는 중 오류가 발생했습니다: {e}")
        else:
            st.warning("유량, 양정, 축동력 컬럼을 모두 선택해주세요.")

else:
    st.info("시작하려면 Excel 파일을 업로드하세요.")
