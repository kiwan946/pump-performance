import streamlit as st
import pandas as pd

# 페이지 기본 설정
st.set_page_config(page_title="파일 구조 진단 도구", layout="wide")
st.title("🔬 Excel 파일 구조 진단 도구")
st.info("업로드된 파일의 시트와 컬럼 구조를 확인합니다.")

# --- 메인 로직 ---
uploaded_file = st.file_uploader("분석할 Excel 파일을 업로드하세요 (.xlsx 또는 .xlsm)", type=["xlsx", "xlsm"])

if uploaded_file:
    try:
        # 1. 파일 전체를 읽어 시트 이름 목록 가져오기
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        
        st.subheader("1. 발견된 시트 목록")
        st.info(f"파일에서 총 {len(sheet_names)}개의 시트를 찾았습니다.")
        st.write(sheet_names)
        
        st.markdown("---")

        # 2. 사용자가 시트를 선택할 수 있도록 드롭다운 메뉴 제공
        st.subheader("2. 시트 선택 및 컬럼 확인")
        selected_sheet = st.selectbox("내용을 확인할 시트를 선택하세요:", sheet_names)

        if selected_sheet:
            try:
                # 3. 선택된 시트를 DataFrame으로 읽기
                df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                
                # 4. 발견된 컬럼 목록 표시
                st.write(f"**'{selected_sheet}'** 시트에서 발견된 컬럼 목록:")
                st.write(df.columns.tolist())
                
                st.markdown("---")
                
                # 5. 데이터 샘플 표시
                st.subheader("3. 데이터 샘플 확인")
                st.write(f"**'{selected_sheet}'** 시트의 상위 5개 데이터 샘플입니다.")
                st.dataframe(df.head())

            except Exception as e:
                st.error(f"'{selected_sheet}' 시트를 읽는 중 오류가 발생했습니다: {e}")

    except Exception as e:
        st.error(f"파일을 처리하는 중 심각한 오류가 발생했습니다: {e}")

else:
    st.info("파일을 업로드하여 진단을 시작하세요.")
