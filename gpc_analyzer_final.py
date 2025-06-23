import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- 함수 정의 (변경 없음) ---
def calculate_gpc_data(df, A, B, C, D, start_time, end_time, mode):
    if mode == "chromatogram":
        df.columns = ['Retention Time', 'RI Signal']; df = df.apply(pd.to_numeric, errors='coerce').dropna()
        df_filtered = df[(df['Retention Time'] >= start_time) & (df['Retention Time'] <= end_time)].copy()
        if df_filtered.empty: return 0, 0, 0, 0, pd.DataFrame()
        min_signal = df_filtered['RI Signal'].min(); df_filtered['RI Signal Corrected'] = df_filtered['RI Signal'] - min_signal
        t = df_filtered['Retention Time']; log_M = A * (t**3) + B * (t**2) + C * t + D
        df_filtered['Molecular Weight'] = 10**log_M; df_filtered['log(M)'] = log_M
    else: # mode == "differential"
        df.columns = ['log(M)', 'RI Signal']; df = df.apply(pd.to_numeric, errors='coerce').dropna()
        df_filtered = df[(df['log(M)'] >= start_time) & (df['log(M)'] <= end_time)].copy()
        if df_filtered.empty: return 0, 0, 0, 0, pd.DataFrame()
        df_filtered['RI Signal Corrected'] = df_filtered['RI Signal']
        df_filtered['Molecular Weight'] = 10**df_filtered['log(M)']

    h = df_filtered['RI Signal Corrected'].values; M = df_filtered['Molecular Weight'].values; epsilon = 1e-9
    sum_h, sum_h_M = np.sum(h), np.sum(h * M)
    Mn = sum_h / (np.sum(h / (M + epsilon)) + epsilon) if sum_h > 0 else 0
    Mw = sum_h_M / (sum_h + epsilon) if sum_h > 0 else 0
    Mz = np.sum(h * M**2) / (sum_h_M + epsilon) if sum_h_M > 0 else 0
    PDI = Mw / Mn if Mn > 0 else 0
    max_signal = df_filtered['RI Signal Corrected'].max()
    df_filtered['RI Signal Normalized'] = df_filtered['RI Signal Corrected'] / max_signal if max_signal > 0 else 0.0
    return Mn, Mw, Mz, PDI, df_filtered

# --- UI 구성 ---
st.set_page_config(layout="wide"); st.title("🔬 GPC 데이터 분석기")

# --- 사이드바 ---
with st.sidebar:
    st.header("⚙️ 분석 설정")
    analysis_mode = st.radio("분석 모드를 선택하세요.", ("From Chromatogram (Raw Data)", "From Differential Curve (Processed)"))
    uploaded_file = st.file_uploader(f"'{analysis_mode.split('(')[0].strip()}' 데이터 업로드", type=['xlsx', 'xls'])
    st.markdown("---")
    if analysis_mode == "From Chromatogram (Raw Data)":
        st.subheader("Calibration Curve 계수")
        coeff_A = st.number_input("A", value=-0.00741883, format="%.8f"); coeff_B = st.number_input("B", value=0.35334440, format="%.8f")
        coeff_C = st.number_input("C", value=-5.92485400, format="%.8f"); coeff_D = st.number_input("D", value=37.64635000, format="%.8f")
    st.markdown("---")
    st.subheader("계산용 범위 설정 (샘플별)")
    st.info("여기서 설정한 범위는 위 '종합 분석 결과' 테이블 계산에만 사용됩니다.")
    if uploaded_file:
        try:
            xls = pd.ExcelFile(uploaded_file); sheet_names = xls.sheet_names
            selected_sheet_for_calc = st.selectbox("범위를 설정할 샘플 선택:", sheet_names)
            if selected_sheet_for_calc:
                df_temp = pd.read_excel(xls, sheet_name=selected_sheet_for_calc, header=1)
                mode_key = "time" if analysis_mode == "From Chromatogram (Raw Data)" else "logm"
                df_temp.columns = ['col1', 'col2']; label = "시간 (min)" if mode_key == "time" else "log(M)"
                min_val, max_val = float(df_temp.iloc[:, 0].min()), float(df_temp.iloc[:, 0].max())
                
                for sheet in sheet_names:
                    range_key = f"range_{mode_key}_{sheet}"
                    if range_key not in st.session_state:
                        temp_df_init = pd.read_excel(xls, sheet_name=sheet, header=1)
                        min_init, max_init = float(temp_df_init.iloc[:, 0].min()), float(temp_df_init.iloc[:, 0].max())
                        st.session_state[range_key] = {'start': min_init, 'end': max_init}
                
                current_range_key = f"range_{mode_key}_{selected_sheet_for_calc}"
                current_range = st.session_state[current_range_key]
                new_start = st.number_input(f"시작 {label}", value=current_range['start'], key=f"start_{current_range_key}", format="%.4f")
                new_end = st.number_input(f"종료 {label}", value=current_range['end'], key=f"end_{current_range_key}", format="%.4f")
                if new_start != current_range['start'] or new_end != current_range['end']:
                    st.session_state[current_range_key] = {'start': new_start, 'end': new_end}; st.rerun()
        except Exception as e: st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

# --- 메인 패널 ---
if uploaded_file:
    results, all_raw_dfs = [], {}
    try:
        xls = pd.ExcelFile(uploaded_file)
        for sheet in xls.sheet_names:
            df_raw = pd.read_excel(xls, sheet_name=sheet, header=1); all_raw_dfs[sheet] = df_raw
            mode_key = "time" if analysis_mode == "From Chromatogram (Raw Data)" else "logm"
            range_key = f"range_{mode_key}_{sheet}"
            start_val, end_val = st.session_state[range_key]['start'], st.session_state[range_key]['end']
            
            Mn, Mw, Mz, PDI, _ = calculate_gpc_data(df_raw, coeff_A if mode_key=="time" else 0, coeff_B if mode_key=="time" else 0, coeff_C if mode_key=="time" else 0, coeff_D if mode_key=="time" else 0, start_val, end_val, "chromatogram" if mode_key=="time" else "differential")
            results.append({"Sample": sheet, "Mn": Mn, "Mw": Mw, "Mz": Mz, "PDI": PDI})
    except Exception as e: st.error(f"데이터를 계산하는 중 오류 발생: {e}"); results = []
    
    if results:
        st.header("📋 종합 분석 결과"); st.dataframe(pd.DataFrame(results).style.format({"Mn":"{:,.0f}","Mw":"{:,.0f}","Mz":"{:,.0f}","PDI":"{:.4f}"}), use_container_width=True)
        st.markdown("---")
        st.header("📊 샘플별 정규화 분자량 분포 곡선 오버레이")
        selected_samples = st.multiselect("그래프에 표시할 샘플을 선택하세요.", options=xls.sheet_names, default=xls.sheet_names)
        
        with st.expander("오버레이 그래프 공통 범위 설정"):
            mode_key = "time" if analysis_mode == "From Chromatogram (Raw Data)" else "logm"
            label = "시간 (min)" if mode_key == "time" else "log(M)"
            min_bound, max_bound = float('inf'), float('-inf')
            for sheet in xls.sheet_names:
                df_b = pd.read_excel(xls, sheet_name=sheet, header=1); min_bound = min(min_bound, df_b.iloc[:,0].min()); max_bound = max(max_bound, df_b.iloc[:,0].max())
            common_range_key = f"common_range_{mode_key}"
            if common_range_key not in st.session_state: st.session_state[common_range_key] = (min_bound, max_bound)
            common_start, common_end = st.slider(f"공통 {label} 범위", min_bound, max_bound, st.session_state[common_range_key])
            st.session_state[common_range_key] = (common_start, common_end)

        graph_dfs = []
        if selected_samples:
            for sheet in selected_samples:
                _, _, _, _, df_p_graph = calculate_gpc_data(all_raw_dfs[sheet], coeff_A if mode_key=="time" else 0, coeff_B if mode_key=="time" else 0, coeff_C if mode_key=="time" else 0, coeff_D if mode_key=="time" else 0, common_start, common_end, "chromatogram" if mode_key=="time" else "differential")
                if not df_p_graph.empty: df_p_graph['Sample'] = sheet; graph_dfs.append(df_p_graph)
        
        if graph_dfs:
            combined_graph_df = pd.concat(graph_dfs, ignore_index=True)
            fig_overlay = px.line(combined_graph_df, x='log(M)', y='RI Signal Normalized', color='Sample', title="정규화 분자량 분포 곡선",
                                  labels={'RI Signal Normalized': 'Normalized RI Signal'}, color_discrete_sequence=px.colors.qualitative.Plotly)
            fig_overlay.update_xaxes(autorange="reversed")
            # [핵심] 맞춤 다운로드 버튼을 제거하고 Plotly 기본 기능을 사용하도록 함
            st.plotly_chart(fig_overlay, use_container_width=True, config={'toImageButtonOptions': {'format': 'png', 'filename': 'gpc_overlay_graph', 'scale': 2}})
            st.info("💡 **이미지 다운로드 방법**: 그래프 오른쪽 상단에 마우스를 올리면 나타나는 **카메라 아이콘**을 클릭하세요.")

        else: st.warning("선택된 범위에 해당하는 데이터가 없습니다.")
    elif uploaded_file: st.warning("분석할 데이터가 없습니다. 파일 형식이나 분석 범위를 확인해주세요.")
else: st.info("👈 왼쪽 사이드바에서 분석 모드를 선택하고 파일을 업로드해주세요.")