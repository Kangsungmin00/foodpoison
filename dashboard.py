import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import json
import folium
import altair as alt
import shap
import streamlit.components.v1 as components
import joblib
import requests
import io
import plotly.express as px
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
from PIL import Image
import base64

st.set_page_config(layout = 'wide'
                   , page_title = '전국 식중독 현황_A04'
                   , page_icon='🩺'
                   , initial_sidebar_state = 'auto')

@st.cache_data
def loading_data(path):
    response = requests.get(path)
    response.raise_for_status() # 요청에 실패하면 오류 발생
    data = pd.read_csv(io.StringIO(response.text))
    return data

@st.cache_data
def loading_json_file():
    state_geo = 'https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/TL_SCCO_CTPRVN_%EB%8F%85%EB%8F%84%ED%91%9C%EA%B8%B0.json'
    response = requests.get(state_geo)
    response.raise_for_status() # 요청에 실패하면 오류 발생
    jsonResult = response.json()
    return jsonResult

@st.cache_resource
def loading_model(model_path):
    response = requests.get(model_path)
    response.raise_for_status() # 요청에 실패하면 오류 발생
    file = io.BytesIO(response.content) # pkl파일 불러오기
    loaded_model = joblib.load(file)
    return loaded_model

def convert_dash_info(x):
    if x == '종합위험도':
        return 'risk'
    elif x == '발생건수':
        return 'OCCRNC_CNT'
    elif x == '발생환자수':
        return 'PATNT_CNT'
    
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def shap_summary_plot(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X, check_additivity=False) # ExplainerError가 나서 check_additivity 기능을 해제해주었음
    plot = shap.force_plot(explainer.expected_value, shap_values.values[-1, :], X.iloc[-1, :], link="logit")
    return st_shap(plot)

def shap_rf_summary_plot(model, X):
    explainer = shap.TreeExplainer(model, X)
    shap_values = explainer(X, check_additivity=False) # ExplainerError가 나서 check_additivity 기능을 해제해주었음
    plot = shap.force_plot(explainer.expected_value[1], shap_values.values[-1,:,1], X.iloc[-1, :], link="logit")
    return st_shap(plot)

def predict_reason(model, X):
    pred_y = model.predict_proba(X.loc[:, model.feature_names_in_].tail(1))[0][1]
    return pred_y

def main():

    # 지역별 데이터 로드 및 캐시 저장
    data = loading_data('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/Foodborne_Region_MasterTable.csv')
    data.index = pd.to_datetime(data['OCCRNC_YEAR'].astype(str) + '-' + data['OCCRNC_MM'].astype(str))
    data = data.sort_index()
    test_X_region = data.loc[data.index == data.index.max()]

    # 원인물질별 데이터 로드 및 캐시 저장
    data_cause = loading_data('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/Foodborne_Cause_MasterTable4.csv')
    data_cause.index = pd.to_datetime(data_cause['OCCRNC_YEAR'].astype(str) + '-' + data_cause['OCCRNC_MM'].astype(str))
    data_cause_2 = data_cause.drop(columns = ['OCCRNC_YEAR', 'OCCRNC_MM', 'PATNT_CNT', 'OCCRNC_IND']).sort_index().rename(columns = {'HOL_DUR':'황금연휴기간','HOL_IND':'황금연휴여부','CPI_VALUE':'소비자물가지수','WTHR_AVG_TEMP':'평균기온'
                          ,'WTHR_AVG_H_TEMP':'평균최고기온','WTHR_AVG_L_TEMP':'평균최저기온','WTHR_AVG_PRECIP':'평균강수량'
                          ,'WTHR_AVG_WNDSPD':'평균풍속','WTHR_MX_WNDSPD':'평균최고풍속','WTHR_AVG_PRESS':'평균기압','WTHR_MX_PRESS':'최고기압'
                          ,'WTHR_MN_PRESS':'최저기압','WTHR_AVG_SEA_PRESS':'평균해면기압','WTHR_MX_SEA_PRESS':'최고해면기압'
                          ,'WTHR_MN_SEA_PRESS':'최저해면기압','WTHR_AVG_RHUM':'평균상대습도','WTHR_MN_RHUM':'최소상대습도','WTHR_SUM_SUNHR':'총일조시간'
                          ,'FST_CNT':'축제횟수','FST_IND':'축제여부','POP_GEN_CNT':'총인구수','POP_ELM_CNT':'초등학생수','POP_MID_CNT':'중학생수'
                          ,'POP_HIGH_CNT':'고등학생수','POP_60P_CNT':'60세이상인구수','POP_ELM_PROB':'초등학생비율','POP_MID_PROB':'중학생비율'
                          ,'POP_HIGH_PROB':'고등학생비율','POP_60P_PROB':'60세이상비율','POP_DENS':'인구밀도','GMS_LIC_CNT':'집단급식소수'})
    test_X_cause = data_cause.loc[data_cause.index == data_cause.index.max()].rename(columns = {'HOL_DUR':'황금연휴기간','HOL_IND':'황금연휴여부','CPI_VALUE':'소비자물가지수','WTHR_AVG_TEMP':'평균기온'
                          ,'WTHR_AVG_H_TEMP':'평균최고기온','WTHR_AVG_L_TEMP':'평균최저기온','WTHR_AVG_PRECIP':'평균강수량'
                          ,'WTHR_AVG_WNDSPD':'평균풍속','WTHR_MX_WNDSPD':'평균최고풍속','WTHR_AVG_PRESS':'평균기압','WTHR_MX_PRESS':'최고기압'
                          ,'WTHR_MN_PRESS':'최저기압','WTHR_AVG_SEA_PRESS':'평균해면기압','WTHR_MX_SEA_PRESS':'최고해면기압'
                          ,'WTHR_MN_SEA_PRESS':'최저해면기압','WTHR_AVG_RHUM':'평균상대습도','WTHR_MN_RHUM':'최소상대습도','WTHR_SUM_SUNHR':'총일조시간'
                          ,'FST_CNT':'축제횟수','FST_IND':'축제여부','POP_GEN_CNT':'총인구수','POP_ELM_CNT':'초등학생수','POP_MID_CNT':'중학생수'
                          ,'POP_HIGH_CNT':'고등학생수','POP_60P_CNT':'60세이상인구수','POP_ELM_PROB':'초등학생비율','POP_MID_PROB':'중학생비율'
                          ,'POP_HIGH_PROB':'고등학생비율','POP_60P_PROB':'60세이상비율','POP_DENS':'인구밀도','GMS_LIC_CNT':'집단급식소수'})

    # 독립변수 Prophet 모델 예측 데이터 로드 및 캐시 저장
    data_cause_forecast = loading_data('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/cause_prediction_12months.csv')
    data_region_forecast = loading_data('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/region_prediction_12months.csv')

    # model = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/test_clf_model.pkl')
    # model = model.best_estimator_

    # 지역별 모델 로드 및 캐시 저장
    model_강원 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EA%B0%95%EC%9B%90_GradientBoostingClassifier.pkl')
    model_경기 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EA%B2%BD%EA%B8%B0%20GradientBoostingClassifier.pkl')
    model_경북 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EA%B2%BD%EB%B6%81%20GradientBoostingClassifier.pkl')
    model_전남 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EC%A0%84%EB%82%A8%20GradientBoostingClassifier.pkl')
    model_경남 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EA%B2%BD%EB%82%A8%20LGBMClassifier.pkl')
    model_대구 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EB%B6%80%EC%82%B0%20LGBMClassifier.pkl')
    model_부산 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EB%B6%80%EC%82%B0%20LGBMClassifier.pkl')
    model_울산 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EC%9A%B8%EC%82%B0%20LGBMClassifier.pkl')
    model_인천 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EC%9D%B8%EC%B2%9C%20LGBMClassifier.pkl')
    model_전북 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EC%A0%84%EB%B6%81%20LGBMClassifier.pkl')
    model_제주 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EC%A0%9C%EC%A3%BC%20LGBMClassifier.pkl')
    model_광주 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EA%B4%91%EC%A3%BC%20LGBMClassifier.pkl')
    model_대전 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EB%8C%80%EC%A0%84%20XGBClassifier.pkl')
    model_서울 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EC%84%9C%EC%9A%B8%20GradientBoostingClassifier.pkl')
    model_충남 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EC%B6%A9%EB%82%A8%20GradientBoostingClassifier.pkl')
    model_세종 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EC%84%B8%EC%A2%85%20RandomForestClassifier.pkl')
    model_충북 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EC%B6%A9%EB%B6%81%20XGBClassifier.pkl')

    # 원인물질별 모델 로드 및 캐시 저장
    model_노로바이러스 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EB%85%B8%EB%A1%9C%EB%B0%94%EC%9D%B4%EB%9F%AC%EC%8A%A4_GradientBoostingClassifier.pkl')
    model_병원성대장균 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EB%B3%91%EC%9B%90%EC%84%B1%EB%8C%80%EC%9E%A5%EA%B7%A0_LGBMClassifier.pkl')
    model_살모넬라 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EC%82%B4%EB%AA%A8%EB%84%AC%EB%9D%BC_RandomForestClassifier.pkl')
    model_장염비브리오 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EC%9E%A5%EC%97%BC%EB%B9%84%EB%B8%8C%EB%A6%AC%EC%98%A4_GradientBoostingClassifier.pkl')
    model_포도상구균 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%ED%99%A9%EC%83%89%ED%8F%AC%EB%8F%84%EC%83%81%EA%B5%AC%EA%B7%A0_RandomForestClassifier.pkl')
    model_캠필로박터제주니 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EC%BA%A0%ED%95%84%EB%A1%9C%EB%B0%95%ED%84%B0%EC%A0%9C%EC%A3%BC%EB%8B%88_GradientBoostingClassifier.pkl')
    model_클로스트리디움퍼프린젠스 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%ED%81%B4%EB%A1%9C%EC%8A%A4%ED%8A%B8%EB%A6%AC%EB%94%94%EC%9B%80%ED%8D%BC%ED%94%84%EB%A6%B0%EC%A0%A0%EC%8A%A4_RandomForestClassifier.pkl')
    model_바실러스세레우스 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EB%B0%94%EC%8B%A4%EB%9F%AC%EC%8A%A4%EC%84%B8%EB%A0%88%EC%9A%B0%EC%8A%A4_GradientBoostingClassifier.pkl')
    model_원충 = loading_model('https://github.com/Kangsungmin00/foodpoision_project/blob/main/data/%EC%9B%90%EC%B6%A9_RandomForestClassifier.pkl')

    pred_노로바이러스 = predict_reason(model_노로바이러스, data_cause_2)
    pred_병원성대장균 = predict_reason(model_병원성대장균, data_cause_2)
    pred_살모넬라 = predict_reason(model_살모넬라, data_cause_2)
    pred_장염비브리오 = predict_reason(model_장염비브리오, data_cause_2)
    pred_포도상구균 = predict_reason(model_포도상구균, data_cause_2)
    pred_캠필로박터제주니 = predict_reason(model_캠필로박터제주니, data_cause_2)
    pred_원충 = predict_reason(model_원충, data_cause_2)
    pred_클로스트리디움퍼프린젠스 = predict_reason(model_클로스트리디움퍼프린젠스, data_cause_2)
    pred_바실러스세레우스 = predict_reason(model_바실러스세레우스, data_cause_2)

    pred_cause_df = pd.DataFrame([pred_노로바이러스, pred_병원성대장균, pred_살모넬라, pred_장염비브리오, pred_포도상구균, pred_캠필로박터제주니, pred_원충, pred_클로스트리디움퍼프린젠스, pred_바실러스세레우스]
                                    , index = ['노로바이러스', '병원성대장균', '살모넬라', '장염비브리오', '포도상구균', '캠필로박터제주니', '원충', '클로스트리디움퍼프린젠스', '바실러스세레우스'], columns = ['prob']).sort_values(by = 'prob', ascending = False)

    # 지역코드별 지역명 mapping
    region_map = {'42': '강원', '41': '경기', '48': '경남', '47': '경북', '29': '광주', '27': '대구', '30': '대전', '26': '부산', '11': '서울', '36': '세종', '31': '울산', '28': '인천', '46': '전남', '45': '전북', '50': '제주', '44': '충남', '43': '충북'}

    # 지역코드별 모델 객체 딕셔너리화

    model_region_dict = {'42': model_강원, '41': model_경기, '48': model_경남, '47': model_경북, '29': model_광주
                         ,'27': model_대구, '30': model_대전, '26': model_부산, '11': model_서울, '36': model_세종
                         ,'31': model_울산, '28': model_인천, '46': model_전남, '45': model_전북, '50': model_제주
                         ,'44': model_충남, '43': model_충북}
    
    # 원인물질별 모델 객체 딕셔너리화

    model_cause_dict = {'노로바이러스': model_노로바이러스, '병원성대장균': model_병원성대장균, '살모넬라': model_살모넬라
                         , '장염비브리오': model_장염비브리오, '황색포도상구균': model_포도상구균, '캠필로박터제주니': model_캠필로박터제주니
                         , '원충': model_원충, '클로스트리디움퍼프린젠스': model_클로스트리디움퍼프린젠스, '바실러스세레우스': model_바실러스세레우스}

    # 행정구역 json 로드 및 캐시 저장
    jsonResult = loading_json_file()

    # 지역별 말월 기준 예측확률 산출    
    before_y = []

    for region in model_region_dict.keys():
        model_apply = model_region_dict[str(region)]
        before_y.append([region, model_apply.predict_proba(test_X_region.loc[test_X_region['CTPRVN_CD'] == int(region), model_apply.feature_names_in_])[0,1]])
    
    before_y = pd.DataFrame(before_y, columns = ['CTPRVN_CD','기준결과'])
    
    navy_colors = ['#404258','#474E68','#50577A','#6B728E']
    grey_colors = ['#F4F4F2','#E8E8E8','#BBBFCA','#495464']
    
    metric_color = navy_colors[2]
    char_color = grey_colors[0]
    white = '#FFFFFF'
    black = '#000000'
    new_col = '#B8BDD0'
    yellow = '#FFEBD4'
    
    with st.sidebar:
        
        st.markdown(
                f"""
                <div style='padding: 5px; border-radius: 5px; text-align: center; color: #000000;'>
                    <p style='font-size: 30px; color:{metric_color};margin: 0;font-weight: bold;'>전국 식중독 현황</p>
                    
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        st.markdown(
                f"""
                <div style='padding: 5px; border-radius: 5px; text-align: right; color: #000000;'>
                    <p style='font-size: 20px; color:{metric_color};margin: 0;font-weight: bold;'>🩺 A04 헬렌켈러</p>
                    
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        #st.write(""" """)
        st.write('--'*3)
        #st.write('기능을 선택하세요.')
                        
        function = option_menu(
        #menu_title="전국 식중독 현황", 
        None,
        options=["현황 모니터링", "시뮬레이션 분석", "향후 12개월 예측"],
        icons=['search', 'search', 'search'],
        #menu_icon="cast",
        default_index=0,
        styles={
        "container": {"padding": "4!important", "background-color": metric_color},
        "icon": {"color": "white", "font-size": "25px"},
        "nav-link": {
            "font-size": "20px",
            "text-align": "left",
            "margin": "0px",
            "color": grey_colors[0],
            "font-weight": "bold",
            "--hover-color": grey_colors[2]
        },
        "nav-link-selected": {
                        "background-color": grey_colors[1],
                        "color": navy_colors[0],
                        "font-weight": "bold"
                    },       
            }
        )
        st.write('--'*3)
    
    #시작 선
    

    if function == '현황 모니터링':

        # sidebar 설정
            
        with st.sidebar:
            # Container for border
            with st.container():
                # Selection for information to view
                dash_info = option_menu(
                menu_title=None,
                options=["발생건수", "발생환자수"],
                icons=['clipboard-check', 'clipboard-check'],
                default_index=0,
                styles={
                    "container": {"padding": "4!important", "background-color": metric_color},
                    "icon": {"color": "white", "font-size": "25px"},
                    "nav-link": {
                        "font-size": "18px",
                        "text-align": "left",
                        "margin": "2px",
                        "color": grey_colors[0],
                        "font-weight": "bold",
                        "--hover-color": grey_colors[2]
                    },
                    "nav-link-selected": {
                        "background-color": grey_colors[1],
                        "color": navy_colors[0],
                        "font-weight": "bold"
                    },                    
                }
            )
            
            # Custom CSS to style the selectbox widget
                st.markdown(
                    """
                    <style>                    
                    div[data-baseweb="select"] > div {
                        background-color: #F4F4F2; 
                        border-radius: 6px; 
                        padding: 0.5px;
                        font-weight: bold;
                        border: 2px solid #495464;
                    }
                    
                    div[data-baseweb="select"] > div > div {
                        color: #495464; /* Text color */
                        font-size: 16px; /* Font size */
                    }


                    div[data-baseweb="select"] .css-1uccc91-singleValue {
                        color: #495579; 
                        font-weight: bold;
                    }
                    
                    div[data-baseweb="select"] .css-26l3qy-menu {
                        background-color: #F4F4F2;
                        font-size: 16px;
                        color: #495579;
                        font-weight: bold;
                    }
                    </style>
                    """, unsafe_allow_html=True
                )

            # Selection for period to view
            st.markdown(
                """
                <div style='
                    font-size: 20px; 
                    color: #495579; 
                    font-weight: bold; 
                    margin-bottom: 5px; 
                    text-align: left; 
                    background-color: #F0F0F5;
                    padding: 10px;
                    border-radius: 8px;
                    '>
                    🔎 조회 기간을 선택하세요
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <style>
                .custom-label {
                    font-size: 18px;
                    color: #495579;
                    font-weight: bold;
                    margin-bottom: -20px; 
                }               
                .stSelectbox {
                    margin-top: -20px;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Your selectbox for start year
            st.markdown("<p style='font-size: 18px; color: #495579; font-weight: bold; margin-bottom: -20px;'>시작년도</p>", unsafe_allow_html=True)
            start_year = st.selectbox("", range(2002, 2023), index=0)
            st.markdown("<p style='font-size: 18px; color: #495579; font-weight: bold; margin-bottom: -20px;'>시작월</p>", unsafe_allow_html=True)
            start_month = st.selectbox('', range(1, 13), index=0)
            st.markdown("<p style='font-size: 18px; color: #495579; font-weight: bold; margin-bottom: -20px;'>종료연도</p>", unsafe_allow_html=True)
            end_year = st.selectbox('', range(2022, start_year - 1, -1), index=0)
            
            # Additional logic for conditional end month
            if end_year == start_year:
                min_end_month = start_month
            else:
                min_end_month = 1
            st.markdown("<p style='font-size: 18px; color: #495579; font-weight: bold; margin-bottom: -20px;'>종료월</p>", unsafe_allow_html=True)
            end_month = st.selectbox('', range(12, min_end_month - 1, -1), index=0)
            
            st.write('--'*3)
            # Option for conditions
            function = option_menu(
                menu_title=None,
                options=["전체", "상위 5개지역", "하위 5개지역"],
                icons=['globe', 'arrow-bar-up', 'arrow-bar-down'],
                default_index=0,
                styles={
                    "container": {"padding": "4!important", "background-color": metric_color},
                    "icon": {"color": "white", "font-size": "25px"},
                    "nav-link": {
                        "font-size": "18px",
                        "text-align": "left",
                        "margin": "0px",
                        "color": grey_colors[0],
                        "font-weight": "bold",
                        "--hover-color": grey_colors[2]
                    },
                    "nav-link-selected": {
                        "background-color": grey_colors[1],
                        "color": navy_colors[0],
                        "font-weight": "bold"
                    }
                }
            )
        
        # sidebar - 기간 조건 기준 데이터 필터링
        target_df_1 = data.loc[data.index <= datetime.datetime(end_year, end_month, 1),:]
        target_df = target_df_1.loc[target_df_1.index >= datetime.datetime(start_year, start_month, 1)]
        
        group_df = target_df.groupby('OCCRNC_REGN')[dash_info].sum().sort_values(ascending = False)
        group_df = group_df.loc[group_df > 0]

        # sidebar - 상/하위 5개지역 설정값에 따라 데이터 필터링
             
        if function == '전체':
            region_filter = group_df.index
        
        elif function == '상위 5개지역':
            region_filter = group_df[:5].index
            
        elif function == '하위 5개지역':
            region_filter = group_df[-5:].index
        
        target_cause_df = data_cause.loc[(data_cause.index >= datetime.datetime(start_year, start_month, 1))
            & (data_cause.index <= datetime.datetime(end_year, end_month, 1))
            & (data_cause['OCCRNC_VIRS'].isin(['노로바이러스','바실러스세레우스','병원성대장균','살모넬라'
                                               ,'원충','장염비브리오','캠필로박터제주니','클로스트리디움퍼프린젠스','황색포도상구균'])),:]
        
        # 종료월 기준 데이터프레임
        target_end_df = target_df.loc[(target_df.index == target_df.index.max())
                                      &(target_df['OCCRNC_REGN'].isin(region_filter))]

        # 종료월 직전월 기준 데이터프레임
        max_yearm = datetime.datetime(end_year, end_month, 1)
        max_yearm_1 = max_yearm - datetime.timedelta(days=1)
        year_1 = datetime.datetime(max_yearm_1.year, max_yearm_1.month, 1)
        target_end_1_df = target_df.loc[(target_df.index == year_1)
                                        &(target_df['OCCRNC_REGN'].isin(region_filter))]

        # 현황 모니터링 화면에서 KPI 표시
        
        col1, col2, col3, col4 = st.columns(4)
        

        with col1:

            OCCRNC_CNT_KPI = target_end_df['발생건수'].sum()
            OCCRNC_CNT_KPI_1 = target_end_1_df['발생건수'].sum()

            KPI_OCCRNC_DF = target_df.loc[target_df.index >= datetime.datetime(max_yearm.year - 1, max_yearm.month, 1), '발생건수']
            grouped_data = KPI_OCCRNC_DF.groupby(KPI_OCCRNC_DF.index).sum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=grouped_data.index,
                y=grouped_data.values,
                mode='lines',
                line=dict(color='#FFEBD4', width = 25)
            ))
            
            fig.update_layout(
                width = 600,
                height = 400,
                plot_bgcolor= navy_colors[2],  # Light blue background for the plot area
                paper_bgcolor="#000000",  # Transparent background for the overall figure
                xaxis=dict(showgrid=True, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                margin=dict(l=0, r=0, t=0, b=0)  # Remove margins to make the plot compact
            )
            
            buf = io.BytesIO()
            fig.write_image(buf, format="png")
            buf.seek(0)
            base64_image = base64.b64encode(buf.read()).decode("utf-8")
            
            delta_value = OCCRNC_CNT_KPI - OCCRNC_CNT_KPI_1
            
            
            
            st.markdown(
                f"""
                <div style='background-color: {metric_color}; padding: 20px; border-radius: 5px; display: flex; align-items: center; flex: 1; min-height:175px; height:100%;'>
                    <div style='flex: 1;'>  
                        <p style='font-size: 30px; margin: 0; color: {char_color}; font-weight: bold; line-height: 1.5;'>발생 건수</p>
                        <h3 style='margin: 0; color: {char_color}; font-weight: bold; line-height: 1.5;'>{OCCRNC_CNT_KPI:,.0f}건</h3>
                        <p style='font-size: 15px; margin: 0; color: {"#FFEBD4" if delta_value >= 0 else char_color}; font-weight: bold; line-height: 1.5;'>
                            전월대비 {"▲" if delta_value >= 0 else "▼"} {abs(delta_value):,.0f}건
                        </p>
                    </div>
                    <div style='flex: 1; display: flex; justify-content: center; margin-left: 10px;'>
                        <img src="data:image/png;base64,{base64_image}" style="width: 200px; height: auto; border-radius: 5px;" />
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            PATNT_CNT_KPI = target_end_df['발생환자수'].sum()
            PATNT_CNT_KPI_1 = target_end_1_df['발생환자수'].sum()
            
            KPI_PATNT_DF = target_df.loc[target_df.index >= datetime.datetime(max_yearm.year - 1, max_yearm.month, 1), '발생환자수']
            grouped_data = KPI_PATNT_DF.groupby(KPI_PATNT_DF.index).sum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=grouped_data.index,
                y=grouped_data.values,
                mode='lines',
                line=dict(color='#FFEBD4', width = 25)
            ))
            
            fig.update_layout(
                width = 600,
                height = 400,
                plot_bgcolor= navy_colors[2],  # Light blue background for the plot area
                paper_bgcolor="#000000",  # Transparent background for the overall figure
                xaxis=dict(showgrid=True, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                margin=dict(l=0, r=0, t=0, b=0)  # Remove margins to make the plot compact
            )
            
            buf = io.BytesIO()
            fig.write_image(buf, format="png")
            buf.seek(0)
            base64_image = base64.b64encode(buf.read()).decode("utf-8")
            
            delta_value = PATNT_CNT_KPI - PATNT_CNT_KPI_1

            st.markdown(
                f"""
                <div style='background-color: {metric_color}; padding: 20px; border-radius: 5px; display: flex; align-items: center; flex: 1; min-height:175px; height:100%;'>
                    <div style='flex: 1; font-family: "mine";'>  <!-- Apply the custom font here -->
                        <p style='font-size: 30px; margin: 0; color: {char_color}; font-weight: bold; line-height: 1.5;'>환자수</p>
                        <h3 style='margin: 0; font-size: 25; color: {char_color}; font-weight: bold; line-height: 1.5;'>{PATNT_CNT_KPI:,.0f}명</h3>
                        <p style='font-size: 16px; margin: 0; color: {"#FFEBD4" if delta_value >= 0 else char_color}; font-weight: bold; line-height: 1.5;'>
                            전월대비 {"▲" if delta_value >= 0 else "▼"} {abs(delta_value):,.0f}건
                        </p>
                    </div>
                    <div style='flex: 1; display: flex; justify-content: center; margin-left: 20px;'>
                        <img src="data:image/png;base64,{base64_image}" style="width: 200px; height: auto; border-radius: 5px;" />
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:

            MOST_OCCRNC_CNT_REGN = target_end_df.groupby('OCCRNC_REGN')['발생건수'].sum().sort_values().index[-1]
            st.markdown(
                f"""
                <div style='background-color: {metric_color}; padding: 20px; border-radius: 5px; text-align: left; color: #000000; min-height:175px; height:100%;'>
                    <p style='font-size: 30px; color:{char_color};margin: 0;font-weight: bold;'>최다 발생지역</p>
                    <h3 style='margin: 0; color:{char_color};font-size: 40px;'>{MOST_OCCRNC_CNT_REGN}</h3>                       
                </div>
                """, 
                unsafe_allow_html=True
            )

        with col4:
            MOST_OCCRNC_CNT_VIRS = target_cause_df.groupby('OCCRNC_VIRS')['OCCRNC_CNT'].sum().sort_values().index[-1]
            st.markdown(
                f"""
                <div style='background-color: {metric_color}; padding: 20px; border-radius: 5px; text-align: left; color: #000000; min-height:175px; height:100%;'>
                    <p style='font-size: 30px;color:{char_color}; margin: 0;font-weight: bold;'>최대 원인물질</p>
                    <h3 style='margin: 0;color:{char_color};font-size: 40px;'>{MOST_OCCRNC_CNT_VIRS}</h3>
                </div>
                """, 
                unsafe_allow_html=True
            )
        #st.write('--'*3)
       
        # 지도 시각화
        dash_info_eng = convert_dash_info(dash_info)

        col1, col2 = st.columns(2)

        # 지도 옆 컬럼
        
        with col1:

            col1_1, = st.columns(1)

            map_df = target_df.loc[(target_df[dash_info] > 0) & (target_df['OCCRNC_REGN'].isin(region_filter))]

            m = folium.Map(location = [36.2, 128.2], tiles = 'Cartodb Positron'
                           , zoom_start = 7, zoom_control = False, min_zoom=7, max_zoom=7
                           , max_bounds=True
                           , min_lat = 33.5
                           , min_lon = 124.4
                           , max_lat = 38.8
                           , max_lon = 132)

            # GeoJSON에 발생 건수와 환자 수 추가???
            for feature in jsonResult['features']:
                region_code = feature['properties']['CTPRVN_CD']
                region_data = map_df[map_df['CTPRVN_CD'] == int(region_code)] if region_code.isdigit() else None

                if not region_data.empty:
                    feature['properties']['발생건수'] = int(region_data['발생건수'].sum())
                    feature['properties']['발생환자수'] = int(region_data['발생환자수'].sum())

                else:
                    # 값이 없을 경우 기본값으로 설정
                    feature['properties']['발생건수'] = 0
                    feature['properties']['발생환자수'] = 0

            folium.Choropleth(
                geo_data = jsonResult
                , name = dash_info
                , data = map_df.groupby('CTPRVN_CD', as_index = False)[dash_info].sum()
                , columns = ['CTPRVN_CD', dash_info]
                , key_on='feature.properties.CTPRVN_CD'
                , fill_color = 'RdBu_r'
                , fill_opacity = 0.7
                , line_opacity = 0.3
                , nan_fill_color = 'white'
                , color = 'white'
                , legend_name = dash_info
            ).add_to(m)

            def on_click(feature):
                return {
                    # 'fillColor': '#FFFFFF',
                    'color': 'lightgray',
                    'weight': 1,
                    'fillOpacity': 0.1
                }
            
            #st.write(""" """)
            folium.GeoJson(
                data = jsonResult,
                name="OCCRNC_REGN",
                style_function=on_click,
                tooltip=folium.features.GeoJsonTooltip(
                    fields=["CTP_KOR_NM","발생건수","발생환자수"],  # GeoJSON의 구역 이름 필드명
                    aliases=["지 역 명: ","발생건수: ","환 자 수: "]
                ),
                highlight_function=lambda x: {"weight": 3, "color": "gray"}
            ).add_to(m)
            
            output = st_folium(m, width=500, height=750, use_container_width=True)
            
            
            with col1_1:
                region_name = pd.Series(output["last_active_drawing"]["properties"]["CTPRVN_CD"]).map(region_map)[0] if output["last_active_drawing"] else "전국"
                st.write(""" """)
                st.markdown(
                    f"""
                    <div style='background-color: {grey_colors[0]}; padding: 5px; border-radius: 1px; text-align: center;border: 2px solid {grey_colors[2]};'>
                        <h5 style='margin: 15px 0 0 0; color: {navy_colors[0]}; font-weight: bold; font-size: 25px;'>️🗺️ 식중독 현황 지도: {region_name}</h5>
                    </div>
                    """,
                    unsafe_allow_html=True
            )
            
            
           
       
        with col2:
        
            tab_container_bg_color = "#f4f4f8"
            tab_border_color = "#cccccc"
            unselected_tab_bg_color = tab_container_bg_color
            unselected_tab_hover_bg_color = "#d5d5d5"
            selected_tab_bg_color = navy_colors[1]
            selected_tab_text_color = "white"
            tab_font_color = "black"
            tab_font_size = "30px"
            tab_border_radius = "10px"

            st.markdown(
                f"""
                <style>                
                div[data-testid="stTabs"] > div {{
                    background-color: {tab_container_bg_color};
                    border-radius: {tab_border_radius};
                    border: 1px solid {tab_border_color};
                    padding: 5px;
                }}
               
                div[data-testid="stTabs"] button[role="tab"] {{
                    background-color: {unselected_tab_bg_color};
                    color: {tab_font_color};
                    font-weight: bold;
                    font-size: {tab_font_size};
                    border-radius: 8px;
                    padding: 10px;
                    margin-right: 1px; 
                    border: none; 
                }}
               
                div[data-testid="stTabs"] button[role="tab"]:hover {{
                    background-color: {unselected_tab_hover_bg_color};
                    color: #333333;
                }}
                
                div[data-testid="stTabs"] button[aria-selected="true"] {{
                    background-color: {selected_tab_bg_color}; 
                    color: {selected_tab_text_color};
                    border: 1px solid {selected_tab_bg_color}; 
                    box-shadow: none; 
                    outline: none; 
                }}
                </style>
                """,
                unsafe_allow_html=True
            )


            tab1, tab2 = st.tabs(['**식중독 알아보기**','**식중독 현황 분석하기**'])

            with tab1:
                
                st.markdown(
                    f"""
                    <div style='background-color: {new_col}; padding: 5px; border-radius: 5px; text-align: left;border: 2px solid {navy_colors[0]};'>
                        <h5 style='margin: 15px 0 0 0; color: {grey_colors[3]}; font-weight: bold;'>️  ✔  식중독이란?</h5>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                #st.write(''' ''')
                st.markdown(
                    """
                    <div style='border: 2px solid #d3d3d3; padding: 10px; border-radius: 5px;'>
                        식품위생법 제2조제14항에 따르면<br>
                        “<strong>식품의 섭취</strong>로 인하여 인체에 유해한 미생물 또는 유독 물질에 의하여 발생하였거나 발생한 것으로 판단되는 <strong>감염성 또는 독소형 질환</strong>”을 의미합니다.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.write(''' ''')
                st.markdown(
                    f"""
                    <div style='background-color: {new_col}; padding: 5px; border-radius: 5px; text-align: left;border: 2px solid {navy_colors[0]};'>
                        <h5 style='margin: 15px 0 0 0; color: {grey_colors[3]}; font-weight: bold;'>️  ✔  식중독 원인</h5>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.markdown(
                    """
                    <div style='border: 2px solid #d3d3d3; padding: 10px; border-radius: 5px;'>
                        <p><strong>오염된 손으로 음식을 조리 또는 섭취</strong>하거나, 하나의 도마에서 <strong>육류와 채소류를 함께 사용하여 교차오염</strong>이 발생하거나, <strong>충분히 가열하여 먹지 않는 경우</strong> 등으로 발생할 수 있습니다.<br>
                        특히, 여름에는 덥고 습하여 다양한 식중독균이 증식하기 쉽고, 겨울철에 주로 발생하는 노로바이러스는 영하 20도에서도 감염력을 유지합니다.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.write(''' ''')
                st.markdown(
                    f"""
                    <div style='background-color: {new_col}; padding: 5px; border-radius: 5px; text-align: left;border: 2px solid {navy_colors[0]};'>
                        <h5 style='margin: 15px 0 0 0; color: {grey_colors[3]}; font-weight: bold;'>️  ✔  식중독 예방 3대 원칙</h5>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    """
                    <div style='border: 2px solid #d3d3d3; padding: 10px; border-radius: 5px; text-align: left;'>
                        <p><strong>1️⃣ 손은 비누로 깨끗이 씻기! <br> 2️⃣ 물은 끓여서 마시기! <br> 3️⃣ 음식은 충분히 익혀서 먹기!</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write(''' ''')
                st.markdown(
                    f"""
                    <div style='background-color: {new_col}; padding: 5px; border-radius: 5px; text-align: left;border: 2px solid {navy_colors[0]};'>
                        <h5 style='margin: 15px 0 0 0; color: {grey_colors[3]}; font-weight: bold;'>️  ✔  주요 식중독 원인물질</h5>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                poisson_cause = [['노로바이러스','바이러스성','']]
                st.dataframe(pd.DataFrame([
                    ['노로바이러스',
                    '바이러스성',
                    '겨울철',
                    '24~48시간',
                    '급성 위장염, 구토, 설사, 복통',
                    '오염된 물 및 어패류',
                    '적은 수로도 감염 가능, 2차 감염 가능'],
                    ['원충', '원충성', '여름철', '7일 이상', '설사, 복통', '오염된 물 및 채소 등', '장내에서 기생하면서 증상 유발'],
                    ['병원성 대장균',
                    '세균성',
                    '여름철',
                    '16시간 이상',
                    '심한 설사, 복통, 발열',
                    '오염된 육류 및유제품',
                    '적은 수로도 감염 가능'],
                    ['살모넬라',
                    '세균성',
                    '여름철 및 가을철',
                    '16시간 이상',
                    '발열, 설사, 복통',
                    '오렴된육류, 가금류, 계란',
                    '작은 양으로도 감염, 2차 감염 가능'],
                    ['황색포도상구균', '세균성', '여름철', '2~6시간', '구토, 복통', '육류 및 유제품 등', '염분 환경에서 증식 가능'],
                    ['바실러스세레우스',
                    '세균성',
                    '여름철',
                    '6~24시간',
                    '구토형(오심, 구토), 설사형(복통, 설사)',
                    '쌀, 파스타 등 전분질 음식',
                    '음식 조리 후 방치로 증식'],
                    ['장염 비브리오',
                    '세균성',
                    '여름철',
                    '16시간 이상',
                    '복통, 설사,오심',
                    '어패류 등 해산물',
                    '높은 염도 환경에서 증식 가능'],
                    ['클로스트리디움퍼프린젠스',
                    '세균성',
                    '연중',
                    '8~24시간',
                    '경미한 복통 및 설사',
                    '조리 후 오래 보관된 음식',
                    '산소가 없는 환경에서 증식'],
                    ['캠필로박터제주니',
                    '세균성',
                    '여름철 및 가을철',
                    '2~7일',
                    '발열, 복통, 설사, (종종 혈변동반)',
                    '가금류',
                    '면역 저하자의 경우 신경계 합병증 유발 가능']], columns = ['병원체', '분류', '주요 발생시기', '잠복기', '주요 증상', '주요 오염원', '특징']), use_container_width=True, hide_index = True, height = 355)
                
                st.markdown(
                    f"""
                    <div style='background-color: {new_col}; padding: 5px; border-radius: 5px; text-align: left;border: 2px solid {navy_colors[0]};'>
                        <h5 style='margin: 15px 0 0 0; color: {grey_colors[3]}; font-weight: bold;'>️  ✔  더 자세히 알기!</h5>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.write(''' ''')
                st.page_link(page = 'https://www.foodsafetykorea.go.kr:443/portal/board/boardDetail.do?menu_no=4418&menu_grp=MENU_NEW02&bbs_no=bbs400', label = '🔗 식품의약품안전처 식품안전나라')
                st.page_link(page = 'https://health.kdca.go.kr/healthinfo/biz/health/gnrlzHealthInfo/gnrlzHealthInfo/gnrlzHealthInfoView.do?cntnts_sn=5239', label = '🔗 질병관리청 국가건강정보포털')

            with tab2:               
                
                st.markdown(
                    f"""
                    <style>
                    .tooltip {{
                        position: relative;
                        display: inline-block;
                        cursor: pointer;
                    }}

                    .tooltip .tooltiptext {{
                        visibility: hidden;
                        width: 200px;
                        background-color: {metric_color};
                        color: {char_color};
                        text-align: center;
                        padding: 5px;
                        border-radius: 5px;
                        position: absolute;
                        z-index: 1;
                        bottom: 125%; 
                        left: 50%;
                        margin-left: -100px;
                        opacity: 0;
                        transition: opacity 0.3s;
                        font-size: 12px;
                    }}

                    .tooltip:hover .tooltiptext {{
                        visibility: visible;
                        opacity: 1;
                    }}
                    </style>

                    <div style='background-color: {new_col}; padding: 5px; border-radius: 5px; text-align: left;border: 2px solid {navy_colors[0]};'>
                        <h5 class="tooltip" style='margin: 15px 0 0 0; color: {grey_colors[3]}; font-weight: bold;'>️ ✔ 원인물질별 발생 현황
                            <span class="tooltiptext">조회 기간 원인물질별 조회정보 ({dash_info})의 합계</span>
                        </h5>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                group_cause_df = target_cause_df.groupby('OCCRNC_VIRS')[dash_info_eng].sum().sort_values(ascending = False)
                group_cause_df = group_cause_df.loc[group_cause_df > 0].reset_index()
                
                group_cause_df_col = group_cause_df.columns

                if group_cause_df_col[1] == 'OCCRNC_CNT':

                    labels = {'OCCRNC_VIRS':'원인물질', 'OCCRNC_CNT':'발생건수'}
                
                elif group_cause_df_col[1] == 'PATNT_CNT':

                    labels = {'OCCRNC_VIRS':'원인물질', 'PATNT_CNT':'발생환자수'}

                fig = px.bar(group_cause_df
                            , x = group_cause_df_col[0]
                            , y = group_cause_df_col[1]
                            , color = group_cause_df_col[1]
                            , color_continuous_scale='Reds' # or RdBu_r 
                            , labels = labels
                            , height=330
                            )
                
                # 그래프 레이아웃 업데이트
                fig.update_layout(
                    xaxis_tickangle=-45,
                    xaxis_title='',
                    yaxis_title='',
                    coloraxis_showscale=False  # 색상 범례 숨기기
                        )

                st.plotly_chart(fig)
                
                st.markdown(
                    f"""
                    <style>
                    .tooltip {{
                        position: relative;
                        display: inline-block;
                        cursor: pointer;
                    }}

                    .tooltip .tooltiptext {{
                        visibility: hidden;
                        width: 200px;
                        background-color: {grey_colors[0]};
                        color: {grey_colors[3]};
                        text-align: center;
                        padding: 5px;
                        border-radius: 5px;
                        position: absolute;
                        z-index: 1;
                        bottom: 125%; 
                        left: 50%;
                        margin-left: -100px;
                        opacity: 0;
                        transition: opacity 0.3s;
                        font-size: 12px;
                    }}

                    .tooltip:hover .tooltiptext {{
                        visibility: visible;
                        opacity: 1;
                    }}
                    </style>

                    <div style='background-color: {new_col}; padding: 5px; border-radius: 5px; text-align: left;border: 2px solid {navy_colors[0]};'>
                        <h5 class="tooltip" style='margin: 15px 0 0 0; color: {grey_colors[3]}; font-weight: bold;'>️ ✔ 식중독 발생 주요 요인
                            <span class="tooltiptext">
                                조회 기간에 대하여 선택한 지역의 식중독 예측 모델과 SHAP로 추출한 변수 중요도를 나타냄<br>
                                <span style='color: red;'>빨간색</span>은 조회 지역의 식중독 발생확률 증가요인, <span style='color: blue;'>파란색</span>은 감소요인을 의미
                            </span>
                        </h5>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.write(""" """)

                
                if output["last_active_drawing"]:

                    # 선택된 행정구역 코드 반환
                    selected_area = output["last_active_drawing"]["properties"]["CTPRVN_CD"]
                    area_name = pd.Series(selected_area).map(region_map)[0]
                    
                    # 원 데이터의 CTPRVN_CD가 정수형 변수여서 강제로 변환해줌, 이후 데이터타입 변경 필요
                    target_df_2 = target_df.loc[target_df['CTPRVN_CD'] == int(selected_area)]
                    
                    X = target_df_2[model_region_dict[selected_area].feature_names_in_]
                    selected_model = model_region_dict[selected_area]
                    if selected_model in [model_세종]:
                        shap_plot = shap_rf_summary_plot(model_region_dict[selected_area], X)

                    else:
                        shap_plot = shap_summary_plot(model_region_dict[selected_area], X)
                    
                    st.markdown(f"""
                    </style>

                    <div style='background-color: {new_col}; padding: 5px; border-radius: 5px; text-align: left;border: 2px solid {navy_colors[0]};'>
                        <h5 class="tooltip" style='margin: 15px 0 0 0; color: {grey_colors[3]}; font-weight: bold;'>️ ✔ 월별 식중독 {dash_info} 추이
                            <span class="tooltiptext">
                                조회 기간 해당 지역의 월별 조회정보({dash_info}) 추이
                            </span>
                        </h5>
                    </div>
                    """,
                    unsafe_allow_html=True
                    )
                    
                    # 선형 그래프 도식화
                    st.line_chart(target_df_2[dash_info], color = '#DF0101', height=200, use_container_width=True) # 기존 색상(회색) : #848484

                else:
                    
                    # 지역을 선택하지 않는 경우 현재 조회 조건에서 가장 수가 많은 지역의 결과를 보여줌
                    target_df_3 = target_df.loc[target_df['OCCRNC_REGN'] == region_filter[0]]
                    selected_area_2 = target_df_3['CTPRVN_CD'].astype(str).unique()[0]

                    X_2 = target_df_3[model_region_dict[selected_area_2].feature_names_in_]

                    selected_model = model_region_dict[selected_area_2]
                    if selected_model in [model_세종]:
                        shap_plot = shap_rf_summary_plot(model_region_dict[selected_area_2], X_2)

                    else:
                        shap_plot = shap_summary_plot(model_region_dict[selected_area_2], X_2)
                    
                    st.markdown(f"""
                    </style>

                    <div style='background-color: {new_col}; padding: 5px; border-radius: 5px; text-align: left;border: 2px solid {navy_colors[0]};'>
                        <h5 class="tooltip" style='margin: 15px 0 0 0; color: {grey_colors[3]}; font-weight: bold;'>️ ✔ 월별 식중독 {dash_info} 추이
                            <span class="tooltiptext">
                                조회 기간 해당 지역의 월별 조회정보({dash_info}) 추이
                            </span>
                        </h5>
                    </div>
                    """,
                    unsafe_allow_html=True
                    )

                    st.line_chart(target_df_3[dash_info], color = '#DF0101', height=200, use_container_width=True, ) # 기존 색상(회색) : #848484

    elif function == '시뮬레이션 분석':

        with st.sidebar:
        
            st.markdown("""
                <style>               
                input[type="number"] {
                    width: 300px !important;
                    padding: 2.5px;
                    font-size: 23px;
                    color: #FFFFFF ; 
                    background-color: #50577A; 
                    border-radius: 2px; 
                    border: 1px solid #50577A; 
                    text-align: center; 
                }

                .step-up, .step-down {
                    color: #000000;
                }
                </style>
            """, unsafe_allow_html=True)

            option_value = {}
            
            
            st.markdown("""
    <style>
    .custom-header {
        font-size: 20px; 
        font-weight: bold; 
        color: #495579; 
        background-color: #f0f0f5; 
        padding: 10px; 
        border-radius: 8px; 
        text-align: center;        
        margin-bottom: 20px;
    }
    </style>
    <div class="custom-header">🔎 기상변수를 설정하세요</div>
""", unsafe_allow_html=True)
            
            
            st.markdown("""
    <style>
    .simple-label {
        font-size: 18px; 
        font-weight: bold; 
        color: #495579; 
        margin-bottom: 10px; 
    }
    </style>
    <div class="simple-label">기온</div>
""", unsafe_allow_html=True)
            option_value['기온'] = st.number_input(f'(기준월 평균기온 : {test_X_region['평균기온'].mean():.1f}˚C)', value = test_X_region['평균기온'].mean(), step = 0.1, format = '%.1f')
            st.markdown("""
    <style>
    .simple-label {
        font-size: 18px; 
        font-weight: bold; 
        color: #495579; 
        
    }
    </style>
    <div class="simple-label">강수량</div>
""", unsafe_allow_html=True)
            option_value['강수량'] = st.number_input(f'(기준월 평균강수량 : {test_X_region['평균강수량'].mean():.1f}mm)', min_value=0.0, value = test_X_region['평균강수량'].mean(), step = 0.1, format = '%.1f')
            st.markdown("""
    <style>
    .simple-label {
        font-size: 18px; 
        font-weight: bold;
        color: #495579; 
        margin-bottom: 10px;
    }
    </style>
    <div class="simple-label">습도</div>
""", unsafe_allow_html=True)
            option_value['습도'] = st.number_input(f'(기준월 평균습도 {test_X_region['평균상대습도'].mean():.1f}%)', min_value=0.0, max_value=100.0, value = test_X_region['평균상대습도'].mean(), step = 0.1, format = '%.1f')
            st.write('--'*3)
            st.markdown("""
    <style>
    .custom-header {
        font-size: 22px; 
        font-weight: bold; 
        color: #495579; 
        background-color: #f0f0f5; 
        padding: 10px; 
        border-radius: 8px; 
        text-align: center; 
        
        margin-bottom: 20px;
    }
    </style>
    <div class="custom-header">🔎 인구변수를 설정하세요</div>
""", unsafe_allow_html=True)
            st.markdown("""
    <style>
    .simple-label {
        font-size: 18px; 
        font-weight: bold;
        color: #495579; 
        margin-bottom: 10px;
    }
    </style>
    <div class="simple-label">초등학생 비율</div>
""", unsafe_allow_html=True)
            option_value['초등학생비율'] = st.number_input(f'(기준월 평균비율 : {test_X_region['초등학생비율'].mean():.1%})', min_value=0.0, value = test_X_region['초등학생비율'].mean()*100, step = 0.1, format = '%.1f')
            st.markdown("""
    <style>
    .simple-label {
        font-size: 18px; 
        font-weight: bold; 
        color: #495579; 
        margin-bottom: 10px; 
    }
    </style>
    <div class="simple-label">중학생 비율</div>
""", unsafe_allow_html=True)
            option_value['중학생비율'] = st.number_input(f'(기준월 평균비율 : {test_X_region['중학생비율'].mean():.1%})', min_value=0.0, value = test_X_region['중학생비율'].mean()*100, step = 0.1, format = '%.1f')
            st.markdown("""
    <style>
    .simple-label {
        font-size: 18px; 
        font-weight: bold; 
        color: #495579; 
        margin-bottom: 10px;
    }
    </style>
    <div class="simple-label">고등학생 비율</div>
""", unsafe_allow_html=True)
            option_value['고등학생비율'] = st.number_input(f'(기준월 평균비율 : {test_X_region['고등학생비율'].mean():.1%})', min_value=0.0, value = test_X_region['고등학생비율'].mean()*100, step = 0.1, format = '%.1f')
            st.markdown("""
    <style>
    .simple-label {
        font-size: 18px; 
        font-weight: bold; 
        color: #495579; 
        margin-bottom: 10px;
    }
    </style>
    <div class="simple-label">60세 이상 비율</div>
""", unsafe_allow_html=True)
            option_value['60세이상비율'] = st.number_input(f'(기준월 평균비율 : {test_X_region['60세이상비율'].mean():.1%})', min_value=0.0, value = test_X_region['60세이상비율'].mean()*100, step = 0.1, format = '%.1f')
            
            st.write(f'(자료 기준월 : {data.index.max().strftime('%Y년 %m월')})')

        test_X_2 = test_X_region.copy()

        for col in option_value.keys():

            if col == '기온':
                test_X_2['평균기온'] = option_value['기온']
                test_X_2['평균최고기온'] = test_X_region['평균최고기온'] + (option_value['기온'] - test_X_region['평균기온'])
                test_X_2['평균최저기온'] = test_X_region['평균최저기온'] + (option_value['기온'] - test_X_region['평균기온'])
            elif col == '강수량':
                test_X_2['평균강수량'] = option_value['강수량']
            elif col == '습도':
                test_X_2['평균상대습도'] = option_value['습도']
                test_X_2['최소상대습도'] = test_X_region['최소상대습도'] + (option_value['습도'] - test_X_region['평균상대습도'])
            elif col == '초등학생비율':
                test_X_2['초등학생비율'] = option_value['초등학생비율']/100
            elif col == '중학생비율':
                test_X_2['중학생비율'] = option_value['중학생비율']/100
            elif col == '고등학생비율':
                test_X_2['고등학생비율'] = option_value['고등학생비율']/100
            elif col == '6세이상비율':
                test_X_2['60세이상비율'] = option_value['60세이상비율']/100

        before_y = []
        pred_y = []

        for region in model_region_dict.keys():
            model_apply = model_region_dict[str(region)]
            before_y.append([region, model_apply.predict_proba(test_X_region.loc[test_X_region['CTPRVN_CD'] == int(region), model_apply.feature_names_in_])[0,1]])
            pred_y.append([region, model_apply.predict_proba(test_X_2.loc[test_X_2['CTPRVN_CD'] == int(region), model_apply.feature_names_in_])[0,1]])

        before_y = pd.DataFrame(before_y, columns = ['CTPRVN_CD','기준결과'])
        pred_y = pd.DataFrame(pred_y, columns = ['CTPRVN_CD','시뮬레이션'])

        col1, col2 = st.columns(2)

        with col1:        
            st.markdown(
                f"""
                <style>
                .tooltip {{
                    position: relative;
                    display: inline-block;
                    cursor: pointer;
                }}

                .tooltip .tooltiptext {{
                    visibility: hidden;
                    width: 250px;
                    background-color: {grey_colors[0]};
                    color: {navy_colors[0]};
                    text-align: center;
                    padding: 5px;
                    border-radius: 5px;
                    position: absolute;
                    z-index: 1;
                    bottom: 125%; 
                    left: 50%;
                    margin-left: -125px; 
                    opacity: 0;
                    transition: opacity 0.3s;
                    font-size: 12px;
                }}

                .tooltip:hover .tooltiptext {{
                    visibility: visible;
                    opacity: 1;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown(
                f"""
                <div style='background-color: {grey_colors[0]}; padding: 5px; border-radius: 1px; text-align: center;border: 2px solid {grey_colors[2]};'>
                    <h5 style='margin: 15px 0 0 0; color: {grey_colors[3]}; font-weight: bold;font-size: 25px;'>
                        <span class="tooltip">🗺️ 식중독 예측 지도
                            <span class="tooltiptext">
                                각 변수 변화에 따른 지역별 식중독 예상 발생확률을 나타냄
                            </span>
                        </span>
                    </h5>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            

            m = folium.Map(location = [36.2, 128.2], tiles = 'Cartodb Positron'
                           , zoom_start = 7, zoom_control = False, min_zoom=7, max_zoom=7
                           , max_bounds=True
                           , min_lat = 33.5
                           , min_lon = 124.4
                           , max_lat = 38.8
                           , max_lon = 132)
            
            # GeoJSON에 발생 건수와 환자 수 추가???
            for feature in jsonResult['features']:
                region_code = feature['properties']['CTPRVN_CD']
                region_data = pred_y[pred_y['CTPRVN_CD'] == region_code] if region_code.isdigit() else None
                if not region_data.empty:
                    feature['properties']['예측확률'] = f'{float(region_data['시뮬레이션']) *100:.1f}%'

                else:
                    # 값이 없을 경우 기본값으로 설정
                    feature['properties']['예측확률'] = 0

            folium.Choropleth(
                geo_data = jsonResult
                , name = '식중독 발생확률'
                , data = pred_y
                , columns = ['CTPRVN_CD', '시뮬레이션']
                , key_on='feature.properties.CTPRVN_CD'
                , fill_color = 'RdBu_r'
                , fill_opacity = 0.7
                , line_opacity = 0.3
                , color = 'gray'
                , legend_name = '식중독 발생확률'
            ).add_to(m)

            def on_click(feature):
                return {
                    # 'fillColor': '#ffaf00',
                    'color': 'lightgray',
                    'weight': 1,
                    'fillOpacity': 0.1
                }

            folium.GeoJson(
                data = jsonResult,
                name="OCCRNC_REGN",
                style_function=on_click,
                tooltip=folium.features.GeoJsonTooltip(
                    fields=["CTP_KOR_NM","예측확률"],  # GeoJSON의 구역 이름 필드명
                    aliases=["지 역 명: ","예측확률: "]
                ),
                highlight_function=lambda x: {"weight": 3, "color": "gray"}
            ).add_to(m)

            output = st_folium(m, width=500, height=700, use_container_width=True)

        with col2:
            
            # 원인물질별 발생 확률
                       
            st.markdown(
                f"""
                <style>
                .tooltip {{
                    position: relative;
                    display: inline-block;
                    cursor: pointer;
                }}

                .tooltip .tooltiptext {{
                    visibility: hidden;
                    width: 250px;
                    background-color: {grey_colors[0]};
                    color: {navy_colors[0]};
                    text-align: center;
                    padding: 5px;
                    border-radius: 5px;
                    position: absolute;
                    z-index: 1;
                    bottom: 125%; 
                    left: 50%;
                    margin-left: -125px; 
                    opacity: 0;
                    transition: opacity 0.3s;
                    font-size: 12px;
                }}
                .tooltip:hover .tooltiptext {{
                    visibility: visible;
                    opacity: 1;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )

            # Create the tooltip content using HTML
            st.markdown(
                f"""
                <div style='background-color: {new_col}; padding: 5px; border-radius: 5px; text-align: left;border: 2px solid {navy_colors[0]};'>
                    <h5 style='margin: 15px 0 0 0; color: {grey_colors[3]}; font-weight: bold;'>
                        <span class="tooltip">✔️ 원인물질별 발생 확률 시뮬레이션 결과
                            <span class="tooltiptext">
                                각 변수 변화에 따른 원인물질별 식중독 예상 발생확률을 나타냄<br>
                                확률이 높을수록 식중독 발생 시 해당 원인물질에 기인할 가능성이 큰 것을 의미
                            </span>
                        </span>
                    </h5>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            
            test_X_3 = test_X_cause.copy()

            for col in option_value.keys():

                if col == '기온':
                    test_X_3['평균기온'] = option_value['기온']
                    test_X_3['평균최고기온'] = test_X_cause['평균최고기온'] + (option_value['기온'] - test_X_cause['평균기온'].mean())
                    test_X_3['평균최저기온'] = test_X_cause['평균최저기온'] + (option_value['기온'] - test_X_cause['평균기온'].mean())
                elif col == '강수량':
                    test_X_3['평균강수량'] = option_value['강수량']
                elif col == '습도':
                    test_X_3['평균상대습도'] = option_value['습도']
                    test_X_3['최소상대습도'] = test_X_cause['최소상대습도'] + (option_value['습도'] - test_X_cause['평균상대습도'].mean())
                elif col == '초등학생비율':
                    test_X_3['초등학생비율'] = option_value['초등학생비율']/100
                elif col == '중학생비율':
                    test_X_3['중학생비율'] = option_value['중학생비율']/100
                elif col == '고등학생비율':
                    test_X_3['고등학생비율'] = option_value['고등학생비율']/100
                elif col == '6세이상비율':
                    test_X_3['60세이상비율'] = option_value['60세이상비율']/100

            pred_cause_df_2 = pd.DataFrame()
            for cause in model_cause_dict.keys():
                model_apply = model_cause_dict[cause]
                target_predict_df = test_X_3.loc[test_X_3['OCCRNC_VIRS'] == cause].copy()
                target_predict_df['예측확률'] = model_apply.predict_proba(target_predict_df.loc[:,model_apply.feature_names_in_])[0,1]
                pred_cause_df_2 = pd.concat([pred_cause_df_2, target_predict_df[['OCCRNC_VIRS','예측확률']]])      

            pred_cause_df_2 = pred_cause_df_2.sort_values(by = '예측확률', ascending = False)
            pred_cause_df_2['예측확률'] = (pred_cause_df_2['예측확률'] * 100).round(1)

            # Plotly로 막대 그래프 생성 (발생 건수에 따른 색상 변경)
            fig = px.bar(pred_cause_df_2
                         , x = 'OCCRNC_VIRS'
                         , y = '예측확률'
                         , color = '예측확률'
                         , color_continuous_scale='Reds' # or RdBu_r 
                         , labels = {'OCCRNC_VIRS':'원인물질'}
                         , height=330
                         )
                      
            # 그래프 레이아웃 업데이트
            fig.update_layout(
                xaxis_tickangle=-45,
                xaxis_title='',
                yaxis_title='',
                coloraxis_showscale=False  # 색상 범례 숨기기
                    )
                        
            # Streamlit에 차트 표시
            st.plotly_chart(fig, use_container_width=True)

            # 시뮬레이션 전/후 비교 선 그래프
            st.markdown(
                f"""
                <div style='background-color: {new_col}; padding: 5px; border-radius: 5px; text-align: left;border: 2px solid {navy_colors[0]};
'>
                    <h5 style='margin: 15px 0 0 0; color: {grey_colors[3]}; font-weight: bold;'>
                        <span class="tooltip">✔️ 지역별 식중독 발생확률 시뮬레이션 결과
                            <span class="tooltiptext">
                                각 변수 변화에 따른 지역별 식중독 예상 발생확률의 변화를 나타냄
                            </span>
                        </span>
                    </h5>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            before_pred_y = pd.merge(before_y, pred_y)
            before_pred_y['CTPRVN_CD'] = before_pred_y['CTPRVN_CD'].map(region_map)
            before_pred_y = before_pred_y.rename(columns = {'CTPRVN_CD':'지역명'}).melt(id_vars='지역명', var_name='구분',value_name='prob')
            before_pred_y['prob'] = (before_pred_y['prob'] * 100).round(1)

            fig = px.line(before_pred_y, x = '지역명', y = 'prob', color = '구분', markers=True
                          , labels = {'prob':'발생확률'}
                          , color_discrete_sequence=['#81DAF5','#DF0101'])
            fig.update_layout(xaxis_title = '', yaxis_title = '', height = 300
                              , legend_x = 0.4, legend_y = -0.2, legend_orientation = 'h')
            st.plotly_chart(fig, use_container_width=True)

    elif function == '향후 12개월 예측':
            
        with st.sidebar:
            
            st.markdown(
                    """
                    <style>                   
                    div[data-baseweb="select"] > div {
                        background-color: #F4F4F2; 
                        border-radius: 6px; 
                        padding: 0.5px;
                        font-weight: bold;
                        border: 2px solid #495464;
                    }
                    div[data-baseweb="select"] > div > div {
                        color: #495464; 
                        font-size: 16px; 
                    }

                    div[data-baseweb="select"] .css-1uccc91-singleValue {
                        color: #495579; 
                        font-weight: bold;
                    }
                    
                    div[data-baseweb="select"] .css-26l3qy-menu {
                        background-color: #F4F4F2; 
                        font-size: 16px; 
                        color: #495579; 
                        font-weight: bold;
                    }
                    </style>
                    """, unsafe_allow_html=True
                )
            
                            
            st.markdown("<p style='font-size: 20px; color: #495579; font-weight: bold; margin-bottom: -20px; text-align: center'>🔎 예측할 시점을 선택하세요</p>", unsafe_allow_html=True)
            forecast_month = st.selectbox('', options= range(1, 13), format_func= lambda x : f'{x}개월 후')

        forecast_y_by_region = pd.DataFrame()

        for region in model_region_dict.keys():
            model_apply = model_region_dict[str(region)]
            target_forecast_df = data_region_forecast.loc[data_region_forecast['CTPRVN_CD'] == int(region)]
            # st.write(region)
            # st.write(model_apply.feature_names_in_)
            target_forecast_df['예측확률'] = model_apply.predict_proba(target_forecast_df[model_apply.feature_names_in_])[:,1]
            forecast_y_by_region = pd.concat([forecast_y_by_region, target_forecast_df[['OCCRNC_YEAR','OCCRNC_MM','CTPRVN_CD','OCCRNC_REGN','예측확률']]])
        
        forecast_y_by_cause = pd.DataFrame()

        for cause in model_cause_dict.keys():
            model_apply = model_cause_dict[cause]
            target_forecast_df = data_cause_forecast.copy()
            target_forecast_df['OCCRNC_VIRS'] = cause
            target_forecast_df['예측확률'] = model_apply.predict_proba(target_forecast_df[model_apply.feature_names_in_])[:,1]
            forecast_y_by_cause = pd.concat([forecast_y_by_cause, target_forecast_df[['OCCRNC_YEAR','OCCRNC_MM','OCCRNC_VIRS','예측확률']]])

        col1, col2 = st.columns(2)

        with col1:
            
            # 주어진 데이터가 12월 말 기준 데이터이므로 예측 대상 시점과 대상 월이 동일하나, 연중 데이터로 업데이트하는 경우 코드 수정 필요

            pred_y = forecast_y_by_region.loc[forecast_y_by_region['OCCRNC_MM'] == forecast_month, ['CTPRVN_CD', '예측확률']]
            st.markdown(
                f"""
                <style>
                .tooltip {{
                    position: relative;
                    display: inline-block;
                    cursor: pointer;
                }}

                .tooltip .tooltiptext {{
                    visibility: hidden;
                    width: 250px;
                    background-color: {grey_colors[0]};
                    color: {navy_colors[0]};
                    text-align: center;
                    padding: 5px;
                    border-radius: 5px;
                    position: absolute;
                    z-index: 1;
                    bottom: 125%; 
                    left: 50%;
                    margin-left: -125px; 
                    opacity: 0;
                    transition: opacity 0.3s;
                    font-size: 12px;
                }}

                .tooltip:hover .tooltiptext {{
                    visibility: visible;
                    opacity: 1;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown(
                f"""
                <div style='background-color: {grey_colors[0]}; padding: 5px; border-radius: 1px; text-align: center;border: 2px solid {grey_colors[2]};
'>
                    <h5 style='margin: 15px 0 0 0; color: {navy_colors[0]}; font-weight: bold;font-size: 25px;'>
                        <span class="tooltip">🗺️ 식중독 예측 지도: 2023년 {forecast_month}월
                            <span class="tooltiptext">
                                예측 시점별로 예상되는 지역별 식중독 발생확률을 나타냄
                            </span>
                        </span>
                    </h5>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            m = folium.Map(location = [36.2, 128.2], tiles = 'Cartodb Positron'
                           , zoom_start = 7, zoom_control = False, min_zoom=7, max_zoom=7
                           , max_bounds=True
                           , min_lat = 33.5
                           , min_lon = 124.4
                           , max_lat = 38.8
                           , max_lon = 132)

            for feature in jsonResult['features']:
                region_code = feature['properties']['CTPRVN_CD']
                region_data = pred_y[pred_y['CTPRVN_CD'] == int(region_code)] if region_code.isdigit() else None
                if not region_data.empty:
                    feature['properties']['예측확률'] = f'{float(region_data['예측확률']) *100:.1f}%'

                else:

            # 값이 없을 경우 기본값으로 설정
                    feature['properties']['예측확률'] = 0

            folium.Choropleth(
                geo_data = jsonResult
                , name = '식중독 발생확률'
                , data = pred_y
                , columns = ['CTPRVN_CD', '예측확률']
                , key_on='feature.properties.CTPRVN_CD'
                , fill_color = 'RdBu_r'
                , fill_opacity = 0.7
                , line_opacity = 0.3
                , color = 'gray'
                , legend_name = '식중독 발생확률'
            ).add_to(m)

            def on_click(feature):
                return {
                    # 'fillColor': '#ffaf00',
                    'color': 'lightgray',
                    'weight': 1,
                    'fillOpacity': 0.1
                }

            folium.GeoJson(
                data = jsonResult,
                name="OCCRNC_REGN",
                style_function=on_click,
                tooltip=folium.features.GeoJsonTooltip(
                    fields=["CTP_KOR_NM","예측확률"],  # GeoJSON의 구역 이름 필드명
                    aliases=["지 역 명: ","예측확률: "]
                ),
                highlight_function=lambda x: {"weight": 3, "color": "gray"}
            ).add_to(m)

            output = st_folium(m, width=500, height=700, use_container_width=True)

        with col2:
                                                           
            st.markdown(
                f"""
                <style>
                .tooltip {{
                    position: relative;
                    display: inline-block;
                    cursor: pointer;
                }}

                .tooltip .tooltiptext {{
                    visibility: hidden;
                    width: 250px;
                    background-color: {grey_colors[0]};
                    color: {navy_colors[0]};
                    text-align: center;
                    padding: 5px;
                    border-radius: 5px;
                    position: absolute;
                    z-index: 1;
                    bottom: 125%;
                    left: 50%;
                    margin-left: -125px; 
                    opacity: 0;
                    transition: opacity 0.3s;
                    font-size: 12px;
                }}

                .tooltip:hover .tooltiptext {{
                    visibility: visible;
                    opacity: 1;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown(
                f"""
                <div style='background-color: {new_col}; padding: 5px; border-radius: 5px; text-align: left;border: 2px solid {navy_colors[0]};
'>
                    <h5 style='margin: 15px 0 0 0; color: {grey_colors[3]}; font-weight: bold;'>
                        <span class="tooltip">✔️ 원인 물질별 발생확률 
                            <span class="tooltiptext">
                                각 시점에 따른 원인물질별 식중독 예상 발생확률을 나타냄 <br>
                                확률이 높을수록 식중독 발생 시 해당 원인물질에 기인할 가능성이 큰 것을 의미
                            </span>
                        </span>
                    </h5>
                </div>
                """,
                unsafe_allow_html=True
            )
                                                                       
            forecast_y_by_cause_month =  forecast_y_by_cause.loc[forecast_y_by_cause['OCCRNC_MM'] == forecast_month].sort_values(by = '예측확률', ascending = False)
            forecast_y_by_cause_month['예측확률'] = (forecast_y_by_cause_month['예측확률'] * 100).round(1)

            # Plotly로 막대 그래프 생성 (발생 건수에 따른 색상 변경)
            fig = px.bar(forecast_y_by_cause_month
                         , x = 'OCCRNC_VIRS'
                         , y = '예측확률'
                         , color = '예측확률'
                         , color_continuous_scale='Reds' # or RdBu_r
                         , labels = {'OCCRNC_VIRS':'원인물질'}
                         , height=330
                         )
                      
            # 그래프 레이아웃 업데이트
            fig.update_layout(
                xaxis_tickangle=-45,
                xaxis_title='',
                yaxis_title='',
                coloraxis_showscale=False  # 색상 범례 숨기기
                    )

            st.plotly_chart(fig, use_container_width=True)

            # 예측 전/후 비교 선 그래프
            forecast_y_by_region_month = forecast_y_by_region.loc[forecast_y_by_region['OCCRNC_MM'] == forecast_month, ['CTPRVN_CD','예측확률']].rename(columns = {'예측확률':'예측결과'})
            forecast_y_by_region_month['CTPRVN_CD'] = forecast_y_by_region_month['CTPRVN_CD'].astype(str)
    
                      
            st.markdown(
                f"""
                <div style='background-color: {new_col}; padding: 5px; border-radius: 5px; text-align: left;border: 2px solid {navy_colors[0]};
'>
                    <h5 style='margin: 15px 0 0 0; color: {grey_colors[3]}; font-weight: bold;'>
                        <span class="tooltip">✔️ 지역별 예측 전/후 식중독 발생확률 비교 
                            <span class="tooltiptext">
                                시점 변화에 따른 지역별 식중독 예상 발생확률의 변화를 나타냄
                            </span>
                        </span>
                    </h5>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            before_pred_y = pd.merge(before_y, forecast_y_by_region_month, on = 'CTPRVN_CD')
            before_pred_y['CTPRVN_CD'] = before_pred_y['CTPRVN_CD'].map(region_map)
            before_pred_y = before_pred_y.rename(columns = {'CTPRVN_CD':'지역명'}).melt(id_vars='지역명', var_name='구분',value_name='prob')
            before_pred_y['prob'] = (before_pred_y['prob'] *100).round(1)

            fig = px.line(before_pred_y
                          , x = '지역명'
                          , y = 'prob'
                          , color = '구분'
                          , markers=True
                          , labels = {'prob':'예측확률'}
                          , color_discrete_sequence=['#81DAF5','#DF0101'])
            fig.update_layout(xaxis_title = '', yaxis_title = '', height = 300
                              , legend_x = 0.4, legend_y = -0.2, legend_orientation = 'h')
            st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()