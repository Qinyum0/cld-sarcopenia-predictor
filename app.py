#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# 必须在所有Streamlit命令之前设置页面配置
st.set_page_config(
    page_title="Sarcopenia Risk Prediction in CLD Patients",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载预训练模型
try:
    # 确保model.pkl文件存在于同一目录
    best_rf_model = joblib.load("cld_model.pkl")  
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()  # 如果模型加载失败则停止应用

def predict_prevalence(patient_data):
    """使用预训练模型进行预测"""
    try:
        input_df = pd.DataFrame([patient_data])
        # 确保输入字段与模型训练时完全一致
        proba = best_rf_model.predict_proba(input_df)[0]
        prediction = best_rf_model.predict(input_df)[0]
        return prediction, proba
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

def main():
    st.title('Sarcopenia Risk Prediction in CLD Patients')
    st.markdown("""
    This tool is used to predict the risk of sarcopenia in patients with chronic lung disease(CLD).
    """)
    
    # 侧边栏输入
    st.sidebar.header('Patient Parameters')
    age = st.sidebar.slider('Age', 18, 100, 50)
    education = st.sidebar.selectbox('Educational level', ['Below junior high school level', 'Junior high school and above'])
    smoke = st.sidebar.radio('Smoking Status', ['No', 'Yes'])
    drink = st.sidebar.radio('History of Drinking Alcohol', ['No', 'Yes'])
    hypertension = st.sidebar.radio('Hypertension', ['No', 'Yes'])
    dyslipidemia = st.sidebar.radio('Dyslipidemia', ['No', 'Yes'])
    diabetes = st.sidebar.radio('Diabetes', ['No', 'Yes'])
    kidney = st.sidebar.radio('Kidney Disease', ['No', 'Yes'])
    stomach = st.sidebar.radio('Stomach Disease', ['No', 'Yes'])
    psychiatric = st.sidebar.radio('Psychiatric Disease', ['No', 'Yes'])
    memory = st.sidebar.radio('Memory Disorder', ['No', 'Yes'])
    arthritis = st.sidebar.radio('Arthritis', ['No', 'Yes'])
    asthma = st.sidebar.radio('Asthma', ['No', 'Yes'])
    
       
    if st.sidebar.button('Predict'):
        patient_data = {
            'Age': age,
            'Educational level': education,
            'Smoking Status': smoke,
            'History of Drinking Alcohol': drink,
            'Hypertension': hypertension,
            'Dyslipidemia': dyslipidemia,
            'Diabetes': diabetes,
            'Kidney Disease': kidney,
            'Stomach Disease': stomach,
            'Psychiatric Disease': psychiatric,
            'Memory Disorder': memory,
            'Arthritis': arthritis,
            'Asthma': asthma
        }
        
        prediction, proba = predict_prevalence(patient_data)
        
        if prediction is not None:
            st.subheader('Prediction Results')
            if prediction == 1:
                st.error(f'High Risk: Sarcopenia probability {proba[1]*100:.2f}%')
            else:
                st.success(f'Low Risk: Sarcopenia probability {proba[0]*100:.2f}%')
            
            st.progress(proba[1])
            st.write(f'Survival: {proba[0]*100:.2f}% | Sarcopenia: {proba[1]*100:.2f}%')

if __name__ == '__main__':
    main()


# In[2]:




# In[ ]:




