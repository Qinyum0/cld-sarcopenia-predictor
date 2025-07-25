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
    age = st.sidebar.slider('age', 18, 100, 50)
    education = st.sidebar.selectbox('education', ['Below junior high school level', 'Junior high school and above'])
    smoke = st.sidebar.radio('smoke', ['No', 'Yes'])
    drink = st.sidebar.radio('drink', ['No', 'Yes'])
    hypertension = st.sidebar.radio('hypertension', ['No', 'Yes'])
    dyslipidemia = st.sidebar.radio('dyslipidemia', ['No', 'Yes'])
    diabetes = st.sidebar.radio('diabetes', ['No', 'Yes'])
    kidney = st.sidebar.radio('kidney', ['No', 'Yes'])
    stomach = st.sidebar.radio('stomach', ['No', 'Yes'])
    psychiatric = st.sidebar.radio('psychiatric', ['No', 'Yes'])
    memory = st.sidebar.radio('memory', ['No', 'Yes'])
    arthritis = st.sidebar.radio('arthritis', ['No', 'Yes'])
    asthma = st.sidebar.radio('asthma', ['No', 'Yes'])
    
       
    if st.sidebar.button('Predict'):
        patient_data = {
            'age': age,
            'education': 0 if education == 'Below junior high school level' else 1,
            'smoke': 1 if smoke == 'Yes' else 0,
            'drink_alcohol': 1 if drink == 'Yes' else 0,
            'hypertension': 1 if hypertension == 'Yes' else 0,
            'dyslipidemia': 1 if dyslipidemia == 'Yes' else 0,
            'diabetes': 1 if diabetes == 'Yes' else 0,
            'kidney_disease': 1 if kidney == 'Yes' else 0,
            'stomach_disease': 1 if stomach == 'Yes' else 0,
            'psychiatric_disease': 1 if psychiatric == 'Yes' else 0,
            'memory_disease': 1 if memory == 'Yes' else 0,
            'arthritis': 1 if arthritis == 'Yes' else 0,
            'asthma': 1 if asthma == 'Yes' else 0
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




