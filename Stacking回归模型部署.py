# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 16:49:14 2025

@author: 86185
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import joblib
import shap

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 加载模型
model_path = "stacking_regressor_model.pkl"
stacking_regressor = joblib.load(model_path)

# 设置页面配置和标题
st.set_page_config(layout="wide", page_title="Stacking 模型预测与 SHAP 可视化", page_icon="📊")

st.title("📊 Stacking 模型预测与 SHAP 可视化分析")
st.write("""
通过输入特征值进行模型预测，并结合 SHAP 分析结果，了解特征对模型预测的贡献。
""")

# 左侧侧边栏输入区域
st.sidebar.header("特征输入区域")
st.sidebar.write("请输入特征值：")

# 定义特征输入范围
Resilience = st.sidebar.number_input("Resilience (范围: 6-36)", min_value=6, max_value=36, value=18)
Depression = st.sidebar.number_input("Depression (范围: 0-3)", min_value=0, max_value=3, value=3)
Anxiety = st.sidebar.number_input("Anxiety (范围: 0-3)", min_value=0, max_value=3, value=3)
Family_support = st.sidebar.number_input("Family support (范围: 0-10)", min_value=0, max_value=10, value=5)
Age = st.sidebar.number_input("Age (范围: 21-63)", min_value=21, max_value=63, value=21)
Occupation = st.sidebar.selectbox("Occupation", options=["Full-time job", "Part-time job"])
Method_of_delivery = st.sidebar.selectbox("Method of delivery", options=["Vaginal delivery", "Cesarean section"])
Marital_status = st.sidebar.selectbox("Marital status", options=["Married", "Unmarried"])
Educational_degree = st.sidebar.selectbox("Educational degree", options=["Associate degree or below", "Bachelor's degree or above"])
Average_monthly_household_income = st.sidebar.selectbox("Average monthly household income", options=["Average monthly household income less than or equal to 5000 yuan", "Average monthly household income greater than 5000 yuan"])
Medical_insurance = st.sidebar.selectbox("Medical_insurance", options=["No", "Yes"])
Mode_of_conception = st.sidebar.selectbox("Mode of conception", options=["Natural conception", "Assisted conception"])
Pregnancy_complications = st.sidebar.selectbox("Pregnancy complications", options=["Yes", "No"])
Breastfeeding = st.sidebar.selectbox("Breastfeeding", options=["Yes", "No"])
Rooming_in = st.sidebar.selectbox("Rooming-in", options=["Yes", "No"])
Planned_pregnancy = st.sidebar.selectbox("Planned pregnancy", options=["Yes", "No"])
Intrapartum_pain = st.sidebar.number_input("Intrapartum pain (范围: 0-10)", min_value=0, max_value=10, value=5)
Postpartum_pain = st.sidebar.number_input("Postpartum pain (范围: 0-10)", min_value=0, max_value=10, value=5)

# 添加预测按钮
predict_button = st.sidebar.button("进行预测")

# 主页面用于结果展示
if predict_button:
    st.header("预测结果")
    try:
        # 将输入特征转换为模型所需格式
        input_array = np.array([
            Resilience, Depression, Anxiety, Family_support, Age, Intrapartum_pain, Postpartum_pain,
            # 对于分类特征，需要将其转换为数值
            1 if Occupation == "Full-time job" else 0,
            1 if Method_of_delivery == "Vaginal delivery" else 0,
            1 if Marital_status == "Married" else 0,
            1 if Educational_degree == "Associate degree or below" else 0,
            1 if Average_monthly_household_income == "Average monthly household income less than or equal to 5000 yuan" else 0,
            1 if Medical_insurance == "No" else 0,
            1 if Mode_of_conception == "Natural conception" else 0,
            1 if Pregnancy_complications == "Yes" else 0,
            1 if Breastfeeding == "Yes" else 0,
            1 if Rooming_in == "Yes" else 0,
            1 if Planned_pregnancy == "Yes" else 0
        ]).reshape(1, -1)

        # 模型预测
        prediction = stacking_regressor.predict(input_array)[0]

        # 显示预测结果
        st.success(f"预测结果：{prediction:.2f}")

        # 生成 SHAP 瀑布图
        st.header("SHAP 瀑布图分析")
        st.write("根据用户输入的数值生成的单个样本 SHAP 瀑布图，用于解释特征对预测结果的贡献。")
        st.write("注意：由于回归模型不能输出预测概率，因此 SHAP 分析基于模型的预测值进行解释。")

        # 创建SHAP解释器并计算SHAP值
        try:
            # 尝试使用TreeExplainer (适用于树模型)
            explainer = shap.TreeExplainer(stacking_regressor)
            shap_values = explainer.shap_values(input_array)
        except:
            # 如果TreeExplainer失败，使用通用的KernelExplainer
            explainer = shap.KernelExplainer(stacking_regressor.predict, input_array)
            shap_values = explainer.shap_values(input_array)

        # 特征名称列表
        feature_names = [
            "Resilience", "Depression", "Anxiety", "Family_support", "Age", 
            "Intrapartum_pain", "Postpartum_pain", "Occupation", 
            "Method_of_delivery", "Marital_status", "Educational_degree",
            "Average_monthly_household_income", "Medical_insurance",
            "Mode_of_conception", "Pregnancy_complications", "Breastfeeding",
            "Rooming_in", "Planned_pregnancy"
        ]

        # 确保shap_values格式正确
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # 创建画布
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制SHAP瀑布图
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0], 
                base_values=explainer.expected_value,
                data=input_array[0],
                feature_names=feature_names
            ),
            max_display=18,
            show=False
        )
        
        # 显示图形
        st.pyplot(fig)

    except Exception as e:
        st.error(f"预测时发生错误：{e}")
        
# 可视化展示
st.header("SHAP 可视化分析")
st.write("""
以下图表展示了模型的 SHAP 分析结果，包括第一层基学习器、第二层元学习器以及整个 Stacking 模型的特征贡献。
""")


# 整体 Stacking 模型 SHAP 可视化
st.subheader("3. 整体 Stacking 模型")
st.write("整个 Stacking 模型的特征贡献分析。")
overall_img = "Based on the overall feature contribution analysis of SHAP to the stacking model.png"
try:
    img3 = Image.open(overall_img)
    st.image(img3, caption="整体 Stacking 模型的 SHAP 贡献分析", use_container_width=True)
except FileNotFoundError:
    st.warning("未找到整体 Stacking 模型的 SHAP 图像文件。")

