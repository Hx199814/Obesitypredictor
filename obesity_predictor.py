import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import seaborn as sns

# ========== 新增：配置Matplotlib支持中文 ==========
# 配置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定黑体
plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

# 设置Seaborn样式，去除背景栅格
sns.set_style("white", {"font.sans-serif": ['SimHei', 'Droid Sans Fallback']})
# ================================================

# Load the trained model
model = joblib.load('CatBoost.pkl')  # 加载训练好的CatBoost模型

# Define the feature options
GENDER_options = {
    1: '男生', 
    2: '女生'
}

AP_options = {
    1: '优异', 
    2: '中等偏上',  
    3: '中等',
    4: '中等偏下',
    5: '差',
}

FES_options = {
    1: '很好', 
    2: '较好', 
    3: '一般',
    4: '较差', 
    5: '很差'  # 修正：将重复的3改为5
}

FrFF_options = {
    1: '从来不吃',  
    2: '少于每天1次',  
    3: '每天1次',
    4: '每天2次及以上'
}

DFT_options = {
    1: '从来不吃或少于每天1种',  
    2: '每天1种',  
    3: '每天2种',
    4: '每天3次及以上'
}

WMED_options = {
    1: '0天',  
    2: '1天',  
    3: '2天',
    4: '3天',
    5: '4天',  
    6: '5天',  
    7: '6天',
    8: '7天'
}

PEC_options = {
    1: '0节',  
    2: '1节',  
    3: '2节',
    4: '3节',
    5: '4节',  
    6: '5节及以上'
}

TVDU_options = {
    1: '我没有看过',  
    2: '不到1小时',  
    3: '1-2（不含2）小时',
    4: '2-3（不含3）小时',
    5: '3-4（不含4）小时',  
    6: '4小时及以上'
}

CDU_options = {
    1: '我没有看过',  
    2: '不到1小时',  
    3: '1-2（不含2）小时',
    4: '2-3（不含3）小时',
    5: '3-4（不含4）小时',  
    6: '4小时及以上'
}

D1_options = {
    1: '没有或偶尔',  
    2: '有时',  
    3: '时常或一半时间',
    4: '多数时间或持续',
    5: '不清楚'
}

D5_options = {
    1: '没有或偶尔',  
    2: '有时',  
    3: '时常或一半时间',
    4: '多数时间或持续',
    5: '不清楚'
}

D16_options = {
    1: '没有或偶尔',  
    2: '有时',  
    3: '时常或一半时间',
    4: '多数时间或持续',
    5: '不清楚'
}

# Streamlit UI
st.title("学生肥胖风险预测")  # 修改为中文标题

# Sidebar for input options
st.sidebar.header("请输入学生信息")  # 侧边栏输入样本数据

# 按照模型训练时的特征顺序收集输入
GENDER = st.sidebar.selectbox("性别:", options=list(GENDER_options.keys()), format_func=lambda x: GENDER_options[x])

AP = st.sidebar.selectbox("与同学相比，您如何评价自己的学业表现:", options=list(AP_options.keys()), format_func=lambda x: AP_options[x])

FES = st.sidebar.selectbox("您如何描述家庭经济状况:", options=list(FES_options.keys()), format_func=lambda x: FES_options[x])

FrFF = st.sidebar.selectbox("过去七天您吃新鲜水果的次数:", options=list(FrFF_options.keys()), format_func=lambda x: FrFF_options[x])

DFT = st.sidebar.selectbox("您通常每天吃多少种新鲜水果:", options=list(DFT_options.keys()), format_func=lambda x: DFT_options[x])

WMED = st.sidebar.selectbox("工作日中高强度运动天数（每天≥60分钟）:", options=list(WMED_options.keys()), format_func=lambda x: WMED_options[x])

PEC = st.sidebar.selectbox("每周体育课节数:", options=list(PEC_options.keys()), format_func=lambda x: PEC_options[x])

TVDU = st.sidebar.selectbox("平均每天看电视时间:", options=list(TVDU_options.keys()), format_func=lambda x: TVDU_options[x])

CDU = st.sidebar.selectbox("平均每天使用电脑时间:", options=list(CDU_options.keys()), format_func=lambda x: CDU_options[x])

D1 = st.sidebar.selectbox("以前从不困扰我的事情现在让我烦恼:", options=list(D1_options.keys()), format_func=lambda x: D1_options[x])

D5 = st.sidebar.selectbox("我发现很难集中注意力:", options=list(D5_options.keys()), format_func=lambda x: D5_options[x])

D16 = st.sidebar.selectbox("我过着幸福的生活:", options=list(D16_options.keys()), format_func=lambda x: D16_options[x])

DST = st.sidebar.selectbox("每日睡眠时长（小时）:", options=[6,7,8,9,10,11,12], format_func=lambda x: f"{x}小时")

# Process the input and make a prediction
# 按照模型训练时的特征顺序收集所有输入
feature_values = [GENDER, AP, FES, FrFF, DFT, WMED, PEC, TVDU, CDU, D1, D5, D16, DST]
features = np.array([feature_values])  # 转换为NumPy数组

if st.button("开始预测"):  # 如果点击了预测按钮
    try:
        # Predict the class and probabilities
        predicted_class = model.predict(features)[0]  # 修正：使用[0]而不是[1]
        predicted_proba = model.predict_proba(features)[0]  # 预测肥胖的概率

        # Display the prediction results
        st.write(f"**预测结果:** {'肥胖风险高' if predicted_class == 1 else '肥胖风险低'}")  # 显示肥胖类别
        
        # Generate advice based on the prediction result
        probability = predicted_proba[predicted_class] * 100  # 根据预测类别获取对应的概率，并转化为百分比

        if predicted_class == 1:  # 如果预测为肥胖高风险
            advice = (
                f"根据我们的模型预测，您的肥胖风险较高。"
                f"肥胖风险概率为 {probability:.1f}%。"
                "建议：增加体育锻炼，改善饮食习惯，控制屏幕使用时间，保证充足睡眠。"
            ) 
        else: 
            advice = (
                f"根据我们的模型预测，您的肥胖风险较低。"
                f"非肥胖概率为 {probability:.1f}%。"
                "建议：继续保持健康的生活方式，均衡饮食，适量运动。"
            ) 

        st.write(advice)  # 显示建议

        # Visualize the prediction probabilities
        sample_prob = {
            '非肥胖': predicted_proba[0],  # 类别0的概率
            '肥胖': predicted_proba[1]  # 类别1的概率
        }

        # Set figure size
        plt.figure(figsize=(10, 3))  # 设置图形大小

        # Create bar chart
        bars = plt.barh(['非肥胖', '肥胖'], 
                        [sample_prob['非肥胖'], sample_prob['肥胖']], 
                        color=['#512b58', '#fe346e'])  # 绘制水平条形图

        # Add title and labels, set font bold and increase font size
        plt.title("学生肥胖风险预测概率", fontsize=16, fontweight='bold')  # 添加图表标题，并设置字体大小和加粗
        plt.xlabel("概率", fontsize=12, fontweight='bold')  # 添加X轴标签，并设置字体大小和加粗
        plt.ylabel("类别", fontsize=12, fontweight='bold')  # 添加Y轴标签，并设置字体大小和加粗

        # Add probability text labels, adjust position to avoid overlap, set font bold
        for i, v in enumerate([sample_prob['非肥胖'], sample_prob['肥胖']]):  # 为每个条形图添加概率文本标签
            plt.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=12, color='black', fontweight='bold')  # 设置标签位置、字体加粗

        # Hide other axes (top, right, bottom)
        plt.gca().spines['top'].set_visible(False)  # 隐藏顶部边框
        plt.gca().spines['right'].set_visible(False)  # 隐藏右边框

        # Set x-axis limit to 0-1 for probability
        plt.xlim(0, 1)

        # Show the plot
        st.pyplot(plt)  # 显示图表

    except Exception as e:
        st.error(f"预测过程中出现错误: {str(e)}")
        st.write("请检查模型文件和数据输入是否正确")
