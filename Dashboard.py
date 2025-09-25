
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NeuroPredict Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #ffffff;
        text-align: center;
        padding: 1rem;
        background-color: #0e2a47;
        border: 1px solid #cccccc;
        border-radius: 8px;
        margin-bottom: 2rem;
        width : 100%;
    }
    .risk-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem auto;
    }
    .high-risk {
        background-color: #ffebee;
        color: #c62828;
        border: 3px solid #c62828;
    }
    .medium-risk {
        background-color: #fff3e0;
        color: #ef6c00;
        border: 3px solid #ef6c00;
    }
    .low-risk {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 3px solid #2e7d32;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dddddd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for model
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.data_loaded = False

# Generate sample data for demonstration
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n_samples = 5000
    
    data = pd.DataFrame({
        'age': np.random.normal(55, 15, n_samples).clip(18, 90).astype(int),
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.45, 0.55]),
        'hypertension': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
        'heart_disease': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'ever_married': np.random.choice(['Yes', 'No'], n_samples, p=[0.65, 0.35]),
        'work_type': np.random.choice(['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'], 
                                     n_samples, p=[0.4, 0.25, 0.2, 0.1, 0.05]),
        'Residence_type': np.random.choice(['Urban', 'Rural'], n_samples, p=[0.55, 0.45]),
        'avg_glucose_level': np.random.normal(110, 40, n_samples).clip(50, 300),
        'bmi': np.random.normal(28, 5, n_samples).clip(15, 50),
        'smoking_status': np.random.choice(['formerly smoked', 'never smoked', 'smokes', 'Unknown'], 
                                          n_samples, p=[0.2, 0.4, 0.25, 0.15])
    })
    
    # Create stroke outcome with some correlation to risk factors
    stroke_prob = 0.05  # Base probability
    stroke = []
    for idx, row in data.iterrows():
        prob = stroke_prob
        if row['age'] > 65: prob += 0.1
        if row['hypertension'] == 1: prob += 0.15
        if row['heart_disease'] == 1: prob += 0.15
        if row['avg_glucose_level'] > 140: prob += 0.1
        if row['bmi'] > 30: prob += 0.05
        if row['smoking_status'] == 'smokes': prob += 0.1
        
        stroke.append(1 if np.random.random() < prob else 0)
    
    data['stroke'] = stroke
    return data

# Train a simple model
@st.cache_resource
def train_model(data):
    # Prepare features
    df_model = data.copy()
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_married = LabelEncoder()
    le_work = LabelEncoder()
    le_residence = LabelEncoder()
    le_smoking = LabelEncoder()
    
    df_model['gender_encoded'] = le_gender.fit_transform(df_model['gender'])
    df_model['ever_married_encoded'] = le_married.fit_transform(df_model['ever_married'])
    df_model['work_type_encoded'] = le_work.fit_transform(df_model['work_type'])
    df_model['Residence_type_encoded'] = le_residence.fit_transform(df_model['Residence_type'])
    df_model['smoking_status_encoded'] = le_smoking.fit_transform(df_model['smoking_status'])
    
    # Select features for model
    features = ['age', 'gender_encoded', 'hypertension', 'heart_disease', 
                'ever_married_encoded', 'work_type_encoded', 'Residence_type_encoded',
                'avg_glucose_level', 'bmi', 'smoking_status_encoded']
    
    X = df_model[features]
    y = df_model['stroke']
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    return model, le_gender, le_married, le_work, le_residence, le_smoking

# Load data and train model
data = load_sample_data()
model, le_gender, le_married, le_work, le_residence, le_smoking = train_model(data)

# Sidebar navigation
st.sidebar.markdown('<div class="sidebar-header"> Pages:</div>', unsafe_allow_html=True)


page = st.sidebar.radio(
    "Select Page:",
    ["Home", "Descriptive Analytics", "Diagnostic Analytics", "Risk Prediction", 
     "What-If/Preventive"],
    label_visibility="collapsed"
)

# Header
st.markdown('<h1 class="main-header"> Stroke Risk Prediction Dashboard 🧠 </h1>', unsafe_allow_html=True)

# Home Page
if page == "Home":
    st.markdown("## Guide to the Dashboard")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Welcome to NeuroPredict
        
        This comprehensive dashboard helps healthcare professionals and individuals assess stroke risk 
        using machine learning algorithms and data visualization.
        
        #### Available Features:
        
        1. Descriptive Analytics :
        - View key statistics and analysis about the stroke dataset
        - Understand demographic distributions and risk factor prevalence
        
        2. Diagnostic Analytics:
        - Explore statistical relationships between variables
        - Identify important risk factors through correlation analysis
        
        3. Risk Prediction :
        - Input patient data to generate personalized stroke risk assessments
        - Get immediate risk classification (Low/Medium/High)
        
        4. What-If/Preventive :
        - Explore prevention strategies and lifestyle modifications
        - Understand how changing risk factors affects stroke probability
        
        5. About :
        - Learn more about the project and methodology
        - View model performance metrics and limitations
        """)
    
    with col2:
        
        st.success("""
        ### Risk Factors Considered: 
        - Age
        - Gender
        - Hypertension
        - Heart Disease
        - Glucose Level
        - BMI
        - Smoking Status
        - Marital Status
        - Work Type
        - Residence Type
        """)

# Descriptive Analysis Page
elif page == "Descriptive Analytics":
    st.markdown("## Descriptive Analytics")
    st.markdown("This page shows key analysis about the dataset we trained")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Demographics", "Risk Factors", "Correlations"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            fig_age = px.histogram(data, x='age', nbins=20, title='Age Distribution',
                                   color_discrete_sequence=['#1f77b4'])
            fig_age.update_layout(height=400)
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            # Gender distribution
            gender_counts = data['gender'].value_counts()
            fig_gender = px.pie(values=gender_counts.values, names=gender_counts.index,
                               title='Gender Distribution', color_discrete_map={'Male': '#1f77b4', 'Female': '#ff7f0e'})
            fig_gender.update_layout(height=400)
            st.plotly_chart(fig_gender, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Stroke distribution
            stroke_counts = data['stroke'].value_counts()
            fig_stroke = px.pie(values=stroke_counts.values, 
                               names=['No Stroke', 'Stroke'],
                               title='Stroke Distribution',
                               color_discrete_sequence=['#2ca02c', '#d62728'])
            fig_stroke.update_layout(height=400)
            st.plotly_chart(fig_stroke, use_container_width=True)
        
        with col2:
            # Heart Disease and Stroke
            fig_heart = px.bar(data.groupby(['heart_disease', 'stroke']).size().reset_index(name='count'),
                              x='heart_disease', y='count', color='stroke',
                              title='Heart Disease and Stroke',
                              labels={'heart_disease': 'Heart Disease', 'count': 'Count'})
            fig_heart.update_layout(height=400)
            st.plotly_chart(fig_heart, use_container_width=True)
    
    with tab3:
        # Correlation matrix for numerical features
        numerical_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']
        corr_matrix = data[numerical_cols].corr()
        
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           title='Correlation Matrix of Risk Factors',
                           color_continuous_scale='RdBu')
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)

# Diagnostic Page
elif page == "Diagnostic Analytics":
    st.markdown("## Diagnostic Analytics")
    st.markdown("This page includes statistical analysis about the stroke data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Age", f"{data['age'].mean():.1f} years", 
                 f"±{data['age'].std():.1f}")
    
    with col2:
        st.metric("Hypertension Rate", 
                 f"{(data['hypertension'].mean() * 100):.1f}%",
                 f"{len(data[data['hypertension']==1])} patients")
    
    with col3:
        st.metric("Stroke Rate", 
                 f"{(data['stroke'].mean() * 100):.1f}%",
                 f"{len(data[data['stroke']==1])} cases")
    
    st.markdown("### Risk Factor Analysis")
    
    # Create subplots for risk factor analysis
    risk_factors = ['hypertension', 'heart_disease', 'smoking_status', 'work_type']
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=risk_factors)
    
    # Hypertension vs Stroke
    hyp_data = data.groupby(['hypertension', 'stroke']).size().reset_index(name='count')
    fig.add_trace(go.Bar(x=['No Hypertension', 'Hypertension'], 
                         y=hyp_data[hyp_data['stroke']==1]['count'].values,
                         name='Stroke Cases'), row=1, col=1)
    
    # Heart Disease vs Stroke
    heart_data = data.groupby(['heart_disease', 'stroke']).size().reset_index(name='count')
    fig.add_trace(go.Bar(x=['No Heart Disease', 'Heart Disease'], 
                         y=heart_data[heart_data['stroke']==1]['count'].values,
                         name='Stroke Cases'), row=1, col=2)
    
    # Smoking Status
    smoke_data = data.groupby(['smoking_status', 'stroke']).size().reset_index(name='count')
    fig.add_trace(go.Bar(x=smoke_data[smoke_data['stroke']==1]['smoking_status'], 
                         y=smoke_data[smoke_data['stroke']==1]['count'],
                         name='Stroke Cases'), row=2, col=1)
    
    # Work Type
    work_data = data.groupby(['work_type', 'stroke']).size().reset_index(name='count')
    fig.add_trace(go.Bar(x=work_data[work_data['stroke']==1]['work_type'], 
                         y=work_data[work_data['stroke']==1]['count'],
                         name='Stroke Cases'), row=2, col=2)
    
    fig.update_layout(height=700, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Risk Prediction Page
elif page == "Risk Prediction":
    st.markdown("## Risk Prediction")
    st.markdown("Generate stroke risk prediction based on patient data entered")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Patient Data for Prediction")
        
        # Input fields
        col_a, col_b = st.columns(2)
        
        with col_a:
            age = st.number_input("Age", min_value=1, max_value=120, value=42, step=1)
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
            marital_status = st.selectbox("Marital Status", ["Married", "Single"])
            work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        
        with col_b:
            residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
            avg_glucose = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=228.69, step=0.1)
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=34.4, step=0.1)
            smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
            gender = st.selectbox("Gender", ["Male", "Female"])
        
        # Predict button
        if st.button("🔮 Predict Risk", type="primary"):
            # Prepare input data
            input_data = pd.DataFrame({
                'age': [age],
                'gender_encoded': [le_gender.transform([gender])[0]],
                'hypertension': [1 if hypertension == "Yes" else 0],
                'heart_disease': [1 if heart_disease == "Yes" else 0],
                'ever_married_encoded': [le_married.transform([marital_status])[0] if marital_status in ["Married", "Single"] else 0],
                'work_type_encoded': [le_work.transform([work_type])[0]],
                'Residence_type_encoded': [le_residence.transform([residence_type])[0]],
                'avg_glucose_level': [avg_glucose],
                'bmi': [bmi],
                'smoking_status_encoded': [le_smoking.transform([smoking_status])[0]]
            })
            
            # Make prediction
            prediction_proba = model.predict_proba(input_data)[0]
            risk_percentage = prediction_proba[1] * 100
            
            # Store in session state for display
            st.session_state.prediction = risk_percentage
    
    with col2:
        st.markdown("### Output")
        
        if 'prediction' in st.session_state:
            risk = st.session_state.prediction
            
            if risk < 30:
                risk_level = "Low Risk"
                risk_class = "low-risk"
                color = "#2e7d32"
            elif risk < 70:
                risk_level = "Medium Risk"
                risk_class = "medium-risk"
                color = "#ef6c00"
            else:
                risk_level = "High Risk"
                risk_class = "high-risk"
                color = "#c62828"
            
            st.markdown(f'<div class="risk-box {risk_class}">{risk_level}<br><span style="font-size: 1.2rem;">{risk:.1f}% Risk</span></div>', 
                       unsafe_allow_html=True)
            
            # Risk gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Stroke Risk Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 30], 'color': "#e8f5e9"},
                        {'range': [30, 70], 'color': "#fff3e0"},
                        {'range': [70, 100], 'color': "#ffebee"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.info("Enter patient data and click 'Predict Risk' to see results")

# What-If/Preventive Page
elif page == "What-If/Preventive":
    st.markdown("## What-If Analysis & Prevention Measures")
    st.markdown("Explore how lifestyle changes can reduce stroke risk")
    
    tab1, tab2 = st.tabs(["What-If Scenarios", "Prevention Guidelines"])
    
    with tab1:
        st.markdown("### Interactive Risk Calculator")
        st.info("Adjust the sliders to see how different factors affect stroke risk")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Interactive sliders
            age_sim = st.slider("Age", 18, 90, 55)
            glucose_sim = st.slider("Average Glucose Level", 50, 300, 110)
            bmi_sim = st.slider("BMI", 15, 50, 28)
            
            col_a, col_b = st.columns(2)
            with col_a:
                hyp_sim = st.checkbox("Hypertension", value=False)
                heart_sim = st.checkbox("Heart Disease", value=False)
            with col_b:
                smoke_sim = st.selectbox("Smoking Status", 
                                        ["never smoked", "formerly smoked", "smokes"],
                                        index=0)
            
            # Calculate simulated risk
            base_risk = 5
            sim_risk = base_risk
            
            if age_sim > 65: sim_risk += 15
            elif age_sim > 50: sim_risk += 8
            
            if glucose_sim > 140: sim_risk += 12
            elif glucose_sim > 110: sim_risk += 5
            
            if bmi_sim > 30: sim_risk += 8
            elif bmi_sim > 25: sim_risk += 4
            
            if hyp_sim: sim_risk += 20
            if heart_sim: sim_risk += 18
            if smoke_sim == "smokes": sim_risk += 15
            elif smoke_sim == "formerly smoked": sim_risk += 5
            
            sim_risk = min(sim_risk, 95)  # Cap at 95%
        
        with col2:
            # Display simulated risk
            if sim_risk < 30:
                color = "🟢"
                message = "Low Risk - Keep maintaining healthy habits!"
            elif sim_risk < 70:
                color = "🟡"
                message = "Medium Risk - Consider lifestyle changes"
            else:
                color = "🔴"
                message = "High Risk - Immediate intervention recommended"
            
            st.markdown(f"### {color} Risk Level: {sim_risk}%")
            st.markdown(f"**{message}**")
            
            # Recommendations based on risk factors
            st.markdown("### Recommendations:")
            recommendations = []
            if glucose_sim > 140:
                recommendations.append("- Monitor and control blood sugar levels")
            if bmi_sim > 30:
                recommendations.append("- Achieve healthy weight through diet and exercise")
            if smoke_sim == "smokes":
                recommendations.append("- Quit smoking immediately")
            if hyp_sim:
                recommendations.append("- Take prescribed hypertension medications")
            if not recommendations:
                recommendations.append("- Continue healthy lifestyle choices")
            
            for rec in recommendations:
                st.markdown(rec)
    
    with tab2:
        st.markdown("### Prevention Guidelines")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Lifestyle Modifications
            - **Diet:** Follow a Mediterranean or DASH diet
            - **Exercise:** At least 150 minutes of moderate activity per week
            - **Weight:** Maintain BMI between 18.5-24.9
            - **Sleep:** 7-9 hours of quality sleep nightly
            - **Stress:** Practice stress management techniques
            
            #### Medical Management
            - Regular blood pressure monitoring
            - Cholesterol level checks
            - Diabetes screening and management
            - Atrial fibrillation detection
            - Medication adherence
            """)
        
        with col2:
            st.markdown("""
            #### Risk Factor Control
            - **Blood Pressure:** Keep below 120/80 mmHg
            - **Cholesterol:** LDL below 100 mg/dL
            - **Blood Sugar:** HbA1c below 7%
            - **Smoking:** Complete cessation
            - **Alcohol:** Limit to moderate consumption
            
            #### Warning Signs (F.A.S.T.)
            - **F**ace drooping
            - **A**rm weakness
            - **S**peech difficulty
            - **T**ime to call emergency services
            """)
        
        st.warning("""
        **Important:** These are general guidelines. Always consult with healthcare 
        professionals for personalized medical advice and treatment plans.
        """)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    © 2024 NeuroPredict - Group 6 Dashboard | Stroke Risk Prediction System
</div>
""", unsafe_allow_html=True)