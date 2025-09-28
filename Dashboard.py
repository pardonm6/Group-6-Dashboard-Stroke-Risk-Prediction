import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

# Loading dataset (making sure the CSV is in the same folder as Dashboard.py)


df = pd.read_csv("jupyter-notebooks/stroke_data_cleaned.csv")


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

# Use the cleaned CSV (uploaded) instead of synthetic data
@st.cache_data
def load_data():
    df = pd.read_csv("jupyter-notebooks/stroke_data_cleaned.csv")
    return df

# Loading trained model
@st.cache_resource
def load_model():
    with open('stroke_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Initialize scaler for new predictions
@st.cache_resource
def initialize_scaler(data):
    from sklearn.preprocessing import MinMaxScaler  
    scaler = MinMaxScaler()  
    numerical_cols = ['age', 'avg_glucose_level', 'bmi']
    scaler.fit(data[numerical_cols])
    return scaler, numerical_cols

# Load everything
data = load_data()
data_original = pd.read_csv("jupyter-notebooks/stroke_data_cleaned_unnormalized.csv")
model = load_model()
scaler, numerical_cols = initialize_scaler(data)

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
        using machine learning algorithms and data visualization.""")
        
        st.markdown("### Available Features:")
        
        # Create expanders for each feature
        with st.expander("1. Descriptive Analytics"):
            st.write("""
            • Understand demographic distributions and risk factor prevalence.
            """)
        
        with st.expander("2. Diagnostic Analytics"):
            st.write("""
            • Explore statistical relationships between variables.\n
            • Identify important risk factors through correlation analysis
            """)
        
        with st.expander("3. Risk Prediction"):
            st.write("""
            • Input patient data to generate personalized stroke risk assessments.\n
            • Get immediate risk classification (Low/Medium/High)
            """)
        
        with st.expander("4. What-If/Preventive"):
            st.write("""
            • Explore prevention strategies and lifestyle modifications.\n
            • Understand how changing risk factors affects stroke probability.
            """)
        
        with st.expander("5. About"):
            st.write("""
            • Learn more about the project methodology and future directions.
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
    st.markdown("Explore patterns and distributions in the stroke dataset")
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["Demographics", "Risk Factors", "Distributions",])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution with stroke overlay
            fig_age = px.histogram(data_original, x='age', color='stroke', 
                                   nbins=30, barmode='overlay',
                                   title='Age Distribution by Stroke Status',
                                   labels={'age': 'Age (years)', 'count': 'Count', 'stroke': 'Stroke'},
                                   color_discrete_map={0: "#c2dbeb", 1: "#c92210"},
                                   opacity=0.5)
            fig_age.update_layout(height=400)
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            # Age box plot comparing stroke vs no stroke
            fig_box = px.box(data_original, y='age', x='stroke', 
                    title='Age Distribution: Stroke vs No Stroke',
                    labels={'age': 'Age (years)', 'stroke': 'Had Stroke'},
                    color='stroke',
                    color_discrete_map={0: '#3498db', 1: '#e74c3c'})
            fig_box.update_xaxes(ticktext=['No Stroke', 'Stroke'], tickvals=[0, 1])  # Fixed: update_xaxes
            fig_box.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
    
    with tab2:
        # Risk Factor Heatmap
        st.markdown("### Risk Factor Prevalence by Stroke Status")
        
        

        # Calculate percentages for each risk factor
        risk_data = []
        for stroke_val in [0, 1]:
            subset = data_original[data_original['stroke'] == stroke_val]
            risk_data.append({
                'Stroke Status': 'Stroke' if stroke_val == 1 else 'No Stroke',
                'Hypertension': (subset['hypertension'].mean() * 100),
                'Heart Disease': (subset['heart_disease'].mean() * 100),
                'High Glucose (>140)': ((subset['avg_glucose_level'] > 140).mean() * 100),
                'Obesity (BMI>30)': ((subset['bmi'] > 30).mean() * 100),
                'Ever Married': (subset['ever_married'].mean() * 100)
            })
        
        risk_df = pd.DataFrame(risk_data)
        
        # Create heatmap
        fig_heat = px.imshow(risk_df.set_index('Stroke Status').T,
                            text_auto='.1f',
                            title='Risk Factor Prevalence (%) by Stroke Status',
                            color_continuous_scale='RdYlBu_r',
                            aspect='auto')
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, use_container_width=True)
        
        # Comparative bar chart
        col1, col2 = st.columns(2)
        
        with col1:
            # Smoking status distribution
            smoke_stroke = data_original.groupby(['smoking_status', 'stroke']).size().unstack(fill_value=0)
            smoke_pct = smoke_stroke.div(smoke_stroke.sum(axis=1), axis=0) * 100
            
            fig_smoke = px.bar(smoke_pct.reset_index(), 
                              x='smoking_status', y=1,
                              title='Stroke Rate by Smoking Status',
                              labels={'1': 'Stroke Rate (%)', 'smoking_status': 'Smoking Status'},
                              color_discrete_sequence=['#e74c3c'])
            fig_smoke.update_xaxes(ticktext=['Unknown', 'Never smoked', 'Formerly smoked', 'Smokes'], 
                          tickvals=[0, 1, 2, 3])
            fig_smoke.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_smoke, use_container_width=True)
        
        with col2:
            # Work type distribution
            work_stroke = data_original.groupby(['work_type', 'stroke']).size().unstack(fill_value=0)
            work_pct = work_stroke.div(work_stroke.sum(axis=1), axis=0) * 100
            
            fig_work = px.bar(work_pct.reset_index(), 
                             x='work_type', y=1,
                             title='Stroke Rate by Work Type',
                             labels={'1': 'Stroke Rate (%)', 'work_type': 'Work Type'},
                             color_discrete_sequence=['#e74c3c'])
            fig_work.update_xaxes(ticktext=['Children', 'Never worked', 'Govt job', 'Private', 'Self'],
                         tickvals=[0, 1, 2, 3, 4])
            fig_work.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_work, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Glucose level distribution
            fig_glucose = px.violin(data_original, y='avg_glucose_level', x='stroke',
                                   title='Glucose Level Distribution by Stroke Status',
                                   labels={'avg_glucose_level': 'Average Glucose Level (mg/dL)', 
                                          'stroke': 'Stroke Status'},
                                   color='stroke',
                                   color_discrete_map={0: '#3498db', 1: '#e74c3c'},
                                   box=True)
            fig_glucose.update_xaxes(ticktext=['No Stroke', 'Stroke'], tickvals=[0, 1])
            fig_glucose.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_glucose, use_container_width=True)
        
        with col2:
            # BMI distribution
            fig_bmi = px.violin(data_original, y='bmi', x='stroke',
                               title='BMI Distribution by Stroke Status',
                               labels={'bmi': 'Body Mass Index', 'stroke': 'Stroke Status'},
                               color='stroke',
                               color_discrete_map={0: '#3498db', 1: '#e74c3c'},
                               box=True)
            fig_bmi.update_xaxes(ticktext=['No Stroke', 'Stroke'], tickvals=[0, 1])
            fig_bmi.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bmi, use_container_width=True)
        
   
    
   
    

# Diagnostic Page
elif page == "Diagnostic Analytics":
    st.markdown("## Diagnostic Analytics")
    st.markdown("This page includes key statistical insights from the stroke data analysis.")
        
    col1, col2, col3 = st.columns(3)
        
    with col1:
            avg_age_stroke = data_original[data_original['stroke']==1]['age'].mean()
            avg_age_no_stroke = data_original[data_original['stroke']==0]['age'].mean()
            st.info(f"""
            **Age Impact**
            - Stroke patients: {avg_age_stroke:.1f} years
            - No stroke: {avg_age_no_stroke:.1f} years
            - Difference: {avg_age_stroke - avg_age_no_stroke:.1f} years
            """)
        
    with col2:
            hyp_stroke_rate = data_original[data_original['hypertension']==1]['stroke'].mean() * 100
            no_hyp_stroke_rate = data_original[data_original['hypertension']==0]['stroke'].mean() * 100
            st.warning(f"""
            **Hypertension Impact**
            - With hypertension: {hyp_stroke_rate:.1f}% stroke rate
            - Without: {no_hyp_stroke_rate:.1f}% stroke rate
            - Risk multiplier: {hyp_stroke_rate/no_hyp_stroke_rate:.1f}x
            """)
        
    with col3:
            heart_stroke_rate = data_original[data_original['heart_disease']==1]['stroke'].mean() * 100
            no_heart_stroke_rate = data_original[data_original['heart_disease']==0]['stroke'].mean() * 100
            st.error(f"""
            **Heart Disease Impact**
            - With heart disease: {heart_stroke_rate:.1f}% stroke rate
            - Without: {no_heart_stroke_rate:.1f}% stroke rate
            - Risk multiplier: {heart_stroke_rate/no_heart_stroke_rate:.1f}x
            """)
        
        # Additional insights
    st.markdown("### Distribution Summary")
    summary_df = pd.DataFrame({
            'Metric': ['Total Patients', 'Stroke Cases', 'Stroke Rate', 'Avg Age', 'Avg BMI', 'Avg Glucose'],
            'Value': [
                f"{len(data_original):,}",
                f"{data_original['stroke'].sum():,}",
                f"{data_original['stroke'].mean()*100:.2f}%",
                f"{data_original['age'].mean():.1f} years",
                f"{data_original['bmi'].mean():.1f}",
                f"{data_original['avg_glucose_level'].mean():.1f} mg/dL"
            ]
        })
    st.table(summary_df)

# Risk Prediction Page
elif page == "Risk Prediction":
    st.markdown("## Risk Prediction")
    st.markdown("Predicting stroke risk based on data entered")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Enter Patient Information")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            age = st.number_input("Age", min_value=1, max_value=120, value=45)
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
            ever_married = st.selectbox("Ever Married", ["No", "Yes"])
        
        with col_b:
            work_type = st.selectbox("Work Type", 
                                    ["Private", "Self-employed", "Government", "Children", "Never worked"])
            residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
            avg_glucose = st.number_input("Average Glucose Level", 
                                         min_value=50.0, max_value=300.0, value=100.0)
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
            smoking_status = st.selectbox("Smoking Status", 
                                         ["Never smoked", "Formerly smoked", "Smokes", "Unknown"])
        
        if st.button("🔮 Predict Risk", type="primary"):
            # Prepare input data

        

            # Replace the input_data creation in your Risk Prediction section with this:
            input_data = pd.DataFrame({
                        'gender': [0 if gender == "Female" else 1 if gender == "Male" else 2],
                        'age': [age],
                        'hypertension': [1 if hypertension == "Yes" else 0],
                        'heart_disease': [1 if heart_disease == "Yes" else 0],
                        'ever_married': [0 if ever_married == "No" else 1],  
                        'work_type': [  0 if work_type == "Children" else 
                                        2 if work_type == "Government" else 
                                        1 if work_type == "Never worked" else 
                                        3 if work_type == "Private" else 
                                        4],  
                        'Residence_type': [0 if residence_type == "Rural" else 1],  
                        'avg_glucose_level': [avg_glucose],
                        'bmi': [bmi],
                        'smoking_status':   [0 if smoking_status == "Unknown" else 
                                            2 if smoking_status == "Formerly smoked" else 
                                            1 if smoking_status == "Never smoked" else 
                                            3] 
})



            # Normalize numerical features
            input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            risk_percentage = prediction_proba[1] * 100
            
            st.session_state.prediction = risk_percentage
            st.session_state.risk_class = prediction
    
    with col2:
        st.markdown("### Risk Assessment")
        
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
            
            st.markdown(f'<div class="risk-box {risk_class}">{risk_level}<br>'
                       f'<span style="font-size: 1.2rem;">{risk:.1f}% Risk</span></div>', 
                       unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("### Recommendations")
            if risk >= 70:
                st.error("""
                ⚠️ **High Risk Detected**
                - Immediate medical consultation recommended
                - Monitor blood pressure and glucose
                - Consider lifestyle changes
                """)
            elif risk >= 30:
                st.warning("""
                ⚡ **Moderate Risk**
                - Schedule regular check-ups
                - Focus on risk factor management
                - Maintain healthy lifestyle
                """)
            else:
                st.success("""
                ✅ **Low Risk**
                - Continue healthy habits
                - Annual health screening
                - Stay active
                """)
  

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
            st.success(f"""
            ### Risk Assessment
            - **Risk Level:** {sim_risk}%
            - **Status:** Low Risk ✅
            - **Outlook:** Excellent health profile
            """)
        elif sim_risk < 70:
            st.warning(f"""
            ### Risk Assessment
            - **Risk Level:** {sim_risk}%
            - **Status:** Moderate Risk ⚠️
            - **Outlook:** Room for improvement
            """)
        else:
            st.error(f"""
            ### Risk Assessment
            - **Risk Level:** {sim_risk}%
            - **Status:** High Risk 🚨
            - **Outlook:** Immediate action needed
            """)
        
        # Recommendations in a container
        with st.container():
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
            
            # Put recommendations in an expander or container
            with st.expander("View Personalized Recommendations", expanded=True):
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