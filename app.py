import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Decision Support System for Depression Level Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        text-align: center;
        margin-top: 0;
        padding-top: 0;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 10px 0;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #10B981;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #F59E0B;
        margin: 10px 0;
    }
    .danger-box {
        background-color: #FEE2E2;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #EF4444;
        margin: 10px 0;
    }
    .recommendation-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
        border: 1px solid #E5E7EB;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .stButton > button {
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #1E3A8A;
        box-shadow: 0 5px 15px rgba(59,130,246,0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    """Load the trained Logistic Regression model"""
    try:
        model = joblib.load('Logistic_Regression.joblib')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_data():
    """Load the dataset for reference"""
    try:
        df = pd.read_csv('depression-dataset-new.csv')
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'probability_scores' not in st.session_state:
    st.session_state.probability_scores = None

# Load resources
model = load_model()
df = load_data()

# Define frequency mapping
freq_mapping = {
    'Never': 0,
    'Rarely (less than one day)': 1,
    'Occasionally (1-2 days)': 2,
    'Frequently (3-4 days)': 3,
    'Most of the time (5-7 days)': 4
}

# Define all symptom features that need encoding
symptom_features = [
    'chronicFatigue', 'studyFocus', 'universityStress', 'futureHopelessness',
    'lostInterest', 'failureFeeling', 'decisionDifficulty', 'restlessness',
    'sleepDisruption', 'morningFatigue', 'persistentSadness', 'futureApathy',
    'selfWorth', 'motivationDeficit', 'selfBlame', 'burdenFeelings',
    'lossOfAppetite', 'weightChange', 'relaxationDifficulty', 'irritability',
    'loneliness', 'alienation', 'pleasureLoss', 'friendshipDifficulty',
    'empathyGap', 'familyDisappointment', 'selfGuilt', 'lossOfControl',
    'selfHarm', 'hopelessness'
]

def preprocess_input(input_data):
    """
    Preprocess input data to match the exact 38 features expected by the model
    """
    # Initialize a dictionary with all expected features set to 0
    encoded_data = {
        'ageGroup_encoded': 0,
        'studyLevel_encoded': 0,
        'gender_Male': 0,
        'universityType_Public': 0,
        'department_EEE': 0,
        'department_Pharmacy': 0,
        'department_SWE': 0,
        'universityName_University of Dhaka': 0
    }
    
    # Add all symptom features with _encoded suffix
    for symptom in symptom_features:
        encoded_data[f'{symptom}_encoded'] = 0
    
    # Encode age group
    age_mapping = {
        '15-20': 0,
        '20-25': 1,
        '25-30': 2,
        '30-35': 3,
        '35+': 4
    }
    encoded_data['ageGroup_encoded'] = age_mapping.get(input_data['ageGroup'], 1)
    
    # Encode study level
    level_mapping = {
        '1st Year': 0,
        '2nd Year': 1,
        '3rd Year': 2,
        '4th Year': 3,
        'MSc': 4,
        'PhD': 5
    }
    encoded_data['studyLevel_encoded'] = level_mapping.get(input_data['studyLevel'], 0)
    
    # Encode gender (one-hot)
    if input_data['gender'] == 'Male':
        encoded_data['gender_Male'] = 1
    
    # Encode university type (one-hot)
    if input_data['universityType'] == 'Public':
        encoded_data['universityType_Public'] = 1
    
    # Encode department (one-hot)
    if input_data['department'] == 'EEE':
        encoded_data['department_EEE'] = 1
    elif input_data['department'] == 'Pharmacy':
        encoded_data['department_Pharmacy'] = 1
    elif input_data['department'] == 'SWE':
        encoded_data['department_SWE'] = 1
    
    # Encode university name (one-hot)
    if input_data['universityName'] == 'University of Dhaka':
        encoded_data['universityName_University of Dhaka'] = 1
    
    # Encode all symptom features
    for symptom in symptom_features:
        if symptom in input_data:
            encoded_data[f'{symptom}_encoded'] = freq_mapping.get(input_data[symptom], 2)
    
    # Create DataFrame with features in the exact order expected by model
    feature_order = [
        'ageGroup_encoded', 'studyLevel_encoded',
        'chronicFatigue_encoded', 'studyFocus_encoded', 'universityStress_encoded',
        'futureHopelessness_encoded', 'lostInterest_encoded', 'failureFeeling_encoded',
        'decisionDifficulty_encoded', 'restlessness_encoded', 'sleepDisruption_encoded',
        'morningFatigue_encoded', 'persistentSadness_encoded', 'futureApathy_encoded',
        'selfWorth_encoded', 'motivationDeficit_encoded', 'selfBlame_encoded',
        'burdenFeelings_encoded', 'lossOfAppetite_encoded', 'weightChange_encoded',
        'relaxationDifficulty_encoded', 'irritability_encoded', 'loneliness_encoded',
        'alienation_encoded', 'pleasureLoss_encoded', 'friendshipDifficulty_encoded',
        'empathyGap_encoded', 'familyDisappointment_encoded', 'selfGuilt_encoded',
        'lossOfControl_encoded', 'selfHarm_encoded', 'hopelessness_encoded',
        'gender_Male', 'universityType_Public', 'department_EEE', 'department_Pharmacy',
        'department_SWE', 'universityName_University of Dhaka'
    ]
    
    df_pred = pd.DataFrame([encoded_data])[feature_order]
    
    return df_pred

# Header
st.markdown("<h1 class='main-header'>Decision Support System for Depression Level Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Early Detection & Personalized Recommendations</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/mental-health.png", width=100)
    st.markdown("## About the System")
    st.markdown("""
    This Decision Support System (DSS) helps in early detection of depression levels using Machine Learning.
    
    **Features:**
    - Predict depression severity
    - Personalized recommendations
    - Visual analytics
  
    
    **Depression Levels:**
    - 🟢 No Depression
    - 🟡 Mild Depression
    - 🟠 Moderate Depression
    - 🔴 Severe Depression
    """)
    
    st.markdown("---")
   

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([" Assessment", " Results", " Recommendations", " About"])

with tab1:
    st.markdown("## Depression Assessment Form")
    st.markdown("Please fill in the following details for accurate assessment:")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Personal Information")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age_group = st.selectbox("Age Group", ["15-20", "20-25", "25-30", "30-35", "35+"])
        university_type = st.selectbox("University Type", ["Private", "Public"])
        department = st.selectbox("Department", 
                                  ["CSE", "EEE", "Pharmacy", "SWE", "Other"])
        study_level = st.selectbox("Study Level", 
                                   ["1st Year", "2nd Year", "3rd Year", "4th Year", "MSc", "PhD"])
        university_name = st.selectbox("University Name", ["DIU", "University of Dhaka", "Other"])
    
    with col2:
        st.markdown("### Health & Wellbeing")
        chronic_fatigue = st.select_slider(
            "Chronic Fatigue",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        study_focus = st.select_slider(
            "Difficulty in Study Focus",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        university_stress = st.select_slider(
            "University Stress",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        future_hopelessness = st.select_slider(
            "Future Hopelessness",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
    
    st.markdown("### Psychological Symptoms")
    col3, col4 = st.columns(2)
    
    with col3:
        lost_interest = st.select_slider(
            "Lost Interest in Activities",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        failure_feeling = st.select_slider(
            "Feeling of Failure",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        decision_difficulty = st.select_slider(
            "Difficulty in Making Decisions",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        restlessness = st.select_slider(
            "Restlessness",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        sleep_disruption = st.select_slider(
            "Sleep Disruption",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        morning_fatigue = st.select_slider(
            "Morning Fatigue",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        persistent_sadness = st.select_slider(
            "Persistent Sadness",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        future_apathy = st.select_slider(
            "Future Apathy",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
    
    with col4:
        self_worth = st.select_slider(
            "Low Self Worth",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        motivation_deficit = st.select_slider(
            "Motivation Deficit",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        self_blame = st.select_slider(
            "Self Blame",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        burden_feelings = st.select_slider(
            "Feelings of Being a Burden",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        loss_of_appetite = st.select_slider(
            "Loss of Appetite",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        weight_change = st.select_slider(
            "Weight Change",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        relaxation_difficulty = st.select_slider(
            "Difficulty Relaxing",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        irritability = st.select_slider(
            "Irritability",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
    
    st.markdown("### Social & Emotional Factors")
    col5, col6 = st.columns(2)
    
    with col5:
        loneliness = st.select_slider(
            "Loneliness",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        alienation = st.select_slider(
            "Alienation",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        pleasure_loss = st.select_slider(
            "Loss of Pleasure",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        friendship_difficulty = st.select_slider(
            "Difficulty in Friendships",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        empathy_gap = st.select_slider(
            "Empathy Gap",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
    
    with col6:
        family_disappointment = st.select_slider(
            "Family Disappointment",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        self_guilt = st.select_slider(
            "Self Guilt",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        loss_of_control = st.select_slider(
            "Loss of Control",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
        self_harm = st.select_slider(
            "Self Harm Thoughts",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Never"
        )
        hopelessness = st.select_slider(
            "Hopelessness",
            options=["Never", "Rarely (less than one day)", "Occasionally (1-2 days)", 
                    "Frequently (3-4 days)", "Most of the time (5-7 days)"],
            value="Occasionally (1-2 days)"
        )
    
    st.markdown("---")
    
    # Predict button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button(" Predict Depression Level", use_container_width=True)
    
    if predict_button:
        # Collect all inputs
        input_data = {
            'gender': gender,
            'ageGroup': age_group,
            'universityType': university_type,
            'department': department,
            'studyLevel': study_level,
            'universityName': university_name,
            'chronicFatigue': chronic_fatigue,
            'studyFocus': study_focus,
            'universityStress': university_stress,
            'futureHopelessness': future_hopelessness,
            'lostInterest': lost_interest,
            'failureFeeling': failure_feeling,
            'decisionDifficulty': decision_difficulty,
            'restlessness': restlessness,
            'sleepDisruption': sleep_disruption,
            'morningFatigue': morning_fatigue,
            'persistentSadness': persistent_sadness,
            'futureApathy': future_apathy,
            'selfWorth': self_worth,
            'motivationDeficit': motivation_deficit,
            'selfBlame': self_blame,
            'burdenFeelings': burden_feelings,
            'lossOfAppetite': loss_of_appetite,
            'weightChange': weight_change,
            'relaxationDifficulty': relaxation_difficulty,
            'irritability': irritability,
            'loneliness': loneliness,
            'alienation': alienation,
            'pleasureLoss': pleasure_loss,
            'friendshipDifficulty': friendship_difficulty,
            'empathyGap': empathy_gap,
            'familyDisappointment': family_disappointment,
            'selfGuilt': self_guilt,
            'lossOfControl': loss_of_control,
            'selfHarm': self_harm,
            'hopelessness': hopelessness
        }
        
        # Store in session state
        st.session_state.input_data = input_data
        
        # Preprocess input
        with st.spinner("Analyzing your responses..."):
            try:
                # Preprocess the input to match model expectations
                X_pred = preprocess_input(input_data)
                
                # Make prediction
                prediction = model.predict(X_pred)[0]
                probabilities = model.predict_proba(X_pred)[0]
                
                # Map prediction to class
                class_mapping = {0: 'No Depression', 1: 'Mild Depression', 
                               2: 'Moderate Depression', 3: 'Severe Depression'}
                
                # Store results
                st.session_state.prediction_made = True
                st.session_state.prediction_result = prediction
                st.session_state.prediction_class = class_mapping[prediction]
                st.session_state.probability_scores = probabilities
                st.session_state.class_mapping = class_mapping
                
                # Success message
                st.success("✅ Analysis complete! Check the Results tab for details.")
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.exception(e)

with tab2:
    st.markdown("## Assessment Results")
    
    if st.session_state.prediction_made:
        # Get prediction details
        pred_class = st.session_state.prediction_class
        probabilities = st.session_state.probability_scores
        class_mapping = st.session_state.class_mapping
        
        # Determine color and icon
        if pred_class == 'No Depression':
            box_class = "success-box"
            icon = "🟢"
            color = "#10B981"
        elif pred_class == 'Mild Depression':
            box_class = "info-box"
            icon = "🟡"
            color = "#F59E0B"
        elif pred_class == 'Moderate Depression':
            box_class = "warning-box"
            icon = "🟠"
            color = "#F97316"
        else:  # Severe Depression
            box_class = "danger-box"
            icon = "🔴"
            color = "#EF4444"
        
        # Display main result
        st.markdown(f"""
        <div class='{box_class}' style='text-align: center;'>
            <h2>{icon} Depression Level: {pred_class}</h2>
        </div>
        """, unsafe_allow_html=True)
        
  
        
        # Create probability gauge chart
        st.markdown("### Probability Distribution")
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(class_mapping.values()),
                y=probabilities * 100,
                marker_color=['#10B981' if c == 'No Depression' else '#F59E0B' if c == 'Mild Depression' 
                             else '#F97316' if c == 'Moderate Depression' else '#EF4444' 
                             for c in class_mapping.values()],
                text=[f'{p*100:.1f}%' for p in probabilities],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Depression Level Probability Distribution",
            xaxis_title="Depression Level",
            yaxis_title="Probability (%)",
            yaxis_range=[0, 100],
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key indicators
        st.markdown("### Key Indicators")
        
        # Get input data
        input_data = st.session_state.input_data
        
        # Identify high-risk symptoms
        high_risk_symptoms = []
        for symptom in symptom_features:
            if symptom in input_data and input_data[symptom] in ["Frequently (3-4 days)", "Most of the time (5-7 days)"]:
                # Format symptom name for display
                display_name = ' '.join([word.capitalize() for word in symptom.split('_')])
                high_risk_symptoms.append(display_name)
        
        if high_risk_symptoms:
            st.markdown("**Areas of Concern:**")
            for symptom in high_risk_symptoms[:5]:  # Show top 5
                st.markdown(f"- {symptom}")
        else:
            st.markdown("No high-risk symptoms detected.")
        
        # Add to session state for recommendations
        st.session_state.high_risk_symptoms = high_risk_symptoms
        
        # Download report button
        report_data = f"""
        DEPRESSION ASSESSMENT REPORT
        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Depression Level: {pred_class}
        Confidence: {probabilities[list(class_mapping.values()).index(pred_class)]*100:.1f}%
        
        Probability Distribution:
        - No Depression: {probabilities[0]*100:.1f}%
        - Mild Depression: {probabilities[1]*100:.1f}%
        - Moderate Depression: {probabilities[2]*100:.1f}%
        - Severe Depression: {probabilities[3]*100:.1f}%
        
        Key Symptoms to Monitor:
        {chr(10).join([f'- {symptom}' for symptom in high_risk_symptoms[:10]])}
        
        Recommendations:
        Please check the Recommendations tab for personalized suggestions.
        """
        
        st.download_button(
            label=" Download Assessment Report",
            data=report_data,
            file_name=f"depression_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        
    else:
        st.info("Please complete the assessment form in the 'Assessment' tab first.")

with tab3:
    st.markdown("## Personalized Recommendations")
    
    if st.session_state.prediction_made:
        pred_class = st.session_state.prediction_class
        high_risk_symptoms = st.session_state.get('high_risk_symptoms', [])
        
        # Import recommendations function
        from utils.recommendations import get_recommendations
        recommendations = get_recommendations(pred_class, high_risk_symptoms)
        
        # Display recommendations based on depression level
        for category, items in recommendations.items():
            with st.expander(f"### {category}", expanded=True):
                for item in items:
                    st.markdown(f"""
                    <div class='recommendation-card'>
                        <p>{item}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
       
        
        # Self-care tips
        st.markdown("### 🌿 Daily Self-Care Tips")
        
        tips = [
            "🧘 Practice mindfulness or meditation for 10 minutes daily",
            "🏃 Engage in physical activity - even a short walk helps",
            "🥗 Maintain a balanced diet and stay hydrated",
            "😴 Aim for 7-9 hours of quality sleep",
            "📝 Keep a gratitude journal",
            "👥 Connect with supportive friends or family",
            "🎯 Set small, achievable goals each day",
            "🎨 Engage in activities you enjoy"
        ]
        
        cols = st.columns(4)
        for i, tip in enumerate(tips):
            with cols[i % 4]:
                st.markdown(f"<div class='recommendation-card'>{tip}</div>", unsafe_allow_html=True)
        
    else:
        st.info("Complete the assessment to receive personalized recommendations.")

with tab4:
    st.markdown("## About This System")
    
    st.markdown("""
    ###  Purpose
    This Decision Support System (DSS) is designed to assist in early detection of depression levels among university students using Machine Learning techniques. It provides:
    
    - **Early identification** of depression symptoms
    - **Personalized recommendations** based on depression severity
    - **Evidence-based insights** from trained ML models
    - **Actionable suggestions** for mental health support
   
    """)
    

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 10px;'>
    <p> Depression Level Detection DSS | For educational and screening purposes only</p>
    <p> If you're in crisis, please call your local emergency services or a suicide prevention hotline immediately.</p>
</div>

""", unsafe_allow_html=True)
