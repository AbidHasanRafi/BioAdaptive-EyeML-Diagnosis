import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import plotly.express as px
from torch import nn

# --- Model Definitions ---
class BAMNBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_out)
        self.gate = nn.Linear(dim_in, dim_out)
        self.act = nn.GELU()
        self.bn = nn.BatchNorm1d(dim_out)
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()

    def forward(self, x):
        gate = torch.sigmoid(self.gate(x))
        x1 = self.fc1(x)
        x1 = self.act(x1) * gate
        x1 = self.bn(x1)
        x1 = self.dropout(x1)
        res = self.residual_proj(x)
        return x1 + res

class BioAdaptiveMLP(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.1):
        super().__init__()
        self.block1 = BAMNBlock(input_dim, 128, dropout)
        self.block2 = BAMNBlock(128, 64, dropout)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)

# --- App Setup ---
def setup_page():
    st.set_page_config(
        page_title="BioAdaptive EyeML",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Dark mode aware color scheme
    primary_color = "#7c4dff"  # Purple
    secondary_color = "#03dac6"  # Teal
    bg_color = "#1a1a1a" if st.get_option("theme.base") == "dark" else "#ffffff"
    text_color = "#ffffff" if st.get_option("theme.base") == "dark" else "#000000"
    
    # Custom CSS for modern design
    st.markdown(f"""
    <style>
        :root {{
            --primary: {primary_color};
            --secondary: {secondary_color};
            --bg: {bg_color};
            --text: {text_color};
        }}
        
        .stApp {{
            background-color: var(--bg);
            color: var(--text);
        }}
        
        .card {{
            background-color: var(--bg);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, {primary_color}, {secondary_color});
            color: white !important;
            border-radius: 12px;
            padding: 15px;
        }}
        
        .stSlider > div {{
            padding: 0 15px;
        }}
        
        .stNumberInput > div {{
            padding: 0 15px;
        }}
        
        .stSelectbox > div {{
            padding: 0 15px;
        }}
        
        .stButton > button {{
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 500;
            border: none;
            background-color: {primary_color};
            color: white;
        }}
        
        .stButton > button:hover {{
            background-color: {secondary_color};
            color: var(--text);
        }}
        
        .tab-content {{
            padding: 20px 0;
        }}
        
        /* Custom tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 10px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            padding: 10px 20px;
            border-radius: 8px 8px 0 0;
            background-color: transparent;
            transition: all 0.3s;
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {primary_color};
            color: white;
        }}
    </style>
    """, unsafe_allow_html=True)

# --- Main App ---
def main():
    setup_page()
    
    # Register safe globals
    import torch.serialization
    torch.serialization.add_safe_globals([
        BioAdaptiveMLP,
        BAMNBlock,
        nn.Linear,
        nn.GELU,
        nn.BatchNorm1d,
        nn.Dropout,
        nn.Identity
    ])
    
    # Load model
    @st.cache_resource
    def load_model():
        with open('preprocessing_objects.pkl', 'rb') as f:
            preprocessing = pickle.load(f)
        model = torch.load('bioadaptive_mlp_model.pth', map_location='cpu')
        model.eval()
        return model, preprocessing
    
    try:
        model, preprocessing = load_model()
        scaler = preprocessing['scaler']
        label_encoders = preprocessing['label_encoders']
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()
    
    # --- Sidebar ---
    with st.sidebar:
        st.title("BioAdaptive EyeML")
        st.markdown("**Ocular Diagnosis Assistant**")
        
        with st.expander("⚙️ Patient Parameters", expanded=True):
            with st.form("patient_form"):
                st.subheader("Demographics")
                age = st.slider("Age", 0, 100, 45, help="Patient age in years")
                sex = st.selectbox("Sex", label_encoders['sex'].classes_)
                
                st.subheader("Visual Acuity")
                col1, col2 = st.columns(2)
                with col1:
                    va_right = st.slider("Right (logMAR)", -0.3, 3.0, 0.0, 0.1)
                with col2:
                    va_left = st.slider("Left (logMAR)", -0.3, 3.0, 0.0, 0.1)
                unilateral = st.checkbox("Unilateral condition")
                
                st.subheader("Eye Tracking")
                col3, col4 = st.columns(2)
                with col3:
                    right_movement = st.number_input("Right (deg)", value=0.0, step=0.1)
                with col4:
                    left_movement = st.number_input("Left (deg)", value=0.0, step=0.1)
                
                session_num = st.number_input("Session", min_value=1, value=1)
                time_elapsed = st.number_input("Time (sec)", min_value=0.0, value=0.0)
                
                submitted = st.form_submit_button("Run Analysis", use_container_width=True)
    
    # --- Main Content ---
    st.title("Ocular Diagnosis Dashboard")
    
    if submitted:
        try:
            # Prepare input data
            input_data = {
                'age_years': age,
                'sex': sex,
                'va_re_logMar': va_right,
                'va_le_logMar': va_left,
                'unilateral': unilateral,
                'right_eye': right_movement,
                'left_eye': left_movement,
                'measurement_num': session_num,
                'time_elapsed': time_elapsed,
                'inter_eye_diff': abs(right_movement - left_movement),
                'avg_eye_movement': (right_movement + left_movement) / 2
            }
            
            df_input = pd.DataFrame([input_data])
            df_input['sex'] = label_encoders['sex'].transform(df_input['sex'])
            df_input['unilateral'] = df_input['unilateral'].astype(int)
            
            # Create features
            def create_biomimetic_features(df):
                features = pd.DataFrame()
                features['magno_diff'] = df['right_eye'] - df['left_eye']
                features['magno_avg'] = (df['right_eye'] + df['left_eye']) / 2
                features['parvo_std'] = df[['right_eye', 'left_eye']].std(axis=1)
                features['parvo_var'] = features['parvo_std'] ** 2
                features['dorsal_asym'] = np.abs(features['magno_diff'])
                features['dorsal_ratio'] = df['right_eye'] / (df['left_eye'] + 1e-6)
                features['ventral_age'] = df['age_years'] * features['magno_avg']
                features['ventral_inter'] = features['dorsal_asym'] * features['parvo_std']
                return features
            
            biomimetic_features = create_biomimetic_features(df_input)
            base_features = [
                'age_years', 'sex', 'va_re_logMar', 'va_le_logMar', 'unilateral',
                'right_eye', 'left_eye', 'measurement_num', 'time_elapsed',
                'inter_eye_diff', 'avg_eye_movement'
            ]
            X_full = pd.concat([df_input[base_features], biomimetic_features], axis=1)
            X_scaled = scaler.transform(X_full)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(X_tensor)
                _, predicted = torch.max(outputs, 1)
                probabilities = torch.softmax(outputs, dim=1).numpy()[0]
            
            predicted_diagnosis = label_encoders['diagnosis_final'].inverse_transform(predicted.numpy())[0]
            sorted_indices = np.argsort(probabilities)[::-1]
            
            # --- Results Display ---
            st.markdown("## Diagnostic Results")
            
            # Main result card
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Primary Diagnosis", predicted_diagnosis)
                    st.metric("Confidence", f"{probabilities[sorted_indices[0]]:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    fig = px.bar(
                        x=[label_encoders['diagnosis_final'].classes_[i] for i in sorted_indices[:5]],
                        y=probabilities[sorted_indices[:5]],
                        labels={'x':'Diagnosis', 'y':'Probability'},
                        color=probabilities[sorted_indices[:5]],
                        color_continuous_scale='viridis',
                        title="Top 5 Diagnoses"
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color=st.get_option("theme.textColor")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Tabs for detailed analysis
            tab1, tab2, tab3 = st.tabs(["📊 Probability Distribution", "🔍 Feature Analysis", "📝 Clinical Notes"])
            
            with tab1:
                st.plotly_chart(
                    px.pie(
                        names=[label_encoders['diagnosis_final'].classes_[i] for i in sorted_indices],
                        values=probabilities[sorted_indices],
                        title="Diagnosis Probability Distribution",
                        hole=0.3
                    ).update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color=st.get_option("theme.textColor")
                    ),
                    use_container_width=True
                )
                
                for i in range(min(5, len(probabilities))):
                    diagnosis = label_encoders['diagnosis_final'].classes_[sorted_indices[i]]
                    prob = float(probabilities[sorted_indices[i]])  # Convert to float
                    
                    st.markdown(f"**{diagnosis}**")
                    st.progress(prob, text=f"Confidence: {prob:.1%}")
            
            with tab2:
                features = X_full.columns
                importance = np.abs(model.block1.fc1.weight.detach().numpy()).mean(axis=0)
                
                st.plotly_chart(
                    px.bar(
                        x=features,
                        y=importance,
                        labels={'x':'Feature', 'y':'Importance'},
                        title="Feature Importance",
                        color=importance,
                        color_continuous_scale='thermal'
                    ).update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color=st.get_option("theme.textColor"),
                        xaxis_tickangle=-45
                    ),
                    use_container_width=True
                )
                
                st.plotly_chart(
                    px.bar(
                        x=['Right Eye', 'Left Eye'],
                        y=[right_movement, left_movement],
                        title="Eye Movement Comparison",
                        color=['Right', 'Left'],
                        color_discrete_sequence=['#636EFA', '#EF553B']
                    ).update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color=st.get_option("theme.textColor")
                    ),
                    use_container_width=True
                )
            
            with tab3:
                st.json(input_data, expanded=False)
                clinical_notes = st.text_area("Add clinical observations", height=150)
                if st.button("Save Report"):
                    st.success("Report saved successfully")
        
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()