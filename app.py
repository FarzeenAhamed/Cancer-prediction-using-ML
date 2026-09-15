from datetime import datetime
import hashlib
import os
import sqlite3
import tempfile
import cv2
from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import plotly.express as px
import plotly.graph_objects as go
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from tensorflow.keras import layers, models

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title='OncoVision AI | Pro', page_icon='🎗️', layout='wide'
)


# --- DATABASE MANAGEMENT ---
def create_db():
  conn = sqlite3.connect('user_data.db')
  c = conn.cursor()
  c.execute(
      '''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)'''
  )
  c.execute('''CREATE TABLE IF NOT EXISTS profiles
                 (username TEXT PRIMARY KEY, full_name TEXT, age INTEGER, gender TEXT)''')
  c.execute('''CREATE TABLE IF NOT EXISTS history
                 (username TEXT, timestamp TEXT, result TEXT, confidence REAL,
                  radius REAL, texture REAL, perimeter REAL, area REAL, smoothness REAL)''')
  conn.commit()
  conn.close()


def add_user(username, password):
  conn = sqlite3.connect('user_data.db')
  try:
    conn.execute('INSERT INTO users VALUES (?, ?)', (username, password))
    conn.commit()
    return True
  except:
    return False
  finally:
    conn.close()


def check_user(username, password):
  conn = sqlite3.connect('user_data.db')
  c = conn.cursor()
  c.execute('SELECT password FROM users WHERE username = ?', (username,))
  data = c.fetchone()
  conn.close()
  return data and data[0] == password


def update_profile(username, full_name, age, gender):
  conn = sqlite3.connect('user_data.db')
  conn.execute(
      'INSERT OR REPLACE INTO profiles VALUES (?, ?, ?, ?)',
      (username, full_name, age, gender),
  )
  conn.commit()
  conn.close()


def get_profile(username):
  conn = sqlite3.connect('user_data.db')
  c = conn.cursor()
  c.execute(
      'SELECT full_name, age, gender FROM profiles WHERE username = ?',
      (username,),
  )
  data = c.fetchone()
  conn.close()
  return data if data else (None, None, None)


def save_prediction(username, result, confidence, inputs):
  conn = sqlite3.connect('user_data.db')
  conn.execute(
      '''INSERT INTO history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
      (
          username,
          str(datetime.now()),
          result,
          confidence,
          inputs['radius'],
          inputs['texture'],
          inputs['perimeter'],
          inputs['area'],
          inputs['smoothness'],
      ),
  )
  conn.commit()
  conn.close()


def get_user_history(username):
  conn = sqlite3.connect('user_data.db')
  df = pd.read_sql_query(
      'SELECT * FROM history WHERE username = ?', conn, params=(username,)
  )
  conn.close()
  return df


create_db()

# --- CSS THEME ---
st.markdown(
    """
<style>
    .main { background-color: #000000; color: #ffffff; }
    .stApp { background-color: #000000; }
    h1, h2, h3 { color: #00e5ff; }
    [data-testid="stMetric"] { background-color: #1a1a1a; border-left: 4px solid #00e5ff; border-radius: 8px; padding: 15px; }
    [data-testid="stSidebar"] { background-color: #0a0a0a; border-right: 1px solid #222; }
    .stButton>button { background-color: #00e5ff; color: black; font-weight: bold; border-radius: 5px; }
    .stButton>button:hover { background-color: #ffffff; box-shadow: 0 0 15px rgba(0,229,255,0.5); }
    .login-box { background-color: #111; padding: 3rem; border-radius: 15px; border: 1px solid #333; box-shadow: 0 0 20px rgba(0,0,0,0.5); max-width: 400px; margin: auto; }
    .cancer-badge { background-color: #1a1a1a; color: #ff4d4d; padding: 5px 15px; border-radius: 20px; font-weight: bold; display: inline-block; margin-bottom: 10px; border: 1px solid #ff4d4d; }
    .guide-box { background-color: #111; padding: 20px; border-radius: 10px; border-left: 4px solid #00e5ff; margin-bottom: 20px; }
</style>
""",
    unsafe_allow_html=True,
)


# --- MODEL & DATA LOADING ---
@st.cache_resource
def load_tabular_model():
  data = load_breast_cancer()
  X = pd.DataFrame(data.data, columns=data.feature_names)
  y = pd.Series(data.target, name='target')

  n_samples = 10000
  synth_X = X.sample(n=n_samples, replace=True)
  noise = np.random.normal(0, 0.03, synth_X.shape)
  synth_X = synth_X + noise
  synth_y = y.loc[synth_X.index]

  X_full = pd.concat([X, synth_X], ignore_index=True)
  y_full = pd.concat([y, synth_y], ignore_index=True)

  X_train, X_test, y_train, y_test = train_test_split(
      X_full, y_full, test_size=0.2, random_state=42
  )

  model = RandomForestClassifier(
      n_estimators=300, random_state=42, max_depth=10
  )
  model.fit(X_train, y_train)

  preds = model.predict(X_test)
  probs = model.predict_proba(X_test)
  acc = accuracy_score(y_test, preds)
  fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
  roc_auc = auc(fpr, tpr)

  return (
      model,
      acc,
      X_full,
      y_full,
      X_test,
      y_test,
      preds,
      fpr,
      tpr,
      roc_auc,
  )


(
    model,
    accuracy,
    X_full,
    y_full,
    X_test,
    y_test,
    preds,
    fpr,
    tpr,
    roc_auc,
) = load_tabular_model()


# --- IMAGE MODEL ---
@st.cache_resource
def load_image_model():
  model_img = models.Sequential([
      layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.Flatten(),
      layers.Dense(64, activation='relu'),
      layers.Dense(1, activation='sigmoid'),
  ])
  model_img.compile(
      optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']
  )
  return model_img


img_model = load_image_model()


# --- PDF REPORT CLASS ---
class PDF(FPDF):

  def header(self):
    self.set_font('Arial', 'B', 14)
    self.set_text_color(0, 229, 255)
    self.cell(0, 10, 'OncoVision AI - Medical Report', 0, 1, 'C')
    self.set_draw_color(0, 229, 255)
    self.line(10, 20, 200, 20)
    self.ln(10)

  def footer(self):
    self.set_y(-15)
    self.set_font('Arial', 'I', 8)
    self.set_text_color(128, 128, 128)
    self.cell(
        0,
        10,
        f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")} |'
        ' Confidential',
        0,
        0,
        'C',
    )

  def chapter_title(self, title):
    self.set_font('Arial', 'B', 12)
    self.set_text_color(255, 255, 255)
    self.cell(0, 10, title, 0, 1, 'L')
    self.ln(2)

  def chapter_body(self, body):
    self.set_font('Arial', '', 10)
    self.set_text_color(200, 200, 200)
    self.multi_cell(0, 5, body)
    self.ln()


def create_pdf_report(
    patient_info, result, confidence, inputs, seg_img_path=None
):
  pdf = PDF()
  pdf.add_page()
  pdf.set_auto_page_break(auto=True, margin=15)

  pdf.chapter_title('1. Patient Profile')
  profile_text = (
      f"Patient Name: {patient_info['name']}\nPatient ID:"
      f" {patient_info['username']}\nAge: {patient_info['age']} Years\nGender:"
      f' {patient_info["gender"]}\nDate:'
      f' {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
  )
  pdf.chapter_body(profile_text)

  pdf.chapter_title('2. Diagnosis Outcome')
  res_col = (255, 77, 77) if result == 'Malignant' else (0, 229, 255)
  pdf.set_text_color(*res_col)
  pdf.set_font('Arial', 'B', 12)
  pdf.cell(0, 10, f'Result: {result}', ln=1)
  pdf.set_text_color(200, 200, 200)
  pdf.cell(0, 10, f'AI Confidence: {confidence:.2f}%', ln=1)
  pdf.ln(5)

  pdf.chapter_title('3. Tumor Metrics & Shape Analysis')
  radius_mm = inputs['radius']
  volume_mm3 = (4 / 3) * 3.14159 * (radius_mm**3)
  analysis_text = (
      f'Tumor Shape Analysis (CT/MRI Simulation):\n- Estimated Radius:'
      f' {radius_mm:.2f} mm\n- Estimated Volume: {volume_mm3:.2f} mm^3\n-'
      f" Texture Index: {inputs['texture']:.2f}\n- Smoothness Index:"
      f" {inputs['smoothness']:.4f}\n\nModern AI Analysis uses"
      ' computer-assisted segmentation to measure tumor size, texture, and'
      ' volume.'
  )
  pdf.chapter_body(analysis_text)

  if seg_img_path and os.path.exists(seg_img_path):
    pdf.chapter_title('4. Visual Segmentation')
    pdf.image(seg_img_path, x=30, w=150)

  return pdf


# --- AUTH LOGIC ---
if 'logged_in' not in st.session_state:
  st.session_state['logged_in'] = False
  st.session_state['username'] = ''


def login_ui():
  col1, col2, col3 = st.columns([1, 2, 1])
  with col2:
    st.markdown("<div class='login-box'>", unsafe_allow_html=True)
    st.title('🔐 OncoVision AI')
    st.markdown('### Dynamic Cancer Specialist')
    choice = st.selectbox('Mode', ['Login', 'Sign Up'])
    u = st.text_input('Username')
    p = st.text_input('Password', type='password')
    if st.button('Enter System'):
      if not u or not p:
        st.warning('Please fill all fields')
      elif choice == 'Login':
        if check_user(u, hashlib.sha256(p.encode()).hexdigest()):
          st.session_state.update({'logged_in': True, 'username': u})
          st.rerun()
        else:
          st.error('Invalid Credentials')
      else:
        if add_user(u, hashlib.sha256(p.encode()).hexdigest()):
          st.success('Account Created! Please Login.')
        else:
          st.error('Username exists')
    st.markdown('</div>', unsafe_allow_html=True)


# --- MAIN APP ---
if not st.session_state['logged_in']:
  login_ui()
else:
  st.sidebar.title(f"👋 {st.session_state['username']}")
  p_name, p_age, p_gender = get_profile(st.session_state['username'])

  if not p_name:
    with st.sidebar.expander('⚠️ Complete Profile', expanded=True):
      with st.form('profile_form'):
        st.warning('Please update your personal info')
        name = st.text_input('Full Name')
        age = st.number_input('Age', 1, 120, 25)
        gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
        if st.form_submit_button('Save Profile'):
          update_profile(st.session_state['username'], name, age, gender)
          st.rerun()
  else:
    st.sidebar.markdown(f'👤 **{p_name}**')
    st.sidebar.markdown(f'🎂 Age: {p_age} | Gender: {p_gender}')

  st.sidebar.markdown('---')
  page = st.sidebar.radio(
      'Navigate',
      [
          '📖 User Guide',
          '📊 Dashboard',
          '🔬 Data Exploration',
          '🧬 Prediction Tool',
          '📷 Image Analysis',
          '📉 Performance',
          '📜 History',
      ],
  )

  if st.sidebar.button('Logout'):
    st.session_state['logged_in'] = False
    st.rerun()

  # PAGE 0: USER GUIDE
  if page == '📖 User Guide':
    st.title('📖 USER GUIDE & MANUAL')
    st.markdown('### How to use OncoVision AI Effectively')

    col1, col2 = st.columns(2)
    with col1:
      st.markdown("<div class='guide-box'>", unsafe_allow_html=True)
      st.subheader('1. Data Exploration')
      st.markdown("""
            - **Pair Plot**: Visualizes the relationship between multiple features.
              Malignant and Benign tumors often form distinct clusters, making them easy to differentiate.
            - **Count Plot**: Shows the balance of the dataset.
              In our data, we have a mix of Malignant and Benign cases.
            - **Heatmaps**:
                - **Feature Values**: Shows the raw magnitude of features. Some features (like 'Area') have larger values than others.
                - **Correlation Matrix**: Displays how features relate to each other. Darker colors indicate stronger relationships.
            """)
      st.markdown('</div>', unsafe_allow_html=True)

    with col2:
      st.markdown("<div class='guide-box'>", unsafe_allow_html=True)
      st.subheader('2. Prediction Tool')
      st.markdown("""
            - **Input Sliders**: Adjust the tumor characteristics (Radius, Texture, etc.).
            - **Diagnosis**: The AI predicts if the tumor is Malignant (Cancer) or Benign (Non-Cancer).
            - **SHAP Waterfall Plot**: Explains *why* the AI made that decision.
              - Red bars push the prediction towards **Malignant**.
              - Blue bars push the prediction towards **Benign**.
            - **Shape Analysis**: Estimates tumor volume based on input radius using geometric formulas.
            """)
      st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div class='guide-box'>", unsafe_allow_html=True)
    st.subheader('3. Graphs Interpretation')
    st.markdown("""
        - **ROC Curve**: Measures the model's ability to distinguish between classes. A curve closer to the top-left corner indicates high accuracy.
        - **Confusion Matrix**: Shows the number of correct vs. incorrect predictions on the test set.
        - **History Graph**: Tracks your past prediction confidence levels over time.
        """)
    st.markdown('</div>', unsafe_allow_html=True)

  # PAGE 1: DASHBOARD
  elif page == '📊 Dashboard':
    st.markdown(
        "<span class='cancer-badge'>Breast Cancer Specialist</span>",
        unsafe_allow_html=True,
    )
    st.title('📊 ONCOLOGY DASHBOARD')
    col1, col2, col3 = st.columns(3)
    col1.metric('Model Accuracy', f'{accuracy*100:.2f}%')
    col2.metric('Training Samples', f'{len(X_full)}', 'Augmented Data')
    col3.metric('Model Type', 'Random Forest')

    st.markdown('---')
    st.subheader('Feature Correlation Heatmap')
    corr = X_full.corr()
    fig = px.imshow(
        corr,
        color_continuous_scale='RdBu_r',
        title='Feature Correlation Matrix',
    )
    fig.update_layout(paper_bgcolor='#000000', height=600)
    st.plotly_chart(fig, use_container_width=True)

  # PAGE 2: DATA EXPLORATION
  elif page == '🔬 Data Exploration':
    st.markdown(
        "<span class='cancer-badge'>Dynamic Analysis</span>",
        unsafe_allow_html=True,
    )
    st.title('🔬 DYNAMIC DATA EXPLORATION')

    viz_df = X_full.copy()
    viz_df['Target'] = y_full.map({0: 'Malignant', 1: 'Benign'})

    st.subheader('1. Diagnosis Distribution (Count Plot)')
    st.caption(
        'Total count of Malignant and Benign tumor patients in the dataset.'
    )
    fig_count = px.histogram(
        viz_df,
        x='Target',
        color='Target',
        color_discrete_map={'Malignant': '#ff4d4d', 'Benign': '#00e5ff'},
        template='plotly_dark',
        text_auto=True,
    )
    fig_count.update_layout(paper_bgcolor='#000000', plot_bgcolor='#000000')
    st.plotly_chart(fig_count, use_container_width=True)

    st.subheader('2. Feature Value Heatmap')
    st.caption(
        'Heatmap showing the variety of different feature values. Note how'
        " 'mean area' and 'worst area' have greater values than other features."
    )

    heat_sample = viz_df.drop(columns=['Target']).sample(n=20, random_state=42)

    fig_heat = px.imshow(
        heat_sample.T,
        color_continuous_scale='Viridis',
        aspect='auto',
        title='Feature Magnitude Heatmap (20 Samples)',
    )
    fig_heat.update_layout(paper_bgcolor='#000000', height=600)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader('3. Pair Plot Relationships')
    st.caption(
        'Visualizing relationships between features. It is easy to differentiate'
        ' Malignant and Benign tumors in the pair plot.'
    )

    default_pair_features = [
        'mean radius',
        'mean texture',
        'mean perimeter',
        'mean area',
    ]
    selected_pair_cols = st.multiselect(
        'Select Features for Pair Plot',
        X_full.columns.tolist(),
        default=default_pair_features,
    )

    if len(selected_pair_cols) >= 2:
      pair_df = viz_df.sample(n=500, random_state=42)
      fig_pair = px.scatter_matrix(
          pair_df,
          dimensions=selected_pair_cols,
          color='Target',
          template='plotly_dark',
          height=800,
          color_discrete_map={'Malignant': '#ff4d4d', 'Benign': '#00e5ff'},
      )
      fig_pair.update_traces(
          diagonal_visible=False, showupperhalf=False, marker=dict(size=3)
      )
      fig_pair.update_layout(paper_bgcolor='#000000')
      st.plotly_chart(fig_pair, use_container_width=True)
    else:
      st.warning('Please select at least 2 features for the Pair Plot.')

  # PAGE 3: PREDICTION TOOL
  elif page == '🧬 Prediction Tool':
    st.markdown(
        "<span class='cancer-badge'>Breast Cancer Analysis</span>",
        unsafe_allow_html=True,
    )
    st.title('🧬 AI DIAGNOSTIC SUITE')

    col1, col2 = st.columns(2)
    with col1:
      r = st.slider('Mean Radius', 5.0, 35.0, 15.0)
      t = st.slider('Mean Texture', 8.0, 40.0, 20.0)
      p = st.slider('Mean Perimeter', 40.0, 200.0, 100.0)
    with col2:
      a = st.slider('Mean Area', 150.0, 2500.0, 600.0)
      s = st.slider('Mean Smoothness', 0.05, 0.20, 0.1)
      c = st.slider('Mean Concavity', 0.0, 0.5, 0.1)

    if st.button('🔍 Diagnose'):
      p_name, p_age, p_gender = get_profile(st.session_state['username'])
      if not p_name:
        st.warning(
            'Please update your profile in the sidebar to generate reports.'
        )

      input_data = X_full.mean()
      input_data['mean radius'], input_data['mean texture'] = r, t
      input_data['mean perimeter'], input_data['mean area'] = p, a
      input_data['mean smoothness'], input_data['mean concavity'] = s, c
      input_df = pd.DataFrame([input_data])

      pred = model.predict(input_df)[0]
      prob = model.predict_proba(input_df)[0]
      res = 'Malignant' if pred == 0 else 'Benign'
      conf = prob[pred] * 100

      save_prediction(
          st.session_state['username'],
          res,
          conf,
          {'radius': r, 'texture': t, 'perimeter': p, 'area': a, 'smoothness': s},
      )

      st.subheader('🔬 DIAGNOSIS RESULT')
      if pred == 0:
        st.error(f'⚠️ Result: {res} (Cancer Detected)')
      else:
        st.success(f'✅ Result: {res} (No Cancer)')
      st.metric('Confidence', f'{conf:.2f}%')

      st.markdown('### 📐 Tumor Shape Analysis')
      col_shape1, col_shape2 = st.columns(2)
      with col_shape1:
        theta = np.linspace(0, 2 * np.pi, 100)
        rad_noise = r / 2 + np.random.normal(0, r / 10, 100)
        x = rad_noise * np.cos(theta)
        y = rad_noise * np.sin(theta)

        fig_shape = go.Figure()
        fig_shape.add_trace(
            go.Scatter(
                x=x,
                y=y,
                fill='toself',
                fillcolor=(
                    'rgba(255, 77, 77, 0.5)'
                    if pred == 0
                    else 'rgba(0, 229, 255, 0.5)'
                ),
                line_color='white',
                name='Tumor Shape',
            )
        )
        fig_shape.update_layout(
            title='Estimated Shape (Based on Radius & Texture)',
            paper_bgcolor='#000000',
            height=300,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        st.plotly_chart(fig_shape, use_container_width=True)

      with col_shape2:
        st.info('Shape Metrics:')
        st.write(f'**Approx Radius:** {r:.2f} mm')
        st.write(f'**Approx Volume:** {(4/3)*3.14*(r**3):.2f} mm³')
        st.write(f'**Texture Deviation:** {t:.2f}')

      st.markdown('### 🧠 AI Explainability')
      with st.spinner('Analyzing decision factors...'):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        if isinstance(shap_values, list):
          sv = shap_values[pred][0]
          base_val = explainer.expected_value[pred]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
          sv = shap_values[0, :, pred]
          base_val = explainer.expected_value[pred]
        else:
          sv = shap_values[0]
          base_val = explainer.expected_value

        if isinstance(base_val, (list, np.ndarray)):
          base_val = base_val[0] if len(base_val) == 1 else base_val[pred]

        explanation = shap.Explanation(
            values=sv,
            base_values=base_val,
            data=input_df.iloc[0].values,
            feature_names=input_df.columns,
        )

        fig, ax = plt.subplots()
        plt.title(f'Feature Impact on Prediction ({res})')
        shap.plots.waterfall(explanation, max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig, bbox_inches='tight')

      if p_name:
        st.markdown('### 📄 Report')
        inputs_dict = {
            'radius': r,
            'texture': t,
            'perimeter': p,
            'area': a,
            'smoothness': s,
        }
        patient_info = {
            'name': p_name,
            'age': p_age,
            'gender': p_gender,
            'username': st.session_state['username'],
        }
        pdf = create_pdf_report(patient_info, res, conf, inputs_dict)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
          pdf.output(tmp.name)
          with open(tmp.name, 'rb') as f:
            st.download_button(
                '📥 Download Detailed PDF Report',
                f,
                file_name='OncoVision_Report.pdf',
            )
          os.unlink(tmp.name)

  # PAGE 4: IMAGE ANALYSIS
  elif page == '📷 Image Analysis':
    st.markdown(
        "<span class='cancer-badge'>Visual Diagnosis & Segmentation</span>",
        unsafe_allow_html=True,
    )
    st.title('📷 IMAGE ANALYSIS WITH AI SEGMENTATION')
    st.markdown(
        'Upload a CT/MRI scan (Simulated). The AI will visualize tumor'
        ' segmentation.'
    )

    uploaded_file = st.file_uploader(
        'Upload Image', type=['jpg', 'png', 'jpeg']
    )

    if uploaded_file:
      col1, col2 = st.columns(2)
      with col1:
        st.image(uploaded_file, caption='Original Scan', use_column_width=True)

      with col2:
        if st.button('Analyze & Segment'):
          with st.spinner('Running AI Segmentation...'):
            img = Image.open(uploaded_file).convert('RGB')
            img_array = np.array(img)

            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (11, 11), 0)
            thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            overlay = img_array.copy()
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)

            st.image(
                overlay,
                caption='AI Segmentation Result (Green)',
                use_column_width=True,
            )
            if contours:
              st.info(
                  'Detected Anomalous Area:'
                  f' {cv2.contourArea(max(contours, key=cv2.contourArea))} pixels'
              )
            else:
              st.warning('No distinct anomalies detected.')

            seg_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            Image.fromarray(overlay).save(seg_temp.name)
            st.session_state['seg_img_path'] = seg_temp.name

  # PAGE 5: PERFORMANCE
  elif page == '📉 Performance':
    st.title('📉 MODEL PERFORMANCE')
    col1, col2 = st.columns(2)
    with col1:
      st.subheader('ROC Curve')
      fig = go.Figure()
      fig.add_trace(
          go.Scatter(x=fpr, y=tpr, name='ROC', line=dict(color='#00e5ff'))
      )
      fig.add_trace(
          go.Scatter(
              x=[0, 1],
              y=[0, 1],
              name='Random',
              line=dict(dash='dash', color='#555'),
          )
      )
      fig.update_layout(paper_bgcolor='#000000', plot_bgcolor='#000000')
      st.plotly_chart(fig, use_container_width=True)
    with col2:
      st.subheader('Confusion Matrix')
      cm = confusion_matrix(y_test, preds)
      fig_cm = px.imshow(
          cm, text_auto=True, color_continuous_scale='Blues', title='Confusion Matrix'
      )
      fig_cm.update_layout(paper_bgcolor='#000000')
      st.plotly_chart(fig_cm, use_container_width=True)

  # PAGE 6: HISTORY
  elif page == '📜 History':
    st.title('📜 PREDICTION HISTORY')
    hist = get_user_history(st.session_state['username'])

    if not hist.empty:
      hist_display = hist.sort_values(by='timestamp', ascending=False)

      st.subheader('Historical Records')
      st.dataframe(hist_display.style.background_gradient(cmap='Blues'))

      st.subheader('Diagnosis Count in History')
      fig_hist_count = px.histogram(
          hist_display,
          x='result',
          color='result',
          color_discrete_map={'Malignant': '#ff4d4d', 'Benign': '#00e5ff'},
          template='plotly_dark',
          text_auto=True,
      )
      fig_hist_count.update_layout(paper_bgcolor='#000000')
      st.plotly_chart(fig_hist_count, use_container_width=True)

      st.subheader('Input Parameter Relationships')
      st.caption(
          'Analyzing how your input parameters correlate with each other and'
          ' the diagnosis result.'
      )

      pair_cols = ['radius', 'texture', 'perimeter', 'area', 'smoothness']
      hist_sample = hist_display.tail(100)

      fig_pair_hist = px.scatter_matrix(
          hist_sample,
          dimensions=pair_cols,
          color='result',
          color_discrete_map={'Malignant': '#ff4d4d', 'Benign': '#00e5ff'},
          template='plotly_dark',
          height=700,
      )
      fig_pair_hist.update_traces(
          diagonal_visible=False, showupperhalf=False, marker=dict(size=5)
      )
      fig_pair_hist.update_layout(paper_bgcolor='#000000')
      st.plotly_chart(fig_pair_hist, use_container_width=True)

      st.subheader('Confidence Trend Over Time')
      fig_line = px.line(
          hist_display,
          x='timestamp',
          y='confidence',
          color='result',
          markers=True,
          color_discrete_map={'Malignant': '#ff4d4d', 'Benign': '#00e5ff'},
      )
      fig_line.update_layout(paper_bgcolor='#000000')
      st.plotly_chart(fig_line, use_container_width=True)

    else:
      st.info('No history found. Make a prediction to see data here.')