import streamlit as st
import pandas as pd
import base64, random
import plotly.graph_objects as go
from io import BytesIO
import time, datetime
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter
import io, random
from streamlit_tags import st_tags
from PIL import Image
import pymysql
from Courses import sde_course, ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos
from pytube import YouTube
import plotly.express as px
import nltk
nltk.download('stopwords')
import spacy
spacy.cli.download("en_core_web_sm")
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Set Streamlit page config as the very first command.
st.set_page_config(
    page_title="ResuMate Pro",
    page_icon='./Logo/logo3.png',
    layout="wide"
)

# ------------------------ AI Configuration ------------------------
@st.cache_resource
def load_ai_models():
    with st.spinner("🚀 Loading AI engines (first time may take 2-5 minutes)..."):
        return {
            'analysis_model': pipeline(
                "text-generation",
                model="gpt2",  
                device_map="auto"
            ),
            'embedding_model': SentenceTransformer('all-MiniLM-L6-v2')
        }

ai_models = load_ai_models()

# ------------------------ Enhanced AI Functions ------------------------
def advanced_ai_analysis(resume_text, job_desc):
    try:
        prompt = f"""
Analyze this resume for the job: {job_desc[:2000]}
Provide structured analysis with:
1. Professional summary (3 sentences)
2. Top 3 technical strengths with examples
3. Top 3 improvement areas with actionable advice
4. ATS optimization checklist
5. Estimated job match percentage (0-100%)

Resume: {resume_text[:3000]}
Format using markdown headings
"""
        response = ai_models['analysis_model'](
            prompt,
            max_new_tokens=1200,
            temperature=0.65,
            do_sample=True,
            top_p=0.95
        )
        st.write("DEBUG: Raw AI Response:", response)
        if not response or not isinstance(response, list) or len(response) == 0:
            st.error("AI analysis returned an empty response.")
            return ""
        generated_text = response[0].get('generated_text', "")
        st.write("DEBUG: Generated Text:", generated_text)
        return generated_text.strip()
    except Exception as e:
        st.error(f"AI analysis failed: {str(e)}")
        return ""



def skill_match_analysis(resume_skills, job_desc):
    """Advanced skill matching with embeddings."""
    try:
        if not resume_skills or not job_desc:
            return 0.0
        # Encode skills and job description
        resume_emb = ai_models['embedding_model'].encode(
            ", ".join(resume_skills),
            convert_to_tensor=True
        )
        job_emb = ai_models['embedding_model'].encode(
            job_desc,
            convert_to_tensor=True
        )
        # Calculate cosine similarity (scaled to percentage)
        return util.pytorch_cos_sim(resume_emb, job_emb).item() * 100
    except Exception as e:
        st.error(f"Skill analysis error: {str(e)}")
        return 0.0

# ------------------------ Core Application Functions ------------------------
def fetch_yt_video(link):
    try:
        yt = YouTube(link)
        return yt.title
    except Exception as e:
        return f"Video Title ({str(e)})"

def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'

def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
    text = fake_file_handle.getvalue()
    converter.close()
    fake_file_handle.close()
    return text

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def course_recommender(course_list):
    st.subheader("🎓 Recommended Courses & Certifications")
    no_of_reco = st.slider('Select number of recommendations:', 1, 10, 4)
    random.shuffle(course_list)
    cols = st.columns(2)
    for idx, (c_name, c_link) in enumerate(course_list[:no_of_reco]):
        cols[idx % 2].markdown(f"• [{c_name}]({c_link})")
    # Return just the course names for record (if needed)
    return [course_list[i][0] for i in range(no_of_reco)]

# ------------------------ Database Operations ------------------------
def init_database():
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='Soumya2004',
        db='cv'
    )
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_data (
            ID INT AUTO_INCREMENT PRIMARY KEY,
            Name VARCHAR(500) NOT NULL,
            Email_ID VARCHAR(500) NOT NULL,
            resume_score VARCHAR(8) NOT NULL,
            Timestamp VARCHAR(50) NOT NULL,
            Page_no VARCHAR(5) NOT NULL,
            Predicted_Field TEXT NOT NULL,
            User_level TEXT NOT NULL,
            Actual_skills TEXT NOT NULL,
            Recommended_skills TEXT NOT NULL,
            Recommended_courses TEXT NOT NULL,
            AI_analysis TEXT
        )
    """)
    return connection, cursor

# ------------------------ Helper Functions ------------------------
def display_basic_info(resume_data, save_path):
    with st.container():
        cols = st.columns([1, 2])
        cols[0].subheader("👤 Basic Information")
        cols[0].write(f"**Name:** {resume_data.get('name', 'N/A')}")
        cols[0].write(f"**Email:** {resume_data.get('email', 'N/A')}")
        cols[0].write(f"**Pages:** {resume_data.get('no_of_pages', 0)}")
        cols[1].subheader("🔍 Resume Preview")
        show_pdf(save_path)

def calculate_resume_score(text):
    st.subheader("📝 Resume Tips & Ideas")
    score = 0
    if 'Objective' in text:
        score += 20
        st.markdown('<h5 style="text-align: left; color: #1ed760;">[+] Objective found.</h5>', unsafe_allow_html=True)
    else:
        st.markdown('<h5 style="text-align: left; color: red;">[-] Please add a career objective.</h5>', unsafe_allow_html=True)
    if 'Certifications' in text:
        score += 20
        st.markdown('<h5 style="text-align: left; color: #1ed760;">[+] Certifications found.</h5>', unsafe_allow_html=True)
    else:
        st.markdown('<h5 style="text-align: left; color: red;">[-] Please add Certifications.</h5>', unsafe_allow_html=True)
    if 'Soft skills' in text or 'Interests' in text:
        score += 20
        st.markdown('<h5 style="text-align: left; color: #1ed760;">[+] Soft skills present.</h5>', unsafe_allow_html=True)
    else:
        st.markdown('<h5 style="text-align: left; color: red;">[-] Please add Soft skills.</h5>', unsafe_allow_html=True)
    if 'Achievements' in text:
        score += 20
        st.markdown('<h5 style="text-align: left; color: #1ed760;">[+] Achievements present.</h5>', unsafe_allow_html=True)
    else:
        st.markdown('<h5 style="text-align: left; color: red;">[-] Please add Achievements.</h5>', unsafe_allow_html=True)
    if 'Projects' in text:
        score += 20
        st.markdown('<h5 style="text-align: left; color: #1ed760;">[+] Projects found.</h5>', unsafe_allow_html=True)
    else:
        st.markdown('<h5 style="text-align: left; color: red;">[-] Please add Projects.</h5>', unsafe_allow_html=True)
    
    st.subheader("⏳ Calculating Resume Score")
    my_bar = st.progress(0)
    for i in range(score):
        time.sleep(0.05)
        my_bar.progress(i + 1)
    st.success(f"*Your Resume Writing Score: {score}*")
    st.warning("*Note: This score is based on the content found in your resume.*")
    return score

def get_recommendations(skills):
    # Define keywords for each field
    sde_keyword = ['dsa', 'cp', 'data structures', 'algorithms', 'contest', 'competitive programming',
                   'problem solving', 'c++', 'java', 'python', 'programming', 'coding', 'codeforces', 'leetcode',
                   'hackerrank', 'hackerearth', 'geeksforgeeks', 'interviewbit', 'topcoder', 'spoj', 'competitive coding',
                   'competitive coder', 'website', 'projects']
    ds_keyword = ['tensorflow', 'keras', 'pytorch', 'machine learning', 'deep learning', 'flask', 'streamlit']
    web_keyword = ['react', 'django', 'node jS', 'react js', 'php', 'laravel', 'magento', 'wordpress',
                   'javascript', 'angular js', 'c#', 'flask']
    android_keyword = ['android', 'android development', 'flutter', 'kotlin', 'xml', 'kivy']
    ios_keyword = ['ios', 'ios development', 'swift', 'cocoa', 'cocoa touch', 'xcode']
    uiux_keyword = ['ux', 'adobe xd', 'figma', 'zeplin', 'balsamiq', 'ui', 'prototyping', 'wireframes',
                    'storyframes', 'adobe photoshop', 'photoshop', 'editing', 'adobe illustrator', 'illustrator',
                    'adobe after effects', 'after effects', 'adobe premier pro', 'premier pro', 'adobe indesign', 'indesign',
                    'wireframe', 'solid', 'grasp', 'user research', 'user experience']
    
    recommended_skills = []
    reco_field = ''
    rec_course = ''
    
    for i in skills:
        low = i.lower()
        if low in ds_keyword:
            reco_field = 'Data Science'
            st.success("*Our analysis says you are looking for Data Science Jobs.*")
            recommended_skills = ['Data Visualization', 'Predictive Analysis', 'Statistical Modeling', 'Data Mining',
                                  'Clustering & Classification', 'Data Analytics', 'Quantitative Analysis', 'Web Scraping',
                                  'ML Algorithms', 'Keras', 'Pytorch', 'Probability', 'Scikit-learn', 'Tensorflow',
                                  'Flask', 'Streamlit']
            rec_course = course_recommender(ds_course)
            break
        elif low in sde_keyword:
            reco_field = 'Software Development'
            st.success("*Our analysis says you are looking for Software Development Jobs.*")
            recommended_skills = ['DSA', 'CP', 'Data Structures', 'Algorithms', 'Problem Solving', 'C++', 'Java',
                                  'Python', 'Programming', 'Coding', 'Codeforces', 'Leetcode', 'Hackerrank', 'Hackerearth',
                                  'Geeksforgeeks', 'Interviewbit', 'Topcoder', 'Spoj', 'Competitive Coding', 'Competitive Coder',
                                  'Website', 'Projects']
            rec_course = course_recommender(sde_course)
            break
        elif low in web_keyword:
            reco_field = 'Web Development'
            st.success("*Our analysis says you are looking for Web Development Jobs.*")
            recommended_skills = ['React', 'Django', 'Node JS', 'React JS', 'PHP', 'Laravel', 'Magento', 'WordPress',
                                  'Javascript', 'Angular JS', 'C#', 'Flask', 'SDK']
            rec_course = course_recommender(web_course)
            break
        elif low in android_keyword:
            reco_field = 'Android Development'
            st.success("*Our analysis says you are looking for Android App Development Jobs.*")
            recommended_skills = ['Android', 'Android Development', 'Flutter', 'Kotlin', 'XML', 'Java', 'Kivy', 'GIT', 'SDK', 'SQLite']
            rec_course = course_recommender(android_course)
            break
        elif low in ios_keyword:
            reco_field = 'IOS Development'
            st.success("*Our analysis says you are looking for IOS App Development Jobs.*")
            recommended_skills = ['IOS', 'IOS Development', 'Swift', 'Cocoa', 'Cocoa Touch', 'Xcode', 'Objective-C', 'SQLite', 'Plist', 'StoreKit', 'UI-Kit', 'AV Foundation', 'Auto-Layout']
            rec_course = course_recommender(ios_course)
            break
        elif low in uiux_keyword:
            reco_field = 'UI-UX Development'
            st.success("*Our analysis says you are looking for UI-UX Development Jobs.*")
            recommended_skills = ['UI', 'User Experience', 'Adobe XD', 'Figma', 'Zeplin', 'Balsamiq', 'Prototyping', 'Wireframes',
                                  'Storyframes', 'Adobe Photoshop', 'Editing', 'Illustrator', 'After Effects', 'Premier Pro',
                                  'Indesign', 'Wireframe', 'Solid', 'Grasp', 'User Research']
            rec_course = course_recommender(uiux_course)
            break
    return reco_field, recommended_skills, rec_course

def plot_pie_chart(df, column_name, title):
    # Ensure the column is decoded as string if needed.
    df[column_name] = df[column_name].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x))
    fig = px.pie(df, names=column_name, title=title)
    st.plotly_chart(fig)

# ------------------------ Main Application ------------------------
def main():
    img = Image.open('./Logo/logo-removebg-preview.png').resize((768, 140))
    st.image(img)
    st.title("AI-Powered Resume Analyzer")
    st.markdown("---")
    
    st.sidebar.markdown("# User Type")
    choice = st.sidebar.selectbox("", ["User", "Admin"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed by [Soumya Ranjan Jena](https://www.linkedin.com/in/srj2004/)")
    
    connection, cursor = init_database()
    
    if choice == "User":
        user_interface(connection, cursor)
    else:
        admin_interface(connection, cursor)

# ------------------------ User Interface ------------------------
def user_interface(connection, cursor):
    st.header("📄 Smart Resume Analysis")
    with st.expander("Upload Your Resume", expanded=True):
        pdf_file = st.file_uploader("Choose PDF", type=["pdf"])
    if pdf_file:
        process_resume(pdf_file, connection, cursor)

def process_resume(pdf_file, connection, cursor):
    save_path = f'./Uploaded_Resumes/{pdf_file.name}'
    with open(save_path, "wb") as f:
        f.write(pdf_file.getbuffer())
    with st.spinner("Analyzing Resume..."):
        try:
            resume_data = ResumeParser(save_path).get_extracted_data()
            resume_text = pdf_reader(save_path)
            if not resume_data:
                st.error("Failed to parse resume")
                return
            display_basic_info(resume_data, save_path)
            display_ai_analysis(resume_text, resume_data, connection, cursor)
        except Exception as e:
            st.error(f"Processing error: {str(e)}")

def display_ai_analysis(resume_text, resume_data, connection, cursor):
    tab1, tab2, tab3 = st.tabs(["Basic Analysis", "AI Insights", "Career Tools"])
    with tab1:
        basic_analysis(resume_text, resume_data, connection, cursor)
    with tab2:
        ai_insights(resume_text, resume_data)
    with tab3:
        career_tools()

def basic_analysis(resume_text, resume_data, connection, cursor):
    st.subheader("📝 Resume Evaluation")
    exp_level = "Fresher" if resume_data['no_of_pages'] == 1 else \
                "Intermediate" if resume_data['no_of_pages'] == 2 else "Experienced"
    st.markdown(f"*Career Level:* {exp_level}")
    current_skills = st_tags(label="Your Current Skills", 
                             value=resume_data['skills'],
                             text="See recommendations below")
    # Get recommendations based on skills
    reco_field, recommended_skills, rec_course = get_recommendations(resume_data['skills'])
    score = calculate_resume_score(resume_text)
    
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    insert_tuple = (
        resume_data['name'],
        resume_data['email'],
        str(score),
        ts,
        str(resume_data['no_of_pages']),
        reco_field,
        exp_level,
        str(resume_data['skills']),
        str(recommended_skills),
        str(rec_course),
        ""  # Placeholder for AI_analysis
    )
    cursor.execute(
        "INSERT INTO user_data (Name, Email_ID, resume_score, Timestamp, Page_no, Predicted_Field, User_level, Actual_skills, Recommended_skills, Recommended_courses, AI_analysis) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        insert_tuple
    )
    connection.commit()

def ai_insights(resume_text, resume_data):
    st.subheader("🤖 Advanced AI Analysis")
    with st.spinner("Generating deep insights..."):
        analysis = advanced_ai_analysis(resume_text, "Software Engineering Position")
        st.markdown(analysis)
    st.subheader("📊 Skill Compatibility")
    match_score = skill_match_analysis(resume_data['skills'], "Software Engineering Requirements")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=match_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Skill Match Score"},
        gauge={'axis': {'range': [None, 100]},
               'steps': [
                   {'range': [0, 50], 'color': "lightgray"},
                   {'range': [50, 75], 'color': "gray"}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}}
    ))
    st.plotly_chart(fig)

def career_tools():
    st.subheader("🛠️ Career Toolkit")
    with st.expander("📝 Cover Letter Generator"):
        if st.button("Generate Cover Letter"):
            with st.spinner("Creating professional letter..."):
                st.write("AI-generated cover letter...")
    with st.expander("🎥 Interview Prep Videos"):
        vid = random.choice(interview_videos)
        st.video(vid)
    with st.expander("📚 Learning Resources"):
        course_recommender(sde_course + ds_course)

# ------------------------ Admin Interface ------------------------
def admin_interface(connection, cursor):
    st.subheader("🔒 Admin Dashboard")
    if authenticate_admin():
        display_admin_panels(connection)

def authenticate_admin():
    cols = st.columns(2)
    user = cols[0].text_input("Username")
    pwd = cols[1].text_input("Password", type="password")
    if st.button("Login"):
        if user == "Soumya" and pwd == "Soumya2004":
            return True
        st.error("Invalid credentials")
    return False

def display_admin_panels(connection):
    st.subheader("📊 User Analytics")
    user_data = pd.read_sql("SELECT * FROM user_data", connection)
    st.dataframe(user_data)
    cols = st.columns(2)
    cols[0].plotly_chart(px.pie(user_data, names='Predicted_Field', title='Field Distribution'))
    cols[1].plotly_chart(px.histogram(user_data, x='resume_score', title='Score Distribution'))
    st.markdown(get_table_download_link(user_data, "user_data.csv", "📥 Export Data"), unsafe_allow_html=True)

# ------------------------ Main Execution ------------------------
if __name__ == "__main__":
    main()
