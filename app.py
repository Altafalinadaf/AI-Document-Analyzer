import streamlit as st
import PyPDF2
import io
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import plotly.express as px
import pandas as pd
import time

# Set page config
st.set_page_config(
    page_title="Resume Screening Agent",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'api_key_valid' not in st.session_state:
    st.session_state.api_key_valid = False
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False
if 'api_key_checked' not in st.session_state:
    st.session_state.api_key_checked = False
if 'api_key_error' not in st.session_state:
    st.session_state.api_key_error = ""

def validate_api_key(api_key):
    """Test if the API key is valid"""
    # Clean the API key
    api_key = api_key.strip()
    
    if not api_key:
        return False, "Please enter an API key"
    
    if not api_key.startswith('sk-'):
        return False, "API key must start with 'sk-'"
    
    try:
        client = OpenAI(api_key=api_key)
        # Test with a simple, low-cost request (compatible with all versions)
        models = client.models.list()
        # Just check if we got any response at all
        if hasattr(models, 'data') and len(models.data) > 0:
            return True, "API key is valid"
        else:
            return False, "API key test failed - no models returned"
    except Exception as e:
        error_msg = str(e)
        if "Incorrect API key" in error_msg:
            return False, "Incorrect API key provided"
        elif "invalid_api_key" in error_msg:
            return False, "Invalid API key"
        elif "rate limit" in error_msg.lower():
            return False, "Rate limit exceeded. Please try again later."
        elif "authentication" in error_msg.lower():
            return False, "Authentication error. Please check your API key."
        elif "organization" in error_msg.lower():
            return False, "Organization billing issue. Please check your OpenAI account."
        else:
            return False, f"API validation error: {error_msg}"

def initialize_openai_client():
    if st.session_state.api_key and st.session_state.api_key_valid:
        try:
            return OpenAI(api_key=st.session_state.api_key)
        except:
            st.session_state.api_key_valid = False
    return None

def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def clean_resume_text(text):
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    return text

def get_embedding(text, client):
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        error_msg = str(e)
        st.session_state.api_key_error = f"Embedding error: {error_msg}"
        st.error(f"Error getting embedding: {error_msg}")
        return None

def calculate_similarity(job_embedding, resume_embedding):
    if job_embedding is None or resume_embedding is None:
        return 0
    
    # Convert to numpy arrays
    job_embedding = np.array(job_embedding).reshape(1, -1)
    resume_embedding = np.array(resume_embedding).reshape(1, -1)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(job_embedding, resume_embedding)[0][0]
    
    # Convert to percentage
    return round(similarity * 100, 2)

def generate_summary(job_desc, resume_text, client):
    prompt = f"""
    Analyze this resume against the provided job description and provide:
    1. A brief summary of the candidate's most relevant qualifications
    2. Key skills that match the job requirements
    3. Any potential gaps or missing qualifications
    4. An overall suitability assessment
    
    JOB DESCRIPTION:
    {job_desc}
    
    RESUME TEXT:
    {resume_text[:4000]}  # Limit text length to avoid token limits
    
    Provide a concise analysis:
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not generate analysis: {str(e)}"

def display_results(candidates, job_desc, client):
    # Display top candidates in a table
    st.subheader("📊 Ranking Results")
    
    # Create dataframe for visualization
    df = pd.DataFrame({
        'Candidate': [c['name'] for c in candidates],
        'Score': [c['score'] for c in candidates]
    })
    
    # Create bar chart
    fig = px.bar(df, x='Score', y='Candidate', orientation='h', 
                 title='Candidate Ranking by Match Score',
                 color='Score', color_continuous_scale='Blues')
    st.plotly_chart(fig)
    
    # Display details for each candidate
    for i, candidate in enumerate(candidates):
        with st.expander(f"{i+1}. {candidate['name']} - Score: {candidate['score']}%"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Visual score indicator
                score = candidate['score']
                st.metric("Match Score", f"{score}%")
                
                # Progress bar for score visualization
                st.progress(score/100)
                
                # Color code based on score
                if score >= 80:
                    st.success("Highly Qualified")
                elif score >= 60:
                    st.warning("Moderately Qualified")
                else:
                    st.error("Not Well Qualified")
            
            with col2:
                if st.button(f"Generate Detailed Analysis", key=f"analyze_{i}"):
                    with st.spinner("Generating analysis..."):
                        if st.session_state.demo_mode:
                            # Demo mode analysis
                            demo_analysis = f"""
                            **Demo Analysis for {candidate['name']}**
                            
                            This is a sample analysis that would be generated by OpenAI GPT in a real scenario.
                            
                            Based on the resume content, this candidate appears to have:
                            - Strong Java development experience
                            - Experience with Spring Boot and Microservices
                            - Some exposure to cloud platforms
                            
                            **Recommendation**: This candidate would be worth interviewing for the Java Developer position.
                            """
                            st.write(demo_analysis)
                        else:
                            analysis = generate_summary(job_desc, candidate['text'], client)
                            st.write(analysis)

def api_key_input():
    st.sidebar.header("🔑 API Key Configuration")
    
    # Option 1: Input API key manually
    api_key_input = st.sidebar.text_input(
        "Enter your OpenAI API Key", 
        type="password",
        placeholder="sk-...",
        value=st.session_state.api_key,
        help="You can get your API key from https://platform.openai.com/api-keys",
        key="api_key_input_field"
    )
    
    # Update session state when input changes
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        st.session_state.api_key_valid = False
        st.session_state.api_key_checked = False
        st.session_state.api_key_error = ""
    
    # Validate button
    if st.sidebar.button("Validate API Key", key="validate_btn"):
        with st.sidebar:
            with st.spinner("Validating API key..."):
                is_valid, message = validate_api_key(st.session_state.api_key)
                st.session_state.api_key_checked = True
                st.session_state.api_key_error = message
                if is_valid:
                    st.session_state.api_key_valid = True
                    st.success("✅ API key is valid!")
                else:
                    st.session_state.api_key_valid = False
                    st.error(f"❌ {message}")
    
    # Show current API key status
    if st.session_state.api_key_checked:
        if st.session_state.api_key_valid:
            st.sidebar.success("✅ API key is set and valid")
        else:
            st.sidebar.error(f"❌ {st.session_state.api_key_error}")
    
    # Clear API key button
    if st.session_state.api_key:
        if st.sidebar.button("Clear API Key"):
            st.session_state.api_key = ""
            st.session_state.api_key_valid = False
            st.session_state.api_key_checked = False
            st.session_state.api_key_error = ""
            st.rerun()
    
    # Debug info (collapsible)
    with st.sidebar.expander("Debug Info"):
        st.write(f"API Key (first 10 chars): {st.session_state.api_key[:10] + '...' if st.session_state.api_key else 'None'}")
        st.write(f"API Key Valid: {st.session_state.api_key_valid}")
        st.write(f"API Key Checked: {st.session_state.api_key_checked}")
        st.write(f"Demo Mode: {st.session_state.demo_mode}")
        st.write(f"Error: {st.session_state.api_key_error}")
    
    # Demo mode toggle
    st.sidebar.markdown("---")
    use_demo = st.sidebar.checkbox("Use Demo Mode (without API key)", value=st.session_state.demo_mode)
    if use_demo != st.session_state.demo_mode:
        st.session_state.demo_mode = use_demo
        st.rerun()
    
    # Show current mode status
    if st.session_state.demo_mode:
        st.sidebar.info("🔶 Running in demo mode")
    elif not st.session_state.api_key_valid:
        st.sidebar.warning("⚠️ Valid API key is required for full functionality")
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **How it works:**
    1. Set your OpenAI API key or use demo mode
    2. Paste a job description
    3. Upload multiple resumes (PDF)
    4. The AI will analyze and rank them
    5. Review the results and detailed analysis
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Your API key is only stored in your browser session and is not saved anywhere.")

def run_demo_analysis(job_desc, uploaded_files):
    """Run a demo analysis with sample data"""
    st.info("🔶 Running in demo mode with sample data")
    
    # Sample resume data for demo
    sample_resumes = [
        {
            "name": "Altaf Java Developer.pdf",
            "score": 87,
            "text": "Java Developer with 10 years of experience. Expertise in Java 17, Spring Boot, Microservices, Docker, Kubernetes. Strong background in building enterprise applications. Experience with CI/CD pipelines and cloud platforms."
        },
        {
            "name": "Senior Java Engineer.pdf",
            "score": 72,
            "text": "Senior Java Engineer with 8 years of experience. Proficient in Java, Spring Framework, REST APIs. Some experience with microservices and Docker. Looking for challenging opportunities in backend development."
        },
        {
            "name": "Full Stack Developer.pdf", 
            "score": 65,
            "text": "Full Stack Developer with 6 years of experience. Strong frontend skills with React and Angular. Backend experience with Java and Node.js. Familiar with basic Docker concepts."
        },
        {
            "name": "Backend Specialist.pdf",
            "score": 93,
            "text": "Backend Specialist with 12 years of Java experience. Deep expertise in Microservices architecture, Spring Boot, Kubernetes, CI/CD pipelines. AWS certified with extensive cloud experience."
        }
    ]
    
    # If files were uploaded, use their names
    if uploaded_files:
        for i, file in enumerate(uploaded_files):
            if i < len(sample_resumes):
                sample_resumes[i]["name"] = file.name
    
    # Display results
    display_results(sample_resumes, job_desc, None)

def main():
    st.title("📄 AI Resume Screening Agent")
    st.write("Upload job description and resumes to automatically rank candidates based on relevance")
    
    # API key input
    api_key_input()
    
    # Initialize OpenAI client if not in demo mode
    client = None
    if not st.session_state.demo_mode and st.session_state.api_key_valid:
        client = initialize_openai_client()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Job Description")
        job_desc = st.text_area(
            "Paste the job description here", 
            height=250,
            placeholder="Enter the job requirements, skills needed, qualifications, etc.",
            key="job_desc"
        )
    
    with col2:
        st.subheader("Resume Upload")
        uploaded_files = st.file_uploader(
            "Upload resumes (PDF files)", 
            type="pdf", 
            accept_multiple_files=True,
            help="You can select multiple PDF files at once",
            key="resume_upload"
        )
    
    if st.button("🚀 Analyze Resumes", type="primary", key="analyze_btn") and job_desc:
        if not job_desc.strip():
            st.error("Please enter a job description")
            return
        
        # Check if we're in demo mode
        if st.session_state.demo_mode:
            run_demo_analysis(job_desc, uploaded_files)
            return
            
        # Check if we have a valid API key
        if not st.session_state.api_key_valid:
            st.error("Please enter a valid OpenAI API key and click 'Validate API Key' or enable demo mode")
            return
            
        if not uploaded_files:
            st.error("Please upload at least one resume")
            return
        
        with st.spinner("Analyzing resumes..."):
            # Get job description embedding
            job_embedding = get_embedding(job_desc, client)
            
            if job_embedding is None:
                st.error(f"Failed to process job description. Error: {st.session_state.api_key_error}")
                return
            
            candidates = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing {i+1} of {len(uploaded_files)}: {uploaded_file.name}")
                
                # Extract text from PDF
                resume_text = extract_text_from_pdf(uploaded_file)
                
                if not resume_text.strip():
                    st.warning(f"Could not extract text from {uploaded_file.name}. Skipping...")
                    continue
                
                # Clean the text
                cleaned_text = clean_resume_text(resume_text)
                
                # Get resume embedding
                resume_embedding = get_embedding(cleaned_text, client)
                
                if resume_embedding is None:
                    st.warning(f"Failed to process {uploaded_file.name}. Error: {st.session_state.api_key_error}")
                    continue
                
                # Calculate similarity score
                score = calculate_similarity(job_embedding, resume_embedding)
                
                candidates.append({
                    "name": uploaded_file.name,
                    "score": score,
                    "text": cleaned_text
                })
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            if not candidates:
                st.error("No resumes could be processed. Please check your files and try again.")
                return
            
            # Sort candidates by score
            candidates.sort(key=lambda x: x["score"], reverse=True)
            
            # Display results
            display_results(candidates, job_desc, client)
    
    elif not job_desc:
        st.info("👈 Please enter a job description to get started")

if __name__ == "__main__":
    main()