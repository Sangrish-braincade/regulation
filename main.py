import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from mistralai import Mistral
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import json
import  requests
from datetime import datetime

MODEL_ID = "gemini-2.5-flash"
PROJECT_ID = st.secrets["VERTEX_PROJECT_ID"]
LOCATION = st.secrets["VERTEX_LOCATION"]

# The JSON string for credentials will need to be parsed
google_applications_credentials_json_str = st.secrets["GOOGLE_APPLICATIONS_CREDENTIALS_JSON"]
GOOGLE_APPLICATIONS_CREDENTIALS_JSON = google_applications_credentials_json_str

# Configure page
st.set_page_config(
    page_title="AI Regulatory Intelligence Platform",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        margin-left: auto;
    }
    .ai-message {
        background-color: #f8f9fa;
        border-left: 4px solid #28a745;
        color: #333;
    }
    .regulatory-alert {
        background-color: #000000;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .status-prohibited { background-color: #dc3545; color: white; padding: 3px 8px; border-radius: 15px; font-size: 0.8em; }
    .status-allowed { background-color: #28a745; color: white; padding: 3px 8px; border-radius: 15px; font-size: 0.8em; }
    .status-restricted { background-color: #ffc107; color: black; padding: 3px 8px; border-radius: 15px; font-size: 0.8em; }
</style>
""", unsafe_allow_html=True)

def get_access_token():
    """Get access token using service account credentials"""
    try:
        service_account_info = json.loads(GOOGLE_APPLICATIONS_CREDENTIALS_JSON)
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        credentials.refresh(Request())
        return credentials.token
    except Exception as e:
        print(f"Error getting access token: {e}")
        return None

def make_gemini_request(prompt: str, max_tokens: int = 1024, temperature: float = 0.1, system_prompt: str = None):
    """Make a request to Gemini API with proper system prompt handling"""
    access_token = get_access_token()
    
    if not access_token:
        raise Exception("Failed to get access token")
    
    url = f"https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:generateContent"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [
            {
                "role": "model",
                "parts": [{"text": f"Please follow this system prompt to the end: {system_prompt}"}]
            },
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature
        }
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    result = response.json()
    
    if "candidates" in result and len(result["candidates"]) > 0:
        candidate = result["candidates"][0]
        if "content" in candidate and "parts" in candidate["content"]:
            return candidate["content"]["parts"][0].get("text", "")
    
    return ""

# Load regulatory data
@st.cache_data
def load_regulatory_data():
    """Load regulatory data from CSV file"""
    try:
        # Try CSV first
        df = pd.read_csv("Sample_Banned_Herbal_Ingredients_USA_Canada_.csv")
        return df
    except FileNotFoundError:
        try:
            # Fallback to Excel if CSV not found
            df = pd.read_excel("Sample_Banned_Herbal_Ingredients_USA_Canada_.xlsx")
            return df
        except FileNotFoundError:
            st.error("❌ Data file not found. Please ensure 'Sample_Banned_Herbal_Ingredients_USA_Canada_.csv' or '.xlsx' is in the same directory.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

# Load data
df = load_regulatory_data()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'regulatory_context' not in st.session_state:
    st.session_state.regulatory_context = df.to_json(orient='records')

def get_regulatory_context(ingredient_name=None, country=None):
    """Get relevant regulatory context for the query"""
    context_df = df.copy()
    
    if ingredient_name:
        context_df = context_df[context_df['Ingredient Name'].str.contains(ingredient_name, case=False, na=False)]
    
    if country:
        context_df = context_df[context_df['Country'].str.contains(country, case=False, na=False)]
    
    return context_df.to_json(orient='records')

def query_mistral_ai(user_message, regulatory_context):
    """Query Mistral AI with regulatory context"""
    system_prompt = f"""
    You are an expert AI regulatory consultant specializing in botanical extracts and nutraceutical compliance.
    
    Your knowledge includes current regulatory data:
    {regulatory_context}
    
    Guidelines:
    1. Always reference specific regulatory data when available
    2. Provide clear, actionable compliance advice
    3. Mention specific restrictions (import, cultivation, sale)
    4. Include citation links when relevant
    5. Flag high-risk ingredients clearly
    6. Suggest alternative regulatory pathways when applicable
    7. Be precise about jurisdictional differences
    
    Answer queries about:
    - Ingredient regulatory status
    - Import/export restrictions
    - Compliance pathways
    - Registration requirements
    - Claims substantiation
    - Risk assessments
    """
    
    try:
        response = make_gemini_request(
            prompt=user_message,
            max_tokens=1024,
            temperature=0.3,
            system_prompt=system_prompt
        )
        return response
    except Exception as e:
        return f"Error querying AI: {str(e)}"

# Sidebar - Regulatory Database Overview
st.sidebar.title("🌿 Regulatory Database")
st.sidebar.markdown("---")

# Quick stats
total_ingredients = len(df)
prohibited_count = len(df[df['Prohibited to Import'] == 'Yes'])
banned_count = len(df[df['Banned'] == 'Yes'])
countries = df['Country'].nunique()

st.sidebar.metric("Total Ingredients", total_ingredients)
st.sidebar.metric("Import Prohibited", prohibited_count)
st.sidebar.metric("Banned Ingredients", banned_count)
st.sidebar.metric("Countries Covered", countries)

# Country filter
selected_country = st.sidebar.selectbox(
    "Filter by Country",
    options=['All'] + sorted(df['Country'].unique().tolist()),
    index=0
)

# Status filter
status_filter = st.sidebar.multiselect(
    "Filter by Status",
    options=['Prohibited to Import', 'Banned', 'Cannot be Grown'],
    default=[]
)

# Apply filters
filtered_df = df.copy()
if selected_country != 'All':
    filtered_df = filtered_df[filtered_df['Country'] == selected_country]

if status_filter:
    for status in status_filter:
        filtered_df = filtered_df[filtered_df[status] == 'Yes']

# Display filtered data
st.sidebar.markdown("### Filtered Results")
st.sidebar.dataframe(filtered_df[['Ingredient Name', 'Country', 'Prohibited to Import', 'Banned']], height=300)

# Main content
st.title("🤖 AI-Enabled Regulatory Intelligence Platform")
st.markdown("### Botanical Extracts & Nutraceutical Compliance Assistant")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔍 AI Regulatory Chat", "📊 Regulatory Dashboard", "📋 Ingredient Database"])

with tab1:
    st.markdown("### Ask our AI Regulatory Expert")
    st.markdown("*Get instant answers about ingredient compliance, registration pathways, and regulatory requirements*")
    
    # Sample questions
    st.markdown("**Sample Questions:**")
    sample_questions = [
        "Is Ephedra banned in the USA?",
        "What are the import restrictions for Kava?",
        "Can I grow Aristolochia commercially?",
        "What's the regulatory pathway for Turmeric in Canada?",
        "Are there any alternatives to banned ingredients?"
    ]
    
    cols = st.columns(len(sample_questions))
    for i, question in enumerate(sample_questions):
        if cols[i].button(f"💡 {question[:20]}...", key=f"sample_{i}"):
            st.session_state.messages.append({"role": "user", "content": question})
            # Get response
            context = get_regulatory_context()
            response = query_mistral_ai(question, context)
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Chat interface
    st.markdown("---")
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message ai-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Ask about regulatory compliance...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get AI response
        with st.spinner("Analyzing regulatory requirements..."):
            # Extract potential ingredient and country from query
            ingredient_mentioned = None
            country_mentioned = None
            
            for ingredient in df['Ingredient Name'].unique():
                if ingredient.lower() in user_input.lower():
                    ingredient_mentioned = ingredient
                    break
            
            for country in df['Country'].unique():
                if country.lower() in user_input.lower():
                    country_mentioned = country
                    break
            
            # Get relevant context
            context = get_regulatory_context(ingredient_mentioned, country_mentioned)
            
            # Query AI
            response = query_mistral_ai(user_input, context)
            
            # Add AI response
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()

with tab2:
    st.markdown("### Regulatory Compliance Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "High Risk Ingredients",
            len(df[(df['Prohibited to Import'] == 'Yes') | (df['Banned'] == 'Yes')]),
            delta=f"{prohibited_count + banned_count} total"
        )
    
    with col2:
        st.metric(
            "Import Prohibited",
            prohibited_count,
            delta=f"{round(prohibited_count/total_ingredients*100, 1)}% of total"
        )
    
    with col3:
        st.metric(
            "Completely Banned",
            banned_count,
            delta=f"{round(banned_count/total_ingredients*100, 1)}% of total"
        )
    
    with col4:
        cultivation_banned = len(df[df['Cannot be Grown'] == 'Yes'])
        st.metric(
            "Cultivation Banned",
            cultivation_banned,
            delta=f"{round(cultivation_banned/total_ingredients*100, 1)}% of total"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Status distribution
        status_data = []
        for _, row in df.iterrows():
            if row['Prohibited to Import'] == 'Yes':
                status_data.append('Import Prohibited')
            elif row['Banned'] == 'Yes':
                status_data.append('Banned')
            elif row['Cannot be Grown'] == 'Yes':
                status_data.append('Cultivation Banned')
            else:
                status_data.append('Allowed')
        
        status_df = pd.DataFrame({'Status': status_data})
        fig1 = px.pie(status_df, names='Status', title="Regulatory Status Distribution")
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Country-wise restrictions
        country_stats = df.groupby('Country').agg({
            'Prohibited to Import': lambda x: (x == 'Yes').sum(),
            'Banned': lambda x: (x == 'Yes').sum(),
            'Cannot be Grown': lambda x: (x == 'Yes').sum()
        }).reset_index()
        
        fig2 = px.bar(country_stats, x='Country', 
                     y=['Prohibited to Import', 'Banned', 'Cannot be Grown'],
                     title="Restrictions by Country",
                     barmode='group')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Risk analysis
    st.markdown("### High-Risk Ingredients Analysis")
    high_risk = df[(df['Prohibited to Import'] == 'Yes') | (df['Banned'] == 'Yes')]
    
    if not high_risk.empty:
        for _, row in high_risk.iterrows():
            risk_level = "HIGH RISK" if row['Banned'] == 'Yes' else "MEDIUM RISK"
            st.markdown(f"""
            <div class="regulatory-alert">
                <strong>⚠️ {row['Ingredient Name']} ({row['Country']})</strong> - <span class="risk-high">{risk_level}</span><br>
                Import: {'❌ Prohibited' if row['Prohibited to Import'] == 'Yes' else '✅ Allowed'} | 
                Sale: {'❌ Banned' if row['Banned'] == 'Yes' else '✅ Allowed'} | 
                Cultivation: {'❌ Banned' if row['Cannot be Grown'] == 'Yes' else '✅ Allowed'}<br>
                <a href="{row['Citations']}" target="_blank">📖 View Official Citation</a>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    st.markdown("### Complete Ingredient Database")
    
    # Search functionality
    search_term = st.text_input("🔍 Search ingredients...", placeholder="Enter ingredient name")
    
    if search_term:
        filtered_df = df[df['Ingredient Name'].str.contains(search_term, case=False, na=False)]
    else:
        filtered_df = df
    
    # Display data with styling
    def style_status(val):
        if val == 'Yes':
            return 'background-color: #ff4757; color: white; font-weight: bold'
        elif val == 'No':
            return 'background-color: #2ed573; color: white; font-weight: bold'
        else:
            return 'background-color: #ffa726; color: white; font-weight: bold'
    
    styled_df = filtered_df.style.applymap(style_status, subset=['Prohibited to Import', 'Banned', 'Cannot be Grown'])
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Export functionality
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name=f"regulatory_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        json_data = filtered_df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download as JSON",
            data=json_data,
            file_name=f"regulatory_data_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    🌿 AI-Enabled Regulatory Intelligence Platform | Powered by Mistral AI<br>
    For botanical extracts and nutraceutical compliance | Last updated: {date}
</div>
""".format(date=datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)
