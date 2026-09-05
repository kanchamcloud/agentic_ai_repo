import os
import certifi
import requests
import streamlit as st


from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_agent

# ----------------------------------------------------
# 1. Environment & API Setup
# ----------------------------------------------------
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHER_STACK_API = os.getenv("WEATHER_STACK_API")
# --------------------------------------------------

# ----------------------------------------------------
# 2. Tools & Agent Initialization (Cached for performance)
# ----------------------------------------------------
@tool
def get_weather_data(city: str) -> str:
    """Fetch weather information for a given city."""
    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHER_STACK_API}&query={city}"
    )
    try:
        response = requests.get(url)
        data = response.json()
    except Exception as e:
        return f"Failed to connect to weather service: {str(e)}"

    if "current" not in data:
        return f"Could not fetch weather data for {city}. Response error: {data.get('error', {}).get('info', 'Unknown error')}"
    
    return (
        f"city: {city}\n"
        f"Temperature: {data['current']['temperature']} degrees C\n"
        f"Condition: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )

@st.cache_resource
def init_agent():
    search_tool = TavilySearchResults(max_results=2) 
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY
    )
    tools = [search_tool, get_weather_data]
    return create_agent(model=llm, tools=tools)

agent = init_agent()

# ----------------------------------------------------
# 3. Streamlit Page Configuration & Background Styling
# ----------------------------------------------------
st.set_page_config(page_title="Scenic AI Assistant", page_icon="🌿", layout="centered")

# CSS injection for a crisp hills and fields scenery background with an opacity layer for readability
scenery_background_css = """
<style>
.stApp {
    background-image: linear-gradient(rgba(255, 255, 255, 0.4), rgba(255, 255, 255, 0.2)), 
                      url('https://unsplash.com');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Glassmorphism panel styling to make content legible over the background image */
[data-testid="stHeader"], .main .block-container {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 2rem;
    margin-top: 2rem;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
}

/* Fix chat components design inside the glass template */
[data-testid="stChatMessage"] {
    background-color: rgba(255, 255, 255, 0.85) !important;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    margin-bottom: 10px;
}
</style>
"""
st.markdown(scenery_background_css, unsafe_allow_html=True)

# ----------------------------------------------------
# 4. Streamlit Main Interface Core
# ----------------------------------------------------
st.title("🌱 Scenic Weather & Search Assistant")
st.caption("Ask questions about world facts, current events, or real-time weather information!")

# Track conversation history inside Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages from session state
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input via Streamlit Chat Input component
if user_prompt := st.chat_input("Type your question here (e.g., What is the capital of India and its current weather?)"):
    
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # 2. Process Response with Agent Loader Spinner
    with st.chat_message("assistant"):
        with st.spinner("Searching the web and analyzing data..."):
            try:
                response = agent.invoke({
                    "messages": [("user", user_prompt)]
                })
                final_answer = response["messages"][-1].content
                st.markdown(final_answer)
                
                # Save Assistant response to history
                st.session_state.chat_history.append({"role": "assistant", "content": final_answer})
            except Exception as e:
                st.error(f"An execution error occurred: {str(e)}")
