import os
import streamlit as st
from dotenv import load_dotenv
from langchain_helper import get_few_shot_db_chain, get_gemini_model, DomainError

# Load environment variables
load_dotenv()

st.set_page_config(page_title="AtliQ T Shirts: Database Q&A", page_icon="👕", layout="centered")
st.title("AtliQ T Shirts: Database Q&A 👕")

# Check for API key in environment
env_api_key = os.getenv("GEMINI_API_KEY", "").strip()
if env_api_key in ("YOUR_GEMINI_API_KEY", "your api key here"):
    env_api_key = ""

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    if not env_api_key:
        user_api_key = st.text_input(
            "Gemini API Key:",
            type="password",
            help="Enter your Gemini API key or set GEMINI_API_KEY in the .env file"
        )
        api_key = user_api_key.strip()
    else:
        st.success("✅ GEMINI_API_KEY loaded from .env")
        api_key = env_api_key

    configured_model = get_gemini_model()
    model_options = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    default_index = model_options.index(configured_model) if configured_model in model_options else 0
    selected_model = st.selectbox("Gemini Model", options=model_options, index=default_index)

    st.markdown("---")
    st.markdown("**Database Status:** `atliq_tshirts` (MySQL)")

question = st.text_input("Question: ")

if question:
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY is not configured. Please add it to your `.env` file or enter it in the sidebar.")
    else:
        with st.spinner("Searching schema, retrieving few-shot examples, and querying database..."):
            try:
                chain = get_few_shot_db_chain(api_key=api_key, model=selected_model)
                response = chain.run(question)

                st.header("Answer")
                st.write(response)

                with st.expander("🔍 View Technical Details (SQL & Semantic Retrieval)"):
                    if chain.last_sql:
                        st.subheader("Generated SQL Query")
                        st.code(chain.last_sql, language="sql")
                    if chain.last_result is not None:
                        st.subheader("Database Result")
                        st.code(chain.last_result)
                    if chain.last_examples:
                        st.subheader("Retrieved Few-Shot Examples")
                        for idx, ex in enumerate(chain.last_examples, 1):
                            st.markdown(f"**Example {idx}:** {ex.get('Question')}")
                            st.code(ex.get('SQLQuery'), language="sql")
            except DomainError as e:
                st.warning(f"🚫 {e}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
