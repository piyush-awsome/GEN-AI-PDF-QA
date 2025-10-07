import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------------
# Load AI Model
# -------------------------------
@st.cache_resource
def load_model():
    st.info("⏳ Loading AI model... This may take 2-5 minutes the first time")
    model_name = "google/flan-t5-large"  # larger, free, better outputs
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    st.success("✅ Model loaded successfully!")
    return generator

generator = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AI Test Case Generator", page_icon="🧠", layout="centered")
st.title("🧠 AI Requirement-to-Test Case Generator")
st.markdown(
    "Paste a **user story or requirement**, and this AI will generate structured **test scenarios and test cases** for you."
)

# Input Text Area
user_story = st.text_area(
    "📜 Enter User Story / Requirement",
    height=200,
    placeholder="e.g., As a user, I should be able to log in using email and password."
)

# Instructions for user
st.markdown(
    """
**Prompt Tip:** Enter a clear user story.  
Example:  
`As a user, I want to reset my password via email so that I can recover my account if I forget my password.`
"""
)

# Generate Button
if st.button("Generate Test Cases"):
    if user_story.strip():
        with st.spinner("Generating test cases... ⏳"):
            # Structured prompt
            prompt = f"""
You are a highly experienced QA engineer. For the following requirement, generate exactly 2 detailed test scenarios, 
each with 1-3 test cases. Each test case must include:

- Title
- Preconditions
- Steps:
  1.
  2.
  3.
- Expected Result

Requirement:
{user_story}
"""
            try:
                # Generate output
                result = generator(
                    prompt,
                    max_new_tokens=700,  # remove max_length to avoid warning
                    num_return_sequences=1
                )
                output_text = result[0]['generated_text']
                st.subheader("✅ Generated Test Cases:")
                st.code(output_text, language="text")
            except Exception as e:
                st.error(f"❌ Error generating test cases: {e}")
    else:
        st.warning("Please enter a valid user story or requirement.")

# Download Button
if 'output_text' in locals():
    st.download_button(
        label="⬇️ Download Test Cases",
        data=output_text,
        file_name="test_cases.txt",
        mime="text/plain"
    )
