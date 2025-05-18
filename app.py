
!pip install openai pdfplumber PyMuPDF  pytesseract pdf2image --quiet
!pip install reportlab PyPDF2  beautifulsoup4
!pip install --upgrade openai
!pip install groq
!pip install streamlit


!pip install requests

!pip install pycountry python-dotenv


# %%writefile app.py

import os
import io
import re
import requests
import pdfplumber
import pytesseract
import fitz  # PyMuPDF
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import NameObject, BooleanObject
from pdf2image import convert_from_bytes
import streamlit as st
import json, time, base64, tempfile
from io import BytesIO
import requests, fitz, pycountry
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote_plus
from dotenv import load_dotenv
from groq import Groq
import ast
from google.colab import files


# API Keys
os.environ["GROQ_API_KEY"] = "Replace with your GROQ API "  # Replace with your GROQ API 
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama3-8b-8192"
load_dotenv()
SERPER_API_KEY = "Replace with your GROQ API key"

GROQ_API_KEY = "Replace with your GROQ API key"
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize session state
# if 'search_history' not in st.session_state:
#     st.session_state.search_history = []
# if 'mode' not in st.session_state:
#     st.session_state.mode = "Assistant"
# if 'search_results' not in st.session_state:
#     st.session_state.search_results = []

def classify_query_mode(query: str) -> int:
    """
    Uses a Hugging-Face Llama-3 model hosted by Groq to decide
    whether the user wants the tax assistant (0) or the general
    chatbot (1).  **Returns 0 or 1 only.**
    """
    if not GROQ_API_KEY:
        # fallback — assume assistant if no key
        return 0
    prompt = (
        "You are a classifier.  Output **only** the single digit 0 or 1:\n"
        "0 → the query is explicitly about Pakistani tax forms, filing, "
        "deductions, or other tax-assistant topics.\n"
        "1 → any other type of conversational request.\n\n"
        f"User query:\n{query}\n\nAnswer:"
    )
    try:
        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1,
        )
        result = completion.choices[0].message.content.strip()
        return 0 if result.startswith("0") else 1
    except Exception:
        return 0

# Search helpers
def serper_search(query, country_code="pk"):
    url = "https://google.serper.dev/search"
    country_domain = "site:.gov.pk OR site:.fbr.gov.pk"
    search_query = f"{query} tax form {country_domain} filetype:pdf"
    data = {"q": search_query, "gl": "pk", "hl": "en"}
    try:
        if not SERPER_API_KEY:
            print("SERPER API key not found. Search disabled.")
            # st.warning("SERPER API key not found. Search disabled.")
            return []
        headers = {"X-API-KEY": SERPER_API_KEY}
        r = requests.post(url, json=data, headers=headers, timeout=15)
        if r.status_code == 200:
            print(r.status_code)
            return r.json().get("organic", [])
        # st.error(f"Search error: {r.status_code}")
    except Exception as e:
        print(f"Search failed: {e}")
        # st.error(f"Search failed: {e}")
    return []

# Fallback search method (limited, but free)
def fallback_search(query, country_code=""):
    try:
        # Format country code for search
        country_name = next((country.name for country in pycountry.countries if country.alpha_2.lower() == country_code.lower()), "")

        # Use a different free API or direct scraping approach
        search_query = quote_plus(f"{query} {country_name} tax form pdf")
        url = f"https://ddg-api.herokuapp.com/search?query={search_query}&limit=5"

        response = requests.get(url)
        if response.status_code == 200:
            results = response.json()
            # Convert to a format similar to Serper
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", "")
                })
            return formatted_results
        return []
    except Exception as e:
        st.error(f"Fallback search failed: {str(e)}")
        return []

# Use LLM to extract relevant information from search results
def analyze_search_results(results, query, country):
    if not GROQ_API_KEY:
        return results  # Return unprocessed results if no LLM available

    try:
        # Prepare results for LLM analysis
        results_text = json.dumps(results[:5], indent=2)

        prompt = f"""
        I'm looking the tax forms for Pakistan".

        Here are search results:
        {results_text}

        Please analyze these results and tell me:
        1. Which result is most likely the official tax form I need?
        2. Is this result from an official government source?
        3. What specific form number or name should I be looking for?
        4. Any additional forms I might need based on this search intent?

        Format your response as JSON with the following keys:
        {{
          "best_result_index": 0-4 or -1,
          "is_official": true/false,
          "form_name": "",
          "form_description": "",
          "additional_forms": ["FormA", "FormB", ...],
          "action_recommendation": "short summary of next steps for user"
        }}
        """
        # Call Groq API with mixed model approach (prefer cheaper model)
        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192",  # Free/cheaper model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=800
        )

        try:
            # Parse the response as JSON
            analysis = json.loads(completion.choices[0].message.content)
            return results, analysis
        except json.JSONDecodeError:
            # If parsing fails, return the original results
            return results, None

    except Exception as e:
        st.error(f"LLM analysis failed: {str(e)}")
        return results, None

# Try to download PDF
def fetch_pdf(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        st.write(f"Attempting to download PDF from: {url}")
        r = requests.get(url, headers=headers, timeout=15)
        st.write(f"Response status code: {r.status_code}")
        st.write(f"Content-Type: {r.headers.get('Content-Type', 'Not specified')}")

        if r.status_code == 200:
            if 'application/pdf' in r.headers.get('Content-Type', ''):
                st.success("Successfully retrieved PDF!")
                return BytesIO(r.content)
            else:
                st.info("URL doesn't point directly to a PDF. Searching for PDF links on the page...")
                # Try to find PDF links if this is an HTML page
                pdf_url = find_pdf_in_html_page(url, r.text)
                if pdf_url:
                    st.info(f"Found PDF link: {pdf_url}")
                    return fetch_pdf(pdf_url)
                else:
                    st.warning("No PDF links found on the page")
        else:
            st.error(f"Failed to retrieve URL: {r.status_code}")
        return None
    except Exception as e:
        st.error(f"Error fetching PDF: {str(e)}")
        return None

# Scrape .pdf links from HTML page
def find_pdf_in_html_page(url, html_content=None):
    try:
        if not html_content:
            r = requests.get(url, timeout=10)
            html_content = r.text

        soup = BeautifulSoup(html_content, "html.parser")
        pdf_links = []

        # Look for PDF links
        st.write("Scanning page for PDF links...")
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.lower().endswith('.pdf'):
                full_url = href if href.startswith("http") else urljoin(url, href)
                pdf_links.append((full_url, link.text.strip()))
                st.write(f"Found PDF link: {full_url} - {link.text.strip()}")

        st.write(f"Total PDF links found: {len(pdf_links)}")

        # First, look for links with "tax", "form", or "return" in them
        for link_url, link_text in pdf_links:
            combined_text = (link_url + " " + link_text).lower()
            if any(keyword in combined_text for keyword in ["tax", "form", "return", "income"]):
                st.success(f"Selected most relevant PDF: {link_url}")
                return link_url

        # If no specific tax links, return the first PDF link
        if pdf_links:
            st.info(f"No tax-specific PDFs found. Using first PDF: {pdf_links[0][0]}")
            return pdf_links[0][0]

        st.warning("No PDF links found on the page")
        return None
    except Exception as e:
        st.error(f"Error finding PDF links: {str(e)}")
        return None

# Add to search history
# def add_to_history(country, query, pdf_url=None):
#     timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
#     st.session_state.search_history.append({
#         "timestamp": timestamp,
#         "country": country,
#         "query": query,
#         "pdf_url": pdf_url
#     })
#     # Keep only the last 10 searches
#     if len(st.session_state.search_history) > 10:
#         st.session_state.search_history = st.session_state.search_history[-10:]

# Get country code from name
def get_country_code(country_name):
    try:
        country = pycountry.countries.get(name=country_name)
        return country.alpha_2 if country else ""
    except:
        return ""

# Function to suggest other forms
def suggest_other_forms():
    """Suggest other forms from search results if current form isn't fillable"""
    if ('search_results' in st.session_state and
        st.session_state.search_results and
        'selected_pdf' in st.session_state):

        selected_idx = st.session_state.selected_pdf
        st.markdown("### 🔎 Try These Other Forms")

        # Display up to 5 alternative forms
        other_forms = [r for i, r in enumerate(st.session_state.search_results[:5])
                      if i != selected_idx]

        for idx, result in enumerate(other_forms):
            title = result.get('title', 'Untitled Form')
            link = result.get('link', '')

            st.markdown(f"**{idx+1}. {title}**")
            if st.button(f"Try Form #{idx+1}", key=f"try_form_{idx}"):
                with st.spinner(f"Fetching alternative form #{idx+1}..."):
                    pdf_bytes = fetch_pdf(link)
                    if pdf_bytes:
                        st.session_state.pdf_bytes = pdf_bytes
                        st.session_state.selected_pdf = idx
                        st.session_state.form_fields = extract_form_fields(pdf_bytes)

# Extract form fields
def extract_form_fields(pdf_bytes):
    # Implementation needed
    return []

# Tax agent response
def tax_agent_response(user_query, tax_form_type=None, form_fields=None):
    """Generate an agent-like response to user tax questions using LLM"""
    if not GROQ_API_KEY:
        return "I need an LLM API key to provide detailed assistance. Please upload a PDF or search for forms directly."

    try:
        # Create context from available information
        context = f"The user is asking about Pakistani tax: '{user_query}'\n"

        if tax_form_type:
            context += f"They previously selected tax form type: {tax_form_type}\n"

        if form_fields and len(form_fields) > 0:
            fields_sample = ", ".join([f["name"] for f in form_fields[:5]])
            context += f"They are looking at a form with fields including: {fields_sample}\n"

        # Pakistan-specific tax information to help ground the response
        context += """
        Pakistan tax information:
        - FBR (Federal Board of Revenue) is the main tax authority
        - Common tax forms include income tax returns, sales tax returns, and withholding tax statements
        - The tax year in Pakistan typically runs from July to June
        - NTN (National Tax Number) is required for filing taxes in Pakistan
        """

        prompt = f"""
        {context}

        As a LifePilot Tax Agent specialized in Pakistani taxation, provide a helpful response to their query.
        If they are asking about which tax form they need, explain the options and help them decide.
        If they are asking about how to fill a specific field, provide guidance.
        If they need information about tax filing deadlines or procedures, provide accurate information.

        Your response should be:
        1. Conversational and helpful
        2. Specific to Pakistan's tax system
        3. Brief but informative
        """

        # Call Groq API
        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192",  # Or "mixtral-8x7b-32768" if available
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"I encountered an error while processing your question: {str(e)}"

# Recommend tax form type
def recommend_tax_form_type(user_query):
    """Recommend appropriate tax form type based on user's situation"""
    if not GROQ_API_KEY:
        return ["Income Tax Return", "Sales Tax Return", "Withholding Tax Statement"]

    try:
        prompt = f"""
        The user is asking about Pakistani taxes: "{user_query}"

        Based on their query, which of these Pakistani tax form types would be most relevant?
        - Income Tax Return
        - Sales Tax Return
        - Withholding Tax Statement
        - Property Tax
        - Customs Duty
        - Advance Tax
        - Wealth Statement

        Return only the names of the top 3 most relevant form types as a JSON array:
        ["Form Type 1", "Form Type 2", "Form Type 3"]
        """

        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100
        )

        try:
            # Parse the response as JSON
            return json.loads(completion.choices[0].message.content)
        except json.JSONDecodeError:
            # Fallback to default options
            return ["Income Tax Return", "Sales Tax Return", "Withholding Tax Statement"]

    except Exception as e:
        return ["Income Tax Return", "Sales Tax Return", "Withholding Tax Statement"]

# PDF processing functions
def extract_text_from_pdf(file_bytes):
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def extract_text_with_ocr(file_bytes):
    images = convert_from_bytes(file_bytes)
    return "\n".join(pytesseract.image_to_string(img) for img in images)

def extract_labels_from_text(file_bytes):
    labels = set()
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.splitlines():
                matches = re.findall(r"(?:\d+\.\s*)?([A-Za-z\s\/\(\)]+):", line)
                for match in matches:
                    clean = match.strip()
                    if len(clean) > 1:
                        labels.add(clean)
    return list(labels)

def extract_all_fields_from_text(text):
    """
    Extract key-value pairs from bullet-style text with detailed logging.
    Fixes space-matching to preserve multi-word labels like 'Full Name'.
    """
    data = {}
    print("🔍 Starting field extraction...")

    for line in text.splitlines():
        original_line = line
        line = line.strip("• ").strip()
        if not line:
            continue

        print(f"\n📄 Processing line: '{original_line}'")

        # Colon-based format
        match_colon = re.match(r"^([A-Za-z\s\/\(\)\[\]\-\.]+?)[:：]\s*(.+)$", line)
        if match_colon:
            label = match_colon.group(1).strip()
            value = match_colon.group(2).strip()
            data[label] = value
            print(f"✅ Matched colon format → '{label}': '{value}'")
            continue

        # Dotted-line format
        match_dots = re.match(r"^([A-Za-z\s\/\(\)\[\]\-\.]+?)\s*[\.\-]{2,}\s*(.+)$", line)
        if match_dots:
            label = match_dots.group(1).strip()
            value = match_dots.group(2).strip()
            data[label] = value
            print(f"✅ Matched dotted-line format → '{label}': '{value}'")
            continue

        # Space-separated format — try longest match first
        words = line.split()
        found = False
        for i in reversed(range(1, min(5, len(words)))):
            label_candidate = " ".join(words[:i])
            value_candidate = " ".join(words[i:])
            if len(label_candidate) > 1 and len(value_candidate) > 1:
                data[label_candidate] = value_candidate
                print(f"✅ Matched space format → '{label_candidate}': '{value_candidate}'")
                found = True
                break

        if not found:
            print(f"⚠️ No match found for line: '{original_line}'")

    print("\n✅ Final parsed user data dictionary:", data)
    return data
def extract_info_from_text(text, field_names):
    data = {}
    for line in text.splitlines():
        for field in field_names:
            pattern = re.compile(rf"{re.escape(field)}\s*:\s*(.*)", re.IGNORECASE)
            match = pattern.search(line)
            if match:
                data[field] = match.group(1).strip()
    print("data",data)
    return data

def get_field_mapping_from_llm(form_fields, user_data):
    print("form_fields:", form_fields)
    print("user_data:", user_data)
    prompt = f"""
You are a form-filling assistant. Match the FORM FIELD NAMES with the closest values from USER DATA.

FORM FIELDS:
{form_fields}

USER DATA (key-value pairs):
{user_data}

Return a Python dict that maps each FORM FIELD to:
- The best matching USER DATA field name, or
- A Python expression using user fields (e.g., "First Name + ' ' + Last Name"), or
- `None` if no match is found.

Example:
{{
  "Name of Candidate": "Full Name",
  "Cell No": "Phone Number",
  "CNIC No": "CNIC",
  ...
}}
"""
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a form-matching assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    content = response.choices[0].message.content.strip()
    print("LLM Mapping Response:\n", content)

    # Extract dictionary from the response text
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        dict_str = match.group(0)
        try:
            field_mapping = ast.literal_eval(dict_str)
            print("Parsed field_mapping:", field_mapping)
            return field_mapping
        except Exception as e:
            print("Error parsing dictionary:", e)
    else:
        print("No dictionary found in LLM response.")

    return {}

def reconstruct_user_data(field_mapping, raw_user_data):
    print("field_mapping in reconstruct_user_data", field_mapping)
    print("raw_user_data in reconstruct_user_data", raw_user_data)
    final_data = {}
    for form_field, user_key_expr in field_mapping.items():
        if user_key_expr is None:
            continue
        try:
            value = eval(user_key_expr, {}, raw_user_data)
            print("value", value)
        except Exception:
            value = raw_user_data.get(user_key_expr) if isinstance(user_key_expr, str) else None
        if value:
            final_data[form_field] = value
    return final_data

def extract_acroform_fields(reader):
    fields = reader.get_fields()
    return list(fields.keys()) if fields else []

def fill_pdf_acroform(template_bytes, user_data):
    reader = PdfReader(io.BytesIO(template_bytes))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.update_page_form_field_values(writer.pages[0], user_data)
    writer._root_object.update({NameObject("/NeedAppearances"): BooleanObject(True)})
    output_stream = io.BytesIO()
    writer.write(output_stream)
    output_stream.seek(0)
    return output_stream

def auto_fill_flat_pdf_smart(pdf_bytes, output_path, user_data):
    coords_map = {}
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            words = page.extract_words()
            lines = {}
            for w in words:
                line_key = round(w['top'], 1)
                lines.setdefault(line_key, []).append(w)

            for label_key in user_data:
                for line_top, line_words in lines.items():
                    line_text = " ".join(word['text'] for word in line_words)
                    if label_key.lower() in line_text.lower():
                        first_label_word = label_key.split()[0].lower()
                        for word in line_words:
                            if word['text'].lower() == first_label_word:
                                label_x1 = word['x1']
                                label_top = word['top']
                                break
                        else:
                            label_x1 = line_words[0]['x1']
                            label_top = line_words[0]['top']
                        insert_x = label_x1 + 50
                        coords_map[label_key] = (i, insert_x, label_top)
                        break

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for key, value in user_data.items():
        if key in coords_map:
            page_num, x, y = coords_map[key]
            page = doc[page_num]
            rect = fitz.Rect(x, y, x + 150, y + 20)
            page.draw_rect(rect, color=(1, 0, 0), width=0.5)
            page.insert_text((x + 2, y + 2), str(value), fontsize=12, fontname="helv", color=(0, 0, 0))
    doc.save(output_path)
    with open(output_path, "rb") as file:
        return file.read()

# Constants for services
SERVICES = {
    "fbr_tax": {
        "name": "FBR Tax Registration",
        "form_url": "https://www.sindheducation.gov.pk/Contents/Careers/APPLICATION%20FORM%20ME.pdf"
    },
    "property": {
        "name": "Property Registration (Sindh)",
        "form_url": "https://www.sindheducation.gov.pk/Contents/Careers/APPLICATION%20FORM%20ME.pdf"
    },
      "actual_form": {
         "name": "FBR Form",
         "form_url": "https://www.themodernfirm.com/wp-content/uploads/2017/12/Sample-Fillable-PDF.pdf"
      }

}

# Main app layout
# st.set_page_config(
#     page_title="LifePilot - Pakistan Tax Form Finder",
#     page_icon="📋",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# Sidebar
# with st.sidebar:
#     st.title("📋LifePilot")
#     st.caption("Pakistan Tax Assistant")


#     st.info("Your search history will appear here👉")

#     st.divider()
#     st.markdown("📋LifePilot| Made with Streamlit")

# st.title("Pakistan Tax Form Finder")
# st.markdown("Search, preview, and get assistance with official Pakistani tax forms.")

# Display active mode
# mode = st.session_state.get("mode", "Assistant")
# st.markdown(f"### 🔄 Active Mode: Assistant")

# user_query = st.text_input("Ask your tax question:", placeholder="Which tax form do I need as a salaried employee?")

# if user_query:
#     session = "Assistant" if any(x in user_query.lower() for x in ["which form", "do i need", "recommend"]) else "Chatbot"
query = input("How can I assist you today? (e.g., FBR tax registration or Property registration): ").lower()
service_key = "fbr_tax" if "fbr" in query or "tax" in query else "property" if "property" in query else None
session = "Assistant" if "which form" in query.lower() or "what form" in query.lower() or "do i need" in query.lower() else "Chatbot"
print("service_key",service_key)
if service_key:

    agent_response = tax_agent_response(query)
    print("Agent response", agent_response)
    print("session", session)
    if not agent_response:
        mode = session
    if agent_response and "form" in agent_response.lower() or "need" in agent_response.lower():
        mode = "Assistant"
    else:
        mode = "Chatbot"

    if mode == "Chatbot":
        print("mode of Chatbot", mode)

    elif mode == "Assistant":
        serper_response=serper_search(query)
        print("serper_search", serper_response)
        print("mode of Chatbot", mode)
        print("agent_Resp", agent_response)

        recommended_types = recommend_tax_form_type(query)
        form_links = []
        print("recommended_types", recommended_types)

        if recommended_types:
            print("before try")

            try:
                print(f"\n🔽 Downloading form from {SERVICES['fbr_tax']['form_url']}...")
                resp = requests.get(SERVICES['actual_form']['form_url'], verify=False, timeout=10)
                resp.raise_for_status()
                form_bytes = resp.content
            except Exception as e:
                print(f"❌ Download failed: {e}")
                form_bytes = None

            if form_bytes and input("Do you want help filling the form? (yes/no): ").lower() == "yes":
                reader = PdfReader(io.BytesIO(form_bytes))
                acro_fields = extract_acroform_fields(reader)

                if acro_fields:
                    print("\n📝 AcroForm Fields Detected:")
                    for field in acro_fields:
                        print(f"• {field}")

                    mode = input("Enter data manually or upload info PDF? [1=Manual, 2=Upload]: ")
                    user_data = {}

                    if mode == "1":
                        for field in acro_fields:
                            user_data[field] = input(f"{field}: ")
                    else:
                        print("📤 Upload your data PDF:")
                        uploaded = files.upload()
                        uploaded_file = next(iter(uploaded.values()))
                        text = extract_text_from_pdf(uploaded_file)
                        user_data = extract_info_from_text(text, acro_fields)

                    filled_pdf = fill_pdf_acroform(form_bytes, user_data)
                    with open("filled_form.pdf", "wb") as f:
                        f.write(filled_pdf.read())
                    files.download("filled_form.pdf")

                else:
                    print("⚠️ No AcroForm fields. Switching to smart flat-form filling.")
                    labels = extract_labels_from_text(form_bytes)
                    print("\n📋 Detected form labels:", labels)
                    print("📤 Upload your data PDF:")
                    uploaded = files.upload()
                    uploaded_file = next(iter(uploaded.values()))
                    print("uploaded file")
                    user_text = extract_text_from_pdf(uploaded_file)
                    print("user_text", user_text)

                    if not user_text.strip():
                        user_text = extract_text_with_ocr(uploaded_file)

                    print("\n🔍 Extracted:", user_text, labels)
                    raw_user_data = extract_all_fields_from_text(user_text)
                    print("raw_user_data", raw_user_data)
                    field_mapping = get_field_mapping_from_llm(labels, raw_user_data)
                    print("field_mapping from function", field_mapping)
                    final_user_data = reconstruct_user_data(field_mapping, raw_user_data)
                    print("final_user_data", final_user_data)
                    auto_fill_flat_pdf_smart(form_bytes, "smart_filled_form.pdf", final_user_data)
            else:
                print("👍 You can fill the form manually:", SERVICES['fbr_tax']['form_url'])
        else:
            print("Else of recommended types")
    else:
        print("Else of Mode")
else:
    print("In else of last")
