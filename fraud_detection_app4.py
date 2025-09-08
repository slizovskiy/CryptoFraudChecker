###Crypto-Fraud detection App. Criteria are selected from the regulators  
#### Authors: Samuel Chan, Sergey Slizovskiy, Aseem Pahuja, Grok and Gemini

import streamlit as st
import google.generativeai as genai
import os
import json
import math
import re
import magic
import pandas as pd

# Maximum file size (100MB in bytes)
MAX_SIZE = 100 * 1024 * 1024  # 100MB

# System prompt for file-based evaluation
file_system_prompt = """
System Prompt — White-paper Fraud-Risk Evaluation

1. Purpose  
Evaluate a project’s white-paper (PDF file) against each fraud criteria supplied in JSON. For every criterion you must:

• Locate supporting or contradictory passages in the white-paper text **and any information conveyed by figures or graphs**.  
• Assign one evidence rating:  
  – abundant evidence  
  – some evidence  
  – no evidence                     (document is silent on the point)  
  – insufficient evidence to make a decision  
• Output the exact sentence(s) or concise figure caption(s) you relied on, each ≤ 20 words.  
• Write an evaluation (≤ 60 words) explaining how the quoted material supports the rating.  
• Choose the result:  
  – yes                       → criterion met → fraud indicator  
  – no                        → criterion not met  
  – insufficient evidence to make decision → cannot judge  
  Important:  
    ▸ If Evidence = “no evidence”, you MUST still set result to either “yes” or “no”.  
    ▸ Result may be “insufficient evidence to make decision” only when Evidence is the same phrase.

You may incorporate any internet or external knowledge sources if necessary.

2. Inputs  

• whitepaper_pdf (file): The complete PDF of the white-paper. Extract all text, figure captions, and, where possible, summarize graph data (e.g., axes labels, key values) before analysis.  
• fraud_criteria (JSON): Contains an array of criteria in the format: 
Example A  
{
  [
    {
      "ID": "FCA-001",
      "criteria": "Misleading Statements Inducing Investment",
      "description": "...",
      "scope": "..."
    },
    {
      "ID": "MISC-001",
      "criteria": "Anonymous Or Pseudonymous Team",
      "summary": "..."
    }
    // …more criteria
  ]
}

3. Required Output  

Return a JSON object with:  
• "whitepaper_name": the title of the white-paper (use first PDF heading or metadata).  
• One key exactly matching the fraud_criteria ID ("FCA-001", "MISC-015", etc.). 
Its value is an array containing one object per original criterion (same ID order).

Schema for every object  
{
  "ID": "FCA-001",
  "Evidence": "abundant evidence | some evidence | no evidence | insufficient evidence to make a decision",
  "quote": "<sentence(s) or figure caption(s) from the white-paper, ≤ 20 words each, separated by ' | ' if multiple>",
  "evaluation": "<≤ 60-word explanation referencing the quote>",
  "result": "yes | no | insufficient evidence to make decision"
}

4. Evidence-Level Definitions  

• abundant evidence: Multiple explicit passages, data tables, or graphs strongly match the criterion.  
• some evidence: One clear but limited passage or graph; partial alignment.  
• no evidence: The white-paper is silent on the point; this absence can justify “yes” or “no” for the fraud.  
• insufficient evidence to make decision: Information is too sparse or conflicting for a reliable choice.

5. Evaluation Steps (internal)  

1. Extract all text, figure captions, and key graph details from whitepaper_pdf.  
2. Iterate through the criteria array.  
3. For each criterion:  
   – Search extracted content for relevant material.  
   – Select an evidence rating.  
   – Populate the quote field with up to two supporting excerpts (sentences or captions). If none, leave quote = "".  
   – Draft evaluation (≤ 60 words).  
   – Choose result following the rules above.  

6. Output Rules  

• Return only valid JSON (no Markdown, comments, or trailing commas).  
• Maintain field order inside each item: ID, Evidence, quote, evaluation, result.  
• All Evidence and result values must be lowercase.  
• Do not exceed 20 words per quoted excerpt or 60 words in any evaluation.

7. Style & Safety  

• Avoid legal conclusions; align facts with criteria.  
• Do not expose internal reasoning steps.  
• If context is inadequate, set Evidence and result to “insufficient evidence to make decision”.
"""

# System prompt for URL-based evaluation
url_system_prompt = """
System Prompt — White-paper Fraud-Risk Evaluation from URL

1. Purpose  
Evaluate a project’s white-paper or project description, fetched from the provided URL, against each fraud criteria supplied in JSON. For every criterion you must:

• Locate supporting or contradictory passages in the text **and any information conveyed by figures or graphs** from the webpage content.  
• Assign one evidence rating:  
  – abundant evidence  
  – some evidence  
  – no evidence                     (document is silent on the point)  
  – insufficient evidence to make a decision  
• Output the exact sentence(s) or concise figure caption(s) you relied on, each ≤ 20 words.  
• Write an evaluation (≤ 60 words) explaining how the quoted material supports the rating.  
• Choose the result:  
  – yes                       → criterion met → fraud indicator  
  – no                        → criterion not met  
  – insufficient evidence to make decision → cannot judge  
  Important:  
    ▸ If Evidence = “no evidence”, you MUST still set result to either “yes” or “no”.  
    ▸ Result may be “insufficient evidence to make decision” only when Evidence is the same phrase.

You may incorporate any internet or external knowledge sources if necessary.

2. Inputs  

• whitepaper_url (string): The URL pointing to the white-paper or project description. Fetch and extract all relevant text, figure captions, and, where possible, summarize graph data (e.g., axes labels, key values) before analysis.  
• fraud_criteria (JSON): Contains an array of criteria in the format: 
Example A  
{
  [
    {
      "ID": "FCA-001",
      "criteria": "Misleading Statements Inducing Investment",
      "description": "...",
      "scope": "..."
    },
    {
      "ID": "MISC-001",
      "criteria": "Anonymous Or Pseudonymous Team",
      "summary": "..."
    }
    // …more criteria
  ]
}

3. Required Output  

Return a JSON object with:  
• "whitepaper_name": the title of the white-paper or project (use webpage title, first heading, or metadata).  
• One key exactly matching the fraud_criteria ID ("FCA-001", "MISC-015", etc.). 
Its value is an array containing one object per original criterion (same ID order).

Schema for every object  
{
  "ID": "FCA-001",
  "Evidence": "abundant evidence | some evidence | no evidence | insufficient evidence to make a decision",
  "quote": "<sentence(s) or figure caption(s) from the white-paper, ≤ 20 words each, separated by ' | ' if multiple>",
  "evaluation": "<≤ 60-word explanation referencing the quote>",
  "result": "yes | no | insufficient evidence to make decision"
}

4. Evidence-Level Definitions  

• abundant evidence: Multiple explicit passages, data tables, or graphs strongly match the criterion.  
• some evidence: One clear but limited passage or graph; partial alignment.  
• no evidence: The white-paper is silent on the point; this absence can justify “yes” or “no” for the fraud.  
• insufficient evidence to make decision: Information is too sparse or conflicting for a reliable choice.

5. Evaluation Steps (internal)  

1. Fetch and extract all text, figure captions, and key graph details from the content at whitepaper_url.  
2. Iterate through the criteria array.  
3. For each criterion:  
   – Search extracted content for relevant material.  
   – Select an evidence rating.  
   – Populate the quote field with up to two supporting excerpts (sentences or captions). If none, leave quote = "".  
   – Draft evaluation (≤ 60 words).  
   – Choose result following the rules above.  

6. Output Rules  

• Return only valid JSON (no Markdown, comments, or trailing commas).  
• Maintain field order inside each item: ID, Evidence, quote, evaluation, result.  
• All Evidence and result values must be lowercase.  
• Do not exceed 20 words per quoted excerpt or 60 words in any evaluation.

7. Style & Safety  

• Avoid legal conclusions; align facts with criteria.  
• Do not expose internal reasoning steps.  
• If context is inadequate, set Evidence and result to “insufficient evidence to make decision”.
"""

# Hardcoded criteria from criteria.json for each regulator
criteria_data = {
  "FCA": [
    {
      "ID": "FCA-001",
      "criteria": "Misleading Statements Inducing Investment",
      "description": "Prohibits making statements, promises, or forecasts known to be false or misleading, dishonestly concealing material facts, or recklessly making misleading statements to induce investment in cryptoassets. This includes creating false impressions about the market or value of such investments.",
      "scope": "Applies to any person making such statements or creating such impressions concerning relevant investments, which explicitly include cryptoassets, to induce others to transact."
    },
    {
      "ID": "FCA-005",
      "criteria": "Manipulating Transactions and Disseminating False Information (MAR)",
      "description": "Prohibits engaging in behaviour that gives, or is likely to give, false or misleading signals as to the supply, demand, or price of qualifying cryptoassets, or secures their price at an artificial level. This includes disseminating false or misleading information when the person knew or should have known it was false/misleading.",
      "scope": "Applies to cryptoassets qualifying as financial instruments under MAR, prohibiting transactions, orders, or dissemination of information that distorts the market or misleads participants."
    },
    {
      "ID": "FCA-007",
      "criteria": "Carrying on Regulated Activities Without Authorisation (General Prohibition)",
      "description": "Prohibits any person from carrying on a regulated activity in the UK by way of business, or purporting to do so, unless they are authorised by the FCA or exempt. Certain cryptoasset activities are defined as regulated activities.",
      "scope": "Applies to persons conducting activities with cryptoassets that fall within the definition of \"regulated activity\" (e.g., certain exchanges, custodianship, dealing, arranging) in the UK without FCA authorisation or exemption."
    }
  ],
  "SEC": [
    {
      "ID": "SEC-003",
      "criteria": "Obtaining Money or Property by Material Misstatements or Omissions in Offer/Sale",
      "description": "Makes it unlawful in the offer or sale of securities, using interstate commerce or mails, to obtain money or property through any untrue statement of a material fact or any omission of a material fact necessary to make the statements made not misleading.",
      "scope": "Prohibits acquiring money or property via material misstatements or omissions when offering or selling securities, including crypto-assets, using jurisdictional means."
    },
    {
      "ID": "SEC-008",
      "criteria": "False or Misleading Statements to Induce Securities Purchase or Sale",
      "description": "Prohibits brokers, dealers, or others from making false or misleading statements about a security (including crypto-assets) to induce its purchase or sale, if they knew or had reasonable grounds to believe the statement was false or misleading.",
      "scope": "Targets the dissemination of false or misleading information by market participants to influence investment decisions in securities, including crypto-assets."
    },
    {
      "ID": "SEC-012",
      "criteria": "Manipulation of Security Prices",
      "description": "Prohibits a series of transactions in any security creating actual or apparent active trading in such security or raising or depressing the price of such security, for the purpose of inducing the purchase or sale of such security by others.",
      "scope": "Applies to manipulative practices that affect the price or trading volume of securities, including crypto-assets deemed securities, to deceive or induce investors."
    },
    {
      "ID": "SEC-014",
      "criteria": "General Prohibition of Manipulative and Deceptive Devices in Securities Transactions",
      "description": "Broadly prohibits the use or employment of any manipulative or deceptive device or contrivance in connection with the purchase or sale of any security, in contravention of SEC rules (most notably Rule 10b-5).",
      "scope": "A catch-all anti-fraud provision applicable to a wide range of deceptive practices in securities transactions, including those involving crypto-assets deemed securities."
    },
    {
      "ID": "SEC-030",
      "criteria": "Prevention of Misuse of Material Nonpublic Information by Investment Advisers",
      "description": "Requires investment advisers to establish, maintain, and enforce written policies and procedures reasonably designed to prevent the misuse of material, nonpublic information by the adviser or its associates, in violation of securities laws.",
      "scope": "Mandates preventive measures against insider trading for investment advisers dealing with any securities, including crypto-assets, where they might possess material nonpublic information."
    }
  ],
  "HKSFC": [
    {
      "ID": "HKSFC-001",
      "criteria": "FMSM.1: Fraudulent or Reckless Inducement to Invest",
      "description": "Making fraudulent or reckless misrepresentations (e.g., false statements, unfulfillable promises, unjustified forecasts, or material omissions) to persuade individuals to invest in securities, structured products, collective investment schemes, or virtual assets. This conduct is an offence.",
      "scope": "Prohibits inducing investment in securities, structured products, collective investment schemes (under SFO), or virtual assets (under AMLO) through fraudulent or reckless misrepresentations."
    },
    {
      "ID": "HKSFC-003",
      "criteria": "FMSM.3: Advertising Unlicensed Virtual Asset Services",
      "description": "Issuing or possessing for issue an advertisement that, to the issuer's knowledge, promotes a person or entity as prepared to provide a Virtual Asset (VA) service when that person or entity is not licensed as required under the AMLO. This is an offence.",
      "scope": "Prohibits the advertisement of VA services offered by persons known to be unlicensed to provide such services."
    },
    {
      "ID": "HKSFC-005",
      "criteria": "MM.2: Price Rigging",
      "description": "Engaging in wash sales of securities that affect their price, or using fictitious or artificial transactions or devices, with the intention or recklessness of maintaining, increasing, reducing, stabilizing, or causing fluctuations in the price of securities or futures contracts. This conduct constitutes market misconduct and a criminal offence.",
      "scope": "Prohibits manipulating the price of securities or futures contracts through wash sales affecting price or other fictitious or artificial transactions or devices."
    },
    {
      "ID": "HKSFC-006",
      "criteria": "MM.3: Stock Market Manipulation",
      "description": "Entering into two or more transactions in a corporation's securities that increase, reduce, maintain, or stabilize their price, with the intention of inducing others to buy, sell, or subscribe for those securities or related corporation's securities. This conduct constitutes market misconduct and a criminal offence.",
      "scope": "Prohibits series of transactions in a corporation's securities intended to manipulate their price and thereby induce others to trade."
    },
    {
      "ID": "HKSFC-009",
      "criteria": "FSDP.1: Use of Fraudulent or Deceptive Devices or Schemes in Transactions",
      "description": "Employing any device, scheme, or artifice with intent to defraud or deceive, or engaging in any act, practice, or course of business which is fraudulent or deceptive, in transactions involving securities, futures contracts, leveraged foreign exchange trading, or virtual assets. This is an offence.",
      "scope": "Prohibits broadly any fraudulent or deceptive schemes, devices, acts, or business practices in transactions involving securities, futures contracts, leveraged foreign exchange trading (under SFO), or virtual assets (under AMLO)."
    },
    {
      "ID": "HKSFC-011",
      "criteria": "UOUA.1: Unauthorized Public Offering or Promotion of Investments",
      "description": "Issuing, or possessing for issue, an advertisement, invitation, or document containing an invitation to the public to enter into agreements for securities, structured products, or to acquire an interest in a collective investment scheme, without authorization from the Securities and Futures Commission (SFC). This is an offence.",
      "scope": "Prohibits the unauthorized public offering or promotion of specified investment products (securities, structured products, collective investment schemes) to the public."
    }
  ]
}

# Selected Set criteria (from previous)
selected_criteria = [
    {
      "ID": "SEC-030",
      "criteria": "Prevention of Misuse of Material Nonpublic Information by Investment Advisers",
      "description": "Requires investment advisers to establish, maintain, and enforce written policies and procedures reasonably designed to prevent the misuse of material, nonpublic information by the adviser or its associates, in violation of securities laws.",
      "scope": "Mandates preventive measures against insider trading for investment advisers dealing with any securities, including crypto-assets, where they might possess material nonpublic information."
    },
    {
      "ID": "SEC-008",
      "criteria": "False or Misleading Statements to Induce Securities Purchase or Sale",
      "description": "Prohibits brokers, dealers, or others from making false or misleading statements about a security (including crypto-assets) to induce its purchase or sale, if they knew or had reasonable grounds to believe the statement was false or misleading.",
      "scope": "Targets the dissemination of false or misleading information by market participants to influence investment decisions in securities, including crypto-assets."
    },
    {
      "ID": "SEC-014",
      "criteria": "General Prohibition of Manipulative and Deceptive Devices in Securities Transactions",
      "description": "Broadly prohibits the use or employment of any manipulative or deceptive device or contrivance in connection with the purchase or sale of any security, in contravention of SEC rules (most notably Rule 10b-5).",
      "scope": "A catch-all anti-fraud provision applicable to a wide range of deceptive practices in securities transactions, including those involving crypto-assets deemed securities."
    },
    {
      "ID": "FCA-005",
      "criteria": "Manipulating Transactions and Disseminating False Information (MAR)",
      "description": "Prohibits engaging in behaviour that gives, or is likely to give, false or misleading signals as to the supply, demand, or price of qualifying cryptoassets, or secures their price at an artificial level. This includes disseminating false or misleading information when the person knew or should have known it was false/misleading.",
      "scope": "Applies to cryptoassets qualifying as financial instruments under MAR, prohibiting transactions, orders, or dissemination of information that distorts the market or misleads participants."
    },
    {
      "ID": "HKSFC-001",
      "criteria": "FMSM.1: Fraudulent or Reckless Inducement to Invest",
      "description": "Making fraudulent or reckless misrepresentations (e.g., false statements, unfulfillable promises, unjustified forecasts, or material omissions) to persuade individuals to invest in securities, structured products, collective investment schemes, or virtual assets. This conduct is an offence.",
      "scope": "Prohibits inducing investment in securities, structured products, collective investment schemes (under SFO), or virtual assets (under AMLO) through fraudulent or reckless misrepresentations."
    },
    {
      "ID": "MISC-015",
      "criteria": "Undisclosed Or Misleading Centralization",
      "summary": "The project is marketed as decentralized but retains significant centralized control over funds, governance, or operations, creating an illusion of community ownership."
    }
]

# Coefficients for each set
coefficients_data = {
    "Selected Set": {
        "const": -7.2956,
        "SEC-014": 1.9420,
        "SEC-030": 2.4312,
        "HKSFC-001": 2.4255,
        "SEC-008": 2.0294,
        "MISC-015": 2.0220,
        "FCA-005": 1.7266
    },
    "FCA": {
        "const": -4.0132,
        "FCA-001": 1.9203,
        "FCA-005": 3.4054,
        "FCA-007": 1.0274
    },
    "SEC": {
        "const": -5.2809,
        "SEC-003": 1.6597,
        "SEC-008": 2.4216,
        "SEC-014": 2.1707,
        "SEC-030": 1.9726,
        "SEC-012": 1.4019
    },
    "HKSFC": {
        "const": -4.7737,
        "HKSFC-001": 2.1551,
        "HKSFC-009": 2.3465,
        "HKSFC-006": 1.5324,
        "HKSFC-003": 1.1324,
        "HKSFC-005": 0.4604,
        "HKSFC-011": 0.7876
    }
}

# Sanitize filename function
def sanitize_filename(filename):
    safe_name = re.sub(r'[^\w\-_. ]', '', filename)  # Alphanumeric, underscore, hyphen, dot, space
    safe_name = safe_name.replace('..', '')
    return safe_name.strip() or "default"

# Mapping function
def map_to_result(evidence, result):
    if evidence == "no evidence" and result == "no":
        return 0
    elif evidence == "no evidence" and result == "yes":
        return 1
    elif evidence == "some evidence" and result == "yes":
        return 0.75
    elif evidence == "some evidence" and result == "no":
        return 0.25
    elif evidence == "insufficient evidence to make decision" and result == "insufficient evidence to make decision":
        return 0.5
    elif evidence == "some evidence" and result == "insufficient evidence to make decision":
        return 0.5
    elif evidence == "abundant evidence" and result == "yes":
        return 1
    elif evidence == "abundant evidence" and result == "no":
        return 0
    else:
        raise ValueError(f"Invalid combination: Evidence={evidence}, Result={result}")

# Function to process Gemini response
def process_response(response_text):
    response_text = response_text.strip()
    if response_text.startswith('```json'):
        response_text = response_text[7:].strip()
    if response_text.endswith('```'):
        response_text = response_text[:-3].strip()
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = response_text[start_idx:end_idx + 1]
    else:
        json_str = response_text
    def fix_trailing_commas(js):
        return re.sub(r',\s*([}\]])', r'\1', js)
    json_str = fix_trailing_commas(json_str)
    return json_str


# Streamlit app
st.title("Crypto Project Fraud Detection App")

# API key input
st.write("Enter your Gemini API key. Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey).")
api_key = st.text_input("Gemini API Key", type="password")

# Configure API key
if api_key:
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"Invalid user-provided API key: {e}")
        st.stop()
else:
    try:
        # Try to use key from st.secrets
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except (KeyError, Exception):
        # Fall back to hardcoded key
        st.error(f"Invalid API key. Please provide a valid API key.")
        st.stop()

# Select criteria set
selected_set = st.selectbox("Select Criteria Set", ["Selected Set", "FCA", "SEC", "HKSFC"], index=0)

# Set criteria list and coefficients based on selection
if selected_set == "Selected Set":
    criteria_list = selected_criteria
    coefficients = coefficients_data["Selected Set"]
elif selected_set == "FCA":
    criteria_list = criteria_data["FCA"]
    coefficients = coefficients_data["FCA"]
elif selected_set == "SEC":
    criteria_list = criteria_data["SEC"]
    coefficients = coefficients_data["SEC"]
else:
    criteria_list = criteria_data["HKSFC"]
    coefficients = coefficients_data["HKSFC"]

# Create lookup for criterion names
criteria_lookup = {item["ID"]: item["criteria"] for item in criteria_list}

# Tabs for file upload and URL input
tab1, tab2 = st.tabs(["Upload File", "Submit URL"])

# Initialize variables
evaluations = []
whitepaper_name = None
temp_path = None

with tab1:
    uploaded_file = st.file_uploader("Upload a PDF or TXT whitepaper/description", type=["pdf", "txt"])
    if uploaded_file is not None:
        # Sanitize filename
        safe_filename = sanitize_filename(uploaded_file.name)
        
        # Check file size
        if uploaded_file.size > MAX_SIZE:
            st.error(f"{safe_filename} exceeds 100MB limit!")
            st.stop()

        # Validate file content
        mime_checker = magic.Magic(mime=True)
        file_content = uploaded_file.read()
        file_type = mime_checker.from_buffer(file_content)
        if file_type not in ["application/pdf", "text/plain"]:
            st.error(f"Invalid file type for {safe_filename}. Only PDF or TXT files are allowed.")
            st.stop()

        # Save uploaded file temporarily
        temp_path = f"temp_{safe_filename}"
        with open(temp_path, "wb") as f:
            f.write(file_content)

        # Upload to Gemini and evaluate
        try:
            gemini_file = genai.upload_file(temp_path)
            st.success(f"File {safe_filename} uploaded to Gemini for evaluation. \n  Please, wait for 40 seconds for the result to appear")
            model = genai.GenerativeModel(model_name='gemini-2.5-pro', system_instruction=file_system_prompt)
            criteria_str = json.dumps({"criteria": criteria_list}, indent=2)
            user_prompt = f"Evaluate the attached whitepaper against the following criteria:\n{criteria_str}"
            response = model.generate_content([gemini_file, user_prompt])
            json_str = process_response(response.text)

            # Parse response
            try:
                evaluation_json = json.loads(json_str)
                whitepaper_name = evaluation_json.get("whitepaper_name", safe_filename.replace('.pdf', '').replace('.txt', ''))
                evaluations = []
                for key, value in evaluation_json.items():
                    if key != "whitepaper_name":
                        if isinstance(value, list):
                            evaluations.extend(value)
                        else:
                            st.warning(f"Expected a list for key {key}, got {type(value)}. Skipping: {value}")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON response from Gemini: {e}. Unable to process evaluation.")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                st.stop()

        except Exception as e:
            st.error(f"Error processing file {safe_filename}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            st.stop()
        except Exception as e:
            st.error(f"Error processing file {safe_filename}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            st.stop()

with tab2:
    url = st.text_input("Enter the URL to a crypto-project whitepaper or description")
    if st.button("Submit Link") and url:
        # Validate URL format
        url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
        if not url_pattern.match(url):
            st.error("Invalid URL format. Please enter a valid URL starting with http:// or https://")
            st.stop()

        st.success(f"URL {url} uploaded to Gemini for evaluation. \n  Please, wait for 40 seconds for the result to appear")  
      # Evaluate URL content with Gemini
        try:
            model = genai.GenerativeModel(model_name='gemini-2.5-pro', system_instruction=url_system_prompt)
            criteria_str = json.dumps({"criteria": criteria_list}, indent=2)
            user_prompt = f"Evaluate the whitepaper or project description at the following URL against the provided criteria:\nURL: {url}\nCriteria:\n{criteria_str}"
            response = model.generate_content([user_prompt])
            json_str = process_response(response.text)

            # Parse response
            try:
                evaluation_json = json.loads(json_str)
                whitepaper_name = evaluation_json.get("whitepaper_name", url.split('/')[-1] or "unknown_project")
                evaluations = []
                for key, value in evaluation_json.items():
                    if key != "whitepaper_name":
                        if isinstance(value, list):
                            evaluations.extend(value)
                        else:
                            st.warning(f"Expected a list for key {key}, got {type(value)}. Skipping: {value}")
                st.success(f"URL content evaluated successfully.")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON response from Gemini: {e}. Unable to process evaluation.")
                st.stop()
        except Exception as e:
            st.error(f"Error processing URL {url}: {e}")
            st.stop()

# Process evaluations if available
if evaluations:
    # Compute scores and collect quotes
    scores = {}
    fraud_criteria = []  # Criteria with result="yes"
    table_data = []  # For the table display

    for eval_item in evaluations:
        id_ = eval_item["ID"]
        evidence = eval_item["Evidence"].lower()
        result = eval_item["result"].lower()
        quote = eval_item["quote"]
        try:
            score = map_to_result(evidence, result)
            scores[id_] = score
            if result == "yes":
                fraud_criteria.append(id_)
            # Add to table data
            criterion_name = criteria_lookup.get(id_, id_)  # Fallback to ID if not found
            table_data.append({
                "Criterion Name": criterion_name,
                "Score": round(score, 2),
                "Quote": quote
            })
        except ValueError:
            st.warning(f"Invalid evidence/result for {id_}. Skipping.")

    # Logistic regression
    logit = coefficients["const"]
    for id_, weight in coefficients.items():
        if id_ != "const":
            logit += weight * scores.get(id_, 0)  # Default to 0 if missing

    prob = 1 / (1 + math.exp(-logit))
    is_fraud = "Yes" if prob > 0.7 else "No" if prob<0.3 else "Unsure, human intervention is required"

    # Display results
    st.subheader("Evaluation Results")
    st.write(f"**Fraud Probability**: {prob:.4f}")
    st.write(f"**Fraud Result**: {is_fraud}")
    st.write("**Criteria deemed true (fraud indicators)**:")
    st.write(", ".join(fraud_criteria) if fraud_criteria else "None")
    
    # Display table
    if table_data:
        st.write("**Details for Evaluated Criteria**:")
        df = pd.DataFrame(table_data)
        st.table(df)

# Clean up temp file if it exists
if temp_path and os.path.exists(temp_path):
    os.remove(temp_path)
