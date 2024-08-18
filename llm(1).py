from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests
from io import BytesIO
import json
from openai import OpenAI
from predibase import Predibase
import os
from dotenv import load_dotenv

load_dotenv()

UPSTAGE_API_KEY = os.environ['UPSTAGE_API_KEY']
PB_API_KEY = os.environ['PB_API_KEY']
tenant_id = os.environ['TENANT_ID']
base_model = "solar-1-mini-chat-240612"
adapter_id = 'daechul-ai-modelv4/7'
pb = Predibase(api_token=PB_API_KEY)
pb
client = OpenAI(
    api_key=UPSTAGE_API_KEY,
    base_url="https://api.upstage.ai/v1/solar"
)

app = FastAPI()

class LoanRequest(BaseModel):
    file_url: str

def query_adapter(context, adapter_id=adapter_id, tenant_id=tenant_id, base_model=base_model, PB_API_KEY = PB_API_KEY):
    prompt = f"""
            <|im_start|> system
            You are a bank loan assistant tasked with determining the suitability of a loan applicant based on their provided financial statements. You must:

            1. Analyze the latest financial data provided.
            2. Decide whether the applicant is suitable for a loan.
            3. Provide three reasons or insights supporting your decision.

            Your insights must be backed up with financial figures.

            The output must be in JSON format, following the example below STRICTLY:

            {{
                "stance": true,
                "insight_1": "Example of the first insight",
                "insight_2": "Example of the second insight",
                "insight_3": "Example of the third insight"
            }}

            <|im_start|> user
            These are the applicant's financial statements:
            {context}
    """
    # Send POST request
    url = f"https://serving.app.predibase.com/{tenant_id}/deployments/v2/llms/{base_model}/generate"
    payload = {
        "inputs": prompt,
        "parameters": {
            "adapter_id": adapter_id,
            "adapter_source": "pbase",
            "temperature": 0.1,
            "max_new_tokens": 300
        }
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {PB_API_KEY}"
    }
    response = requests.post(
        url=url, 
        data=json.dumps(payload),
        headers=headers
    )
    return json.loads(json.loads(response.text)['generated_text'])

def prompt_llm(context, client=client):
    stream = client.chat.completions.create(
        model='solar-1-mini-chat',
        messages=[
            {
                "role": "system",
                "content": """
                You are a bank loan assistant tasked with determining the suitability of a loan applicant based on their provided financial statements. You must:

                    1. Analyze the latest financial data provided.
                    2. Decide whether the applicant is suitable for a loan.
                    3. Provide three reasons or insights supporting your decision.

                Your insights must be backed up with financial figures. Be as critical as possible

                Your response MUST be in JSON format, following the example below:

                {
                    "stance": true,
                    "insight_1": "Example of the first insight",
                    "insight_2": "Example of the second insight",
                    "insight_3": "Example of the third insight"
                }
                """
            }, 
            {
                "role": "user",
                "content": f"""
                These are the applicant's financial statements provided:
                {context}
                """
            }
        ],
        stream=False
    )
    try:
        response = json.loads(stream.choices[0].message.content.replace("\n",""))
    except:
        return None
    return response

def prompt_summarize(context, client=client):
    stream = client.chat.completions.create(
        model='solar-1-mini-chat',
        messages=[
            {
                "role": "system",
                "content": """
                You are an expert in financial analysis. Below is a page from a financial statement document. 
                Your task is to summarize the key financial information, including all relevant numerical figures, trends, and notable observations. 
                Ensure that no important numerical data is omitted. Your summary should be clear, concise, and no longer than 200 words, focusing only on the most significant details. 
                If you find the page lacks sufficient information for a summary, you MUST return an empty response.
                """
            }, 
            {
                "role": "user",
                "content": f"""
                This is the text you are to summarize:
                {context}
                """
            }
        ],
        stream=False
    )
    response = stream.choices[0].message.content
    return response

def loan_evaluation(file_url,url="https://api.upstage.ai/v1/document-ai/layout-analysis", API_KEY=UPSTAGE_API_KEY):
    response = requests.get(file_url)
    response.raise_for_status()  

    # Load the file into a BytesIO object
    file_data = BytesIO(response.content)

    headers = {"Authorization": f"Bearer {API_KEY}"}
    files = {"document": file_data}

    # Post request to the API
    api_response = requests.post(url, headers=headers, files=files)
    api_response.raise_for_status()  
    
    obj = api_response.json()

    # Extract information
    context = ''
    for page in range(obj['billed_pages']):
        page_content = ''
        for element in obj['elements']:
            if element['page'] == page:
                if element['category'] == 'table':
                    page_content += f"\n{element['html']}"
                else:
                    page_content += f"\n{element['text']}"
        context += prompt_summarize(page_content)

    loan_results=query_adapter(context)
    
    return loan_results

@app.post("/evaluate-loan")
async def evaluate_loan(request: LoanRequest):
    try:
        result = loan_evaluation(request.file_url)
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to process the loan application")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)