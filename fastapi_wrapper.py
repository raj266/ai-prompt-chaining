from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
from call_groq import call_groq

app = FastAPI(title="Hospitality Prompt Chaining")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChainRequest(BaseModel):
    query: str

class ChainResponse(BaseModel):
    final_answer: str
    steps: list
    elapsed_time: float

@app.post("/chain", response_model=ChainResponse)
async def chain_endpoint(request: ChainRequest):
    start_time = time.time()
    steps = []

    # Step 1: Extract
    extract_prompt = f"""You are a hospitality query extractor. From the user's request, extract:
- destination (city or area)
- budget_per_night (in INR, as a number or range)
- room_type (e.g., deluxe, suite, standard)
- amenities (list any: pool, gym, breakfast, wifi, spa, beachfront)
- cuisine_preference (if restaurant related)

Query: {request.query}

Output ONLY valid JSON like:
{{"destination": "Goa", "budget_per_night": "5000-8000", "room_type": "deluxe", "amenities": ["pool", "breakfast"], "cuisine_preference": null}}
"""
    extract = call_groq(extract_prompt, node_name="EXTRACT")
    steps.append({"step": "extract", "output": extract})

    # Step 2: Recommend
    recommend_prompt = f"""Based on the following extracted data, recommend a specific hotel or restaurant in that destination.

Extracted data: {extract}

Write a short recommendation (2-3 sentences) including the property name, why it fits the user's needs, and any notable feature.
"""
    recommend = call_groq(recommend_prompt, node_name="RECOMMEND")
    steps.append({"step": "recommend", "output": recommend})

    # Step 3: Polish
    polish_prompt = f"""Rewrite the following recommendation in a warm, enthusiastic, and persuasive tone, as if for a travel magazine. Keep it under 100 words.

Original recommendation: {recommend}

Polished version:
"""
    polished = call_groq(polish_prompt, node_name="POLISH")
    steps.append({"step": "polish", "output": polished})

    elapsed = time.time() - start_time
    return ChainResponse(final_answer=polished, steps=steps, elapsed_time=elapsed)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)