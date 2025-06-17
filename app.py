# Full app.py with all original features adapted for OpenAI SDK >= 1.0

import os
import json
import sqlite3
import numpy as np
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import aiohttp
import asyncio
import logging
import traceback
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

DB_PATH = "knowledge_base.db"
SIMILARITY_THRESHOLD = 0.5
MAX_RESULTS = 10
MAX_CONTEXT_CHUNKS = 4

# Models
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure database exists
if not os.path.exists(DB_PATH):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS discourse_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER,
            topic_id INTEGER,
            topic_title TEXT,
            post_number INTEGER,
            author TEXT,
            created_at TEXT,
            likes INTEGER,
            chunk_index INTEGER,
            content TEXT,
            url TEXT,
            embedding BLOB)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS markdown_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_title TEXT,
            original_url TEXT,
            downloaded_at TEXT,
            chunk_index INTEGER,
            content TEXT,
            embedding BLOB)''')
        conn.commit()

# Utility functions
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

async def get_embedding(text):
    try:
        response = client.embeddings.create(model="text-embedding-3-small", input=text)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_multimodal_query(question, image_base64):
    if not image_base64:
        return await get_embedding(question)

    try:
        image_url = f"data:image/jpeg;base64,{image_base64}"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"What do you see? Question: {question}"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ]
        )
        description = response.choices[0].message.content
        return await get_embedding(f"{question}\nImage context: {description}")
    except Exception as e:
        logger.warning("Multimodal fallback to text")
        return await get_embedding(question)

async def find_similar_content(query_embedding, conn):
    cursor = conn.cursor()
    results = []

    cursor.execute("SELECT * FROM discourse_chunks WHERE embedding IS NOT NULL")
    for row in cursor.fetchall():
        embedding = json.loads(row["embedding"])
        similarity = cosine_similarity(query_embedding, embedding)
        if similarity >= SIMILARITY_THRESHOLD:
            results.append({**row, "similarity": similarity, "source": "discourse"})

    cursor.execute("SELECT * FROM markdown_chunks WHERE embedding IS NOT NULL")
    for row in cursor.fetchall():
        embedding = json.loads(row["embedding"])
        similarity = cosine_similarity(query_embedding, embedding)
        if similarity >= SIMILARITY_THRESHOLD:
            results.append({**row, "similarity": similarity, "source": "markdown"})

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:MAX_RESULTS]

async def enrich_with_adjacent_chunks(conn, results):
    cursor = conn.cursor()
    enriched = []

    for r in results:
        content = r["content"]
        if r["source"] == "discourse":
            post_id, index = r["post_id"], r["chunk_index"]
            cursor.execute("SELECT content FROM discourse_chunks WHERE post_id=? AND chunk_index=?", (post_id, index + 1))
            next_chunk = cursor.fetchone()
            if next_chunk: content += " " + next_chunk["content"]
        else:
            title, index = r["doc_title"], r["chunk_index"]
            cursor.execute("SELECT content FROM markdown_chunks WHERE doc_title=? AND chunk_index=?", (title, index + 1))
            next_chunk = cursor.fetchone()
            if next_chunk: content += " " + next_chunk["content"]

        r["content"] = content
        enriched.append(r)

    return enriched

async def generate_answer(question, context):
    context_str = "\n\n".join([f"URL: {r['url']}\n{r['content']}" for r in context])
    messages = [
        {"role": "system", "content": "Answer based only on provided context. Always include sources."},
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {question}"}
    ]
    try:
        response = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0.3)
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def parse_llm_response(response):
    parts = response.split("Sources:", 1)
    answer = parts[0].strip()
    links = []
    if len(parts) > 1:
        for line in parts[1].split("\n"):
            url_match = re.search(r'(https?://\S+)', line)
            if url_match:
                links.append({"url": url_match.group(1), "text": line.strip()})
    return {"answer": answer, "links": links}

@app.post("/query")
async def query_knowledge_base(req: QueryRequest):
    try:
        conn = get_db_connection()
        embedding = await process_multimodal_query(req.question, req.image)
        similar = await find_similar_content(embedding, conn)
        enriched = await enrich_with_adjacent_chunks(conn, similar)
        llm_response = await generate_answer(req.question, enriched)
        result = parse_llm_response(llm_response)

        if not result["links"]:
            result["links"] = [{"url": r["url"], "text": r["content"][:100]} for r in similar[:5]]

        return result
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def root():
    return {"message": "RAG API running"}

@app.get("/health")
async def health_check():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
        discourse_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
        markdown_count = cursor.fetchone()[0]
        return {
            "status": "healthy",
            "discourse_chunks": discourse_count,
            "markdown_chunks": markdown_count,
            "api_key_set": bool(API_KEY)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "status": "unhealthy"})
