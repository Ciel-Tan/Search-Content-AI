import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from dependencies import get_whisper_model, get_db_session, get_cohere_embeddings
from video_processor import VideoProcessor
from langchain_postgres import PGVector
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Search Content AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

vector_store = PGVector(
    embeddings=get_cohere_embeddings(),
    collection_name="video_content",
    connection=f"postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}?sslmode=require",
    use_jsonb=True
)

def get_video_processor(
    whisper_model = Depends(get_whisper_model),
    db_session = Depends(get_db_session),
    embeddings = Depends(get_cohere_embeddings)
):
    return VideoProcessor(whisper_model, db_session, embeddings, vector_store)

@app.get("/search")
async def search(
    query: str,
    processor: VideoProcessor = Depends(get_video_processor)
):
    results = processor.search(query)
    return results

@app.post("/process")
async def process(
    poster_url: str,
    youtube_url: str,
    language: str,
    processor: VideoProcessor = Depends(get_video_processor)
):
    try:
        result = processor.process_youtube_link(poster_url, youtube_url, language)
        return {
            "message": "Processing successful",
            "content": result
        }
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete")
async def delete(
    video_id: str,
    processor: VideoProcessor = Depends(get_video_processor)
):
    return processor.delete_video(video_id)