import threading
import torch
import whisper
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_cohere import CohereEmbeddings
from functools import lru_cache
import os

class Singleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
class WhisperModel(Singleton):
    def __init__(self):
        if not hasattr(self, 'initialized'):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = whisper.load_model("large", device=device)
            self.initialized = True

    def transcribe(self, audio_path, language="en"):
        use_fp16 = torch.cuda.is_available()
        return self.model.transcribe(audio_path, fp16=use_fp16, language=language)
    
class DatabasePool(Singleton):
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.engine = create_engine(f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}", connect_args={'sslmode': "allow"})
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            self._initialized = True

    def get_session(self):
        return self.SessionLocal()   

@lru_cache()
def get_whisper_model():
    return WhisperModel()

def get_db_session():
    db_pool = DatabasePool()
    db = db_pool.get_session()
    try:
        yield db
    finally:
        db.close()

@lru_cache()
def get_cohere_embeddings():
    return CohereEmbeddings(
        cohere_api_key=os.getenv('COHERE_API_KEY'),
        model="embed-multilingual-v3.0"
    )