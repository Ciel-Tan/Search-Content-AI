from sqlalchemy import text
from dotenv import load_dotenv
import yt_dlp
from dependencies import WhisperModel
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

class VideoProcessor:
    def __init__(self, whisper_model, db_session, embeddings, vector_store):
        self.whisper_model = whisper_model
        self.db_session = db_session
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.executor = ThreadPoolExecutor(max_workers=4)

    def get_youtube_title(self, url):
        # Options to ensure we only extract info without downloading
        ydl_opts = {
            'quiet': True,         # suppress output
            'skip_download': True, # do not download the video
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=False)
                return info_dict.get('title', None)
        except Exception as e:
            print("Error:", e)
            return None

    def download_audio(self, youtube_url, output_filename="downloaded_audio.mp3"):
        print("Downloading audio using yt-dlp...")

        ydl_opts = {
            'ffmpeg_location': 'D:\\graduation thesis\\ffmpeg\\bin',
            'format': 'bestaudio/best',
            'outtmpl': 'downloaded_audio.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '128',
            }],
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        print(f"Audio downloaded and saved as {output_filename}")
        return output_filename
    
    def transcribe_audio_whisper(self, audio_file, language):
        result = WhisperModel().transcribe(audio_file, language=language)
        print(f"Transcription completed")
        return result
    
    def store_transcription_in_db(self, transcription, poster_url, youtube_url):
        segments = transcription["segments"]
        video_id = youtube_url.split("=")[-1]
        title = self.get_youtube_title(youtube_url)
        
        text_chunks = []
        metadata_list = []
        current_chunk = []
        current_start = None
        current_end = None
        chunk_size = 512
        chunk_overlap = 128

        for segment in segments:
            text = segment["text"]
            start_time = segment.get("start", None)
            end_time = segment.get("end", None)

            if current_start is None:
                current_start = start_time

            current_chunk.append(text)
            current_end = end_time

            if sum(len(t) for t in current_chunk) > chunk_size:
                text_chunks.append(" ".join(current_chunk))
                metadata_list.append({
                    "video_id": video_id,
                    "poster_url": poster_url,
                    "file_name": title,
                    "chunk_index": len(text_chunks) - 1,
                    "start_time": current_start,
                    "end_time": current_end,
                })
                
                current_chunk = current_chunk[-(chunk_overlap // len(text)):]
                current_start = start_time

        if current_chunk:
            text_chunks.append(" ".join(current_chunk))
            metadata_list.append({
                "video_id": video_id,
                "poster_url": poster_url,
                "file_name": title,
                "chunk_index": len(text_chunks) - 1,
                "start_time": current_start,
                "end_time": current_end,
            })

        documents = [
            Document(page_content=chunk, metadata=meta)
            for chunk, meta in zip(text_chunks, metadata_list)
        ]
        print(f"{video_id}: Created {len(documents)} documents with timestamps")

        self.vector_store.add_documents(documents)
        print(f"{video_id}: Stored {len(documents)} documents in Vector Store")

    def process_youtube_link(self, poster_url: str, youtube_url: str, language: str):
        audio_file = self.download_audio(youtube_url)
        transcription = self.transcribe_audio_whisper(audio_file, language)

        self.store_transcription_in_db(transcription, poster_url, youtube_url)
        return transcription["text"]

    def search(self, query: str):
        results = self.vector_store.similarity_search_with_score(query, k=10)
        return [
            {
                "video_id": doc.metadata["video_id"],
                "content": doc.page_content,
                "poster_url": doc.metadata["poster_url"],
                "file_name": doc.metadata["file_name"],
                "chunk_index": doc.metadata["chunk_index"],
                "start_time": round(doc.metadata["start_time"], 0),
                "end_time": round(doc.metadata["end_time"], 0),
                "similarity_score": float(score)
            }
            for doc, score in results
        ]
    
    def delete_video(self, video_id):        
        try:
            query = text(f"DELETE FROM langchain_pg_embedding WHERE (cmetadata->>'video_id') = '{video_id}'")
            self.db_session.execute(query)
            self.db_session.commit()
            print(f"{video_id}: Deleted documents from vector store")
            db_result = {
                "deleted": True,
                "message": "Documents successfully deleted from vector store"
            }
        except Exception as e:
            print(f"{video_id}: Error deleting from vector store: {str(e)}")
            db_result = {
                "deleted": False,
                "error": str(e)
            }
        return {
            "database": db_result
        }