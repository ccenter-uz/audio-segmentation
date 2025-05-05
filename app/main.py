from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pathlib import Path
from uuid import uuid4
import os
import shutil
import uvicorn
from pydub import AudioSegment
from app.silero_vad_utils import get_speech_timestamps, read_audio, model
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
app = FastAPI()

ffmpeg_path = shutil.which("ffmpeg")

if not ffmpeg_path:
    raise EnvironmentError("ffmpeg not found. Please install and add to PATH.")


AudioSegment.converter = ffmpeg_path

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHUNKS_BASE_DIR = Path("chunks")
UPLOAD_DIR = Path("uploads")
CHUNKS_BASE_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

from pydub import AudioSegment
import subprocess

def convert_spx_to_mp3(input_path: str, output_path: str):
    try:
        # Try loading as Ogg container
        audio = AudioSegment.from_file(input_path, format="ogg")
        audio.export(output_path, format="mp3")
        print(f"✅ Converted via pydub: {input_path} -> {output_path}")
    except Exception as e:
        print(f"⚠️ Pydub failed to decode as ogg: {e}")
        # fallback: use ffmpeg CLI directly
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", input_path,
                "-acodec", "libmp3lame",
                output_path
            ], check=True)
            print(f"✅ Converted via ffmpeg CLI fallback: {input_path} -> {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ FFMPEG CLI fallback also failed: {e}")
            raise RuntimeError("Cannot decode SPX file: invalid format or corrupt.")


@app.post("/vad-chunk")
async def vad_chunk(
    audio_file: UploadFile = File(...),
    min_duration: float = Form(1.0),
    max_duration: float = Form(20.0)
):
    if not audio_file.filename.endswith((".mp3", ".wav", ".spx")):
        raise HTTPException(status_code=400, detail="Only .mp3, .spx or .wav files are supported")

    job_id = str(uuid4())
    job_chunk_dir = CHUNKS_BASE_DIR / job_id
    job_chunk_dir.mkdir(exist_ok=True)

    temp_audio_path = UPLOAD_DIR / f"{job_id}_{audio_file.filename}"
    with open(temp_audio_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    if audio_file.filename.endswith(".spx"):
        mp3_path = UPLOAD_DIR / f"{job_id}.mp3"
        convert_spx_to_mp3(str(temp_audio_path), str(mp3_path))
        os.remove(temp_audio_path)  # Remove the original SPX file
        temp_audio_path = mp3_path

    SAMPLING_RATE = 16000
    print("Running Voice Activity Detection...")
    wav = read_audio(str(temp_audio_path), sampling_rate=SAMPLING_RATE)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)

    print("Exporting VAD-based speech chunks...")
    if audio_file.filename.endswith(".mp3") or audio_file.filename.endswith(".spx"):
        # Converted to mp3 if the input is mp3 or spx
        audio = AudioSegment.from_mp3(str(temp_audio_path))
    else:
        audio = AudioSegment.from_wav(str(temp_audio_path))

    chunks = []
    for i, ts in enumerate(speech_timestamps):
        start_sec = ts['start'] / SAMPLING_RATE
        end_sec = ts['end'] / SAMPLING_RATE
        duration_sec = end_sec - start_sec

        if duration_sec < min_duration or duration_sec > max_duration:
            continue

        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        chunk_audio = audio[start_ms:end_ms]

        chunk_filename = f"{job_id}_chunk_{i+1:03}.wav"
        chunk_path = job_chunk_dir / chunk_filename
        chunk_audio.export(chunk_path, format="wav")

        chunks.append({
            "start": round(start_sec, 2),
            "end": round(end_sec, 2),
            "chunk_id": chunk_filename
        })

    os.remove(temp_audio_path)  # Remove the original audio file
    return {"job_id": job_id, "chunks": chunks}

@app.get("/download/{job_id}/{chunk_id}")
async def download_chunk(job_id: str, chunk_id: str):
    chunk_path = CHUNKS_BASE_DIR / job_id / chunk_id
    if not chunk_path.exists():
        raise HTTPException(status_code=404, detail="Chunk not found")

    def iterfile():
        with open(chunk_path, mode="rb") as file:
            yield from file

    def delete_file():
        try:
            os.remove(chunk_path)
            parent = chunk_path.parent
            if parent.exists() and not list(parent.glob("*.wav")):
                parent.rmdir()
            print(f"✅ Deleted: {chunk_path}")
        except Exception as e:
            print(f"❌ Error cleaning up file: {e}")

    return StreamingResponse(
        iterfile(),
        media_type="audio/wav",
        headers={"Content-Disposition": f"attachment; filename={chunk_id}"},
        background=BackgroundTask(delete_file)
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)