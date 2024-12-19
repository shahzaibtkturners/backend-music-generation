import os
import re
import logging
import torch
from faster_whisper import WhisperModel
import shutil
import subprocess
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global configuration
MODEL_NAME = "large-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac"}

# Load Whisper model with error handling
try:
    whisper_model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type="int8")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    raise

app = FastAPI()


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and remove unsafe characters.
    """
    filename = os.path.basename(filename)
    filename = re.sub(r"[^\w\-\.]", "", filename)
    return filename


def validate_file(file: UploadFile):
    """Validate uploaded file."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {ext} not allowed. Supported types: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    file.file.seek(0, 2)  # Go to end of file
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size is {MAX_FILE_SIZE / (1024*1024)} MB",
        )


@app.post("/process-video")
async def process_audio(
    audio: UploadFile = File(...),
    desired_duration: int = Form(..., gt=0, le=3600),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Process audio file:
    1. Validate input
    2. Transcribe audio
    3. Find vocal segments
    4. Trim audio
    """
    try:
        validate_file(audio)
        safe_filename = sanitize_filename(audio.filename)
        os.makedirs("temp", exist_ok=True)
        audio_path = f"temp/{safe_filename}"
        with open(audio_path, "wb") as f:
            f.write(await audio.read())

        try:
            segments, info = whisper_model.transcribe(
                audio_path,
                beam_size=5,
                # task="translate", language="en", verbose=False
            )
            print(
                "Detected language '%s' with probability %f"
                % (info.language, info.language_probability)
            )

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise HTTPException(
                status_code=500, detail="Transcription processing failed"
            )

        # return

        # Identify vocal segment
        vocal_start_time = None
        vocal_end_time = None
        skip_text = [
            "Music",
            "music",
            "[Music]",
            "[music]",
            "Intro",
            "intro",
            "🎵",
            "🎵🎵",
            "🎵🎵🎵",
            "[🎵]",
            "[🎵🎵]",
            "[🎵🎵🎵]",
            "♫",
            "♫♫",
            "♫♫♫",
            "[♫]",
            "[♫♫]",
            "[♫♫♫]",
            "♪",
            "♪♪",
            "♪♪♪",
            "[♪]",
            "[♪♪]",
            "[♪♪♪]",
            "Instrumental",
            "instrumental",
            "[Instrumental]",
            "[instrumental]",
            "Beat",
            "beat",
            "[Beat]",
            "[beat]",
            "Chorus",
            "chorus",
            "[Chorus]",
            "[chorus]",
            "Background",
            "background",
            "[Background]",
            "[background]",
            "Melody",
            "melody",
            "[Melody]",
            "[melody]",
            "Bridge",
            "bridge",
            "[Bridge]",
            "[bridge]",
            "Outro",
            "outro",
            "[Outro]",
            "[outro]",
            "Hook",
            "hook",
            "[Hook]",
            "[hook]",
            "Theme",
            "theme",
            "[Theme]",
            "[theme]",
            "Interlude",
            "interlude",
            "[Interlude]",
            "[interlude]",
            "Solo",
            "solo",
            "[Solo]",
            "[solo]",
            "Vocals",
            "vocals",
            "[Vocals]",
            "[vocals]",
            "Refrain",
            "refrain",
            "[Refrain]",
            "[refrain]",
        ]

        for segment in segments:
            text = segment.text.strip()
            print(f"text is:{text}")

            if text and text not in skip_text:
                if vocal_start_time is None:
                    vocal_start_time = segment.start

                adjusted_end_time = vocal_start_time + desired_duration
                if (
                    adjusted_end_time >= segment.start
                    and adjusted_end_time <= segment.end
                ):
                    vocal_end_time = segment.start
                    break
        if vocal_start_time is None or vocal_end_time is None:
            return {"error": "No valid vocal segment detected."}

        # Adjust duration if needed
        actual_duration = vocal_end_time - vocal_start_time
        if actual_duration < desired_duration:
            vocal_start_time = max(
                0, vocal_start_time - (desired_duration - actual_duration)
            )

        logger.info(
            f"actual_duration = vocal_end_time - vocal_start_time: {actual_duration} ,{vocal_end_time} ,{vocal_start_time}"
        )
        trimmed_audio_path = f"temp/trimmed_{safe_filename}"
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                audio_path,
                "-ss",
                f"{vocal_start_time:.2f}",
                "-to",
                f"{vocal_end_time:.2f}",
                "-af",
                f"afade=t=out:st={vocal_end_time -1}:d=1",  # Fade out last 1 second
                "-b:a",
                "192k",
                trimmed_audio_path,
            ],
            check=True,
        )

        # Schedule cleanup task
        background_tasks.add_task(cleanup_temp_folder)
        logger.info("Calling cleanup functions")

        return FileResponse(
            trimmed_audio_path,
            media_type="audio/mp3",
            filename=f"trimmed_{safe_filename}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def cleanup_temp_folder():
    """
    Delete all files in the 'temp' folder after processing.
    """
    try:
        shutil.rmtree("temp")
        logger.info("Temporary files cleaned up successfully.")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
