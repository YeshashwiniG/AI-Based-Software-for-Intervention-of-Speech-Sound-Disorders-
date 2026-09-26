"""FastAPI service for continuous recording-level speech analysis."""
import json
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from backend.audio_processor import AudioProcessor
from backend.ssd_detector import SpeechSoundDisorderDetector
from backend.speech_recognizer import SpeechRecognizer
from backend.word_pronunciation import WordPronunciationAnalyzer
from backend.phoneme_pronunciation import PhonemePronunciationAnalyzer

app = FastAPI(title="English Pronunciation Practice Analyzer", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
detector = SpeechSoundDisorderDetector()
speech_recognizer = SpeechRecognizer()
word_analyzer = WordPronunciationAnalyzer()
phoneme_analyzer = PhonemePronunciationAnalyzer()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SENTENCE_BANK = [
    {"id": "speak-three", "text": "I speak three times."},
    {"id": "think-this", "text": "She thinks this is easy."},
    {"id": "red-very", "text": "The red car is very fast."},
    {"id": "please-close", "text": "Please close the door."},
    {"id": "fresh-fish", "text": "The boy likes fresh fish."},
    {"id": "we-call-bear", "text": "We call it bear."},
]
SENTENCE_BY_ID = {item["id"]: item["text"] for item in SENTENCE_BANK}


@app.get("/api/health")
def health_check():
    return {"status": "ok", "trained_model_loaded": detector.model is not None,
            "word_model_loaded": word_analyzer.model is not None,
            "phoneme_model_loaded": phoneme_analyzer.model is not None,
            "analysis_scope": "expected_sentence_target_phone_rating_proxy"}


@app.get("/api/sentences")
def get_sentence_bank():
    return {"sentences": SENTENCE_BANK}


@app.get("/api/datasets")
def get_datasets():
    metrics_path = os.path.join(BASE_DIR, "models", "training_metrics.json")
    if not os.path.isfile(metrics_path):
        return {"datasets": [], "message": "Train the model to produce a report from the local dataset."}
    with open(metrics_path, encoding="utf-8") as f:
        info = json.load(f)
    word_metrics_path = os.path.join(BASE_DIR, "models", "word_training_metrics.json")
    if os.path.isfile(word_metrics_path):
        with open(word_metrics_path, encoding="utf-8") as f:
            word_info = json.load(f)
        labels = word_info["label_definition"]
    else:
        labels = info["label_definition"]
    return {"datasets": [{"name": info["dataset_name"], "source": info["dataset_source"],
            "participants": info["participants_in_dataset"], "recordings": info["total_parquet_recordings"],
            "usable_recordings": info["usable_recordings"], "labels": labels}]}


@app.get("/api/target-sounds")
def get_target_sounds():
    metrics_path = os.path.join(BASE_DIR, "models", "training_metrics.json")
    if not os.path.isfile(metrics_path):
        return {"targets": {}, "message": "Run model training to calculate counts from the local data."}
    with open(metrics_path, encoding="utf-8") as f:
        info = json.load(f)
    return {"targets": info.get("target_phone_samples", {}),
            "note": "Dataset annotation counts, not predictions. The current sound model uses whole-word crops because the dataset has no phone timestamps.",
            "model_test_metrics": (phoneme_analyzer.metrics or {}).get("per_phone_test_metrics", {})}


@app.post("/api/analyze-audio")
async def analyze_audio(audio_file: UploadFile = File(...), prompt_text: str = Form("")):
    try:
        content = await audio_file.read()
        audio, sr = AudioProcessor.load_audio_from_bytes(content)
        return JSONResponse(content=detector.analyze_continuous_speech(audio, sr, prompt_text))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Audio analysis error: {exc}")


@app.post("/api/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Recognize continuous English audio and return ordered word timestamps."""
    try:
        content = await audio_file.read()
        audio, sr = AudioProcessor.load_audio_from_bytes(content)
        result = await run_in_threadpool(speech_recognizer.transcribe, audio, sr)
        result["waveform_envelope"] = AudioProcessor.compute_waveform_envelope(audio, points=120)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Speech recognition error: {exc}")


@app.post("/api/analyze-pronunciation")
async def analyze_pronunciation(audio_file: UploadFile = File(...), sentence_id: str = Form(...)):
    """Analyze a recording against an application-provided sentence-bank entry."""
    expected_text = SENTENCE_BY_ID.get(sentence_id)
    if expected_text is None:
        raise HTTPException(status_code=422, detail="Choose a sentence from the provided practice list.")
    try:
        content = await audio_file.read()
        audio, sr = AudioProcessor.load_audio_from_bytes(content)
        recognized = await run_in_threadpool(speech_recognizer.transcribe, audio, sr)
        results = await run_in_threadpool(phoneme_analyzer.analyze, audio, sr, expected_text, recognized)
        return JSONResponse(content={
            **recognized,
            "expected_text": expected_text.strip(),
            "sentence_id": sentence_id,
            "analysis_scope": "sound_conditioned_word_crop_prediction_of_speechocean_phone_rating_proxy",
            "analysis_note": "Possible pronunciation practice feedback, not a diagnosis. ASR supplies approximate word times only; model sound predictions use whole-word audio because source phone timestamps are unavailable.",
            "word_analysis": results,
            "waveform_envelope": AudioProcessor.compute_waveform_envelope(audio, points=120),
        })
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Pronunciation analysis error: {exc}")


frontend_dir = os.path.join(BASE_DIR, "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8003, reload=False)
