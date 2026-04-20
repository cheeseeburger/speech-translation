import gradio as gr
import torch
import librosa
import numpy as np
import soundfile as sf
import re
import os
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ── Device Setup (M3 MacBook uses MPS) ──────────────────────────────────────
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# ── Load Models ──────────────────────────────────────────────────────────────
print("Loading models... (first run downloads ~4GB, please wait)")

# faster-whisper must stay on CPU (no MPS support)
whisper_model = WhisperModel("large-v3", device="cpu", compute_type="int8")

nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-distilled-1.3B",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device)
nllb_model.eval()

print("Models loaded! Ready.")

# ── Telugu Text Helpers ──────────────────────────────────────────────────────
TELUGU_WORD_FIXES = {
    'పాల్': 'పాలు', 'నికు': 'నీకు', 'నిక్': 'నీకు',
    'నేన్': 'నేను', 'వాడ్': 'వాడు', 'దాన్': 'దాని',
    'చేస్': 'చేస్తు', 'వస్తా': 'వస్తాను',
}

def clean_text(text):
    for prefix in ["ఈ సంభాషణ సాధారణ తెలుగులో ఉంది", "ఈ సంభాషణ తెలుగులో ఉంది"]:
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip('.').strip()
    text = re.sub(r'(.)\1{3,}', r'\1', text)
    text = re.sub(r'\b(\w+)\b(?:\s+\1\b){1,}', r'\1', text, flags=re.UNICODE)
    words = text.split()
    if len(words) > 4 and len(set(words)) / len(words) < 0.35:
        return ''
    return text.strip()

def fix_telugu_text(text):
    for wrong, right in TELUGU_WORD_FIXES.items():
        text = text.replace(wrong, right)
    return text

def is_telugu(text):
    return any('\u0C00' <= c <= '\u0C7F' for c in text)

# ── Audio Processing (no ffmpeg needed — librosa handles it directly) ────────
def process_audio(audio_path):
    speech, _ = librosa.load(audio_path, sr=16000, mono=True)
    out_path = '/tmp/clean.wav'
    sf.write(out_path, speech, 16000)
    return out_path

# ── Transcription (Whisper large-v3 on CPU) ───────────────────────────────────
def transcribe_telugu(audio_path):
    segments, _ = whisper_model.transcribe(
        audio_path,
        language='te',
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        without_timestamps=True,
    )
    text = " ".join([s.text for s in segments])
    text = clean_text(text.strip())
    text = fix_telugu_text(text)
    if text and not is_telugu(text):
        return ""
    return text

# ── Translation (NLLB 1.3B on MPS) ───────────────────────────────────────────
def translate_to_english(telugu_text):
    nllb_tokenizer.src_lang = "tel_Telu"
    inputs = nllb_tokenizer(
        telugu_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    tgt_id = nllb_tokenizer.lang_code_to_id["eng_Latn"]
    with torch.no_grad():
        outputs = nllb_model.generate(
            **inputs,
            forced_bos_token_id=tgt_id,
            num_beams=5,
            max_length=256,
        )
    return nllb_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()

# ── Main Pipeline ─────────────────────────────────────────────────────────────
def translate_audio(audio):
    if audio is None:
        return "No audio received", ""
    try:
        cleaned = process_audio(audio)
        telugu = transcribe_telugu(cleaned)
        if not telugu:
            return "Could not detect Telugu speech", ""
        english = translate_to_english(telugu)
        return telugu, english
    except Exception as e:
        import traceback
        traceback.print_exc()  # prints full error in terminal
        return f"Error: {str(e)}", ""

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="Telugu Speech Translator") as demo:
    gr.Markdown("""
    # 🎙️ Telugu Speech Translator
    ### Real-time Telugu → English speech translation
    *Whisper large-v3 + NLLB-200 1.3B | Apple M3 optimised*
    ---
    """)
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Speak in Telugu",
            )
            translate_btn = gr.Button("Translate", variant="primary", size="lg")
        with gr.Column():
            telugu_output = gr.Textbox(
                label="Telugu (transcribed)",
                lines=3,
                interactive=False,
            )
            english_output = gr.Textbox(
                label="English (translated)",
                lines=3,
                interactive=False,
            )
    translate_btn.click(
        fn=translate_audio,
        inputs=audio_input,
        outputs=[telugu_output, english_output],
    )
    

demo.launch(share=True)
# before runnig it in termin al, make sure to install the required libraries:
# pip install gradio torch torchvision torchaudio transformers faster-whisper librosa soundfile 
# make sure to run this on an Apple M3 MacBook for best performance (MPS support)
# also note that the first run will download the models (~4GB) and may take a few minutes to load. Subsequent runs will be much faster. 
# in your teminal to patch the transformers library for MPS support, run:
# pip install --upgrade transformers accelerate
# if you encounter any issues with MPS, try setting the environment variable before running the app:
# export PYTORCH_ENABLE_MPS_FALLBACK=1
# in the terminal to allow PyTorch to fallback to CPU for unsupported operations while still using MPS where possible.
# make sure to run this line to patch the way "sed -i '' 's/nllb_tokenizer.lang_code_to_id\["eng_Latn"\]/nllb_tokenizer.convert_tokens_to_ids("eng_Latn")/g' /Users/natanyagettineni/Documents/telugu_speech_translator/app.py"
