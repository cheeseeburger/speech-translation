 Telugu Speech Translator
A real-time Telugu to English Speech Translation System built without any paid APIs or cloud services. Everything runs locally on your machine using open-source models.

 Models Used
ModelPurposeVersionOpenAI WhisperTelugu Speech → Telugu Textlarge-v3Meta NLLB-200Telugu Text → English Textnllb-200-distilled-1.3B

 System Architecture
Microphone Input (Telugu Audio)
        ↓
  Audio Processing (librosa)
        ↓
  Whisper large-v3 (Speech to Text)
        ↓
  Telugu Text
        ↓
  NLLB-200 1.3B (Translation)
        ↓
  English Text Output

 Requirements

Python 3.9+
Apple M3 Mac (MPS) or any machine with 8GB+ RAM
No NVIDIA GPU required
No internet needed after first model download

Install dependencies
bashpip install faster-whisper transformers torch librosa soundfile sentencepiece sacremoses accelerate "gradio==4.44.1" "gradio-client==0.9.1"

 Running the App
bashpython3 app.py
Then open http://127.0.0.1:7860 in your browser.

First run will download models (~4GB). After that it loads in ~30 seconds.


 Known Fixes Applied
1. Gradio bool schema bug
Patched gradio_client/utils.py — get_type() and _json_schema_to_python_type() functions crash when schema is a bool instead of a dict. Fixed by adding isinstance guard.
2. lang_code_to_id AttributeError
NllbTokenizerFast does not have lang_code_to_id attribute in newer transformers versions. Fixed by using convert_tokens_to_ids("eng_Latn") instead.
3. ffmpeg dependency removed
Original pipeline used ffmpeg for audio conversion which caused FileNotFoundError. Replaced with librosa.load() which reads audio directly — no ffmpeg needed.
4. Translation repetition
NLLB-200 was repeating phrases. Fixed by adding no_repeat_ngram_size=3 and repetition_penalty=1.3 to the generate call.

💻 Apple M3 Optimisation

Whisper runs on CPU (faster-whisper does not support MPS yet)
NLLB-200 runs on MPS (Apple GPU) using torch.float16
Unified memory architecture means 8GB M3 handles ~6GB of models comfortably

pythondevice = "mps" if torch.backends.mps.is_available() else "cpu"

 Project Structure
telugu_speech_translator/
├── app.py              # Main application
├── README.md           # This file
└── Telugu_AI/
    └── cv-corpus-24.0-2025-12-05/
        └── te/
            ├── clips/          # Audio files
            ├── validated.tsv   # Dataset
            ├── train.tsv
            ├── dev.tsv
            └── test.tsv

Dataset
Mozilla Common Voice 24.0 — Telugu (December 2025)

165 clean validated samples used for evaluation
Filtered by: word count (3–20), vote quality, no duplicates
Download: https://commonvoice.mozilla.org/en/datasets


Team — Batch 4
MemberContributionNatanyaPipeline integration, NLLB translation, MPS optimisation, bug fixesMember 2Gradio UI, frontend designMember 3Dataset selection, model comparison, evaluation

 Key Design Decisions

No paid APIs — fully open source, runs offline
No LLMs — Whisper is a speech model, NLLB is a translation model
No plugins — pure Python, standard libraries only
Local inference — data never leaves your machine

 License
Open source for academic purposes.
