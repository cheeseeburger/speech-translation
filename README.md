Telugu Speech Translator
A real-time Telugu to English speech translation system built entirely using open-source models.
No paid APIs, no cloud services, and no internet required after the initial setup.
All processing happens locally on your machine.
Overview
This project converts spoken Telugu into English text in real time using a two-stage pipeline:
Speech-to-text transcription (Telugu audio → Telugu text)
Machine translation (Telugu text → English text)
The system runs fully offline after the first model download.
Models Used
Model	Purpose	Version
OpenAI Whisper	Telugu speech → Telugu text	large-v3
Meta NLLB-200	Telugu text → English text	nllb-200-distilled-1.3B
System Architecture
Microphone Input (Telugu Audio)
↓
Audio Processing (librosa)
↓
Whisper large-v3 (Speech to Text)
↓
Telugu Text
↓
NLLB-200 (Translation)
↓
English Text Output
Requirements
Python 3.9 or higher
Minimum 8 GB RAM (16 GB recommended)
No GPU required
Hardware Notes
Works on any standard laptop or desktop
Performance depends on RAM
Apple Silicon (M1/M2/M3) can use MPS acceleration
Lower-end systems will run slower but still work

Installation
pip install faster-whisper transformers torch librosa soundfile sentencepiece sacremoses accelerate "gradio==4.44.1" "gradio-client==0.9.1"
Running the Application
python3 app.py
Open in browser:
http://127.0.0.1:7860
First Run
Downloads models (~4 GB)
Later runs load in ~30 seconds
Known Issues and Fixes
Gradio Schema Bug
Crash caused by boolean JSON schema
Fixed using isinstance checks in:
get_type()
_json_schema_to_python_type()
Tokenizer Attribute Error
lang_code_to_id removed in newer versions
Fixed using:
convert_tokens_to_ids("eng_Latn")
FFmpeg Dependency Removed
Removed external dependency
Replaced with:
librosa.load()
Translation Repetition
Model repeating phrases
Fixed using:
no_repeat_ngram_size=3
repetition_penalty=1.3
Performance Notes
Whisper runs on CPU
NLLB runs on GPU if available (MPS on Apple Silicon, otherwise CPU)
Memory usage: ~5–6 GB
Project Structure
telugu_speech_translator/
├── app.py
├── README.md
└── Telugu_AI/
    └── cv-corpus-24.0-2025-12-05/
        └── te/
            ├── clips/
            ├── validated.tsv
            ├── train.tsv
            ├── dev.tsv
            └── test.tsv
Dataset
Mozilla Common Voice 24.0 — Telugu (December 2025)
165 validated samples used
Filtered by:
Word count (3–20)
Vote quality
No duplicates
https://commonvoice.mozilla.org/en/datasets
Design Decisions
No paid APIs
No cloud usage
No LLMs
Fully offline after setup
Pure Python implementation
