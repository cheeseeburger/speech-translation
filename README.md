# Telugu Speech Translator

A real-time Telugu to English speech translation system built entirely using open-source models. No paid APIs, no cloud services, and no internet required after the initial setup. All processing happens locally on your machine.

---

## Overview

This project converts spoken Telugu into English text in real time using a two-stage pipeline:

1. **Speech-to-Text Transcription**: Telugu audio → Telugu text
2. **Machine Translation**: Telugu text → English text

The system runs fully offline after the first model download. Models used provide high-quality multilingual support with minimal hardware requirements.

---

## System Architecture

### Pipeline

```
Microphone Input (Telugu Audio)
    ↓
Audio Processing (librosa)
    ↓
Whisper (Speech to Text)
    ↓
Telugu Text
    ↓
NLLB-200 (Translation)
    ↓
English Text Output
```

### Components

| Component | Purpose | Model |
|-----------|---------|-------|
| **Microphone Input** | Capture Telugu speech | Hardware microphone |
| **Audio Processing** | Preprocess and normalize audio | librosa |
| **Speech Recognition** | Transcribe Telugu audio to text | Whisper large-v3 |
| **Translation** | Convert Telugu text to English | NLLB-200 (distilled-1.3B) |
| **English Text Output** | Final translated text | System output |

---

## Requirements

### Software Requirements

- **Python**: 3.9 or higher
- **RAM**: Minimum 8 GB (16 GB recommended)
- **GPU**: Not required (optional for faster processing)

### Hardware Notes

- Works on any standard laptop or desktop
- **Apple Silicon Support**: M1, M2, M3 can use MPS acceleration for faster inference
- **Lower-end systems**: Will run slower but still functional
- **Performance**: Depends on available RAM

---

## Models Used

### 1. Whisper (Speech-to-Text)
- **Size**: Large-v3 version for optimal accuracy
- **Function**: Transcribes Telugu audio to Telugu text
- **Offline**: Fully offline after download

### 2. NLLB-200 (Translation)
- **Version**: Distilled-1.3B (smaller, faster variant)
- **Function**: Translates Telugu text to English
- **Coverage**: Supports 200+ languages

---

## Installation

### Step 1: Install Dependencies

```bash
pip install faster-whisper transformers torch librosa soundfile sentencepiece sacremoses accelerate
```

### Step 2: Specific Version Requirements

```bash
pip install "gradio==4.44.1" "gradio-client==0.9.1"
```

### Step 3: Model Download

Models (~4 GB) download automatically on first run. Subsequent runs load from cache (~30 seconds).

---

## Running the Application

### Start the Application

```bash
python3 app.py
```

### Access in Browser

Open your browser and navigate to:
```
http://127.0.0.1:7860
```

### First Run

- Initial startup downloads models (~4 GB)
- Takes a few minutes on first run
- Subsequent runs are faster (~30 seconds)

---

## Known Issues and Fixes

### Issue 1: Gradio Schema Bug

**Error**: `Crash caused by boolean JSON schema`

**Fix**: Use isinstance checks instead of type() with json_schema checks:
```python
if isinstance(value, bool):  # Correct
    json_schema_to_python_type()
```

### Issue 2: Tokenizer Attribute Error

**Error**: `lang_code_to_id removed in newer versions`

**Fix**: Use convert_tokens_to_ids("eng_Latn") instead for language code tokenization

### Issue 3: FFmpeg Dependency

**Error**: `FFmpeg Dependency Required`

**Solution**: Replace with librosa.load() for audio file loading

### Issue 4: Translation Repetition

**Error**: Model repeating phrases

**Fix**: Use parameters during inference:
- `no_repeat_ngram_size=3`
- `repetition_penalty=1.3`

---

## Performance Notes

### Speed Benchmarks

- **Whisper**: Runs on CPU; NLLB runs on GPU if available (using MPS on Apple Silicon, otherwise CPU)
- **Memory Usage**: Approximately 5-6 GB during operation
- **Processing Time**: Varies based on audio length and hardware

### Optimization Tips

- Use distilled models for faster inference on lower-end systems
- Enable GPU/MPS acceleration when available
- Monitor RAM usage for large audio files

---

## Project Structure

```
telugu_speech_translator/
├── app.py                      # Main application
├── README.md                   # This file
├── telugu_ai/                  # Core translation module
├── cv_corpus-24.0-2025-12-05/  # Dataset folder
│   ├── te/                     # Telugu language data
│   │   ├── clips/              # Audio clips
│   │   ├── validated.tsv       # Validated samples
│   │   ├── train.tsv           # Training data
│   │   ├── dev.tsv             # Development set
│   │   └── test.tsv            # Test set
```

---

## Dataset

### Mozilla Common Voice 24.0

- **Language**: Telugu (December 2025 release)
- **Samples**: 165 validated samples
- **Filtering**: Word count (3-20 words per sample)
- **Quality**: No duplicates, high-quality audio
- **Source**: [Mozilla Common Voice Datasets](https://commonvoice.mozilla.org/en/datasets)

### Data Splits

- **Validated**: 165 samples (quality-checked)
- **Train**: Training dataset
- **Dev**: Development/validation set
- **Test**: Test evaluation set

---

## Design Decisions

### Core Philosophy

- **No Paid APIs**: Fully independent, no vendor lock-in
- **No Cloud Usage**: Complete privacy and offline capability
- **No LLMs**: Lightweight, efficient inference
- **Pure Python**: Easy to understand and modify

### Model Selection

- **Whisper**: Industry-standard for multilingual speech recognition
- **NLLB-200**: Optimized for low-resource languages like Telugu
- **Distilled versions**: Faster inference without sacrificing quality

---

## Requirements Summary

### Python Dependencies

```
faster-whisper
transformers
torch
librosa
soundfile
sentencepiece
sacremoses
accelerate
gradio==4.44.1
gradio-client==0.9.1
```

### Hardware Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Storage | ~5 GB | ~10 GB |
| GPU | Not required | Optional |
| Python | 3.9+ | 3.9+ |

---

## Future Improvements

- Support for additional Indian languages
- Real-time streaming translation
- Fine-tuning on Telugu-specific data
- Mobile app deployment
- Web service deployment

---

## License

This project uses open-source models and is designed for educational and research purposes.

---

## Support & Resources

- **Common Voice Dataset**: https://commonvoice.mozilla.org/en/datasets
- **Whisper Documentation**: OpenAI Whisper GitHub
- **NLLB-200 Model**: Meta's NLLB-200 Documentation

---

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
