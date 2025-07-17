# GenAIBlue
# Speech-to-Speech Conversational AI System Checklist (Team Division)4

---

## **Project Overview**

We are designing a speech-to-speech conversational AI system. The goal is to enable a natural back-and-forth conversation between a user and the AI.

---

## **Datasets**

### Why These Datasets?

- **STinyStories**: Short conversational stories for training/fine-tuning the language generation component.
    
    🔗 [View on Hugging Face](https://huggingface.co/datasets/slprl/sTinyStories)
    
- **Vibravox**: High-quality speech dataset for training/enhancing ASR and TTS models.
    
    🔗 [View on Hugging Face](https://huggingface.co/datasets/vibravox)
    
- **GLOBE_V2**: Multilingual speech and text pairs for improving conversational diversity and multilingual support.

---

## **Models**

### Why These Models?

- **BigVGAN** (TTS): Generates highly natural and expressive human-like voices.
    
    🔗 [View on Hugging Face](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x)
    
- **MP-SENet-DNS** (Speech Enhancement): Removes noise from real-time microphone input to improve ASR and TTS quality.
    
    🔗 [View on Hugging Face](https://huggingface.co/JacobLinCool/MP-SENet-DNS)
    
- **SpeechT5** (Core Model): Supports speech-to-text, text-to-speech, and speech-to-speech tasks.
    
    🔗 [View on Hugging Face](https://huggingface.co/microsoft/speecht5_vc)
    

---

## 📂 **Datasets**

### ✅ Assigned to: Edward & Sharmin

- [ ]  Download and prepare datasets (STinyStories, Vibravox, GLOBE_V2)
- [ ]  Preprocess datasets for ASR and TTS components
    - [ ]  Text: Tokenization, cleaning, encoding
    - [ ]  Audio: Resampling, normalization

---

## 🧑‍💻 **Models**

### ✅ Assigned to: Khusi & Swikrit

- [ ]  Load Hugging Face models:
    - SpeechT5 (Core model)
    - BigVGAN (TTS)
    - MP-SENet-DNS (Noise suppression)
- [ ]  Set up environment and dependencies (Python, PyTorch, Transformers, etc.)
- [ ]  Test models individually with sample inputs

---

## 🔥 **Language Generation & TTS**

### ✅ Assigned to: Manudeep

- [ ]  Fine-tune SpeechT5 for language generation (if needed)
- [ ]  Implement **TTS (Text-to-Speech)** module using BigVGAN
- [ ]  Validate output for naturalness and clarity

---

## 🎙 **ASR, Testing & Optimization**

### ✅ Assigned to: Uday

- [ ]  Build microphone input and preprocessing module
- [ ]  Implement **ASR (Speech-to-Text)** pipeline using SpeechT5
- [ ]  Integrate MP-SENet-DNS for real-time noise suppression
- [ ]  Test pipeline with different accents and environments
- [ ]  Optimize latency for real-time interaction
