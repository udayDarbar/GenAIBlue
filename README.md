# EchoFlow : Speech-to-Speech Conversational AI System

## Project Overview

This project implements a speech-to-speech conversational AI system. It takes audio input, transcribes it to text, generates a response using a language model, and converts the response back to speech.

## Pipeline Components

The pipeline consists of the following main components:

1.  **Speech Enhancement:** Uses MP-SENet to reduce noise in the input audio.
2.  **Automatic Speech Recognition (ASR):** Employs SpeechT5 to convert the denoised audio into text.
3.  **Language Generation:** Uses DeepSeek R1 to generate a text-based response to the transcribed input.
4.  **Text-to-Speech (TTS):** Converts the generated text response back into an audio signal using VITS and BigVGAN.

## Code Structure

*   **[Team_blue.ipynb](Team_blue.ipynb):** Contains the main implementation of the speech pipeline, including model loading, audio processing, and the overall pipeline execution.
*   **MP-SENet-main/:** Contains the implementation of the MP-SENet model for speech enhancement.

## Models Used

*   **MP-SENet:** For denoising audio.
    *   Original paper: [Explicit Estimation of Magnitude and Phase Spectra in Parallel for High-Quality Speech Enhancement](https://arxiv.org/abs/2305.13686)
    *   Implementation: Located in the `MP-SENet-main/` directory.
*   **SpeechT5:** For ASR and potentially TTS.
    *   Hugging Face: [microsoft/speecht5\_asr](https://huggingface.co/microsoft/speecht5_asr)
*   **VITS:** For generating the base waveform in TTS.
    *   Hugging Face: [facebook/mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng)
*   **BigVGAN:** For enhancing the audio quality of the TTS output.
    *   Hugging Face: [nvidia/bigvgan\_v2\_44khz\_128band\_512x](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x)
*   **DeepSeek R1:** For language generation. Loaded from a local GGUF model.

## Setup Instructions

1.  **Install Dependencies:**

    ```bash
    pip install torchaudio bigvgan sounddevice joblib
    pip install llama-cpp-python
    ```

2.  **Download Models:** The notebook assumes the following models are available:

    *   SpeechT5 ASR and TTS models from Hugging Face.
    *   VITS model from Hugging Face.
    *   BigVGAN model from Hugging Face.
    *   DeepSeek R1 GGUF model located at `C:/Users/udayr/.lmstudio/models/lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf`.  You may need to adjust the path in the [Team_blue.ipynb](Team_blue.ipynb) notebook.
    *   MP-SENet pretrained weights (the notebook attempts to load these from `MP-SENet-main/MP-SENet-main/best_ckpt/g_best_vb`).

3.  **Environment Setup:**

    *   Ensure you have Python 3.6 or higher.
    *   Install PyTorch.
    *   Install the `transformers` library.

## Running the Pipeline

1.  Open and run the [Team_blue.ipynb](Team_blue.ipynb) notebook.
2.  The `main()` function in the last cell provides an example of how to use the `SpeechPipeline` class.
3.  The pipeline will record audio from your microphone (if `sounddevice` is installed) or use dummy audio.
4.  The output audio will be saved to `output_speech.wav`.

## Key Classes and Functions

*   **`SpeechPipeline`:** This class encapsulates the entire speech-to-speech pipeline.
    *   `__init__`: Initializes the pipeline, sets up the device (CPU or CUDA), and loads the models.
    *   `setup_models`: Loads all the required models (MP-SENet, SpeechT5, VITS, BigVGAN, DeepSeek R1).
    *   `load_mpsenet`: Loads the MP-SENet model and pretrained weights.
    *   `denoise_audio`: Denoises the input audio using MP-SENet.
    *   `speech_to_text`: Converts audio to text using SpeechT5.
    *   `generate_response`: Generates a text response using DeepSeek R1.
    *   `text_to_speech`: Converts text to speech using VITS and BigVGAN.
    *   `full_pipeline`: Executes the complete pipeline: denoise -> ASR -> generate response -> TTS.

## Troubleshooting

*   **Missing Dependencies:** If you encounter "ModuleNotFoundError", ensure that you have installed all the required dependencies using `pip`.
*   **Model Loading Errors:** Double-check the paths to the pretrained models in the notebook.
*   **CUDA Issues:** If you are using CUDA, make sure that you have the correct drivers installed and that PyTorch is configured to use your GPU.
*   **DeepSeek R1:** Ensure the path to your GGUF model is correct.

## Team 
Udaysinh Rathod, Swikrit Aryal, Sharmin Nahar, Khushi Rana, Manideep Reddy Kunta, and Edward Nicol Kwakye


<img width="772" height="844" alt="image" src="https://github.com/user-attachments/assets/6bfa0cc2-c7d9-4063-bcc5-740773c74483" />
