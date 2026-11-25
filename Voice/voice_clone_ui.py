"""Upgraded Local Hindi / Hinglish Voice Cloner Web UI using XTTS-Hindi-finetuned on CUDA."""
import os
import gradio as gr
import numpy as np
from clone_xtts_hindi import (
    BASE_DIR,
    DEFAULT_REFS,
    MODEL_DIR,
    generate_xtts,
    load_xtts_hindi,
)

print(f"Loading XTTS-Hindi model from {MODEL_DIR}...")
MODEL = load_xtts_hindi(MODEL_DIR)


def synthesize(
    text: str,
    custom_refs: list[str] | None,
    language: str,
    temperature: float,
    speed: float,
    pause_ms: int,
):
    if not text or not text.strip():
        raise gr.Error("कृपया स्क्रिप्ट बॉक्स में टेक्स्ट लिखें।")

    if custom_refs and len(custom_refs) > 0:
        ref_paths = custom_refs
    else:
        ref_paths = [p for p in DEFAULT_REFS if os.path.exists(p)]
        if not ref_paths:
            fallback = os.path.join(BASE_DIR, "ref_voice.wav")
            if os.path.exists(fallback):
                ref_paths = [fallback]

    if not ref_paths:
        raise gr.Error("No reference audio found. Please upload 1-3 clean voice clips (6-10s each).")

    try:
        audio, sr = generate_xtts(
            model=MODEL,
            text=text.strip(),
            ref_paths=ref_paths,
            language=language,
            temperature=float(temperature),
            speed=float(speed),
            pause_ms=int(pause_ms),
        )
        return (sr, audio)
    except Exception as exc:
        raise gr.Error(f"Generation failed: {exc}")


with gr.Blocks(title="XTTS-Hindi Voice Cloner (RTX 3050)") as demo:
    gr.Markdown(
        "# Local Voice Cloner (XTTS-Hindi Fine-Tuned)\n"
        "`New folder (2)` की ऑडियो (`[2026-11-01][19-26-33]-enhanced-v2.mp3`) की 3 साफ रेफरेंस क्लिप्स (`ref_1.wav`, `ref_2.wav`, `ref_3.wav`) स्थायी डिफ़ॉल्ट के रूप में लोडेड हैं।\n\n"
        "**बेस्ट रिजल्ट के लिए टिप:** हिंदी स्क्रिप्ट में इंग्लिश शब्दों को देवनागरी में लिखें (जैसे *ब्रुटैलिटी*, *इनसेन फाइट्स*, *एक्सिस्ट*) और छोटे वाक्य (`।`, `.`, `,`) रखें।"
    )
    with gr.Row():
        with gr.Column():
            text_in = gr.Textbox(
                label="Script (हिंदी देवनागरी में लिखें; इंग्लिश शब्द भी हिंदी में लिखें तो बेस्ट आएगा)",
                lines=8,
                placeholder="यहाँ अपनी स्क्रिप्ट पेस्ट करें...",
            )
            refs_in = gr.File(
                label="Custom Reference Voice Clips (Optional — खाली छोड़ेंगे तो आपकी डिफ़ॉल्ट 3 क्लिप्स यूज़ होंगी)",
                file_count="multiple",
                file_types=["audio"],
                type="filepath",
            )
            with gr.Row():
                lang_in = gr.Dropdown(["hi", "en"], value="hi", label="Language")
                temp_in = gr.Slider(0.2, 0.95, value=0.65, step=0.05, label="Expressiveness (Temp)")
                speed_in = gr.Slider(0.8, 1.3, value=1.0, step=0.05, label="Speed")
                pause_in = gr.Slider(50, 500, value=180, step=20, label="Sentence Pause (ms)")
            btn = gr.Button("Generate Voice", variant="primary")
        with gr.Column():
            audio_out = gr.Audio(label="Cloned Voice Output (24kHz WAV)", type="numpy")

    btn.click(
        synthesize,
        inputs=[text_in, refs_in, lang_in, temp_in, speed_in, pause_in],
        outputs=audio_out,
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
