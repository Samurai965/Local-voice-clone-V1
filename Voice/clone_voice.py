"""Local Hindi / Multilingual Voice Cloning CLI using Chatterbox Multilingual TTS on CUDA."""
import argparse
import os
import re
import numpy as np
import soundfile as sf
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


def split_sentences(text: str, max_chars: int = 220) -> list[str]:
    """Split Hindi/English script into clean sentence chunks for stable TTS generation."""
    raw_parts = re.split(r"(?<=[।.!?])\s+", text.strip())
    chunks = []
    buf = ""
    for part in raw_parts:
        if not part:
            continue
        if len(buf) + len(part) + 1 <= max_chars:
            buf = f"{buf} {part}".strip() if buf else part
        else:
            if buf:
                chunks.append(buf)
            if len(part) <= max_chars:
                buf = part
            else:
                # Split long clause by commas
                subparts = re.split(r"(?<=[,])\s+", part)
                subbuf = ""
                for sp in subparts:
                    if len(subbuf) + len(sp) + 1 <= max_chars:
                        subbuf = f"{subbuf} {sp}".strip() if subbuf else sp
                    else:
                        if subbuf:
                            chunks.append(subbuf)
                        subbuf = sp
                buf = subbuf
    if buf:
        chunks.append(buf)
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Clone a voice locally with Chatterbox Multilingual TTS.")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize.")
    parser.add_argument("--text-file", type=str, default=None, help="Path to UTF-8 text script file.")
    parser.add_argument(
        "--ref",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "ref_voice.wav"),
        help="Path to clean reference voice WAV/MP3 (6-15s).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "output_cloned.wav"),
        help="Output WAV path.",
    )
    parser.add_argument("--lang", type=str, default="hi", help="Language ID: 'hi' (Hindi), 'en' (English), etc.")
    parser.add_argument("--exaggeration", type=float, default=0.55, help="Expressiveness (0.3 - 0.8).")
    parser.add_argument("--cfg", type=float, default=0.5, help="CFG weight (0.3 - 0.7).")
    parser.add_argument("--pause-ms", type=int, default=220, help="Pause in ms between sentences.")
    args = parser.parse_args()

    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    elif args.text:
        text = args.text.strip()
    else:
        raise ValueError("Provide either --text or --text-file.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading ChatterboxMultilingualTTS on {device}...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    sr = model.sr

    chunks = split_sentences(text)
    print(f"Generating {len(chunks)} chunk(s) in language '{args.lang}'...")
    pause = np.zeros(int(sr * (args.pause_ms / 1000.0)), dtype=np.float32)

    pieces = []
    for idx, chunk in enumerate(chunks, 1):
        print(f"  [{idx}/{len(chunks)}] {chunk}")
        wav = model.generate(
            text=chunk,
            language_id=args.lang,
            audio_prompt_path=args.ref,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg,
        )
        wav_np = wav.squeeze().detach().cpu().numpy().astype(np.float32)
        pieces.append(wav_np)
        if idx < len(chunks):
            pieces.append(pause)

    final_audio = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sf.write(args.out, final_audio, sr)
    print(f"Saved cloned audio to: {args.out} ({len(final_audio)/sr:.2f}s @ {sr}Hz)")


if __name__ == "__main__":
    main()
