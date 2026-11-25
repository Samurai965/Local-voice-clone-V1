"""Local Hindi/Hinglish Voice Cloning using Abhinay45/XTTS-Hindi-finetuned on CUDA."""
import argparse
import os
import re
import subprocess
import tempfile
import numpy as np
import soundfile as sf
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "xtts_hindi")
DEFAULT_REFS = [
    os.path.join(BASE_DIR, "refs", "ref_1.wav"),
    os.path.join(BASE_DIR, "refs", "ref_2.wav"),
    os.path.join(BASE_DIR, "refs", "ref_3.wav"),
]

# Hindi number words (0-99) for Indian numbering system (hazaar, lakh, crore)
_HI_0_99 = [
    "शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",
    "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस",
    "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस",
    "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चौवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास",
    "पचास", "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
    "साठ", "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर",
    "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उन्यासी",
    "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अट्ठासी", "नवासी",
    "नब्बे", "इक्यानबे", "बानबे", "तिरानबे", "चौरानबे", "पचानबे", "छियानबे", "सत्तानबे", "अट्ठानबे", "निन्यानबे",
]


def int_to_hindi(n: int) -> str:
    """Convert integer to natural spoken Hindi words (supports crore/lakh/hazaar)."""
    if n < 0:
        return "माइनस " + int_to_hindi(-n)
    if n < 100:
        return _HI_0_99[n]
    parts = []
    crore, n = divmod(n, 10_000_000)
    if crore:
        parts.append(f"{int_to_hindi(crore)} करोड़")
    lakh, n = divmod(n, 100_000)
    if lakh:
        parts.append(f"{int_to_hindi(lakh)} लाख")
    hazaar, n = divmod(n, 1_000)
    if hazaar:
        parts.append(f"{int_to_hindi(hazaar)} हज़ार")
    sau, n = divmod(n, 100)
    if sau:
        parts.append(f"{_HI_0_99[sau]} सौ")
    if n:
        parts.append(_HI_0_99[n])
    return " ".join(parts)


def normalize_hindi_text(text: str) -> str:
    """Replace unsupported characters and expand digits into Hindi words so XTTS never fails."""
    # Convert Devanagari digits (०-९) to ASCII first
    dev_digits = str.maketrans("०१२३४५६७८९", "0123456789")
    text = text.translate(dev_digits)

    # Clean punctuation that can trip XTTS
    text = text.replace("—", ", ").replace("–", ", ").replace("...", ", ").replace("…", ", ")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("%", " प्रतिशत")

    # Expand decimals like 0.04 -> ज़ीरो पॉइंट ज़ीरो चार
    def _rep_decimal(m):
        whole, frac = m.group(1), m.group(2)
        frac_words = " ".join(_HI_0_99[int(d)] for d in frac)
        return f"{int_to_hindi(int(whole))} पॉइंट {frac_words}"

    text = re.sub(r"\b(\d+)\.(\d+)\b", _rep_decimal, text)

    # Expand integers (strip commas in numbers like 4,000 first)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"\d+", lambda m: int_to_hindi(int(m.group(0))), text)

    # Collapse whitespace/newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_reference_clips(ref_paths: list[str], work_dir: str | None = None) -> list[str]:
    """Ensure uploaded references (even long MP3s) become clean 9s 24kHz mono WAV clips."""
    out_clips = []
    tmp_dir = work_dir or tempfile.mkdtemp(prefix="xtts_refs_")
    os.makedirs(tmp_dir, exist_ok=True)
    for idx, p in enumerate(ref_paths):
        if not os.path.exists(p):
            continue
        # If it's already one of our pre-sliced short WAVs (< 1 MB), use directly
        if p.lower().endswith(".wav") and os.path.getsize(p) < 1_200_000:
            out_clips.append(p)
            continue
        # Extract up to 2 clean 9s slices from longer audio/MP3 using ffmpeg
        for s_idx, start_sec in enumerate((4, 35)):
            dst = os.path.join(tmp_dir, f"ref_{idx}_{s_idx}.wav")
            cmd = [
                "ffmpeg", "-y", "-ss", str(start_sec), "-t", "9", "-i", p,
                "-af", "silenceremove=start_periods=1:start_threshold=-40dB:stop_periods=-1:stop_duration=0.4:stop_threshold=-40dB",
                "-ac", "1", "-ar", "24000", dst,
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 40_000:
                out_clips.append(dst)
        # Fallback if audio is shorter than 4s
        if not any(f"ref_{idx}_" in c for c in out_clips):
            dst = os.path.join(tmp_dir, f"ref_{idx}_full.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-t", "12", "-i", p, "-ac", "1", "-ar", "24000", dst],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if os.path.exists(dst) and os.path.getsize(dst) > 10_000:
                out_clips.append(dst)
    return out_clips if out_clips else ref_paths


def split_hindi_sentences(text: str, max_chars: int = 135) -> list[str]:
    """Split Hindi/English script cleanly at sentence/clause boundaries under XTTS 150-char limit."""
    raw = re.split(r"(?<=[।.!?])\s+", text.strip())
    chunks = []
    buf = ""
    for part in raw:
        part = part.strip()
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
                subs = re.split(r"(?<=[,])\s+", part)
                sbuf = ""
                for s in subs:
                    if len(sbuf) + len(s) + 1 <= max_chars:
                        sbuf = f"{sbuf} {s}".strip() if sbuf else s
                    else:
                        if sbuf:
                            chunks.append(sbuf)
                        if len(s) <= max_chars:
                            sbuf = s
                        else:
                            # Hard word-wrap fallback for very long unpunctuated clauses
                            words = s.split()
                            wbuf = ""
                            for w in words:
                                if len(wbuf) + len(w) + 1 <= max_chars:
                                    wbuf = f"{wbuf} {w}".strip() if wbuf else w
                                else:
                                    if wbuf:
                                        chunks.append(wbuf)
                                    wbuf = w
                            sbuf = wbuf
                buf = sbuf
    if buf:
        chunks.append(buf)
    return chunks


def load_xtts_hindi(model_dir: str = MODEL_DIR):
    cfg = XttsConfig()
    cfg.load_json(os.path.join(model_dir, "config.json"))
    model = Xtts(cfg)
    model.load_checkpoint(cfg, checkpoint_dir=model_dir, use_deepspeed=False)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


def generate_xtts(
    model: Xtts,
    text: str,
    ref_paths: list[str],
    language: str = "hi",
    temperature: float = 0.65,
    speed: float = 1.0,
    repetition_penalty: float = 10.0,
    pause_ms: int = 180,
) -> tuple[np.ndarray, int]:
    clean_refs = prepare_reference_clips(ref_paths)
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=clean_refs,
        gpt_cond_len=30,
        gpt_cond_chunk_len=6,
        sound_norm_refs=True,
    )
    proc_text = normalize_hindi_text(text) if language == "hi" else re.sub(r"\s+", " ", text).strip()
    chunks = split_hindi_sentences(proc_text, max_chars=135)
    sr = 24000
    pause = np.zeros(int(sr * pause_ms / 1000.0), dtype=np.float32)
    pieces = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}/{len(chunks)}] {chunk}")
        out = model.inference(
            text=chunk,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_k=50,
            top_p=0.85,
            speed=speed,
            enable_text_splitting=False,
        )
        wav = np.asarray(out["wav"], dtype=np.float32)
        pieces.append(wav)
        if i < len(chunks):
            pieces.append(pause)
    full = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    peak = np.max(np.abs(full))
    if peak > 0.98:
        full = full * (0.95 / peak)
    return full, sr


def main():
    parser = argparse.ArgumentParser(description="XTTS-Hindi Voice Cloning CLI")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--text-file", type=str, default=None)
    parser.add_argument("--refs", nargs="+", default=None, help="One or more clean reference WAV/MP3 files.")
    parser.add_argument("--out", type=str, default=os.path.join(BASE_DIR, "output_xtts.wav"))
    parser.add_argument("--lang", type=str, default="hi")
    parser.add_argument("--temp", type=float, default=0.65)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    elif args.text:
        text = args.text.strip()
    else:
        raise ValueError("Provide --text or --text-file")

    refs = args.refs if args.refs else [p for p in DEFAULT_REFS if os.path.exists(p)]
    if not refs:
        refs = [os.path.join(BASE_DIR, "ref_voice.wav")]
    print(f"Loading XTTS-Hindi from {MODEL_DIR} using {len(refs)} reference clip(s)...")
    model = load_xtts_hindi(MODEL_DIR)
    audio, sr = generate_xtts(model, text, refs, language=args.lang, temperature=args.temp, speed=args.speed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sf.write(args.out, audio, sr)
    print(f"Saved XTTS-Hindi output: {args.out} ({len(audio)/sr:.2f}s @ {sr}Hz)")


if __name__ == "__main__":
    main()
