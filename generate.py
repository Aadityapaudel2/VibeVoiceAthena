#!/usr/bin/env python
"""
Generate speech audio from a text script using Microsoft's VibeVoice model.

This script locates the locally cached VibeVoice model, reads a user‑provided
script from ``input.txt`` and writes the generated audio into the ``audio``
subfolder.  The filenames are zero‑padded to prevent overwriting existing
files (e.g. ``audio/audio000001.wav``, ``audio/audio000002.wav``, …).

The script attempts to make sensible choices regarding speaker voices.  If
multiple speakers are present in the script (identified by lines starting
with ``Speaker 1:``, ``Speaker 2:``, etc.), the first four speakers will be
mapped to the available example voice samples shipped with the model
(``en‑Alice_woman``, ``en‑Carter_man``, ``en‑Frank_man`` and
``en‑Mary_woman_bgm``).  If fewer voices are available, the first voice in
the model's voices directory is used for all speakers.

Usage: simply run this script from the root of the project.  It does not
accept command‑line arguments; configuration happens via the files in the
package (``input.txt`` and the contents of ``model/``).

When generation finishes, the script prints a checkmark message.  Any
exceptions will cause a stack trace to be printed to aid debugging.
"""

import glob
import os
import re
import sys
import traceback
from typing import List, Tuple

import torch  # type: ignore
import warnings

try:
    # Reduce log verbosity for transformers to suppress noisy warnings
    from transformers.utils import logging as hf_logging  # type: ignore
    hf_logging.set_verbosity_error()
except Exception:
    pass

# Silence optional CUDA/Flash warnings that do not affect functionality
warnings.filterwarnings("ignore", message="APEX FusedRMSNorm not available, using native implementation")
warnings.filterwarnings("ignore", message="The tokenizer class you load from this checkpoint is not the same type")

# Import VibeVoice components.  These imports will succeed after the model
# has been installed as an editable package during setup (pip install -e model).
try:
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor  # type: ignore
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,  # type: ignore
    )
except ImportError as e:
    print(
        "[Error] The VibeVoice package could not be imported.\n"
        "Make sure you have run setup.bat successfully so that the model\n"
        "package is installed."
    )
    raise


def parse_script(txt: str) -> Tuple[List[str], List[str]]:
    """Split the input text into speaker segments.

    Each segment must begin with ``Speaker N:`` where N is a number.  If
    no such pattern is found, the entire text is treated as a single
    segment belonging to speaker "1".

    Returns a tuple of (scripts, speaker_numbers), where both lists are
    aligned.  Each ``scripts[i]`` corresponds to ``speaker_numbers[i]``.
    """
    scripts: List[str] = []
    speaker_numbers: List[str] = []
    # Regular expression for "Speaker X: text"
    pattern = re.compile(r"^Speaker\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)

    current_speaker = None
    current_text = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            # Save previous segment
            if current_speaker is not None and current_text:
                scripts.append(f"Speaker {current_speaker}: " + " ".join(current_text))
                speaker_numbers.append(current_speaker)
            current_speaker = match.group(1)
            # Start new segment with the remaining text after the colon
            current_text = [match.group(2).strip()] if match.group(2) else []
        else:
            # Continuation of the current speaker
            if current_speaker is None:
                # No speaker prefix yet: assume speaker 1
                current_speaker = "1"
            current_text.append(line)
    # Append the final segment if present
    if current_speaker is not None and current_text:
        scripts.append(f"Speaker {current_speaker}: " + " ".join(current_text))
        speaker_numbers.append(current_speaker)
    # Fallback: if nothing was parsed, treat entire text as speaker 1
    if not scripts and txt.strip():
        scripts = [txt.strip()]
        speaker_numbers = ["1"]
    return scripts, speaker_numbers

# -----------------------------------------------------------------------------
# Additional helpers for advanced input features
#
# The following functions implement optional syntax extensions for `input.txt`:
#
#   * `[pause]` or `[pause:ms]` tags introduce silence between segments.  A plain
#     `[pause]` inserts one second (1000 ms) of silence.  `[pause:2000]` inserts
#     two seconds (2000 ms), and so on.  Pauses split the script into
#     independent generations; each portion of text before/after a pause is
#     passed to the model separately.  This prevents the model from seeing
#     across pause boundaries, which may slightly reduce contextual coherence.
#
#   * `[tone:STYLE]` tags modify the tone of subsequent speech.  The tag is
#     replaced with a phrase such as "in a STYLE tone," which is inserted at
#     the beginning of the sentence.  For example, `[tone:excited]` becomes
#     "in an excited tone, ".  Supported styles include "excited",
#     "calm", "sad", "whisper", "shout", etc.  Unknown styles are
#     included verbatim.

def preprocess_pauses(text: str) -> Tuple[List[str], List[int]]:
    """Split a script into segments separated by pause tags.

    Returns a pair (segments, pause_ms) where ``segments`` is the list of
    text chunks and ``pause_ms`` is a list of integer durations in
    milliseconds.  The ``i``‑th pause duration applies *before* ``segments[i]``;
    the first element is always 0 (no pause before the first segment).
    """
    # Pattern for [pause] or [pause:1234]
    pattern = re.compile(r"\[pause(?::(\d+))?\]", re.IGNORECASE)
    segments: List[str] = []
    pauses: List[int] = []
    last_end = 0
    for match in pattern.finditer(text):
        # Text before this pause
        segments.append(text[last_end : match.start()])
        # Record the pause duration (default 1000 ms)
        ms_str = match.group(1)
        ms = int(ms_str) if ms_str is not None else 1000
        pauses.append(ms)
        last_end = match.end()
    # Append the final segment
    segments.append(text[last_end:])
    # Insert zero pause for the first segment
    pauses.insert(0, 0)
    return segments, pauses


def apply_tone_tags(segment: str) -> str:
    """Replace [tone:style] tags with descriptive phrases.

    Example: ``[tone:excited] Hello`` -> ``in an excited tone, Hello``.
    If the style starts with a vowel sound, "an" is used; otherwise "a".
    Tags are removed from the output.
    """
    def repl(match: re.Match) -> str:
        style = match.group(1)
        # Determine the article (a/an) based on the first letter
        article = "an" if style and style[0].lower() in "aeiou" else "a"
        return f"in {article} {style.lower()} tone, "

    return re.sub(r"\[tone:([^\]]+)\]", repl, segment, flags=re.IGNORECASE)


def select_voices(model_path: str, speaker_numbers: List[str]) -> Tuple[List[str], List[str]]:
    """Determine voice sample file paths for each unique speaker.

    A simple mapping from speaker index (1..4) to common English names is
    used: Speaker 1 → ``Alice``, Speaker 2 → ``Carter``, Speaker 3 →
    ``Frank``, Speaker 4 → ``Mary``.  The function searches the
    ``model/demo/voices`` directory for files containing these names.  If
    none are found it falls back to the first voice in the directory.

    Returns a tuple ``(voice_samples, speaker_names)``, where
    ``voice_samples`` is a list of file paths (one per unique speaker in
    order of first appearance) and ``speaker_names`` contains the
    corresponding friendly names.  The lengths of the two lists are
    equal.
    """
    # Start with the model paths and add fallbacks to the cloned repository.
    # The VibeVoice model snapshot does not include voice presets.  However, the
    # community repository cloned during setup contains example voices in its
    # `demo/voices` folder.  We therefore probe several locations and use the
    # first one that contains .wav files.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    voices_dir_candidates = [
        os.path.join(model_path, "demo", "voices"),
        os.path.join(model_path, "voices"),
        os.path.join(base_dir, "vibevoice_repo", "demo", "voices"),
        os.path.join(base_dir, "vibevoice_repo", "voices"),
    ]
    voices_dir = None
    for candidate in voices_dir_candidates:
        if os.path.isdir(candidate) and any(f.lower().endswith(".wav") for f in os.listdir(candidate)):
            voices_dir = candidate
            break
    if voices_dir is None:
        # As a last resort, attempt to find any directory named "voices" under the
        # base directory that contains wav files.
        for root, dirs, files in os.walk(base_dir):
            if os.path.basename(root).lower() == "voices" and any(f.lower().endswith(".wav") for f in files):
                voices_dir = root
                break
    if voices_dir is None:
        raise FileNotFoundError(
            "No voices directory found. The model snapshot does not ship with voice samples, "
            "so ensure that the community repository (vibevoice_repo) exists with a demo/voices folder."
        )
    # Build a mapping from voice name (without extension) to file path
    available_files = [
        f for f in os.listdir(voices_dir) if f.lower().endswith(".wav")
    ]
    voice_map = {}
    for filename in available_files:
        name = os.path.splitext(filename)[0]
        voice_map[name.lower()] = os.path.join(voices_dir, filename)
    # Predefined friendly names in order
    default_names = ["alice", "carter", "frank", "mary"]
    voice_samples = []
    speaker_names = []
    seen = set()
    for speaker_num in speaker_numbers:
        if speaker_num in seen:
            continue
        seen.add(speaker_num)
        index = int(speaker_num) - 1
        preferred_name = default_names[index] if index < len(default_names) else default_names[0]
        # Find a file whose name contains the preferred name
        selected_path = None
        for name, path in voice_map.items():
            if preferred_name in name:
                selected_path = path
                break
        # Fallback to the first available voice
        if selected_path is None and voice_map:
            selected_path = next(iter(voice_map.values()))
        if selected_path is None:
            raise FileNotFoundError(
                f"No voice samples found in {voices_dir}. The VibeVoice model "
                "should include example voices."
            )
        voice_samples.append(selected_path)
        # Use the base filename (without extension) as the speaker name
        speaker_names.append(os.path.splitext(os.path.basename(selected_path))[0])
    return voice_samples, speaker_names


def determine_next_filename(audio_dir: str) -> str:
    """Compute the next audio filename based on existing files in the directory."""
    if not os.path.isdir(audio_dir):
        os.makedirs(audio_dir, exist_ok=True)
    existing = glob.glob(os.path.join(audio_dir, "audio*.wav"))
    numbers = []
    for path in existing:
        base = os.path.basename(path)
        match = re.match(r"audio(\d{6})\.wav", base)
        if match:
            numbers.append(int(match.group(1)))
    next_num = max(numbers) + 1 if numbers else 1
    return f"audio{next_num:06d}.wav"


def main() -> None:
    """Main entry point for voice generation.

    This function reads the input script, determines which voice presets to use
    for each speaker, loads the VibeVoice model with sensible defaults and
    generates an audio file in the ``audio/`` directory.  It tries to choose
    optimal settings based on the available hardware.  Errors are
    caught and reported rather than causing the program to crash.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model")
    input_path = os.path.join(base_dir, "input.txt")
    audio_dir = os.path.join(base_dir, "audio")

    # Read the input script
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"[Error] Could not find {input_path}. Make sure the file exists.")
        return
    if not text.strip():
        print("[Warning] input.txt is empty. Please write something to convert to speech.")
        return

    import time  # local import to avoid unused import lint when unused

    # Preprocess pause tags, splitting the text into segments and associated pauses
    segments, pause_ms_list = preprocess_pauses(text)

    # Load processor and model once (outside the per-segment loop) to avoid repeated
    # initialization.  Choose device and attention implementation based on
    # available hardware.
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    # Inform the user about device choice
    if use_cuda:
        print("Running on GPU (CUDA available)")
    else:
        print("Running on CPU (no CUDA detected)")
    if device == "cuda":
        load_dtype = torch.bfloat16
        attn_impl_primary = "flash_attention_2"
    else:
        load_dtype = torch.float32
        attn_impl_primary = "sdpa"

    try:
        processor = VibeVoiceProcessor.from_pretrained(model_path)
    except Exception as exc:
        print(f"[Error] Failed to load the processor from {model_path}: {exc}")
        traceback.print_exc()
        return

    # Obtain the tokenizer from the processor and ensure BOS token is defined
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        print("[Error] Processor did not provide a tokenizer. Check your installation.")
        return
    # Some versions of VibeVoiceTextTokenizerFast set bos_token_id to None.  We
    # derive it from eos_token_id to satisfy the generation API.
    try:
        if getattr(tokenizer, "bos_token_id", None) is None:
            # Mirror the EOS token for BOS
            tokenizer.bos_token_id = tokenizer.eos_token_id
            tokenizer.bos_token = tokenizer.eos_token
    except Exception:
        pass

    # Load the model with preferred attention, falling back if necessary
    try:
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=load_dtype,
            device_map=device,
            attn_implementation=attn_impl_primary,
        )
    except Exception as exc:
        print(f"[Warning] Failed to load model with {attn_impl_primary}: {exc}\n"
              "Falling back to SDPA implementation.  This may be slower.")
        try:
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=load_dtype,
                device_map=device,
                attn_implementation="sdpa",
            )
        except Exception:
            print("[Error] Failed to load the VibeVoice model. Ensure the model is fully downloaded.")
            traceback.print_exc()
            return

    # Move the model to the selected device explicitly.  Although device_map
    # typically handles placement, calling .to(device) guarantees all
    # parameters reside on the intended device.
    model.eval()
    try:
        model.to(device)
    except Exception:
        pass
    try:
        model.set_ddpm_inference_steps(num_steps=10)
    except Exception:
        pass

    # Warn if CUDA is available but the model is not on a CUDA device
    try:
        if use_cuda and not any(p.device.type == 'cuda' for p in model.parameters()):
            print("[Warning] CUDA is available but the model is running on CPU. Check your PyTorch installation.")
    except Exception:
        pass

    # Container for the final audio segments
    audio_segments: List[torch.Tensor] = []

    # Record start time for performance measurement
    total_start = time.time()

    for seg_text, pause_ms in zip(segments, pause_ms_list):
        # Insert silence before the segment if needed
        if pause_ms > 0:
            # Silence duration in samples (VibeVoice audio uses 24 kHz)
            num_samples = int((pause_ms / 1000.0) * 24000)
            silent = torch.zeros(num_samples, dtype=torch.float32)
            audio_segments.append(silent)
        # Skip empty segments (e.g., multiple pauses in a row)
        if not seg_text.strip():
            continue
        # Apply tone tags within the segment
        seg_processed = apply_tone_tags(seg_text)
        # Parse speaker labels within this segment
        seg_scripts, seg_speaker_numbers = parse_script(seg_processed)
        # Determine voice samples for this segment's unique speakers
        try:
            seg_voice_samples, seg_speaker_names = select_voices(model_path, seg_speaker_numbers)
        except Exception as exc:
            print(f"[Error] Failed to select voice samples: {exc}")
            traceback.print_exc()
            return
        # Prepare inputs for the model
        inputs = processor(
            text=["\n".join(seg_scripts)],
            voice_samples=[seg_voice_samples],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        # Move tensors to the correct device
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device)
        # Generate audio for this segment
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=3.0,
                tokenizer=tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
            )
        except RuntimeError as exc:
            # Handle CUDA kernel compatibility errors gracefully by falling
            # back to CPU.  New GPU architectures (e.g. sm_120) may not be
            # supported by the installed PyTorch build.  In that case we
            # reload the model on the CPU and retry generation.
            msg = str(exc)
            if "no kernel image" in msg or "sm_" in msg:
                print(
                    "[Warning] CUDA execution failed due to unsupported GPU architecture. "
                    "Falling back to CPU. This will be slower."
                )
                # Move tensors to CPU
                for k, v in inputs.items():
                    if torch.is_tensor(v):
                        inputs[k] = v.to("cpu")
                try:
                    # Reload model on CPU only once
                    nonlocal_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        model_path,
                        torch_dtype=torch.float32,
                        device_map="cpu",
                        attn_implementation="sdpa",
                    )
                    nonlocal_model.eval()
                    try:
                        nonlocal_model.set_ddpm_inference_steps(num_steps=10)
                    except Exception:
                        pass
                    # Recompute outputs on CPU
                    outputs = nonlocal_model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=3.0,
                        tokenizer=tokenizer,
                        generation_config={"do_sample": False},
                        verbose=False,
                    )
                    # Append to audio segments
                    if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
                        print("[Error] Model did not return any audio output after CPU fallback.")
                        return
                    audio_segments.append(outputs.speech_outputs[0].cpu())
                    # Skip the remainder of the loop (since we already appended)
                    continue
                except Exception as fallback_exc:
                    print(f"[Error] CPU fallback also failed: {fallback_exc}")
                    traceback.print_exc()
                    return
            # Not a CUDA architecture error: rethrow
            print(f"[Error] An error occurred during generation: {exc}")
            traceback.print_exc()
            return
        if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
            print("[Error] Model did not return any audio output for a segment.")
            return
        # Append the generated audio (move to CPU for concatenation)
        audio_segments.append(outputs.speech_outputs[0].cpu())

    # Concatenate all audio segments
    if not audio_segments:
        print("[Error] Nothing was generated. Check your input.")
        return
    try:
        final_audio = torch.cat(audio_segments)
    except Exception as exc:
        print(f"[Error] Failed to concatenate audio segments: {exc}")
        traceback.print_exc()
        return

    # Save the concatenated audio to the next available filename
    next_filename = determine_next_filename(audio_dir)
    output_path = os.path.join(audio_dir, next_filename)
    try:
        processor.save_audio(final_audio, output_path)
    except Exception as exc:
        print(f"[Error] Failed to save the generated audio: {exc}")
        traceback.print_exc()
        return

    # Report total generation time
    total_end = time.time()
    elapsed = total_end - total_start
    print(f"✅ Athena has spoken. Saved to {output_path}")
    print(f"Generation completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()