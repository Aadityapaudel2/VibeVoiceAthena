Athena Voice Generator using Microsoft VibeVoice
================================================

This package provides a simple, repeatable workflow to convert your own
scripts into natural‑sounding podcast audio using Microsoft's open‑source
VibeVoice‑1.5B model.  Everything runs locally after a one‑time setup,
so your text and generated audio never leave your machine.

Contents
--------

The archive you received contains the following files and folders:

* **`input.txt`** – a plain‑text file where you can write the
  conversation or narration you want to convert to speech.  For
  multi‑speaker conversations, prefix each line with `Speaker 1:`,
  `Speaker 2:`, and so on.  You can include up to four speakers.

* **`generate.py`** – a Python script that loads the VibeVoice
  model, reads `input.txt`, generates the corresponding audio and
  writes a numbered `.wav` file into the `audio/` folder.  The
  numbering prevents accidental overwrites; the first file will be
  `audio/audio000001.wav`, the second `audio/audio000002.wav`, etc.  When
  generation succeeds the script prints `✅ Athena has spoken.`.

* **`run.bat`** – a Windows batch script that activates the
  virtual environment and runs `generate.py`.  Double‑click this file
  whenever you want to generate a new piece of audio from your
  current `input.txt`.

* **`setup.bat`** – a one‑time setup script.  Running this will:
    1. Create a Python virtual environment in the local `venv/` folder.
    2. Install all Python packages listed in `requirements.txt`.
    3. Download the VibeVoice‑1.5B model snapshot from Hugging Face
       into the `model/` directory.  The download may take several
       minutes on a fast connection and requires approximately 7–8 GB
       of disk space.
    4. Install the downloaded model as a local Python package so that
       it can be imported by `generate.py`.

* **`requirements.txt`** – the list of Python packages required by
  the project.  These will be installed automatically by `setup.bat`.

* **`audio/`** – a directory where generated `.wav` files are
  stored.  You can safely delete old files or rename them if you
  like, but the numbering of new files will always follow the highest
  existing number.

Quick start
-----------

1. **Extract the archive** to your preferred working directory
   (for example, `D:\TikTok project\`).

2. **Run the setup script**:

   1. Open a command prompt (PowerShell or `cmd`).
   2. `cd` into the extracted directory.
   3. Execute `setup.bat`.

   The script creates a virtual environment, installs the required
   dependencies, clones the community fork of the VibeVoice code and
   installs it as a Python package, then downloads the model snapshot
   from Hugging Face.  This step only needs to be done once.  After
   the model and code are cached locally you can run offline.

   > **GPU installation:** The setup script installs the GPU‑enabled
   > version of PyTorch with CUDA 12.1.  Make sure your system has an
   > NVIDIA GPU and compatible drivers.  If you see messages about
   > running on the CPU despite having a GPU, double‑check the
   > installation logs.

3. **Write your script** in `input.txt`.  For example:

   ```text
   Speaker 1: Hello, Athena. Could you tell us about the latest advances in AI?
   Speaker 2: Of course! I'm excited to share what's new in the world of artificial intelligence.
   ```

4. **Generate audio** by double‑clicking `run.bat`.  This will
   automatically activate the virtual environment, run the Python
   script and place a new `.wav` file inside the `audio/` folder.

   Generation speed depends on your GPU and the length of the input
   script.  The script automatically detects whether a CUDA‑capable
   GPU is available and uses mixed‑precision and flash attention for
   faster inference when possible.  On systems without a suitable
   GPU the model falls back to CPU execution, which can be much
   slower.  If you see warnings about missing APEX/FusedRMSNorm or
   tokenizers, they are benign; the generation will continue using
   native implementations.

5. **Repeat** steps 3–4 whenever you wish to produce new audio.

Supported input syntax
----------------------

The VibeVoice model recognises up to four distinct speakers per script.  To
assign voices and control the narration, follow these conventions when
editing **`input.txt`**:

* **Speaker labels.** Prefix each turn with `Speaker n:` (where `n` is 1–4).  The
  package maps the first four speakers to example voices included with the
  community repository: `Alice` (female), `Carter` (male), `Frank` (male) and
  `Mary` (female)【875183015017572†L146-L150】.  Additional voices (such as
  `Maya_woman`, `Samuel_man`, `Anchen_man_bgm`, `Bowen_man` and
  `Xinran_woman`) can be used by editing `generate.py`’s `default_names` list.

* **Pause tags.** Include `[pause]` to insert a one‑second silence, or
  `[pause:ms]` to specify a custom silence in milliseconds (e.g.
  `[pause:2000]` for a 2 s pause).  Each pause splits the script into
  independent segments; the model cannot see across pause boundaries, so use
  pauses sparingly at natural sentence breaks.

* **Tone tags.** Add `[tone:STYLE]` before a sentence to hint at emotional
  delivery.  The tag is replaced internally with a descriptive phrase such as
  “in an excited tone,”.  Example styles include `excited`, `calm`, `sad`,
  `whisper`, `shout` and `curious`.  Unknown styles are inserted
  verbatim.  Tone tags may improve expressiveness but are not guaranteed to
  change the voice.

* **Formatting.** Ordinary punctuation like quotation marks, ellipses or
  dashes has no special meaning beyond affecting phrasing.  Line breaks
  separate sentences but do not reset the speaker.

Example `input.txt`
-------------------

```text
Speaker 1: [tone:calm] Welcome to our show.[pause] Speaker 2: [tone:excited]
Thank you for having me![pause:1500] Speaker 1: [tone:whisper] Let's share
some secrets…
```

In this example:

* `Speaker 1` speaks calmly, then a one‑second pause is inserted.
* `Speaker 2` replies excitedly.  A 1.5 s pause follows.
* `Speaker 1` continues in a whisper.  Ellipses create a natural trailing off.

You can mix and match speakers, tones and pauses to craft expressive,
podcast‑style dialogues.  Avoid overly long scripts (tens of thousands of
characters) as they may exhaust GPU memory.

Additional notes
----------------

* The first run of `generate.py` will take longer as the model
  weights are loaded into memory.  Subsequent runs are faster.

* If you wish to reset the numbering of your audio files, simply
  delete or move the existing `.wav` files in the `audio/` folder.  The
  script will always pick the next available number when naming new
  outputs.

* All processing happens locally.  After the initial model download
  completes, you can disconnect from the internet and continue
  generating audio.

* Should you encounter problems during setup or generation,
  consult the console output for hints.  Common issues include
  insufficient disk space for the model download or missing GPU
  drivers.  The scripts attempt to fall back to CPU if a suitable
  GPU is not available.
