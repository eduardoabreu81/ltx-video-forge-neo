# 🎬 LTX Video for Forge Neo

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![Version](https://img.shields.io/badge/Version-0.1.0-brightgreen)](https://github.com/eduardoabreu81/ltx-video-forge-neo)
![Status](https://img.shields.io/badge/Status-Work_in_progress-orange)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green)](LICENSE)

> **Extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**

</div>

Generate **videos with sound** with **[LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5)**, **LTX-2.3** and **LTX-2** by Lightricks inside Forge Neo, using the checkpoint, preset and Generate button you already know - describe the scene and its sounds, or start from a picture, and get an MP4 with picture and audio made together.

<div align="center">

[![A cyclist in a yellow raincoat rides down a rainy Tokyo street at night, past a ramen stall](https://raw.githubusercontent.com/wiki/eduardoabreu81/ltx-video-forge-neo/media/tokyo-two-stages.gif)](https://raw.githubusercontent.com/wiki/eduardoabreu81/ltx-video-forge-neo/media/tokyo-two-stages.mp4)

*A rainy night in Tokyo* - LTX-2.5, 5 seconds at 1280×704 with **two stages**, made by this extension in about 30 seconds on a 48 GB card. Click for the MP4 with sound.

</div>

> [!Important]
> **Work in progress - first preview.** This is an early version: it works, but it has only been tested on one card (a 48 GB NVIDIA A40 with plenty of system RAM), and several features of the official LTX workflows are not here yet - see [What is not here yet](#-what-is-not-here-yet). Expect rough edges, and please [report what you find](https://github.com/eduardoabreu81/ltx-video-forge-neo/issues).
>
> This extension requires an up-to-date **Forge Neo** (the `neo` branch, version 2.30 or later, with comfy-kitchen 0.2.37). On an older version it stays disabled and tells you so in the console - Forge Neo itself keeps working as usual.

---

## 📋 Table of Contents

- [What's New](#-whats-new)
- [Examples](#-examples)
- [Features](#-features)
- [What is not here yet](#-what-is-not-here-yet)
- [Installation](#-installation)
- [Recommended Settings](#%EF%B8%8F-recommended-settings)
- [Tips](#-tips)
- [Credits](#-credits)

---

## 🆕 What's New

### v0.1.0 - First Public Preview

- **LTX-2.5, LTX-2.3 and LTX-2** - text-to-video with sound, on Forge Neo's own loader, samplers and memory management
- **First and last frame** - start from a picture, end on another, or both
- **Two stages** - generate at half the size, upscale x2 and refine, as the official LTX workflows do: sharper and faster than one stage at full size
- **Dev checkpoints and LoRAs** - the dev models with real CFG, and the distilled LoRA for 8 steps

---

## 🎥 Examples

Made inside Forge Neo, with sound. The previews are silent - **click one to download the MP4 with sound**.

| Arctic hunter (official example prompt) | Barista, written prompt | Beach |
| :---: | :---: | :---: |
| [![A hunter on an ice floe, a polar bear far away and the LTX-2.5 title over the sea](https://raw.githubusercontent.com/wiki/eduardoabreu81/ltx-video-forge-neo/media/arctic-template.gif)](https://raw.githubusercontent.com/wiki/eduardoabreu81/ltx-video-forge-neo/media/arctic-template.mp4) | [![A barista draws a leaf in a latte and slides the cup toward the camera](https://raw.githubusercontent.com/wiki/eduardoabreu81/ltx-video-forge-neo/media/cafe-long-prompt.gif)](https://raw.githubusercontent.com/wiki/eduardoabreu81/ltx-video-forge-neo/media/cafe-long-prompt.mp4) | [![A golden retriever runs along a beach and stops in front of the camera](https://raw.githubusercontent.com/wiki/eduardoabreu81/ltx-video-forge-neo/media/dog-beach.gif)](https://raw.githubusercontent.com/wiki/eduardoabreu81/ltx-video-forge-neo/media/dog-beach.mp4) |
| LTX-2.5, two stages | LTX-2.5, two stages | LTX-2.5, two stages |

| From a picture | First and last frame | LTX-2.3 |
| :---: | :---: | :---: |
| [![A red car drives off from a farmhouse](https://raw.githubusercontent.com/wiki/eduardoabreu81/ltx-video-forge-neo/media/car-first-frame.gif)](https://raw.githubusercontent.com/wiki/eduardoabreu81/ltx-video-forge-neo/media/car-first-frame.mp4) | [![A red car leaves a farmhouse and ends on a country road](https://raw.githubusercontent.com/wiki/eduardoabreu81/ltx-video-forge-neo/media/car-first-last.gif)](https://raw.githubusercontent.com/wiki/eduardoabreu81/ltx-video-forge-neo/media/car-first-last.mp4) | [![The rainy Tokyo street made with LTX-2.3](https://raw.githubusercontent.com/wiki/eduardoabreu81/ltx-video-forge-neo/media/tokyo-ltx23.gif)](https://raw.githubusercontent.com/wiki/eduardoabreu81/ltx-video-forge-neo/media/tokyo-ltx23.mp4) |
| img2img, first frame | img2img + ImageStitch | Same prompt as the top clip |

Every prompt and the exact settings are in the **[Examples](https://github.com/eduardoabreu81/ltx-video-forge-neo/wiki/Examples)** page of the wiki.

---

## 🎯 Features

> [!Tip]
> Step-by-step guides and every setting explained are in the **[wiki](https://github.com/eduardoabreu81/ltx-video-forge-neo/wiki)**.

### 🔊 Video and Sound Together

- Describe what happens and what it sounds like in one prompt - footsteps, rain, engines, music, a line of dialogue
- Picture and sound are generated together and saved as one MP4, shown in Forge's usual result area
- **FPS** setting (24 by default), and **Frames** for the length - 121 frames are 5 seconds
- **Include generated audio** checkbox for a silent video
- **Still image** output for a single picture from the model

### 🖼️ First and Last Frame

- **img2img** - your input image becomes the first frame of the video
- **Last frame** - turn on **ImageStitch Integrated**, which Forge Neo already has, and add one image to its gallery
- Use both in img2img for a video that goes from one picture to the other; in txt2img the gallery image is the ending
- **Frame strength** and **Image compression** follow the official recipe of each LTX version

### 🔍 Two Stages

- One checkbox: the video is made at half the size, enlarged x2 by the LTX latent upscaler and refined in 3 more steps
- Sharper and richer than one stage at full size, and faster - about 30 seconds for 5 seconds at 1280×704 on an A40
- Works with text-to-video, first frame, last frame, both, and still images, on all three LTX versions
- **Automatic** picks the right upscaler for your checkpoint

### 🧩 Three LTX Versions

- **LTX-2.5** - the newest: separate files for the model, the Gemma 4 text encoder and the two VAEs
- **LTX-2.3** and **LTX-2** - single checkpoints with the VAEs inside, plus the Gemma 3 text encoder
- **Distilled** checkpoints (8 steps, CFG 1) and **dev** checkpoints (20 steps, real CFG), detected by file name
- The **distilled LoRA** turns a dev checkpoint into an 8-step one; other LoRAs use the usual `<lora:name:weight>` syntax

### 🎛️ Familiar Forge Controls

- Runs inside txt2img and img2img - no separate tab, no extra program
- Works with the Forge Neo model folders and the **VAE / Text Encoder** selector you already use
- **Frames** takes the place of Batch Size and shows the length of the video
- The **ltx** UI preset and the **LTX Distilled** schedule set the official sampling recipe
- A small **LTX Video** panel holds the video options

### 🧠 Forge Memory Management

- The model stays loaded between generations, like any Forge checkpoint
- Weights that leave the graphics card point back to their file on disk instead of being copied into system RAM
- **Never OOM Integrated** works with LTX, for the model and for the video decode
- Switching from LTX to another model releases it first, so system RAM does not run out
- Community files in INT8 and W4A8 formats load as they are

### 🛡️ Safe by Design

- Changes no Forge Neo file - remove the extension and everything is as it was
- Checks your Forge Neo at startup and stays disabled if something it needs is missing
- Checks every model file before loading it, and explains clearly when something is not supported
- Other models are not affected

---

## 🚧 What is not here yet

These are part of the official LTX workflows and are planned, but **not in this version**:

- **Prompt enhancer** - the official workflow can rewrite a short prompt into the long description LTX was trained on. Until it is here, write the long description yourself (see [Tips](#-tips))
- **Diffusion video decoder** (`ltx-2.5-video-vae-bf16`) - use the convolutional VAE (`ltx-2.5-video-vae-conv-bf16`) for now
- **GGUF checkpoints** and **NVFP4** files
- **Reference audio and pictures** (ID-LoRA, IC-LoRAs), **pose, depth and edge control**, audio-to-video, extend and retake
- **Smaller cards** - only a 48 GB card has been tested; 24 GB and 16 GB cards may work with Never OOM, but this is not checked yet

---

## 📦 Installation

1. Open Forge Neo WebUI
2. Go to **Extensions** → **Install from URL**
3. Paste: `https://github.com/eduardoabreu81/ltx-video-forge-neo`
4. Click **Install** and restart the WebUI
5. Place the files of the LTX version you want in the usual folders. The Lightricks files need you to accept the license on their Hugging Face page first.

**LTX-2.5** (the newest):

| Part | File | Folder |
| :--- | :--- | :--- |
| Model | [ltx-2.5-22b-distilled-transformer-comfy-int8-convrot](https://huggingface.co/Lightricks/LTX-2.5/blob/main/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors) | `models/Stable-diffusion` |
| Text encoder | [gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot](https://huggingface.co/Lightricks/LTX-2.5/blob/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors) | `models/text_encoder` |
| Video VAE | [ltx-2.5-video-vae-conv-bf16](https://huggingface.co/Lightricks/LTX-2.5/blob/main/vae/ltx-2.5-video-vae-conv-bf16.safetensors) | `models/VAE` |
| Audio VAE | [ltx-2.5-audio-vae-bf16](https://huggingface.co/Lightricks/LTX-2.5/blob/main/vae/ltx-2.5-audio-vae-bf16.safetensors) | `models/VAE` |
| Upscaler, for two stages | [ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0](https://huggingface.co/Lightricks/LTX-2.5/blob/main/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors) | `models/latent_upscale_models` |

**LTX-2.3** and **LTX-2** (one checkpoint with the VAEs inside):

| Part | File | Folder |
| :--- | :--- | :--- |
| LTX-2.3 model | [ltx-2.3-22b-distilled-1.1](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-distilled-1.1.safetensors) | `models/Stable-diffusion` |
| LTX-2 model | [ltx-2-19b-distilled-fp8](https://huggingface.co/Lightricks/LTX-2/blob/main/ltx-2-19b-distilled-fp8.safetensors) | `models/Stable-diffusion` |
| Text encoder, for both | [gemma_3_12B_it_fp8_scaled](https://huggingface.co/Comfy-Org/ltx-2/blob/main/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors) | `models/text_encoder` |
| LTX-2.3 upscaler | [ltx-2.3-spatial-upscaler-x2-1.1](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors) | `models/latent_upscale_models` |
| LTX-2 upscaler | [ltx-2-spatial-upscaler-x2-1.0](https://huggingface.co/Lightricks/LTX-2/blob/main/ltx-2-spatial-upscaler-x2-1.0.safetensors) | `models/latent_upscale_models` |

Create the `models/latent_upscale_models` folder if it is not there. Dev checkpoints, the distilled LoRAs and community files are listed in the **[Models](https://github.com/eduardoabreu81/ltx-video-forge-neo/wiki/Models)** page.

6. Make sure **FFmpeg** is installed (or set its path in **Settings** → **LTX Video**)
7. Pick the **ltx** UI preset and the checkpoint, and under **VAE / Text Encoder** select the text encoder - plus, for LTX-2.5, **both** VAEs

> [!Note]
> LTX is a large model and Forge keeps its files in system RAM. With the LTX-2.5 files above, Forge used about 41-45 GiB of system RAM and about 44 GiB of VRAM on a 48 GB A40 (the models stay loaded on a large card). Plan for at least **64 GB of system RAM**.

---

## ⚙️ Recommended Settings

| Checkpoint | Sampler | Schedule type | Steps | CFG |
| :--- | :--- | :--- | :--- | :--- |
| **Distilled** (the ltx preset) | Euler a | LTX Distilled | 8 | 1 |
| **Dev** | Euler a | LTX | 20 | 4 |
| **Dev + distilled LoRA** | Euler a | LTX Distilled | 8 | 1 |

The extension sets these by itself when you select an LTX checkpoint.

**Width and Height** must be multiples of 32 - or of **64 with two stages**. A good start is **1280×704** (landscape) or **704×1280** (portrait) with two stages.

**Frames** sets the length and follows the pattern 8 × n + 1:

| Frames | Length at 24 FPS |
| :--- | :--- |
| 49 | about 2 seconds |
| 97 | about 4 seconds |
| 121 | about 5 seconds |
| 241 | about 10 seconds |

**FPS** - 24 for LTX-2.5 and LTX-2, 25 for LTX-2.3, as in the official workflows.

---

## 💡 Tips

- **Write long prompts.** LTX was trained on detailed descriptions of 150 to 200 words, in one paragraph, in the order things happen: who is there and what they look like, the shot (close-up, wide shot), how the camera moves, every action one after another, and the sounds. A short prompt leaves the model to guess - see the before and after on the wiki's [Writing Prompts](https://github.com/eduardoabreu81/ltx-video-forge-neo/wiki/Writing-Prompts) page
- Describe the sounds you want - rain, an engine, music, a line of dialogue in quotes
- At CFG 1 the negative prompt is not used - say what you do not want in the prompt itself
- From a picture, describe what happens next, not what the picture already shows
- Use **two stages** for the final video, and one stage at a small size to try ideas quickly
- Start with a short clip to check your setup, then go longer
- Keep the same seed when comparing settings - and remember that the same seed gives a different video in Forge and in ComfyUI
- Keep the MP4 and the JSON file saved next to it - it records the settings
- The LTX models have their own [license](https://huggingface.co/Lightricks/LTX-2.5): free for commercial use below US$ 10 million of yearly revenue - check it before using the videos

---

## 📄 Credits

- **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by Haoming02
- **[LTX-2.5, LTX-2.3 and LTX-2](https://huggingface.co/Lightricks)** by Lightricks
- **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)** - reference implementation of the model
- **[Comfy-Org](https://huggingface.co/Comfy-Org)** - the Gemma 3 text encoders and the official workflows
- **[Google](https://huggingface.co/google)** - the Gemma text encoders

---

## 📜 License

AGPL-3.0 - see [LICENSE](LICENSE). The LTX models have their own license.

---

<div align="center">

Made with ❤️ for the Stable Diffusion community

**[Report Bug](https://github.com/eduardoabreu81/ltx-video-forge-neo/issues)** • **[Request Feature](https://github.com/eduardoabreu81/ltx-video-forge-neo/issues)** • **[☕ Ko-fi](https://ko-fi.com/eduardoabreu81)**

</div>
