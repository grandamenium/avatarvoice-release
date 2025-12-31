"""Simplified single-page Gradio UI for AvatarVoice.

Key design principles:
1. Single page - no tabs
2. Preview audio = exact file sent to VibeVoice
3. Reactive updates - changing actor/emotion updates preview
4. Single source of truth for voice selection state
"""

import asyncio
import os
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv
load_dotenv()  # Load .env file

import gradio as gr

from voicematch import VoiceMatchAPI
from voicematch.prompt_optimizer import PromptOptimizer

from vibevoice_client import VibeVoiceClient, GenerationConfig, VoiceReference


# Global instances
_voicematch_api: Optional[VoiceMatchAPI] = None
_vibevoice_client: Optional[VibeVoiceClient] = None
_prompt_optimizer: Optional[PromptOptimizer] = None


def get_voicematch_api() -> VoiceMatchAPI:
    """Get or create VoiceMatch API instance."""
    global _voicematch_api
    if _voicematch_api is None:
        from voicematch.config import Config
        Config.reset()
        _voicematch_api = VoiceMatchAPI()
    return _voicematch_api


def get_vibevoice_client() -> VibeVoiceClient:
    """Get or create VibeVoice client instance."""
    global _vibevoice_client
    if _vibevoice_client is None:
        endpoint = os.getenv("VIBEVOICE_ENDPOINT", "http://localhost:7860")
        _vibevoice_client = VibeVoiceClient(endpoint=endpoint)
    return _vibevoice_client


def get_prompt_optimizer() -> PromptOptimizer:
    """Get or create PromptOptimizer instance."""
    global _prompt_optimizer
    if _prompt_optimizer is None:
        api_key = os.getenv("GEMINI_API_KEY")
        _prompt_optimizer = PromptOptimizer(api_key=api_key)
    return _prompt_optimizer


# CREMA-D emotion codes
EMOTION_CODES = ["NEU", "HAP", "SAD", "ANG", "FEA", "DIS"]
EMOTION_LABELS = {
    "NEU": "Neutral",
    "HAP": "Happy",
    "SAD": "Sad",
    "ANG": "Angry",
    "FEA": "Fear",
    "DIS": "Disgust",
}


def map_gemini_emotion_to_cremad(emotion: str) -> str:
    """Map Gemini emotion string to CREMA-D code."""
    mapping = {
        "neutral": "NEU",
        "happy": "HAP",
        "sad": "SAD",
        "angry": "ANG",
        "anger": "ANG",
        "fear": "FEA",
        "disgust": "DIS",
        "surprised": "NEU",  # No surprise in CREMA-D, fallback to neutral
    }
    return mapping.get(emotion.lower(), "NEU")


def analyze_and_match(image_path: str):
    """Analyze image and find matching voices.

    Returns:
        Tuple of (age, gender, race, emotion_detected, status_msg,
                 actor_dropdown_update, emotion_value)
    """
    if not image_path:
        return (
            "",
            "",
            "",
            "",
            "Please upload an image",
            gr.update(choices=[], value=None),
            "NEU",
        )

    try:
        api = get_voicematch_api()

        # Analyze image
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        analysis = loop.run_until_complete(api.analyze_image(Path(image_path)))
        loop.close()

        # Find top 5 matching actors
        matches = api.find_matches(analysis, limit=5)

        if not matches:
            return (
                str(analysis.estimated_age),
                str(analysis.gender),
                str(analysis.race),
                str(analysis.emotion),
                "No matching voices found",
                gr.update(choices=[], value=None),
                "NEU",
            )

        # Build actor choices for dropdown (label, value)
        actor_choices = []
        for m in matches:
            label = f"{m.actor.id} - {m.actor.sex}, {m.actor.race}, age {m.actor.age} ({m.score:.0%})"
            actor_choices.append((label, m.actor.id))

        # Best match
        best_actor = matches[0].actor.id

        # Map detected emotion to CREMA-D code
        emotion_code = map_gemini_emotion_to_cremad(str(analysis.emotion))

        status = f"Found {len(matches)} matching voices. Best: {best_actor}"

        return (
            str(analysis.estimated_age),
            str(analysis.gender),
            str(analysis.race),
            str(analysis.emotion),
            status,
            gr.update(choices=actor_choices, value=best_actor),
            emotion_code,
        )

    except Exception as e:
        return (
            "",
            "",
            "",
            "",
            f"Error: {str(e)}",
            gr.update(choices=[], value=None),
            "NEU",
        )


def update_voice_preview(actor_id: str, emotion: str) -> Tuple[Optional[str], str, str]:
    """Generate voice preview sample.

    This creates the EXACT audio file that will be sent to VibeVoice.

    Returns:
        Tuple of (audio_path_for_player, stored_path_for_generation, display_filename)
    """
    if not actor_id:
        return None, "", ""

    try:
        api = get_voicematch_api()

        # Generate sample at 24kHz for VibeVoice
        sample_path = api.get_voice_sample(
            actor_id=actor_id,
            emotion=emotion,
            duration_seconds=10.0,
        )

        if sample_path and sample_path.exists():
            path_str = str(sample_path)
            filename = sample_path.name
            return path_str, path_str, f"📁 {filename}"

        return None, "", "No sample found"

    except Exception as e:
        print(f"Error getting voice preview: {e}")
        return None, "", f"Error: {str(e)}"


def generate_speech(
    text: str,
    stored_audio_path: str,
    cfg_scale: float,
) -> Tuple[Optional[str], str]:
    """Generate speech using the EXACT same audio file as preview.

    Args:
        text: Text to synthesize
        stored_audio_path: The cached audio path from preview (24kHz)
        cfg_scale: Guidance scale (only param our endpoint accepts)

    Returns:
        Tuple of (audio_path, status_message)
    """
    if not text or not text.strip():
        return None, "Please enter text to synthesize"

    if not stored_audio_path:
        return None, "Please select a voice actor first"

    audio_path = Path(stored_audio_path)
    if not audio_path.exists():
        return None, f"Voice sample not found: {stored_audio_path}"

    # Show what we're sending
    audio_filename = audio_path.name
    params_info = f"Sending to VibeVoice: audio={audio_filename}, cfg_scale={cfg_scale}"
    print(params_info)

    try:
        client = get_vibevoice_client()

        # Create voice reference from the SAME file used for preview
        voice_ref = VoiceReference(audio_path=audio_path)

        config = GenerationConfig(
            text=text,
            voice_reference=voice_ref,
            cfg_scale=cfg_scale,
        )

        # Generate
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(client.generate(config))
        loop.close()

        # Save output
        if result.audio_bytes:
            output_dir = Path(os.getenv("OUTPUT_DIR", "./output"))
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "generated_speech.wav"
            output_path.write_bytes(result.audio_bytes)
            return str(output_path), f"✓ Generated with cfg_scale={cfg_scale}, voice={audio_filename}"

        return None, "No audio generated"

    except Exception as e:
        return None, f"Error: {str(e)}"


def optimize_text_for_tts(
    text: str,
    emotion: str,
) -> Tuple[str, str, gr.update, gr.update, str]:
    """Optimize text using Gemini and store original for undo.

    Returns:
        Tuple of (optimized_text, original_text, restore_btn_update,
                  status_update, status_message)
    """
    if not text or not text.strip():
        return (
            text,
            "",
            gr.update(visible=False),
            gr.update(visible=True),
            "Please enter text first",
        )

    try:
        optimizer = get_prompt_optimizer()

        # Use synchronous method to avoid event loop conflicts with Gradio
        optimized = optimizer.optimize_sync(text, emotion)

        # Check if text actually changed
        if optimized.strip() == text.strip():
            return (
                text,
                "",
                gr.update(visible=False),
                gr.update(visible=True),
                "Text already optimized - no changes needed",
            )

        return (
            optimized,           # New text for textbox
            text,                # Store original in state
            gr.update(visible=True),   # Show restore button
            gr.update(visible=True),   # Show status
            f"Optimized for {EMOTION_LABELS.get(emotion, emotion)} tone",
        )

    except Exception as e:
        return (
            text,
            "",
            gr.update(visible=False),
            gr.update(visible=True),
            f"Optimization failed: {str(e)}",
        )


def restore_original_text(original: str) -> Tuple[str, gr.update, gr.update, str]:
    """Restore original text and hide restore button.

    Returns:
        Tuple of (original_text, restore_btn_update, status_update, status_message)
    """
    if not original:
        return (
            "",
            gr.update(visible=False),
            gr.update(visible=True),
            "No original text to restore",
        )

    return (
        original,                    # Restore original text
        gr.update(visible=False),    # Hide restore button
        gr.update(visible=True),     # Show status
        "Original text restored",
    )


def create_app() -> gr.Blocks:
    """Create the single-page Gradio application."""

    with gr.Blocks(
        title="AvatarVoice - AI Voice Cloning",
    ) as app:

        # State to store the audio path for generation
        stored_audio_path = gr.State(value="")
        # State for undo functionality (stores original text before optimization)
        original_text = gr.State(value="")

        gr.Markdown("""
        # AvatarVoice - AI Voice Cloning

        Upload an avatar image → Auto-detect demographics & emotion → Generate speech with matched voice
        """)

        # === Section 1: Image Analysis ===
        with gr.Group():
            gr.Markdown("### 1. Upload Avatar Image")
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="Avatar Image",
                        type="filepath",
                        height=250,
                    )
                    analyze_btn = gr.Button("Analyze & Match", variant="primary", size="lg")

                with gr.Column(scale=1):
                    with gr.Row():
                        age_output = gr.Textbox(label="Age", interactive=False, scale=1)
                        gender_output = gr.Textbox(label="Gender", interactive=False, scale=1)
                    with gr.Row():
                        race_output = gr.Textbox(label="Race", interactive=False, scale=1)
                        emotion_detected = gr.Textbox(label="Emotion", interactive=False, scale=1)
                    analysis_status = gr.Textbox(label="Status", interactive=False)

        # === Section 2: Voice Selection ===
        with gr.Group():
            gr.Markdown("### 2. Voice Selection")
            gr.Markdown("*Adjust actor or emotion to update the preview. This exact audio will be used for generation.*")

            with gr.Row():
                actor_dropdown = gr.Dropdown(
                    label="Voice Actor",
                    choices=[],
                    interactive=True,
                    scale=2,
                )
                emotion_dropdown = gr.Dropdown(
                    label="Emotion",
                    choices=[(EMOTION_LABELS[e], e) for e in EMOTION_CODES],
                    value="NEU",
                    interactive=True,
                    scale=1,
                )

            voice_preview = gr.Audio(
                label="Voice Preview",
                type="filepath",
            )
            current_voice_file = gr.Textbox(
                label="Voice File (sent to VibeVoice)",
                interactive=False,
                placeholder="Select an actor to generate voice sample",
            )

        # === Section 3: Text & Generation ===
        with gr.Group():
            gr.Markdown("### 3. Generate Speech")

            tts_text = gr.Textbox(
                label="Text to Speak",
                placeholder="Enter text to convert to speech...",
                lines=3,
            )

            # Optimization buttons row
            with gr.Row():
                optimize_btn = gr.Button(
                    "Optimize for TTS",
                    variant="secondary",
                    scale=2,
                )
                restore_btn = gr.Button(
                    "Restore Original",
                    variant="secondary",
                    visible=False,  # Hidden until optimization happens
                    scale=1,
                )

            # Status for optimization
            optimize_status = gr.Textbox(
                label="Optimization Status",
                interactive=False,
                visible=False,  # Only shown after optimization
            )

            with gr.Accordion("Generation Parameters", open=True):
                gr.Markdown("*These are the exact parameters sent to VibeVoice.*")
                cfg_scale = gr.Slider(
                    label="CFG Scale",
                    minimum=1.0,
                    maximum=2.0,
                    value=2.0,
                    step=0.1,
                    info="Controls voice consistency (1.0 = creative, 2.0 = closest to reference)",
                )

            generate_btn = gr.Button("Generate Speech", variant="primary", size="lg")
            generation_status = gr.Textbox(label="Status", interactive=False)

        # === Section 4: Output ===
        with gr.Group():
            gr.Markdown("### 4. Generated Audio")
            generated_audio = gr.Audio(
                label="Generated Speech",
                type="filepath",
            )

        # === Event Handlers ===

        # Analyze button - runs analysis and auto-populates dropdowns
        analyze_btn.click(
            fn=analyze_and_match,
            inputs=[image_input],
            outputs=[
                age_output,
                gender_output,
                race_output,
                emotion_detected,
                analysis_status,
                actor_dropdown,  # gr.update with choices and value
                emotion_dropdown,  # Sets value
            ],
        ).then(
            # After analysis, auto-generate preview with best match
            fn=update_voice_preview,
            inputs=[actor_dropdown, emotion_dropdown],
            outputs=[voice_preview, stored_audio_path, current_voice_file],
        )

        # Actor change - update preview
        actor_dropdown.change(
            fn=update_voice_preview,
            inputs=[actor_dropdown, emotion_dropdown],
            outputs=[voice_preview, stored_audio_path, current_voice_file],
        )

        # Emotion change - update preview
        emotion_dropdown.change(
            fn=update_voice_preview,
            inputs=[actor_dropdown, emotion_dropdown],
            outputs=[voice_preview, stored_audio_path, current_voice_file],
        )

        # Generate button - uses stored audio path
        generate_btn.click(
            fn=generate_speech,
            inputs=[tts_text, stored_audio_path, cfg_scale],
            outputs=[generated_audio, generation_status],
        )

        # Optimize button - optimizes text and stores original
        optimize_btn.click(
            fn=optimize_text_for_tts,
            inputs=[tts_text, emotion_dropdown],
            outputs=[
                tts_text,        # Updated with optimized text
                original_text,   # Stores original for undo
                restore_btn,     # Shows restore button
                optimize_status, # Shows status area
                optimize_status, # Status message
            ],
        )

        # Restore button - reverts to original text
        restore_btn.click(
            fn=restore_original_text,
            inputs=[original_text],
            outputs=[
                tts_text,        # Restored original text
                restore_btn,     # Hides restore button
                optimize_status, # Shows status area
                optimize_status, # Status message
            ],
        )

        # Clear restore button when user manually edits text
        tts_text.change(
            fn=lambda: gr.update(visible=False),
            inputs=[],
            outputs=[restore_btn],
        )

    return app


def launch_app(
    server_name: str = "0.0.0.0",
    server_port: int = None,
    share: bool = False,
) -> None:
    """Launch the Gradio application."""
    import os
    port = int(os.getenv("PORT", server_port or 7861))

    app = create_app()
    app.launch(
        server_name=server_name,
        server_port=port,
        share=share,
    )


if __name__ == "__main__":
    launch_app()
