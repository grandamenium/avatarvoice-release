"""Gradio web interface for VoiceMatch pipeline."""

import gradio as gr
from pathlib import Path
from typing import Optional, Tuple

from .config import Config
from .gemini_analyzer import GeminiAnalyzer, AvatarAnalysis
from .database import VoiceDatabase
from .voice_matcher import VoiceMatcher, MatchResult
from .audio_processor import AudioProcessor


# Map UI emotion labels to CREMA-D emotion codes
EMOTION_TO_CODE = {
    "Auto-detect": None,
    "Anger": "ANG",
    "Disgust": "DIS",
    "Fear": "FEA",
    "Happy": "HAP",
    "Neutral": "NEU",
    "Sad": "SAD",
}

# Map Gemini emotion names to CREMA-D codes
GEMINI_EMOTION_TO_CODE = {
    "anger": "ANG",
    "disgust": "DIS",
    "fear": "FEA",
    "happy": "HAP",
    "neutral": "NEU",
    "sad": "SAD",
    "ambiguous": "NEU",  # Default to neutral for ambiguous
}


class VoiceMatchApp:
    """Gradio application for voice matching pipeline."""

    # Supported file extensions
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi"}

    # Available emotion options for the UI dropdown
    EMOTION_OPTIONS = [
        "Auto-detect",
        "Anger",
        "Disgust",
        "Fear",
        "Happy",
        "Neutral",
        "Sad",
    ]

    def __init__(self):
        """Initialize the VoiceMatch application."""
        self.config: Optional[Config] = None
        self.analyzer: Optional[GeminiAnalyzer] = None
        self.database: Optional[VoiceDatabase] = None
        self.matcher: Optional[VoiceMatcher] = None
        self.processor: Optional[AudioProcessor] = None
        self._initialized = False

    def _initialize(self) -> None:
        """Initialize all components.

        Lazy initialization to avoid loading resources until needed.
        """
        if self._initialized:
            return

        self.config = Config.load()

        # Initialize Gemini analyzer
        self.analyzer = GeminiAnalyzer(self.config.gemini_api_key)

        # Initialize database connection
        self.database = VoiceDatabase(self.config.database_path)
        self.database.connect()

        # Initialize voice matcher
        self.matcher = VoiceMatcher(self.database)

        # Initialize audio processor
        self.processor = AudioProcessor()

        self._initialized = True

    def _is_image(self, file_path: str) -> bool:
        """Check if file is an image based on extension."""
        return Path(file_path).suffix.lower() in self.IMAGE_EXTENSIONS

    def _is_video(self, file_path: str) -> bool:
        """Check if file is a video based on extension."""
        return Path(file_path).suffix.lower() in self.VIDEO_EXTENSIONS

    def _format_analysis(self, analysis: AvatarAnalysis) -> str:
        """Format analysis results for display.

        Args:
            analysis: AvatarAnalysis results from Gemini.

        Returns:
            Formatted string for UI display.
        """
        return f"""Avatar Analysis Results:
- Estimated Age: {analysis.estimated_age} (range: {analysis.age_range[0]}-{analysis.age_range[1]})
- Gender: {analysis.gender}
- Race: {analysis.race}
- Ethnicity: {analysis.ethnicity or "Not specified"}
- Detected Emotion: {analysis.emotion}
- Confidence: {analysis.confidence:.0%}"""

    def _format_match(self, match: Optional[MatchResult]) -> str:
        """Format match results for display.

        Args:
            match: MatchResult from voice matcher, or None if no match.

        Returns:
            Formatted string for UI display.
        """
        if match is None:
            return "No matching voice found. Please try a different image/video."

        return f"""Best Match: Actor {match.actor.id}
- Age: {match.actor.age}
- Gender: {match.actor.sex}
- Race: {match.actor.race}
- Ethnicity: {match.actor.ethnicity}
- Match Score: {match.score:.0%}

Scoring Details:
- Gender Score: {match.match_details.get('gender_score', 0):.0%}
- Race Score: {match.match_details.get('race_score', 0):.0%}
- Age Score: {match.match_details.get('age_score', 0):.0%}"""

    def process_upload(
        self,
        file_path: str,
        text_input: str,
        emotion_selection: str = "Auto-detect",
    ) -> Tuple[str, str, Optional[str]]:
        """Process uploaded file and return results.

        Args:
            file_path: Path to uploaded file.
            text_input: Optional text input (for future use).
            emotion_selection: User-selected emotion or "Auto-detect".

        Returns:
            Tuple of (analysis_text, match_text, audio_path)
        """
        # Initialize components if not done
        try:
            self._initialize()
        except Exception as e:
            return f"Initialization Error: {str(e)}", "", None

        # Validate file exists
        if not file_path or not Path(file_path).exists():
            return "Error: File not found. Please upload a valid image or video.", "", None

        file_path_obj = Path(file_path)

        # Analyze the uploaded file
        try:
            if self._is_image(file_path):
                analysis = self.analyzer.analyze_image(file_path_obj)  # type: ignore[union-attr]
            elif self._is_video(file_path):
                analysis = self.analyzer.analyze_video(file_path_obj)  # type: ignore[union-attr]
            else:
                return (
                    f"Error: Unsupported file type '{file_path_obj.suffix}'. "
                    "Please upload a JPG, PNG, MP4, or WebM file.",
                    "",
                    None,
                )
        except Exception as e:
            return f"Analysis Error: {str(e)}", "", None

        # Format analysis results
        analysis_text = self._format_analysis(analysis)

        # Determine which emotion code to use for clip selection
        if emotion_selection == "Auto-detect":
            # Use Gemini's detected emotion
            emotion_code: str = GEMINI_EMOTION_TO_CODE.get(analysis.emotion, "NEU")
        else:
            # Use user's selection
            emotion_code = EMOTION_TO_CODE.get(emotion_selection) or "NEU"

        # Find matching voice actors
        try:
            matches = self.matcher.find_matches(analysis, limit=5)  # type: ignore[union-attr]
        except Exception as e:
            return analysis_text, f"Matching Error: {str(e)}", None

        if not matches:
            return analysis_text, self._format_match(None), None

        best_match = matches[0]
        match_text = self._format_match(best_match)

        # Get audio clips for the matched actor with the selected emotion
        try:
            clips = self.database.get_clips_for_actor(best_match.actor.id, emotion=emotion_code)  # type: ignore[union-attr]

            # If no clips with selected emotion, fallback to any clips
            if not clips:
                clips = self.database.get_clips_for_actor(best_match.actor.id)  # type: ignore[union-attr]

            if not clips:
                return (
                    analysis_text,
                    match_text + "\n\nNote: No audio clips available for this actor.",
                    None,
                )

            # Get clip paths
            clip_paths = [Path(clip.filepath) for clip in clips]

            # Filter to existing files
            existing_clips = [p for p in clip_paths if p.exists()]

            if not existing_clips:
                return (
                    analysis_text,
                    match_text + "\n\nNote: Audio files not found. Please download the CREMA-D dataset.",
                    None,
                )

            # Generate output path
            output_path = self.config.output_dir / f"voice_match_{best_match.actor.id}.wav"  # type: ignore[union-attr]

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Concatenate clips
            audio_path = self.processor.concatenate_clips(existing_clips, output_path)  # type: ignore[union-attr]

            if audio_path is None:
                return analysis_text, match_text + "\n\nError: Failed to generate audio.", None

            return analysis_text, match_text, str(audio_path)

        except Exception as e:
            return analysis_text, match_text + f"\n\nAudio Error: {str(e)}", None

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface.

        Returns:
            Gradio Blocks interface.
        """
        with gr.Blocks(title="VoiceMatch Pipeline") as app:
            gr.Markdown("# VoiceMatch Pipeline")
            gr.Markdown(
                "Upload an image or video of an AI avatar to find a matching voice. "
                "The system will analyze the avatar's demographics and match them to "
                "voice actors from the CREMA-D dataset."
            )

            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        label="Upload Image or Video",
                        file_types=["image", "video"],
                    )
                    emotion_input = gr.Dropdown(
                        label="Emotion",
                        choices=self.EMOTION_OPTIONS,
                        value="Auto-detect",
                        info="Select emotion for voice clips, or let the system auto-detect from the image.",
                    )
                    text_input = gr.Textbox(
                        label="Text to Speak (for future use)",
                        placeholder="Enter text here...",
                        lines=3,
                    )
                    process_btn = gr.Button("Find Matching Voice", variant="primary")

                with gr.Column():
                    analysis_output = gr.Textbox(
                        label="Avatar Analysis",
                        lines=8,
                        interactive=False,
                    )
                    match_output = gr.Textbox(
                        label="Voice Match",
                        lines=10,
                        interactive=False,
                    )
                    audio_output = gr.Audio(
                        label="Matched Voice Sample",
                        type="filepath",
                    )

            # Connect the process button
            process_btn.click(
                fn=self.process_upload,
                inputs=[file_input, text_input, emotion_input],
                outputs=[analysis_output, match_output, audio_output],
            )

            # Add footer
            gr.Markdown(
                "---\n"
                "*Note: This tool uses the CREMA-D dataset for voice matching. "
                "The text input field is reserved for future text-to-speech functionality.*"
            )

        return app

    def launch(self, **kwargs) -> None:
        """Launch the Gradio app.

        Args:
            **kwargs: Additional arguments to pass to gr.Blocks.launch()
        """
        app = self.create_interface()
        app.launch(**kwargs)


def main():
    """Entry point for the application."""
    app = VoiceMatchApp()
    app.launch(share=False)


if __name__ == "__main__":
    main()
