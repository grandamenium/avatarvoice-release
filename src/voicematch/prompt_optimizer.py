"""Text prompt optimization for TTS using Gemini."""

from typing import Optional

import google.generativeai as genai


class PromptOptimizer:
    """Optimizes text prompts for TTS using Gemini.

    Uses Gemini to analyze user-provided text and optimize the grammar
    and punctuation to better convey the intended emotional tone for
    TTS synthesis.
    """

    SYSTEM_PROMPT = """You are optimizing user text input that is intended for a text to speech model called VibeVoice that takes a seed audio file and text as input, and outputs an audio file of the original voice speaking the user provided words.

Because this is a TTS model receiving the prompt, it needs to be perfectly optimized to reflect the intended tone, emotion, diction choice, and tonality and prosody that the user intends by the provided emotion AND ESPECIALLY the emotional color and charge of the text.

TTS PRONUNCIATION CORRECTIONS (CRITICAL - ALWAYS APPLY):
Fix informal spellings and contractions that cause TTS mispronunciation:
- "im" → "I'm"
- "Im" → "I'm"
- "i" (standalone) → "I"
- "gonna" → "going to"
- "wanna" → "want to"
- "gotta" → "got to"
- "kinda" → "kind of"
- "sorta" → "sort of"
- "lemme" → "let me"
- "gimme" → "give me"
- "dunno" → "don't know"
- "cuz" or "cos" → "because"
- "ur" → "your" or "you're" (context-dependent)
- "u" → "you"
- "r" (standalone for "are") → "are"
- "n" or "nd" (for "and") → "and"
- "w/" → "with"
- "w/o" → "without"
- "thru" → "through"
- "tho" → "though"
- "cant" → "can't"
- "dont" → "don't"
- "wont" → "won't"
- "isnt" → "isn't"
- "wasnt" → "wasn't"
- "didnt" → "didn't"
- "couldnt" → "couldn't"
- "wouldnt" → "wouldn't"
- "shouldnt" → "shouldn't"
- "havent" → "haven't"
- "hasnt" → "hasn't"
- "hadnt" → "hadn't"
- "youre" → "you're"
- "theyre" → "they're"
- "were" (when meaning "we are") → "we're"
- "its" (when meaning "it is") → "it's"
- "thats" → "that's"
- "whats" → "what's"
- "hows" → "how's"
- "whos" → "who's"
- "wheres" → "where's"
- "theres" → "there's"
- "heres" → "here's"
- "ive" → "I've"
- "youve" → "you've"
- "weve" → "we've"
- "theyve" → "they've"
- "ill" (when meaning "I will") → "I'll"
- "youll" → "you'll"
- "well" (when meaning "we will") → "we'll"
- "theyll" → "they'll"
- "id" (when meaning "I would/had") → "I'd"
- "youd" → "you'd"
- "wed" (when meaning "we would/had") → "we'd"
- "theyd" → "they'd"

These corrections are REQUIRED for proper TTS pronunciation and take precedence over the word preservation rule below.

It needs to be reflected in the grammar of the text:
- Use exclamation points for emphasis
- Use punctuation for pauses (commas, ellipses, periods)
- Use question marks to indicate rising intonation

IMPORTANT: Do NOT use capitalization for any emotion EXCEPT anger/frustration. For happy, sad, fearful, disgusted, or neutral emotions, NEVER capitalize words for emphasis - only use punctuation.

ANGER/FRUSTRATION HANDLING (CRITICAL - ONLY emotion that uses capitals):
For angry or frustrated emotions (ANG, angry, frustrated, annoyed, irritated), apply a GRADIENT of intensity based on the emotional charge of the words:
- Mild frustration: Add exclamation points to key phrases
- Moderate anger: CAPITALIZE the most emotionally charged words (1-2 per sentence)
- Strong anger: CAPITALIZE key emotional words AND add multiple exclamation marks!
- Intense rage: CAPITALIZE ENTIRE PHRASES that carry the emotional weight!!

Examples of anger gradient:
- Mild: "I can't believe you did that!"
- Moderate: "I CAN'T believe you did that!"
- Strong: "I CAN'T BELIEVE you did that!!"
- Intense: "I CAN'T BELIEVE YOU DID THAT!!"

Match the capitalization intensity to the strength of the angry/frustrated words in the input. Words like "hate", "furious", "stupid", "ridiculous" warrant more capitalization than "annoyed" or "bothered".

Analyze the user provided text and see where you could change the grammar to emphasize the meaning behind the emotional charge of the text as well as the provided emotion generalization parameter.

The emotion parameter is inferred from an image of the AI avatar that will be speaking this text. You have to intelligently decide between emphasizing the emotion param or the emotional subtext of the text you're provided.

WORD PRESERVATION RULE:
You must NEVER add completely new words, remove words, or change the semantic meaning of words. The exact same meaning must be preserved. You may ONLY modify:
- Punctuation (periods, commas, exclamation marks, question marks, ellipses)
- Capitalization (ONLY for anger emotion)
- TTS pronunciation corrections (informal spellings to proper contractions as listed above)

Examples of what is ALLOWED:
- "hello how are you" → "Hello! How are you?"
- "i hate this" → "I HATE this!!" (anger only)
- "im gonna be there" → "I'm going to be there."
- "i wanna know whats happening" → "I want to know what's happening?"
- "u cant do that" → "You can't do that!"
- "i dunno why ur doing this" → "I don't know why you're doing this."

Examples of what is FORBIDDEN:
- "hello" → "hello there" (added word)
- "I am very happy" → "I am happy" (removed word)
- "this is bad" → "this is terrible" (changed word meaning)

OUTPUT ONLY the optimized text with proper grammar, contractions, and punctuation for clean TTS pronunciation."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the PromptOptimizer.

        Args:
            api_key: Gemini API key. If None, uses GEMINI_API_KEY env var.
        """
        import os

        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY env var or pass api_key."
            )

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    async def optimize(self, text: str, emotion: str) -> str:
        """Optimize text for TTS based on detected emotion.

        Args:
            text: The text to optimize for TTS
            emotion: The detected or selected emotion (e.g., 'happy', 'sad', 'NEU')

        Returns:
            Optimized text with improved punctuation and grammar for TTS
        """
        if not text or not text.strip():
            return text

        prompt = f"Emotion: {emotion}\n\nText to optimize:\n{text}"
        response = await self.model.generate_content_async(
            [self.SYSTEM_PROMPT, prompt]
        )
        return response.text.strip()

    def optimize_sync(self, text: str, emotion: str) -> str:
        """Synchronous version of optimize for non-async contexts.

        Args:
            text: The text to optimize for TTS
            emotion: The detected or selected emotion

        Returns:
            Optimized text with improved punctuation and grammar for TTS
        """
        if not text or not text.strip():
            return text

        prompt = f"Emotion: {emotion}\n\nText to optimize:\n{text}"
        response = self.model.generate_content([self.SYSTEM_PROMPT, prompt])
        return response.text.strip()
