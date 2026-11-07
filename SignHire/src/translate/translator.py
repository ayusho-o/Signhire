"""
src/translate/translator.py
----------------------------
Pluggable translator interface for SignHire.

Design
------
BaseTranslator       — abstract interface every translator must implement
NoOpTranslator       — default: returns English text unchanged.
                       Used when provider = "none" in settings.yaml.
LibreTranslateTranslator — calls a LibreTranslate HTTP endpoint.
                       Used when provider = "libretranslate" and a URL
                       is configured.  API key is read from the environment
                       variable LIBRETRANSLATE_API_KEY, never hardcoded.

How to add a new provider
--------------------------
1. Subclass BaseTranslator and implement translate().
2. Add a branch in get_translator() below.
3. Set provider: <your_key> in config/settings.yaml.

Usage
-----
    from src.translate.translator import get_translator
    translator = get_translator(settings_cfg["translation"])
    text = translator.translate("hello", target_lang="hi")
"""

import os
from abc import ABC, abstractmethod


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class BaseTranslator(ABC):
    """Every translator must implement this single method."""

    @abstractmethod
    def translate(self, text: str, target_lang: str) -> str:
        """
        Translate text into target_lang.

        Parameters
        ----------
        text        : source text (always English in SignHire)
        target_lang : BCP-47 language code, e.g. "hi", "en"

        Returns
        -------
        Translated string, or original text with a note on failure.
        """

    def is_configured(self) -> bool:
        """Return True if this translator can actually translate."""
        return False


# ─────────────────────────────────────────────────────────────────────────────
# No-op (default)
# ─────────────────────────────────────────────────────────────────────────────

class NoOpTranslator(BaseTranslator):
    """
    Default translator — returns English text unchanged.

    Used when provider = "none" in settings.yaml, or when no provider
    has been configured.  Hindi and other non-English targets show a
    clear "not configured" message rather than silently returning wrong text.
    """

    def translate(self, text: str, target_lang: str) -> str:
        if target_lang == "en" or not target_lang:
            return text
        # Non-English requested but no provider configured
        return f"{text}  [Translation not configured — set a provider in config/settings.yaml]"

    def is_configured(self) -> bool:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# LibreTranslate
# ─────────────────────────────────────────────────────────────────────────────

class LibreTranslateTranslator(BaseTranslator):
    """
    Calls a LibreTranslate HTTP endpoint.

    Configuration (in settings.yaml):
        translation:
          provider: libretranslate
          libretranslate_url: https://libretranslate.com
          libretranslate_api_key: ""   # leave blank; set LIBRETRANSLATE_API_KEY env var

    The API key is read from the environment variable
    LIBRETRANSLATE_API_KEY so it is never stored in the config file.

    LibreTranslate is open-source (AGPL-3.0).  You can self-host it
    or use the public endpoint at https://libretranslate.com (free tier
    has rate limits).
    """

    def __init__(self, url: str, api_key: str = ""):
        self._url     = url.rstrip("/")
        self._api_key = api_key or os.environ.get("LIBRETRANSLATE_API_KEY", "")

    def translate(self, text: str, target_lang: str) -> str:
        if target_lang == "en":
            return text  # no-op for English

        try:
            import requests
            payload = {
                "q":      text,
                "source": "en",
                "target": target_lang,
                "format": "text",
            }
            if self._api_key:
                payload["api_key"] = self._api_key

            resp = requests.post(
                f"{self._url}/translate",
                json=payload,
                timeout=5,
            )
            resp.raise_for_status()
            return resp.json().get("translatedText", text)

        except Exception as exc:
            # Never crash the main app due to translation failure
            return f"{text}  [Translation failed: {exc}]"

    def is_configured(self) -> bool:
        return bool(self._url)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_translator(translation_cfg: dict) -> BaseTranslator:
    """
    Return the appropriate translator based on settings.yaml.

    Parameters
    ----------
    translation_cfg : the 'translation' section of settings.yaml

    Returns
    -------
    A BaseTranslator instance.
    """
    provider = translation_cfg.get("provider", "none").strip().lower()

    if provider == "none" or not provider:
        return NoOpTranslator()

    if provider == "libretranslate":
        url     = translation_cfg.get("libretranslate_url", "")
        api_key = translation_cfg.get("libretranslate_api_key", "")
        if not url:
            print("WARNING: provider=libretranslate but libretranslate_url "
                  "is not set. Falling back to NoOpTranslator.")
            return NoOpTranslator()
        return LibreTranslateTranslator(url=url, api_key=api_key)

    print(f"WARNING: Unknown translation provider '{provider}'. "
          "Falling back to NoOpTranslator.")
    return NoOpTranslator()
