# Create this as: colette/ui/utils/i18n.py

import yaml
from pathlib import Path
from typing import Dict, Any
import gradio as gr
from colette.ui.utils.logger import logger

class ColetteI18n:
    """Shared internationalization utility for Colette UI"""
    
    def __init__(self, translation_file_path: Path = None):
        self.translations = self._load_translations(translation_file_path)
        self.gradio_i18n = gr.I18n(**self.translations)
        self.current_language = "en"  # Default fallback
    
    def _load_translations(self, translation_file_path: Path = None) -> Dict[str, Dict[str, str]]:
        """Load translations from YAML file with comprehensive fallbacks"""
        
        # Default translations
        default_translations = {
            "en": {
                "chatbot": "Chatbot",
                "sessions": "Sessions",
                "new_session": "New session",
                "available_services": "Available Services", 
                "enter_message": "Enter message...",
                "sources": "Sources",
                "context_chunks": "Context Chunks",
                "logs": "Logs",
                "about": "About",
                "about_content": "About content",
                "french": "french",
                "english": "english",
                "content": "Content"
            },
            "fr": {
                "chatbot": "Chatbot",
                "sessions": "Sessions", 
                "new_session": "Nouvelle session",
                "available_services": "Services Disponibles",
                "enter_message": "Entrez un message...",
                "sources": "Sources",
                "context_chunks": "Morceaux de Contexte", 
                "logs": "Journaux",
                "about": "À propos",
                "about_content": "Contenu à propos",
                "french": "français",
                "english": "anglais",
                "content": "Contenu"
            }
        }
        
        if translation_file_path and translation_file_path.exists():
            try:
                with open(translation_file_path, 'r', encoding='utf-8') as f:
                    loaded_translations = yaml.safe_load(f) or {}
                    
                # Merge loaded translations with defaults
                for lang in default_translations:
                    if lang in loaded_translations:
                        default_translations[lang].update(loaded_translations[lang])
                    
                # Add any new languages from the file
                for lang in loaded_translations:
                    if lang not in default_translations:
                        default_translations[lang] = loaded_translations[lang]
                        
                logger.info(f"Loaded translations from {translation_file_path}")
                        
            except Exception as e:
                logger.warning(f"Could not load translation file {translation_file_path}: {e}")
        
        return default_translations
    
    def get_gradio_i18n(self) -> gr.I18n:
        """Get the Gradio I18n instance"""
        return self.gradio_i18n
    
    def translate(self, key: str, lang: str = None) -> str:
        """
        Translate a key to the specified language or current language
        
        Args:
            key (str): Translation key
            lang (str, optional): Target language. Defaults to current_language.
            
        Returns:
            str: Translated text or the key itself if not found
        """
        target_lang = lang or self.current_language
        
        # Try to get translation
        if target_lang in self.translations and key in self.translations[target_lang]:
            return self.translations[target_lang][key]
        
        # Fallback to English
        if key in self.translations.get("en", {}):
            return self.translations["en"][key]
            
        # Final fallback: return the key itself
        return key
    
    def set_language(self, lang: str):
        """Set the current language"""
        if lang in self.translations:
            self.current_language = lang
        else:
            logger.warning(f"Language {lang} not available, using English")
            self.current_language = "en"

# Global instance
_i18n_instance = None

def get_i18n_instance(translation_file_path: Path = None) -> ColetteI18n:
    """Get or create the global i18n instance"""
    global _i18n_instance
    if _i18n_instance is None:
        _i18n_instance = ColetteI18n(translation_file_path)
    return _i18n_instance

def _(key: str) -> str:
    """Simple translation function for backward compatibility"""
    try:
        instance = get_i18n_instance()
        return instance.translate(key)
    except Exception as e:
        logger.debug(f"Translation error for key '{key}': {e}")
        return key  # Fallback to the key itself