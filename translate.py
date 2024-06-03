from deep_translator import GoogleTranslator
from langdetect import detect

def translate_en(text):
    translator = GoogleTranslator(target="en")
    translation = translator.translate(text)
    return translation

def split_text(text, max_length=5000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_length:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def translate_default(text, target_lang):
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")
    try:
        source_lang = detect(target_lang)
    except Exception as e:
        raise ValueError(f"Error in language detection: {e}")
    chunks = split_text(text, max_length=5000)
    translated_chunks = []
    for chunk in chunks:
        try:
            translator = GoogleTranslator(target=source_lang)
            translation = translator.translate(chunk)
            translated_chunks.append(translation)
        except Exception as e:
            raise ValueError(f"Error in translation: {e}")
    return ' '.join(translated_chunks)
