from pathlib import Path
from enum import Enum
import json

import typer
from typing_extensions import Annotated
from transformers import pipeline
from transformers.pipelines.text2text_generation import TranslationPipeline

from experimental_wsd.data_processing.processed_usas_utils import load_usas_mapper


class TranslationLanguages(str, Enum):
    finnish = "finnish"

def main(english_usas_mapper_path: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False, writable=False, readable=True, resolve_path=True,)],
         language: Annotated[TranslationLanguages, typer.Argument()],
         output_usas_mapper_path: Annotated[Path, typer.Argument(writable=True, readable=False)]):
    
    english_usas_mapper = load_usas_mapper(english_usas_mapper_path, None)

    translation_pipeline: TranslationPipeline | None = None

    if language == TranslationLanguages.finnish:
        translation_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fi")
        assert isinstance(translation_pipeline, TranslationPipeline)
    assert translation_pipeline is not None

    translated_usas_mapper: dict[str, str] = {}
    for tag, description in english_usas_mapper.items():
        translated_description_object = translation_pipeline(description)
        not_transalted = False
        if len(translated_description_object) != 1:
            not_transalted = True
        translated_description = translated_description_object[0]["translation_text"].strip()
        if not translated_description:
            not_transalted = True

        if not_transalted:
            raise ValueError(f"The English description: {description} was not "
                             "translated. Language to be translated too: "
                             f"{language}. The translated object: "
                             f"{translated_description_object}")
        translated_usas_mapper[tag] = translated_description
    
    with output_usas_mapper_path.open("w", encoding="utf-8") as output_fp:
        json.dump(translated_usas_mapper, output_fp)
        

    


if __name__ == "__main__":
    typer.run(main)