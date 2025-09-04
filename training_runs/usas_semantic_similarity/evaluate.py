import logging
from pathlib import Path
from collections import Counter
import time

import typer
from typing_extensions import Annotated
import datasets
import torch
from transformers import AutoTokenizer

from dotenv import load_dotenv
load_dotenv()


from experimental_wsd.config import MosaicoCoreUSAS, DATA_PROCESSING_DIR, USASMapper
from experimental_wsd.data_processing import processed_usas_utils
from experimental_wsd.nn.token_similarity import TokenSimilarityVariableNegatives

logger = logging.getLogger(__name__)


def main(best_model_path: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True, writable=False, resolve_path=True)]):
    start_time = time.perf_counter()
    best_model_path_str = str(best_model_path)
    usas_tag_to_description_mapper = processed_usas_utils.load_usas_mapper(USASMapper)

    if MosaicoCoreUSAS is None:
        error_message = (
            "Please follow the data download instructions and set the "
            "`MOSAICO_CORE_USAS` environment variable. "
            "If you have done that successfully then please ensure the "
            "environment variable is loaded before importing this file."
        )
        raise FileNotFoundError(error_message)
    test_file_path =  MosaicoCoreUSAS.test
    test_processed_file_path = processed_usas_utils.process_file(test_file_path, DATA_PROCESSING_DIR, "variable_mosaico_usas_9")
    test_dataset = datasets.load_dataset("json", data_files=str(test_processed_file_path))['train'].take(20000)
    model = TokenSimilarityVariableNegatives.load_from_checkpoint(best_model_path_str).to(device="cpu")
    tokenizer = AutoTokenizer.from_pretrained(model.base_model_name, add_prefix_space=True)
    top_1_scores = Counter()
    total = 0
    model.eval()
    with torch.inference_mode(mode=True):

        usas_embedded_descriptions = []
        usas_index_to_tag = {}
        for index, usas_tag_description in enumerate(usas_tag_to_description_mapper.items()):
            usas_tag, usas_description = usas_tag_description
            tokenized_usas_description = tokenizer(usas_description, truncation=False, padding=False, return_tensors="pt")
            description_input_ids = tokenized_usas_description.input_ids.to(model.device).unsqueeze(0)
            description_attention_mask = tokenized_usas_description.attention_mask.to(model.device).unsqueeze(0)
            definition_embedding = model.label_definition_encoding(description_input_ids, description_attention_mask)
            usas_embedded_descriptions.append(definition_embedding)
            usas_index_to_tag[index] = usas_tag
        usas_embedded_descriptions_tensor = torch.vstack(usas_embedded_descriptions)
        NUM_DESC, DESC_BATCH, EMBEDDING_DIM = usas_embedded_descriptions_tensor.shape
        usas_embedded_descriptions_tensor = usas_embedded_descriptions_tensor.view(DESC_BATCH, NUM_DESC, EMBEDDING_DIM)
        
        for sentence_index, sentence in enumerate(test_dataset):
            all_usas_tags, usas_token_offsets = sentence['usas'], sentence['usas_token_offsets']
            tokenized_text = tokenizer(sentence['tokens'], truncation=False, padding=False, return_tensors="pt", is_split_into_words=True)
            text_input_ids = tokenized_text.input_ids.to(model.device)
            text_attention_mask = tokenized_text.attention_mask.to(model.device)
            text_embedding = model.text_encoding(text_input_ids, text_attention_mask)
            for usas_tags, usas_token_offset in zip(all_usas_tags, usas_token_offsets):
                skip_tag_as_empty = False
                usas_token_offset_indexes = set(range(*usas_token_offset))
                if len(usas_token_offset_indexes) == 1:
                    if not sentence['tokens'][usas_token_offset[0]]:
                        skip_tag_as_empty = True
                if skip_tag_as_empty:
                    continue

                text_word_ids_mask = []
                for word_id in tokenized_text.word_ids():
                    if word_id is None:
                        text_word_ids_mask.append(0)
                    elif word_id in usas_token_offset_indexes:
                        text_word_ids_mask.append(1)
                    else:
                        text_word_ids_mask.append(0)
                text_word_ids_mask = torch.tensor(text_word_ids_mask, dtype=torch.long)
                text_word_ids_mask = text_word_ids_mask.unsqueeze(0).to(device=model.device)
                if text_word_ids_mask.sum() == 0:
                    raise ValueError("Cannot find the token offsets in the given sample. "
                                     f"Annotation: {sentence}")
                #text_input_ids = tokenized_text.input_ids.to(model.device)
                #text_attention_mask = tokenized_text.attention_mask.to(model.device)
                token_embedding = model.token_encoding_using_text_encoding(text_embedding, text_word_ids_mask)
                label_similarity_score = model.token_label_similarity(usas_embedded_descriptions_tensor, token_embedding)
                predicted_usas_tag = usas_index_to_tag[torch.argmax(label_similarity_score).item()]
                
                usas_tags_set = set(usas_tags)
                if predicted_usas_tag in usas_tags_set:
                    top_1_scores.update([predicted_usas_tag])
                total += 1
            if (sentence_index % 100) == 0 and sentence_index != 0:
                print(f'Progress: {sentence_index}')
    end_time = time.perf_counter()
    print(end_time - start_time)
    print(f"Total: {total:,}")
    accuracy = (sum(top_1_scores.values()) / total) * 100
    print(f"Accuracy: {accuracy:.2f}")
    for tag, count in sorted(top_1_scores.items(), key=lambda x: x[1]):
        print(f"Tag: {tag} Count: {count:,}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    typer.run(main)