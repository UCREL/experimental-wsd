import argparse
from pathlib import Path
import gzip

from huggingface_hub import hf_hub_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("english_usas_mosaico_folder", type=Path)
    args = parser.parse_args()

    dataset_folder = str(args.english_usas_mosaico_folder.resolve())
    for i in range(10):
        # filename=f"data/wikipedia_shard_{i}.jsonl.gz",
        downloaded_compressed_file = hf_hub_download("ucrelnlp/English-USAS-Mosaico",
                                                     filename=f"wikipedia_shard_{i}.jsonl.gz",
                                                     subfolder="data",
                                                     repo_type="dataset",
                                                     local_dir=dataset_folder)
        un_compressed_file_path = Path(dataset_folder, f"wikipedia_export.jsonl.{i}")
        with gzip.open(downloaded_compressed_file, mode="rt", encoding="utf-8") as read_fp:
            with un_compressed_file_path.open("w", encoding="utf-8") as write_fp:
                for line in read_fp:
                    write_fp.write(line)
        Path(downloaded_compressed_file).unlink(missing_ok=False)
