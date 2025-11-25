from pathlib import Path
import csv

import typer
from typing_extensions import Annotated

def main(zh_data: Annotated[Path, typer.Argument(exists=True, readable=True, writable=False, resolve_path=True, file_okay=True, dir_okay=False)],
         export_zh_data: Annotated[Path, typer.Argument(readable=False, writable=True)]):
    sentence_break_token_indexes = set([
        49, 102, 121, 153, 208, 295, 385, 421, 457, 494, 524, 546, 580, 756, 801, 896, 979, 1031, 1045, 1078, 1143, 1157, 1165, 1170, 1202, 1240,
        1293, 1353, 1417, 1443, 1579, 1625, 1654, 1692, 1732, 1855, 1886, 1952, 1986, 2035, 2108, 2141, 2155, 2223, 2284, 2311
    ])
    with export_zh_data.open('w', encoding="utf-8", newline='') as write_fp:
        with zh_data.open("r", encoding="utf-8-sig", newline="") as read_fp:
            reader = csv.DictReader(read_fp)
            field_names = reader.fieldnames
            field_names.append("sentence-break")
            csv_writer = csv.DictWriter(write_fp, fieldnames=field_names)
            csv_writer.writeheader()

            

            for row_index, row in enumerate(reader):
                if row_index in sentence_break_token_indexes:
                    row["sentence-break"] = True
                else:
                    row["sentence-break"] = False
                csv_writer.writerow(row)


if __name__ == "__main__":
    typer.run(main)