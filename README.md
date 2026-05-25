# WarrInt

This is the code for the paper "WarrInt: Integrity Validation for Legal Process."

## Example PDFs

To understand what documents look like throughout the WarrInt process, we recommend looking at `./examples/pdfs`. This folder contains three canonical documents as well as their scanned and modified forms, as well as how WarrInt annotates any (potential) modifications.

We use three documents as examples, and their original canonical forms can be found in `./examples/pdfs/original`. The documents are:

- `casdc-323mj02081wvg-060823`: a document from our training dataset
- `wied-221mj00530SCD`: a document from our test dataset
- `mtd-12024mj00012-1112024`: a document from our test dataset that has the largest number of spurious inconsistencies (72) for an unmodified scanned documents

The versions of these documents with the document description encoded in 2D-barcodes are provided in `./examples/pdfs/original_with_doc_description`.

We also provide different variants of these files in the following subfolders. Each subfolder also contains a PDF ending in `_annotated.pdf` which contains the result of WarrInt.

- `./examples/pdfs/unmodified_scanned`: the canonical documents and document descriptions scanned at DPI 200
- `./examples/pdfs/modified_replacement_digital`: the documents after being digitally modified so that one identifier is replaced with a random string
    - Note that `mtd-12024mj00012-1112024` is unable to be modified using our PDF editing methodology, so it will not be included in any of the `modified` subfolders
- `./examples/pdfs/modified_replacement_scanned`: the files in `./examples/pdfs/modified_replacement_digital` scanned at 200 DPI
- `./examples/pdfs/modified_ocrmistake_digital`: the documents after being digitally modified so that one character of one identifier is replaced with an OCR confusion (e.g. `l` to `i`)
- `./examples/pdfs/modified_ocrmistake_scanned`: the documents in `./examples/pdfs/modified_ocrmistake_digital` scanned at 200 DPI

### Viewing Annotations

WarrInt produces PDFs with annotations describing potential modifications. These PDFs do not require JavaScript or any special features; we have tested that our annotations can be accessed with Mac Preview, Adobe Acrobat, and Google Drive.

On Mac Preview, you can click on the note icon to view the original and modified text. On Adobe Acrobat and Google Drive, the annotation text will be automatically shown in a sidebar. Additionally, on Adobe Acrobat one can hover over an annotation to read the text (as shown in Figure 3 of our paper).

### Generating these PDFs

To run any of our code, please first run `cd optar && make && cd ../ && pip install -r requirements.txt` to install dependencies.

To generate the canonical documents with their document descriptions, you can run `./scripts/gen_doc_description.sh`. To generate the annotated PDFs, you can run `./scripts/test_provider.sh` with a subfolder (e.g. `./scripts/test_provider.sh unmodified_scanned`) or with the `--all` flag to run on all scanned and/or modified subfolders.

Note that we provide precomputed OCR outputs, since the tool we use (Google Document AI) is a paid service. To rerun the OCR, one can set up credentials for Document AI and run `court.py` using the `--processor-id` and `--project-id` flags. Since our OCR library also computes page dimensions, we store the precomputed page dimensions in `examples/dimensions.csv`.

## Code Structure

- The main scripts are `court.py` and `provider.py`, which both take as arguments an input PDF and output a modified version of that PDF. 
    - `court.py` generates a document description for the input PDF and outputs the input PDF with that document description appended.
    - `provider.py` reads the document description attached to the input PDF, verifies the signature, and generates a version of the input PDF with annotation rectangles.
- `lib/` contains our code that supports the above scripts.
    - `lib/ocr.py` contains code used to run Google Document AI and extract OCR output from PDF files.
    - `lib/data_encoding.py` contains code to generate the document description, encode and compress it, and decompress and decode it.
    - `lib/optar_lib.py` is a Python wrapper around the `optar` library used to generate 2D-barcodes of the encoded and compressed document description.
    - `lib/detect_differences.py` contains code to determine how the document has been modified and generate the provider's output PDF with annotations.
- `Seal Detection.ipynb` contains our code to determine whether a document has seals, and to return their bounding boxes. We currently run this as an offline processing step and our code retrieves the seals from `examples/seal_detections.csv`; we plan to integrate this into `court.py` as an online step.
- `examples/` contains example files for running and testing the code.
    - `examples/example_private_key.pem` and `examples/example_public_key.pem` are example Ed25519 keys for use with `court.py` and `provider.py`. New keys can be generated with `python3 scripts/generate_keys.py`.
    - `examples/dimensions.csv` stores precomputed page dimensions for the example PDFs.
    - `examples/seal_detections.csv` stores precomputed seal bounding boxes for the example PDFs.
    - `examples/pdfs/` contains example PDFs as described above.
- `PACER Cases.csv` contains the case numbers and links to the PACER headers that can be used to reconstruct our training and test dataset.
 