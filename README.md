# TIGeR: Text-Image Generative Repair

## Goal
The goal of **TIGeR** is to automatically identify and repair mismatches between text and images in datasets. In many large-scale datasets, text descriptions may no longer match their corresponding images due to corruption, drift, or bad initial data entry. TIGeR provides a pipeline to detect these quality issues and apply repairs, such as replacing the image with a better candidate or patching the text metadata.

## Process
The TIGeR pipeline consists of several stages, moving from raw data to a repaired dataset:

1.  **Data Simulation (Optional)**:
    *   `scripts/make_toy_noisy.py`: Generates a synthetic "noisy" dataset for testing purposes.

2.  **Sieving (Detection)**:
    *   `scripts/run_sieve.py`: Uses **CLIP** (Contrastive Language-Image Pre-Training) to compute similarity scores between text and images.
    *   Identifies "suspicious" rows where the similarity score falls below a certain threshold (calculated via IQR or Quantile methods).
    *   Filters the dataset to isolate potential errors.

3.  **Diagnosis (Analysis)**:
    *   `scripts/mismatch_analyzer.py`: Analyzes the suspicious rows identified by the Sieve.
    *   Determines the type of mismatch and generates a queue of proposed fixes (e.g., proposing a candidate image from another row or assessing if text attributes need updates).
    *   Outputs an `arbiter_queue_*.csv`.

4.  **Arbitration (Decision)**:
    *   `scripts/run_arbiter.py`: Consumes the queue from the analyzer.
    *   Decides on the best course of action based on available fixes and costs ("MVP" logic):
        *   **T2V (Text-to-Visual)**: Suggests `replace_image_from_row` if a valid candidate exists.
        *   **V2T (Visual-to-Text)**: Suggests `apply_text_patch` to update text/attributes.
        *   **Human Review**: Fallback for low-confidence or complex cases.
    *   Outputs an `arbiter_plan_*.csv`.

5.  **Execution (Repair)**:
    *   `scripts/apply_repairs.py`: Executes the plan.
    *   Modifies the dataset (parquet) and organizes image files.
    *   Updates metadata (titles, categories, attributes) and physically copies/replaces images where necessary.

## Prerequisites
To run this project, you need Python installed along with the dependencies listed in `requirements.txt`.

### Installation
```bash
git clone https://github.com/abdullah2142/TIGeR-Text-Image-Generative-Repair.git
cd TIGeR
pip install -r requirements.txt
```

**Key Dependencies:**
*   `torch`, `torchvision`
*   `transformers` (for CLIP)
*   `pandas`, `numpy` (data manipulation)
*   `pillow`, `opencv-python` (image processing)

## Results
After running the pipeline, the system produces:
*   **Repaired Dataset**: A new Parquet file (e.g., `repaired_data.parquet`) containing the corrected text-image pairs.
*   **Repair Logs**: CSV logs detailing the status of every repair attempt (Success, Fail, Skip) and the reasons.
*   **Repaired Images**: A directory containing the actual image files for any replacements that occurred.


## Current Limitations
*   **Heuristic-Only Repairs (MVP)**: The current system relies strictly on swapping images from other existing rows (`replace_image_from_row`) or applying manual text patches. It does not yet generate *new* content.
*   **Unimplemented Generative Actions**: While the `arbiter` has placeholders for Text-to-Visual (T2V) and Visual-to-Text (V2T) generation, these paths currently default to "unimplemented" or fallback to human review.
*   **Sieve Sensitivity**: The detection mechanism uses statistical thresholds (IQR/Quantile) on CLIP scores. This may yield false positives (flagging unique but correct pairs) or false negatives (missing subtle mismatches).
*   **Resource Dependence**: The system assumes the dataset is small enough for local processing or that the user has sufficient compute for CLIP inference on CPU.

## Future Work
The current implementation utilizes "MVP" heuristic repairs (swapping existing images, patching text). Future work aims to integrate **Generative AI** models to synthesize repairs:
*   **Text-to-Visual (T2V)**: Using generative models (like Stable Diffusion) to *create* a new image that matches the text description when no suitable candidate exists.
*   **Visual-to-Text (V2T)**: Using Vision-Language Models (VLMs) to generate accurate text descriptions from images to overwrite incorrect metadata.
