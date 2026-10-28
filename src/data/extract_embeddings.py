"""
Filename: extract_embeddings.py

Author: Hakima Laribi

Description: Downloads the MIMIC-CXR-JPG images and the MIMIC-IV-Note notes, and extracts their embeddings
(DenseNet121-res224-chex for the chest X-rays, Bio_ClinicalBERT for the radiology and discharge notes).
Requires the PHYSIONET_USERNAME and PHYSIONET_PASSWORD environment variables.

Run from the root of the repository:
    python -m src.data.extract_embeddings
"""
from src.data.extraction.extraction_mimic_cxr import MIMICCXRExtractor, group_image_emb
from src.data.extraction.extraction_mimic_notes import MIMICNotesExtractor

IMAGES_DIR = 'physionet.org/image_features'
NOTES_DIR = 'physionet.org/note'

if __name__ == '__main__':
    # Chest X-rays, downloaded, embedded and deleted one patient group (p10 to p19) at a time
    MIMICCXRExtractor(path_to_images=IMAGES_DIR, to_download=True)()
    group_image_emb(feature_files=[f'{IMAGES_DIR}/p{i}_list_densenet_features' for i in range(10, 20)],
                    output_folder=f'{IMAGES_DIR}/cxr_features')

    # Radiology and discharge notes
    notes = MIMICNotesExtractor(path_to_notes=NOTES_DIR, to_download=True)
    notes('radiology')
    notes('discharge')
