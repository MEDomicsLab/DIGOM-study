"""
Filename: extraction_mimic_cxr.py

Author: Hakima Laribi

Description: Defines the classes and functions to extract embeddings from MIMIC-CXR-JPG medical images.
"""
import gzip
import multiprocessing
import os
import pickle
import shutil
import subprocess
from glob import glob
from typing import List, Optional, Union

import cv2
import numpy as np
import pandas as pd
import skimage
import torch
import torch.nn.functional as F
import torchxrayvision as xrv
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class MIMICCXRExtractor:
    """Downloads MIMIC-CXR-JPG images from PhysioNet and extracts their DenseNet embeddings."""

    def __init__(self,
                 path_to_images: str,
                 to_download: bool = False,
                 base_url: Optional[str] = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/",
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 img_to_download: Optional[List[str]] = None):
        """
        Args:
            path_to_images: path to where images are saved if already downloaded, else where they will be saved
            to_download: boolean to specify if to download the data from physionet
            base_url: URL to physionet repository
            username: username to physionet account (defaults to the PHYSIONET_USERNAME environment variable)
            password: password to physionet account (defaults to the PHYSIONET_PASSWORD environment variable)
            img_to_download: relative paths of specific images to download; if None, whole patient groups are used
        """
        # Set protected attributes
        self._path_to_images = path_to_images
        self._to_download = to_download
        self._base_url = base_url
        self._username = username or os.environ.get('PHYSIONET_USERNAME')
        self._password = password or os.environ.get('PHYSIONET_PASSWORD')
        self._model = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._img_to_download = img_to_download

        # If we need to download the data, get the csv files first and meta-data
        if self._to_download:
            # Make sure download directory exists
            os.makedirs(self._path_to_images, exist_ok=True)
            # Download all csv files and meta-data
            cmd = ["wget", "-r", "-l1", "-nd", "-np", "-A", "*.csv*,IMAGE_FILENAMES", "--user", self._username,
                   "--password", self._password, "-c", "-P", self._path_to_images, self._base_url]
            subprocess.run(cmd, check=True)
            # Extract the compressed csv files
            self.extract_compressed_csvs()

    def __call__(self,
                 pretrained_model: str = "densenet121-res224-chex",
                 batch_size: int = 10):
        """
        Extracts the embeddings of all images and saves them per patient group.

        Args:
            pretrained_model: torchxrayvision DenseNet weights used to extract the embeddings
            batch_size: number of images per batch
        """
        self._model = xrv.models.DenseNet(weights=pretrained_model).to(self._device)

        # Iterate over all the images
        if self._img_to_download is None:
            patient_list = list(range(10, 20))
            for (i, patient_grp) in enumerate(patient_list):
                if self._to_download:
                    print(f"Downloading images for patient group {patient_grp} ..")
                    # download the images of the subdirectory
                    delete_previous = patient_list[i-1] if i > 0 else None
                    self.download_from_physionet(patient_grp, delete_previous)
                    path_to_subdir = os.path.join(self._path_to_images, "mimic-cxr-jpg/2.1.0/files/")
                else:
                    path_to_subdir = self._path_to_images

                # Create path to images
                path_to_subdir = os.path.join(path_to_subdir, f"p{patient_grp}")
                images_dirs = []
                for dirpath, dirnames, filenames in os.walk(path_to_subdir):
                    for file in filenames:
                        images_dirs.append(os.path.join(dirpath, file))

                dense_features = self.get_cxr_embeddings(images_dirs, patient_grp, batch_size)
                columns = ['image_name', 'n_study', 'n_patient'] + [f'feature_{i}' for i in
                                                                   range(len(dense_features[0]) - 3)]

                df_dense_features = pd.DataFrame(dense_features, columns=columns)

                # Save to CSV
                output_csv = os.path.join(self._path_to_images, f"p{patient_grp}_densenet_features.csv")
                df_dense_features.to_csv(output_csv, index=False)

        else:
            self.download_from_physionet()
            path_to_subdir = os.path.join(self._path_to_images, "mimic-cxr-jpg/2.1.0/")
            images_dirs = [os.path.join(path_to_subdir, img_path) for img_path in self._img_to_download]
            self.get_cxr_embeddings(images_dirs, '_missing', batch_size)

    def compute_dense_feature(self, img: torch.Tensor) -> np.ndarray:
        """
        Args:
            img: batch of preprocessed images (B, 1, 224, 224)

        Returns:
            (B, 1042) array: the 1024 DenseNet dense features followed by the 18 pathology predictions
        """
        img = img.to(self._device)
        self._model.eval()

        with torch.no_grad():
            # Extract dense features
            feats = self._model.features(img)
            feats = F.relu(feats, inplace=True)
            feats = F.adaptive_avg_pool2d(feats, (1, 1))
            densefeature_embeddings = feats.cpu().detach().numpy().reshape(feats.shape[0], -1)

            # Extract predicted probabilities of considered 18 classes:
            # Get by calling "xrv.datasets.default_pathologies" or "dict(zip(xrv.datasets.default_pathologies,preds[0].detach().numpy()))"
            # ['Atelectasis','Consolidation','Infiltration','Pneumothorax','Edema','Emphysema',Fibrosis',
            #  'Effusion','Pneumonia','Pleural_Thickening','Cardiomegaly','Nodule',Mass','Hernia',
            #  'Lung Lesion','Fracture','Lung Opacity','Enlarged Cardiomediastinum']
            preds = self._model(img).cpu()
            prediction_embeddings = preds.detach().numpy().reshape(feats.shape[0], -1)
            features = np.concatenate((densefeature_embeddings, prediction_embeddings), axis=1)

        return features

    def get_cxr_embeddings(self,
                           images_dirs: List[str],
                           patient_grp: Union[int, str],
                           batch_size: int = 10) -> List[list]:
        """
        Computes the embeddings of a list of images, checkpointing the results in a pickle file after each batch.

        Args:
            images_dirs: paths to the images
            patient_grp: patient group used to name the pickle file
            batch_size: number of images per batch

        Returns:
            list of [image_name, n_study, n_patient, feature_0, ..., feature_n] rows
        """
        dataset = CXRDatasetMIMIC(images_dirs)

        print("Creating dataloader...")
        dataloader = DataLoader(dataset=dataset,
                                shuffle=True,
                                batch_size=batch_size,
                                drop_last=False,
                                num_workers=multiprocessing.cpu_count() - 4,
                                pin_memory=True
                                )

        # Get feature embeddings of all images
        print("Get feature embeddings with DenseNet ..")
        dense_features = []
        for img, img_dir in tqdm(dataloader):
            batch_size = img.shape[0]
            feat = self.compute_dense_feature(img)

            for i in range(batch_size):
                image_path = img_dir[i]
                image_name = os.path.basename(image_path)
                n_study = os.path.basename(os.path.dirname(image_path))
                n_patient = os.path.basename(os.path.dirname(os.path.dirname(image_path)))

                # Flatten the embedding as a list of values
                embedding = feat[i].tolist()

                # Append the information
                dense_features.append([image_name, n_study, n_patient] + embedding)

            with open(os.path.join(self._path_to_images, f"p{patient_grp}_list_densenet_features"), 'wb') as f:
                pickle.dump(dense_features, f)
        return dense_features

    def extract_compressed_csvs(self):
        """
        Extracts all the downloaded compressed csvs
        """
        # Loop through all .csv.gz files
        for filename in os.listdir(self._path_to_images):
            if filename.endswith(".csv.gz"):
                gz_path = os.path.join(self._path_to_images, filename)
                csv_path = gz_path[:-3]  # Remove '.gz' extension

                # Extract the .gz file
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(csv_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                # Delete the original .gz file
                os.remove(gz_path)

    def download_from_physionet(self,
                                subdirectory: Optional[int] = None,
                                delete_previous: Optional[int] = None):
        """
        Downloads a subset of patients images from physionet (~ 6500 patients)

        Args:
            subdirectory: number of the subdirectory containing patients' data; if None, downloads img_to_download
            delete_previous: number of the previously downloaded subdirectory to delete, if any
        """
        if delete_previous is not None:
            # Deletes the directory and all its contents
            repo_path = os.path.join(self._path_to_images, f"mimic-cxr-jpg/2.1.0/files/p{delete_previous}")
            shutil.rmtree(repo_path)
            print(f"Deleted {repo_path} successfully.")

        if type(subdirectory) is int:
            image_files = f'p{subdirectory}'
        else:
            image_files = "\\|".join(self._img_to_download)

        image_filenames = os.path.join(self._path_to_images, 'IMAGE_FILENAMES')
        cmd = f"""
         grep "{image_files}" {image_filenames} | wget -r -N -c -np -nH --cut-dirs=1 -P {self._path_to_images} \
         --user {self._username} --password={self._password} -i - --base={self._base_url}
        """
        subprocess.run(cmd, shell=True, check=True)


class CXRDatasetMIMIC(Dataset):
    """Loads CXR images and preprocesses them as expected by the torchxrayvision DenseNet."""

    def __init__(self, img_dirs: List[str]):
        """
        Args:
            img_dirs: paths to the images
        """
        self.img_dirs = img_dirs

    def __len__(self):
        """Returns the number of images."""
        return len(self.img_dirs)

    def __getitem__(self, idx):
        """
        Args:
            idx: index of the image

        Returns:
            (normalized (1, 224, 224) image, path to the image)
        """
        image = skimage.io.imread(self.img_dirs[idx])
        image = xrv.datasets.normalize(image, 255)
        # For each image check if they are 2D arrays
        if len(image.shape) > 2:
            image = image[:, :, 0]
        if len(image.shape) < 2:
            print("Error: Dimension lower than 2 for image!")
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = image[None, :, :]
        return image, self.img_dirs[idx]


def group_image_emb(feature_files: List[str], output_folder: str = 'image_features/cxr_features'):
    """
    Merges the pickled embeddings of several extraction runs and saves them as one csv per patient group.

    Args:
        feature_files: paths to the pickle files produced by MIMICCXRExtractor.get_cxr_embeddings
        output_folder: directory where cxr_pt_group_{group}.csv files are written
    """
    features = []
    for file_path in feature_files:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                features += pickle.load(file)

    columns = ['image_name', 'n_study', 'n_patient'] + [f'feature_{i}' for i in
                                                       range(len(features[0]) - 3)]

    df_dense_features = pd.DataFrame(features, columns=columns)
    df_dense_features['pt_group'] = df_dense_features['n_patient'].astype(str).str[:3]
    df_dense_features = df_dense_features.drop_duplicates()

    # write
    os.makedirs(output_folder, exist_ok=True)

    for group, df_group in df_dense_features.groupby('pt_group'):
        file_path = os.path.join(output_folder, f"cxr_pt_group_{group}.csv")
        df_group.to_csv(file_path, index=False)


def find_missing_images(csv_folder: str = 'image_features/cxr_features',
                        img_names_file: str = 'image_features/IMAGE_FILENAMES') -> List[str]:
    """
    Finds the images listed in IMAGE_FILENAMES that have no embedding in the saved csvs.

    Args:
        csv_folder: directory of the cxr_pt_group_*.csv files
        img_names_file: path to the MIMIC-CXR-JPG IMAGE_FILENAMES file

    Returns:
        relative paths of the missing images
    """
    # Read image filenames
    expected_paths = {}
    with open(img_names_file, "r") as f:
        for line in f:
            full_path = line.strip()
            filename = os.path.basename(full_path)
            expected_paths[filename] = full_path  # Store full path

    # Read computed image filenames from all CSVs
    computed_filenames = set()
    csv_files = glob(os.path.join(csv_folder, "cxr_pt_group_p1*.csv"))

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        computed_filenames.update(df['image_name'])

    # Find missing images and retrieve their full paths
    missing_images = set(expected_paths.keys()) - computed_filenames
    missing_paths = [expected_paths[img] for img in missing_images]

    print(f"Total expected images: {len(expected_paths)}")
    print(f"Total computed images: {len(computed_filenames)}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Missing images: {missing_paths}")

    return missing_paths


def add_missing_features(missing_path: str = 'image_features/p_missing_list_densenet_features',
                         csv_saved_features: str = 'image_features/cxr_features'):
    """
    Appends the embeddings of previously missing images to the saved per-group csvs.

    Args:
        missing_path: pickle file with the embeddings of the missing images
        csv_saved_features: directory of the cxr_pt_group_*.csv files
    """
    with open(missing_path, 'rb') as file:
        dense_features = pickle.load(file)

    columns = ['image_name', 'n_study', 'n_patient'] + [f'feature_{i}' for i in
                                                        range(len(dense_features[0]) - 3)]

    df_dense_features = pd.DataFrame(dense_features, columns=columns)
    df_dense_features['pt_group'] = df_dense_features['n_patient'].astype(str).str[:3]
    df_dense_features = df_dense_features.drop_duplicates()

    # write
    for group, df_group in df_dense_features.groupby('pt_group'):
        file_path = os.path.join(csv_saved_features, f"cxr_pt_group_{group}.csv")
        saved_group = pd.read_csv(file_path)
        df = pd.concat([df_group, saved_group], ignore_index=True, axis=0)
        df.to_csv(file_path, index=False)
