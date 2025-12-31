import torch
from torch.utils.data import Dataset, Sampler
import tarfile
from io import BytesIO
import requests
import json
import h5py
import os
import numpy as np
import mmap
import lmdb
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from torchvision import transforms
import random
from torch.utils.data.distributed import DistributedSampler


# from concurrent.futures import ThreadPoolExecutor
"""
You’re using ThreadPoolExecutor for parallel processing of patches, which can be efficient.
However, with heavy image-processing tasks, consider moving to a ProcessPoolExecutor if you face CPU bottlenecks,
as GIL can become an issue when using threads in Python with CPU-bound tasks.
"""
from concurrent.futures import ProcessPoolExecutor

class SeededSampler(Sampler):
    def __init__(self, dataset, shuffle, seed):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.num_samples = len(dataset)
        self.indices = self.generate_indices()

    def generate_indices(self):
        indices = list(range(self.num_samples))
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(indices)
        return indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_samples

class ResumableDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        super().__init__(dataset, num_replicas, rank, shuffle, seed)
        self.current_index = 0  # Index to start from

    def set_start_index(self, index):
        self.current_index = index

    def __iter__(self):
        # If shuffling, generate the same shuffle across all processes
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Subsample
        total_size = self.num_samples * self.num_replicas
        indices += indices[:(total_size - len(indices))]
        assert len(indices) == total_size

        # Subset for this process
        indices = indices[self.rank:self.total_size:self.num_replicas]

        # Skip to the current index
        indices = indices[self.current_index:]

        return iter(indices)

    def __len__(self):
        return self.num_samples - self.current_index


class HPADatasetBase(Dataset):
    def __init__(self, hpa11m_data, datadir, data_split="train", processor=None, cache_dir=None, num_workers=4):
        self.hpa11m_data = hpa11m_data
        self.datadir = datadir
        self.data_split = data_split
        self.processor = processor
        self.cache_dir = cache_dir
        self.num_workers = num_workers
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            if not hasattr(self, 'env'):  # Each worker gets its own LMDB environment
                self.env = lmdb.open(os.path.join(self.cache_dir, 'hpa_cache'), map_size=1099511627776)

        # self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self.executor = ProcessPoolExecutor(max_workers=self.num_workers)

        # Initialize augmentations only if in training mode
        self.augmentations = None
        if self.data_split == "train":
            self.augmentations = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3)
            ])
        else:
            self.augmentations = None

    def __len__(self):
        return len(self.hpa11m_data)

    def _load_image_and_metadata(self, name, tar_filename):
        if self.cache_dir:
            with self.env.begin(write=False) as txn:
                image_data = txn.get(name.encode())
                metadata_data = txn.get((name + "_metadata").encode())
                
                if image_data and metadata_data:
                    image = Image.open(BytesIO(image_data))
                    custom_metadata = json.loads(metadata_data)
                    return image, custom_metadata

        with open(tar_filename, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                with tarfile.open(fileobj=mm) as tar:
                    image_member = tar.getmember(name + ".jpg")
                    image_file = tar.extractfile(image_member)
                    image_data = image_file.read()
                    image = Image.open(BytesIO(image_data))

                    json_member = tar.getmember(name + ".json")
                    json_file = tar.extractfile(json_member)
                    json_data = json_file.read()
                    json_annotation = json.loads(json_data)
                    custom_metadata = json_annotation['custom_metadata']

        if self.cache_dir:
            with self.env.begin(write=True) as txn:
                txn.put(name.encode(), image_data)
                txn.put((name + "_metadata").encode(), json.dumps(custom_metadata).encode())

        return image, custom_metadata

    def _process_patch(self, patch):
        # Apply augmentations only during training
        if self.augmentations is not None:
            patch = self.augmentations(patch)  # Apply augmentations

        if self.processor:
            processed_patch = self.processor(images=patch, return_tensors="pt")["pixel_values"].squeeze(0)
            return processed_patch
        else:
            raise ValueError("Processor is not provided.")

    def _pad_patch(self, patch, target_size, fill='white'):
        """
        Pads the input patch to the specified target size with the given fill color.
        """
        width, height = patch.size
        padding = (
            0,  # Left padding
            0,  # Top padding
            target_size - width,  # Right padding
            target_size - height  # Bottom padding
        )

        # Apply padding
        new_patch = ImageOps.expand(patch, border=padding, fill=fill)
        return new_patch

    def get_text_input_output(self, annotation, tissue_folder, cell_type):
        # Mapping dictionaries
        intensity_map = {'negative': [1, 0, 0, 0], 'weak': [0, 1, 0, 0], 'moderate': [0, 0, 1, 0], 'strong': [0, 0, 0, 1]}
        location_map = {'none': [1, 0, 0, 0], 'cytoplasmic/membranous': [0, 1, 0, 0], 'nuclear': [0, 0, 1, 0], 'cytoplasmic/membranous,nuclear': [0, 0, 0, 1]}
        quantity_map = {'none': [1, 0, 0, 0], '<25%': [0, 1, 0, 0], '25%-75%': [0, 0, 1, 0], '>75%': [0, 0, 0, 1]}
        malignancy_map = {'normal': [1, 0], 'cancer': [0, 1]}
        
        """
        # Note: This tissue list is based on:
        np.unique(
                pd.Series(hpa11m_data["tissue"].unique()).str
                    .replace("cancer", "")
                    .replace("tissue", "").str.strip()
            )
            
        It is worth noting that:
        >>> hpa11m_data.loc[hpa11m_data["tissue"].isin(["pancreas"]), "tissue_folder"].unique()
        array(['normal'], dtype=object)
        >>> hpa11m_data.loc[hpa11m_data["tissue"].isin(["pancreatic cancer"]), "tissue_folder"].unique()
        array(['cancer'], dtype=object)
        """
        tissue_list = ['adipose', 'adrenal gland', 'appendix', 'bone marrow', 'breast',
                        'bronchus', 'carcinoid', 'caudate', 'cerebellum',
                        'cerebral cortex', 'cervical', 'cervix', 'colon', 'colorectal',
                        'duodenum', 'endometrial', 'endometrium', 'epididymis',
                        'esophagus', 'fallopian tube', 'gallbladder', 'glioma',
                        'head and neck', 'heart muscle', 'hippocampus', 'kidney', 'liver',
                        'lung', 'lymph node', 'lymphoma', 'melanoma', 'nasopharynx',
                        'oral mucosa', 'ovarian', 'ovary', 'pancreas', 'pancreatic',
                        'parathyroid gland', 'placenta', 'prostate', 'rectum', 'renal',
                        'salivary gland', 'seminal vesicle', 'skeletal muscle', 'skin',
                        'small intestine', 'smooth muscle', 'soft', 'spleen', 'stomach',
                        'testis', 'thyroid', 'thyroid gland', 'tonsil', 'urinary bladder',
                        'urothelial', 'vagina'] # N = 58 (Pancreas and Pancreatic cancer are merged. If not perged, N = 65)
        
        cell_type_list = ['Glandular cells', 'Exocrine glandular cells', 'Tumor cells',
                            'Cholangiocytes', 'Adipocytes', 'Squamous epithelial cells',
                            'Glial cells', 'Cells in endometrial stroma', 'Cells in red pulp',
                            'Alveolar cells', 'Respiratory epithelial cells',
                            'Cells in granular layer', 'Endothelial cells', 'Fibroblasts',
                            'Decidual cells', 'Cells in glomeruli', 'Myocytes',
                            'Cells in seminiferous ducts', 'Hematopoietic cells',
                            'Germinal center cells', 'Cardiomyocytes', 'Urothelial cells',
                            'Trophoblastic cells', 'Smooth muscle cells',
                            'Ovarian stroma cells', 'Follicle cells', 'Epidermal cells',
                            'Chondrocytes', 'Hepatocytes', 'Lymphoid tissue',
                            'Non-germinal center cells', 'Cells in molecular layer',
                            'Keratinocytes', 'Peripheral nerve', 'Cells in tubules',
                            'Neuronal cells', 'Leydig cells', 'Cells in white pulp', 'Langerhans'] # Dec 4 -- added remaining cell types in hpa11m
        # Process additional information
        tissue_name = annotation['tissue'].replace("cancer","").replace("tissue","").strip()
        snomed_text = annotation['snomed_text']
        snomed_code = annotation['snomed_code']
        query_input = f"{tissue_name}. {snomed_text}. Gene: {annotation['gene']} for {annotation['cell_type'].lower()}."
        image_url = annotation['url']
        caption_output = annotation['caption_1']
        
        # staining knowledge
        staining_intensity = intensity_map[annotation['staining_intensity'].lower()]
        staining_location = location_map[annotation['staining_location'].lower()]
        staining_quantity = quantity_map[annotation['staining_quantity'].lower()]
        # tissue knowledge
        malignancy = malignancy_map[tissue_folder]
        # Convert tissue_label to one-hot encoded vector
        tissue_label = tissue_list.index(tissue_name)
        tissue_one_hot = [0] * len(tissue_list)
        tissue_one_hot[tissue_label] = 1
        # cell type knowledge
        cell_type_label = cell_type_list.index(cell_type)
        cell_type_one_hot = [0] * len(cell_type_list)
        cell_type_one_hot[cell_type_label] = 1
        
        
        return query_input, caption_output, snomed_text, snomed_code, image_url, staining_intensity, staining_location, staining_quantity, malignancy, tissue_one_hot, cell_type_one_hot

class HPADatasetMIL(HPADatasetBase):
    def __init__(self, hpa11m_data, datadir, data_split="train", patch_size=224, processor=None, cache_dir=None, num_workers=4):
        super().__init__(hpa11m_data, datadir, data_split, processor, cache_dir, num_workers)
        self.patch_size = patch_size

    def __getitem__(self, idx):
        name = self.hpa11m_data.loc[idx]["name"]
        # tissue_folder = self.hpa11m_data.loc[idx]["tissue_folder"]
        if name.startswith('tissue/'):
            tissue_folder = 'normal'
        elif name.startswith('pathology/'):
            tissue_folder = 'cancer'
        else:
            tissue_folder = self.hpa11m_data.loc[idx]["tissue_folder"]
            
        tar_filename = os.path.join(self.datadir, self.hpa11m_data.loc[idx]["tar_filename"])
        
        use_tar = True
        if use_tar:
            try:
                image, custom_metadata = self._load_image_and_metadata(name, tar_filename)
            except Exception as e:
                print(f"[CorruptionError] Skipping file {name} in {tar_filename}, error: {e}")
                print(f"Index: {idx}")
                return None
        else:
            if self.data_split == "train":
                datadir = "/project/zhihuanglab/zhi/HPA-VLM/20241013_prepare_HPA_100K_files/output"
            else:
                datadir = "/project/zhihuanglab/zhi/HPA-VLM/20241013_prepare_HPA_100K_files/output_test"
            image = Image.open(os.path.join(datadir, f'{name.replace("/", "__")}.jpg'))
            with open(os.path.join(datadir, f'{name.replace("/", "__")}.json'), "r") as f:
                custom_metadata = json.load(f)
        cell_type = custom_metadata["cell_type"]
        # tissue_folder = custom_metadata["tissue_folder"]

        # patches = self._crop_into_patches(image, name.replace("/", "_"), rle_mask=custom_metadata["rle_mask"])
        try:
            patches = self._crop_into_patches(image, name.replace("/", "_"), rle_mask=custom_metadata["rle_mask"])
        except Exception as e:
            print(f"[CorruptionError] Skipping file {name} from {tar_filename} during patching, error: {e}")
            print(f"Index: {idx}")
            return None
        """
        If you prefer concurrent processing (e.g., using ProcessPoolExecutor to bypass Python's GIL for CPU-bound tasks),
        you can re-enable this mapping for potential speedup depending on performance benchmarks.
        """
        # processed_patches = list(self.executor.map(self._process_patch, patches))

        """
        Otherwise, process each patch sequentially without concurrency.
        This is the fallback if you choose not to use parallel processing.
        """
        # Instead of concurrent processing, process each patch sequentially
        processed_patches = [self._process_patch(patch) for patch in patches]
        
        # During training, max we can use with the large models is bag size == 100
        if self.data_split == "train":
            if len(processed_patches) > 100:
                processed_patches = random.sample(processed_patches, 100)

        try:
            processed_image = torch.stack(processed_patches)
        except:
            return None
        
        query_input, caption_output, snomed_text, snomed_code, image_url, staining_intensity, staining_location, staining_quantity, malignancy, tissue_one_hot, cell_type_one_hot = self.get_text_input_output(custom_metadata, tissue_folder, cell_type)
        return processed_image, custom_metadata, query_input, caption_output, snomed_text, snomed_code, image_url, staining_intensity, staining_location, staining_quantity, malignancy, tissue_one_hot, cell_type_one_hot

    def _crop_into_patches(self, image, name, rle_mask, plot_figure=False):
        grid_size_x = int(np.ceil(image.size[0] / self.patch_size))
        grid_size_y = int(np.ceil(image.size[1] / self.patch_size))
        patches = []
        mask = self._rle_decode(rle_mask, image.size)

        if plot_figure:
            # Save the mask as a PNG file
            plt.figure(figsize=(10, 10))
            plt.imshow(mask, cmap='gray')
            plt.title("Mask Visualization")
            plt.axis('off')
            plt.savefig(f'/project/zhihuanglab/zhi/HPA-VLM/20240929_new_method/dataset/extracted_example_image_patches/{name}_mask.png')
            plt.close()
            
            fig, axs = plt.subplots(grid_size_y, grid_size_x, figsize=(20, 20))
            fig.suptitle("Patch Grid Visualization", fontsize=16)

        for i in range(grid_size_y):
            for j in range(grid_size_x):
                left = j * self.patch_size
                top = i * self.patch_size
                right = min(left + self.patch_size, image.size[0])
                bottom = min(top + self.patch_size, image.size[1])
                
                if right <= left or bottom <= top:
                    continue
                
                patch = image.crop((left, top, right, bottom))
                
                patch_mask = mask[top:bottom, left:right]
                mask_percentage = np.sum(patch_mask) / (patch_mask.shape[0] * patch_mask.shape[1])
                is_valid = mask_percentage >= 0.1

                if patch.size != (self.patch_size, self.patch_size):
                    patch = self._pad_patch(patch, target_size=self.patch_size, fill='white')
                
                if is_valid:
                    patches.append(patch)
                    
                if plot_figure:
                    axs[i, j].imshow(patch)
                    if not is_valid:
                        axs[i, j].imshow(np.ones_like(patch) * [1, 0, 0], alpha=0.5)
                    axs[i, j].set_title(f"i={i}, j={j}\n{'invalid' if not is_valid else ''}", fontsize=16)
                    axs[i, j].axis('off')


        if plot_figure:
            plt.tight_layout()
            plt.savefig(f'/project/zhihuanglab/zhi/HPA-VLM/20240929_new_method/dataset/extracted_example_image_patches/{name}_patching.png')
            plt.close()

        return patches
        
    
    def _rle_decode(self, rle, shape):
        """
        Decodes a run-length encoded (RLE) string into a 2D mask.
        
        Parameters:
        - rle: String of RLE-encoded values (alternating start positions and run lengths)
        - shape: Tuple (width, height) from image.size
        
        Returns:
        - 2D numpy array representing the decoded mask
        """
        # Convert the RLE string to a list of integers
        rle = [int(x) for x in rle.split()]
        
        # Swap width and height for numpy array shape
        height, width = shape[1], shape[0]
        total_pixels = width * height
        mask = np.zeros(total_pixels, dtype=np.uint8)
        
        for i in range(0, len(rle), 2):
            start = rle[i] - 1  # Convert 1-based index to 0-based
            length = rle[i + 1]
            mask[start:start + length] = 1
        
        mask = mask.reshape(height, width).T
        # Reshape the 1D mask back into a 2D array with the correct shape
        return mask

    


class HPADatasetDownsample(HPADatasetBase):
    def __init__(self,
                hpa11m_data,
                datadir,
                data_split="train",
                target_size=224,
                n_crop=1,
                processor=None,
                cache_dir=None,
                num_workers=4
                ):
        super().__init__(hpa11m_data, datadir, data_split, processor, cache_dir, num_workers)
        self.target_size = target_size
        self.n_crop = n_crop

    def _process_patch(self, patch):
        if self.processor:
            processed_patch = self.processor(images=patch, return_tensors="pt")["pixel_values"].squeeze(0)
            return processed_patch
        else:
            raise ValueError("Processor is not provided.")

    def __getitem__(self, idx):
        name = self.hpa11m_data.loc[idx]["name"]
        tissue_folder = self.hpa11m_data.loc[idx]["tissue_folder"]
        tar_filename = os.path.join(self.datadir, self.hpa11m_data.loc[idx]["tar_filename"])
        
        use_tar = False
        if use_tar:
            image, custom_metadata = self._load_image_and_metadata(name, tar_filename)
        else:
            if self.data_split == "train":
                datadir = "/project/zhihuanglab/zhi/HPA-VLM/20241013_prepare_HPA_100K_files/output"
            else:
                datadir = "/project/zhihuanglab/zhi/HPA-VLM/20241013_prepare_HPA_100K_files/output_test"
            image = Image.open(os.path.join(datadir, f'{name.replace("/", "__")}.jpg'))
            with open(os.path.join(datadir, f'{name.replace("/", "__")}.json'), "r") as f:
                custom_metadata = json.load(f)
        cell_type = custom_metadata["cell_type"]

        # Downsample and crop the image
        processed_images = self._downsample_and_crop_image(image)
        
        # Process the downsampled image(s)
        if self.n_crop == 1:
            processed_images = self._process_patch(processed_images)
        else:
            processed_images = [self._process_patch(img) for img in processed_images]
            processed_images = torch.stack(processed_images)
        
        query_input, caption_output, snomed_text, snomed_code, image_url, staining_intensity, staining_location, staining_quantity, malignancy, tissue_one_hot, cell_type_one_hot = self.get_text_input_output(custom_metadata, tissue_folder, cell_type)
        return processed_images, custom_metadata, query_input, caption_output, snomed_text, snomed_code, image_url, staining_intensity, staining_location, staining_quantity, malignancy, tissue_one_hot, cell_type_one_hot

    def _downsample_and_crop_image(self, image):
        """
        Downsample the input image and crop if necessary based on n_crop.
        """
        if self.n_crop == 1:
            return [self._downsample_image(image, self.target_size)]
        else:
            larger_size = self.target_size * self.n_crop
            downsampled = self._downsample_image(image, larger_size)
            return self._crop_grid(downsampled)

    def _downsample_image(self, image, target_size):
        """
        Downsample the input image to the target size while maintaining aspect ratio.
        """
        width, height = image.size
        aspect_ratio = width / height

        if aspect_ratio > 1:
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(target_size * aspect_ratio)

        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        final_image = Image.new("RGB", (target_size, target_size), "white")

        paste_x = (target_size - new_width) // 2
        paste_y = (target_size - new_height) // 2
        final_image.paste(resized_image, (paste_x, paste_y))

        return final_image

    def _crop_grid(self, image):
        """
        Crop the image into a grid based on n_crop.
        """
        crops = []
        for i in range(self.n_crop):
            for j in range(self.n_crop):
                left = j * self.target_size
                top = i * self.target_size
                right = left + self.target_size
                bottom = top + self.target_size
                crop = image.crop((left, top, right, bottom))
                crops.append(crop)
        return crops

class StanfordTMAInferenceMILDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, image_dir, mask_dir, cell_type="Tumor cells", patch_size=336, processor=None, mask_threshold=0.9):
        """
        Dataset for inference on multiple images.
        
        Args:
            dataframe: DataFrame with image_id column
            image_dir: Directory containing image files
            mask_dir: Directory containing mask files (background=0, tissue=255)
            cell_type: Cell type for all images (string)
            patch_size: Size of patches to extract
            processor: Image processor for transforming patches
            mask_threshold: Threshold for background percentage to skip patches
        """
        super().__init__()
        self.df = dataframe
        self.image_names = self.df["image_id"].tolist()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.processor = processor
        self.cell_type = cell_type
        self.mask_threshold = mask_threshold
        
        # Pre-load cell type mapping
        self.cell_type_list = ['Glandular cells', 'Exocrine glandular cells', 'Tumor cells',
                              'Cholangiocytes', 'Adipocytes', 'Squamous epithelial cells',
                              'Glial cells', 'Cells in endometrial stroma', 'Cells in red pulp',
                              'Alveolar cells', 'Respiratory epithelial cells',
                              'Cells in granular layer', 'Endothelial cells', 'Fibroblasts',
                              'Decidual cells', 'Cells in glomeruli', 'Myocytes',
                              'Cells in seminiferous ducts', 'Hematopoietic cells',
                              'Germinal center cells', 'Cardiomyocytes', 'Urothelial cells',
                              'Trophoblastic cells', 'Smooth muscle cells',
                              'Ovarian stroma cells', 'Follicle cells', 'Epidermal cells',
                              'Chondrocytes', 'Hepatocytes', 'Lymphoid tissue',
                              'Non-germinal center cells', 'Cells in molecular layer',
                              'Keratinocytes', 'Peripheral nerve', 'Cells in tubules',
                              'Neuronal cells', 'Leydig cells', 'Cells in white pulp', 'Langerhans']
        
        # Prepare cell type one-hot encoding (same for all images)
        if self.cell_type in self.cell_type_list:
            self.cell_type_index = self.cell_type_list.index(self.cell_type)
        else:
            # Default to first cell type if not found
            self.cell_type_index = 0
            print(f"Warning: Cell type '{cell_type}' not found in known list. Using default.")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        Process a single image at the given index.
        
        Returns:
            processed_image: Tensor of shape [N, 3, patch_size, patch_size] where N is number of patches
            cell_type_one_hot: One-hot encoded tensor for cell type
            image_path: Path to the original image (for reference)
        """
        image_name_original = self.image_names[idx]
        image_name = image_name_original.replace("/", "_")
        image_path = os.path.join(self.image_dir, f"{image_name}")  
        mask_path = os.path.join(self.mask_dir, f"{image_name.replace('.jpg', '_mask.jpg')}")
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Load mask from file (instead of generating it)
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path).convert("L"))
            else:
                print(f"Warning: Mask file not found at {mask_path}, using fallback mask")
                mask = np.ones((image.height, image.width), dtype=np.uint8) * 255
            
            # Extract patches
            patches = self._crop_into_patches(image, mask)
            
            # Process patches
            processed_patches = []
            for patch in patches:
                if self.processor is None:
                    raise ValueError("No processor provided for patch transformation.")
                
                patch_tensor = self.processor(
                    images=patch, 
                    return_tensors="pt"
                )["pixel_values"].squeeze(0)
                
                processed_patches.append(patch_tensor)
            
            if len(processed_patches) == 0:
                print(f"Warning: No valid patches found in {image_path}")
                return None
            
            # Stack patches into a single tensor
            processed_image = torch.stack(processed_patches)
            
            # Create cell type one-hot encoding
            cell_type_one_hot = torch.zeros(len(self.cell_type_list), dtype=torch.float)
            cell_type_one_hot[self.cell_type_index] = 1.0
            
            return processed_image, cell_type_one_hot, image_name_original
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    def get_original_data(self, image_name):
        """
        Retrieve the original data row for a given image name.
        
        Args:
            image_name (str): The name of the image to look up
            
        Returns:
            dict: A dictionary containing the original data for the image, or None if not found
        """
        # Find the row in the dataframe where 'name' matches image_name
        matching_rows = self.df[self.df["image_id"] == image_name]
        
        if matching_rows.empty:
            return None
        
        # Convert the first matching row to a dictionary
        row_dict = matching_rows.iloc[0].to_dict()
        
        # Add any additional processing if needed
        # For example, you might want to handle specific fields or format the data
        
        return row_dict
    
    def _crop_into_patches(self, image, mask):
        """
        Splits the image into patches of size patch_size.
        Uses mask where 0=background and 255=tissue.
        """
        width, height = image.size
        grid_size_x = int(np.ceil(width / self.patch_size))
        grid_size_y = int(np.ceil(height / self.patch_size))

        patches = []
        total_patches = 0
        skipped_patches = 0

        for i in range(grid_size_y):
            for j in range(grid_size_x):
                left = j * self.patch_size
                top = i * self.patch_size
                right = min(left + self.patch_size, width)
                bottom = min(top + self.patch_size, height)

                if right <= left or bottom <= top:
                    continue

                patch = image.crop((left, top, right, bottom))
                patch_mask = mask[top:bottom, left:right]

                # Calculate the background percentage (where mask == 0)
                background_percentage = np.mean(patch_mask == 0)

                # Ignore patch if background exceeds threshold
                if background_percentage > self.mask_threshold:
                    skipped_patches += 1
                    continue

                # Pad if needed
                if patch.size != (self.patch_size, self.patch_size):
                    patch = self._pad_patch(patch, target_size=self.patch_size, fill="white")

                patches.append(patch)
                total_patches += 1

        print(f"Image patches: {total_patches} kept, {skipped_patches} skipped")
        return patches

    def _pad_patch(self, patch, target_size, fill="white"):
        """
        Pads the patch to target_size x target_size with 'fill' color
        """
        width, height = patch.size
        padding = (0, 0, target_size - width, target_size - height)
        new_patch = ImageOps.expand(patch, border=padding, fill=fill)
        return new_patch

    def collate_fn(self, batch):
        """
        Custom collate function to handle None values and variable number of patches.
        
        Returns:
            processed_images: List of tensors, each of shape [N_i, 3, patch_size, patch_size]
            cell_type_one_hots: Tensor of shape [batch_size, num_cell_types]
            image_paths: List of image paths
        """
        # Filter out None values
        batch = [item for item in batch if item is not None]
        
        if len(batch) == 0:
            return None
        
        # Unpack the batch
        processed_images, cell_type_one_hots, image_paths = zip(*batch)
        
        # Stack cell type one-hot encodings
        cell_type_one_hots = torch.stack(cell_type_one_hots)
        
        return processed_images, cell_type_one_hots, image_paths

class HPADatasetMIL_url(Dataset):
    def __init__(
        self,
        dataframe,
        rle_map_path,
        rle_hdf5_path,
        data_split="train",
        patch_size=224,
        processor=None,
        cache_dir=None,
        num_workers=4
    ):
        """
        One class to handle everything:
          - Accepts a pandas DataFrame with columns:
              ['name', 'tar_filename', 'intensity', 'location', 'quantity', 'tissue',
               'malignancy', 'staining_intensity', 'staining_location',
               'staining_quantity', 'tissue_name', 'cell_type', 'snomed_text',
               'snomed_code', 'gene', 'image_url', 'caption_output']
          - Accepts a pickle file whose keys = 'name' and values = RLE mask strings.
          - data_split: "train" or "test"/"val" to control augmentations.
          - patch_size: Size (in pixels) of each MIL patch.
          - processor: A callable (e.g., HuggingFace feature extractor) to transform PIL -> torch tensor.
          - cache_dir: If provided, uses LMDB to cache images/metadata for faster repeated loading.
          - num_workers: For potential parallel patch processing (if needed).
        """
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.rle_map = json.load(open(rle_map_path, "r"))
        self.rle_hdf5_base_path = rle_hdf5_path
        self.data_split = data_split
        self.patch_size = patch_size
        self.processor = processor
        self.cache_dir = cache_dir
        self.num_workers = num_workers

        # Load RLE dictionary from pickle
        

        # Optional LMDB cache
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.env = lmdb.open(
                os.path.join(self.cache_dir, 'hpa_cache'),
                map_size=1099511627776
            )

        # Executor for concurrency, if you choose to use it in patching
        self.executor = ProcessPoolExecutor(max_workers=self.num_workers)

        # Augmentations if training
        if self.data_split == "train":
            self.augmentations = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
            ])
        else:
            self.augmentations = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row["name"]

        # 'malignancy' column acts as 'tissue_folder': "normal" or "cancer"
        if self.data_split == "train":
            tissue_folder = "normal" if row["malignancy"] == 0 else "cancer"
        else:
            tissue_folder = row["malignancy"].lower()  # e.g. "normal" or "cancer"


        try:
            image, custom_metadata = self._load_image_and_metadata(row)
        except Exception as e:
            print(f"[CorruptionError] Failed to load {name} from {row['image_url']}, error: {e}")
            print(f"Index: {idx}")
            return None

        try:
            rle_name = name.replace("/", "_")
            # print(f"rle_name: {rle_name}")
            hdf5_path = self.rle_map[rle_name]
            rle_mask = self._load_rle_mask(rle_name, hdf5_path)
        except Exception as e:
            print(f"[Missing RLE] No RLE mask for {name}, error: {e}")
            return None

        try:
            patches = self._crop_into_patches(image, name, rle_mask)
        except Exception as e:
            print(f"[CorruptionError] Skipping file {name} during patching, error: {e}")
            print(f"Index: {idx}")
            return None

        processed_patches = [self._process_patch(patch) for patch in patches]
        if len(processed_patches) == 0:
            return None

        if self.data_split == "train":
            if len(processed_patches) > 100:
                processed_patches = random.sample(processed_patches, 100)

        try:
            processed_image = torch.stack(processed_patches)
        except:
            return None

        cell_type = custom_metadata["cell_type"]
        (
            query_input,
            caption_output,
            snomed_text,
            snomed_code,
            image_url,
            staining_intensity,
            staining_location,
            staining_quantity,
            malignancy,
            tissue_one_hot,
            cell_type_one_hot
        ) = self.get_text_input_output(custom_metadata, tissue_folder, cell_type)

        # Return same format as your original code:
        return (
            processed_image,
            custom_metadata,
            query_input,
            caption_output,
            snomed_text,
            snomed_code,
            image_url,
            staining_intensity,
            staining_location,
            staining_quantity,
            malignancy,
            tissue_one_hot,
            cell_type_one_hot
        )

    # ----------------------------------------------------------------------
    #               HELPER METHODS
    # ----------------------------------------------------------------------

    def _load_image_and_metadata(self, row):
        """
        Replaces the old tar-based method. Here, we:
         1) Check if LMDB cache has the image & metadata by 'name'
         2) If not in cache, download the image from row['image_url']
         3) Construct a custom_metadata dict from the row
         4) Optionally store both in LMDB
         5) Return (image, custom_metadata)
        """
        name = row["name"]
        image_url = row["image_url"]

        # 1) Try loading from LMDB cache
        if self.cache_dir:
            with self.env.begin(write=False) as txn:
                image_data = txn.get(name.encode())
                metadata_data = txn.get((name + "_metadata").encode())

                if image_data and metadata_data:
                    image = Image.open(BytesIO(image_data)).convert("RGB")
                    custom_metadata = json.loads(metadata_data)
                    return image, custom_metadata

        # 2) Otherwise, download
        image_url = image_url.replace('http:', 'https:')
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")

        # 3) Build custom metadata from DataFrame row
        custom_metadata = {
            "name": row["name"],
            "intensity": row["intensity"],
            "location": row["location"],
            "quantity": row["quantity"],
            "tissue": row["tissue"],
            "malignancy": row["malignancy"],
            "staining_intensity": row["staining_intensity"],
            "staining_location": row["staining_location"],
            "staining_quantity": row["staining_quantity"],
            "tissue_name": row["tissue_name"],
            "cell_type": row["cell_type"],
            "snomed_text": row["snomed_text"],
            "snomed_code": row["snomed_code"],
            "gene": row["gene"],
            "url": row["image_url"],
            "caption_1": row["caption_output"],
        }

        # 4) Optionally store in LMDB
        if self.cache_dir:
            with self.env.begin(write=True) as txn:
                # Image
                img_bytes = BytesIO()
                image.save(img_bytes, format="JPEG")
                txn.put(name.encode(), img_bytes.getvalue())
                # Metadata
                txn.put(
                    (name + "_metadata").encode(),
                    json.dumps(custom_metadata).encode()
                )

        # 5) Return
        return image, custom_metadata

    def _process_patch(self, patch):
        """
        Apply augmentations (if train) + transform to tensor using self.processor
        """
        if self.augmentations is not None:
            patch = self.augmentations(patch)

        if self.processor is None:
            raise ValueError("Processor not provided; cannot process patches.")

        processed = self.processor(images=patch, return_tensors="pt")["pixel_values"].squeeze(0)
        return processed

    def _pad_patch(self, patch, target_size, fill='white'):
        """
        Pads a patch to target_size x target_size with 'fill' color
        """
        width, height = patch.size
        padding = (0, 0, target_size - width, target_size - height)
        return ImageOps.expand(patch, border=padding, fill=fill)

    def _crop_into_patches(self, image, name, rle_mask, plot_figure=False):
        """
        Splits the image into patches of size self.patch_size.
        Each patch is considered valid if at least 10% of it is covered by the mask.
        """
        grid_size_x = int(np.ceil(image.size[0] / self.patch_size))
        grid_size_y = int(np.ceil(image.size[1] / self.patch_size))
        patches = []

        mask = self._rle_decode(rle_mask, image.size)

        for i in range(grid_size_y):
            for j in range(grid_size_x):
                left = j * self.patch_size
                top = i * self.patch_size
                right = min(left + self.patch_size, image.size[0])
                bottom = min(top + self.patch_size, image.size[1])

                if right <= left or bottom <= top:
                    continue

                patch = image.crop((left, top, right, bottom))
                patch_mask = mask[top:bottom, left:right]
                mask_percentage = np.sum(patch_mask) / (patch_mask.shape[0] * patch_mask.shape[1])

                # If smaller than patch_size, pad
                if patch.size != (self.patch_size, self.patch_size):
                    patch = self._pad_patch(patch, target_size=self.patch_size, fill='white')

                # Keep patch only if at least 10% covered by mask
                if mask_percentage >= 0.1:
                    patches.append(patch)

        return patches
    def _load_rle_mask(self, name, hdf5_path):
        """
        Loads the RLE mask from the hdf5 file.
        """
        # image name in hdf5 coverted / to _

        # name = name.replace("/", "_")
        with h5py.File(os.path.join(self.rle_hdf5_base_path, hdf5_path), "r") as f:
            rle_mask = f["masks"][name][()]
            # print(rle_mask)
            rle_string = ' '.join(map(str, rle_mask))
        return rle_string
        
    def _rle_decode(self, rle_str, shape):
        """
        Decodes run-length encoding into a 2D mask.
         - rle_str: "start1 length1 start2 length2 ..."
         - shape: (width, height)
        Returns: mask of shape (height, width).
        """
        rle = [int(x) for x in rle_str.split()]
        width, height = shape[0], shape[1]
        total_pixels = width * height
        mask = np.zeros(total_pixels, dtype=np.uint8)

        for i in range(0, len(rle), 2):
            start = rle[i] - 1  # Convert 1-based index to 0-based
            length = rle[i + 1]
            mask[start : start + length] = 1

        # Reshape into (height, width) and transpose if needed
        mask = mask.reshape((height, width)).T
        return mask

    def get_text_input_output(self, annotation, tissue_folder, cell_type):
        """
        Same method from your original code, used to build queries, caption,
        and multi-hot vectors for intensities, location, quantity, malignancy, tissue, and cell type.
        """
        intensity_map = {
            'negative': [1, 0, 0, 0],
            'weak': [0, 1, 0, 0],
            'moderate': [0, 0, 1, 0],
            'strong': [0, 0, 0, 1]
        }
        location_map = {
            'none': [1, 0, 0, 0],
            'cytoplasmic/membranous': [0, 1, 0, 0],
            'nuclear': [0, 0, 1, 0],
            'cytoplasmic/membranous,nuclear': [0, 0, 0, 1]
        }
        quantity_map = {
            'none': [1, 0, 0, 0],
            '<25%': [0, 1, 0, 0],
            '25%-75%': [0, 0, 1, 0],
            '>75%': [0, 0, 0, 1]
        }
        malignancy_map = {
            'normal': [1, 0],
            'cancer': [0, 1]
        }

        # Tissue & cell type reference lists
        tissue_list = [
            'adipose', 'adrenal gland', 'appendix', 'bone marrow', 'breast',
            'bronchus', 'carcinoid', 'caudate', 'cerebellum',
            'cerebral cortex', 'cervical', 'cervix', 'colon', 'colorectal',
            'duodenum', 'endometrial', 'endometrium', 'epididymis',
            'esophagus', 'fallopian tube', 'gallbladder', 'glioma',
            'head and neck', 'heart muscle', 'hippocampus', 'kidney', 'liver',
            'lung', 'lymph node', 'lymphoma', 'melanoma', 'nasopharynx',
            'oral mucosa', 'ovarian', 'ovary', 'pancreas', 'pancreatic',
            'parathyroid gland', 'placenta', 'prostate', 'rectum', 'renal',
            'salivary gland', 'seminal vesicle', 'skeletal muscle', 'skin',
            'small intestine', 'smooth muscle', 'soft', 'spleen', 'stomach',
            'testis', 'thyroid', 'thyroid gland', 'tonsil', 'urinary bladder',
            'urothelial', 'vagina'
        ]
        cell_type_list = [
            'Glandular cells', 'Exocrine glandular cells', 'Tumor cells',
            'Cholangiocytes', 'Adipocytes', 'Squamous epithelial cells',
            'Glial cells', 'Cells in endometrial stroma', 'Cells in red pulp',
            'Alveolar cells', 'Respiratory epithelial cells',
            'Cells in granular layer', 'Endothelial cells', 'Fibroblasts',
            'Decidual cells', 'Cells in glomeruli', 'Myocytes',
            'Cells in seminiferous ducts', 'Hematopoietic cells',
            'Germinal center cells', 'Cardiomyocytes', 'Urothelial cells',
            'Trophoblastic cells', 'Smooth muscle cells',
            'Ovarian stroma cells', 'Follicle cells', 'Epidermal cells',
            'Chondrocytes', 'Hepatocytes', 'Lymphoid tissue',
            'Non-germinal center cells', 'Cells in molecular layer',
            'Keratinocytes', 'Peripheral nerve', 'Cells in tubules',
            'Neuronal cells', 'Leydig cells', 'Cells in white pulp', 'Langerhans'
        ]

        # Prepare text inputs
        tissue_name = annotation['tissue_name']
        snomed_text = annotation['snomed_text']
        snomed_code = annotation['snomed_code']
        gene = annotation['gene']
        url = annotation['url']
        caption_output = annotation['caption_1']

        query_input = f"{tissue_name}. {snomed_text}. Gene: {gene} for {cell_type.lower()}."

        # Staining knowledge
        stain_intensity = intensity_map[annotation['staining_intensity'].lower()]
        stain_location = location_map[annotation['staining_location'].lower()]
        stain_quantity = quantity_map[annotation['staining_quantity'].lower()]

        # Malignancy knowledge
        malignancy_vec = malignancy_map[tissue_folder]  # 'normal' -> [1, 0], 'cancer' -> [0, 1]

        # Tissue one-hot
        tissue_one_hot = [0] * len(tissue_list)
        if tissue_name in tissue_list:
            idx_tissue = tissue_list.index(tissue_name)
            tissue_one_hot[idx_tissue] = 1

        # Cell type one-hot
        cell_type_one_hot = [0] * len(cell_type_list)
        if cell_type in cell_type_list:
            idx_cell_type = cell_type_list.index(cell_type)
            cell_type_one_hot[idx_cell_type] = 1

        return (
            query_input,
            caption_output,
            snomed_text,
            snomed_code,
            url,
            stain_intensity,
            stain_location,
            stain_quantity,
            malignancy_vec,
            tissue_one_hot,
            cell_type_one_hot
        )