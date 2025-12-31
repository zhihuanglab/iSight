import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['NCCL_TIMEOUT'] = '120'
import torch
import torch.nn as nn
from dataset.hpadataset import HPADatasetMIL, HPADatasetDownsample, HPADatasetMIL_url, StanfordTMAInferenceMILDataset
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import numpy as np
import time
import configparser
import argparse
import copy
from datetime import datetime
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score
import torch.cuda.amp  # Import Automatic Mixed Precision
import re
import random
import json

"""
Be sure to init conda env:
conda activate hpa

Note: If running on a non-H100 GPU, there might be an error for pyarrow.
Reinstall pyarrow with following command will resolve the issue:
conda install -c conda-forge pyarrow
"""

# Define loss functions for evaluation
criterion_intensity = nn.CrossEntropyLoss()  # For intensity
criterion_location = nn.CrossEntropyLoss()   # For location
criterion_quantity = nn.CrossEntropyLoss()   # For quantity
criterion_tissue = nn.CrossEntropyLoss()     # For tissue type (58 classes)
criterion_malignancy = nn.CrossEntropyLoss() # For tumor vs non-tumor (binary)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def custom_collate_fn(batch):
    # If no valid samples, return None
    batch_valid = [batch_item for batch_item in batch if batch_item is not None]
    
    if len(batch_valid) == 0:
        return None
    images, metadata, query_input, caption_output, snomed_text, snomed_code, image_url, staining_intensity, staining_location, staining_quantity, malignancy, tissue_one_hot, cell_type_one_hot = zip(*batch_valid)
    # Note: here images is a list of (N, 3, 224, 224), where N is the number of patches per image, and is different for each image.
    
    # Convert to tensors and stack
    staining_intensity = torch.stack([torch.tensor(sublist, dtype=torch.float) for sublist in staining_intensity])
    staining_location = torch.stack([torch.tensor(sublist, dtype=torch.float) for sublist in staining_location])
    staining_quantity = torch.stack([torch.tensor(sublist, dtype=torch.float) for sublist in staining_quantity])
    malignancy = torch.stack([torch.tensor(sublist, dtype=torch.float) for sublist in malignancy])
    tissue_one_hot = torch.stack([torch.tensor(sublist, dtype=torch.float) for sublist in tissue_one_hot])
    cell_type_one_hot = torch.stack([torch.tensor(sublist, dtype=torch.float) for sublist in cell_type_one_hot])
    
    return images, metadata, query_input, caption_output, snomed_text, snomed_code, image_url, staining_intensity, staining_location, staining_quantity, malignancy, tissue_one_hot, cell_type_one_hot


def run_inference(config):
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize model based on configuration
    base_model_name = config.base_model_name

    from model.patch_encoder_with_clam import create_clam_vit
    model, _, _ = create_clam_vit(
        base_model_name=base_model_name,
        training_config=config,
        device=device,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        freeze_vit=config.freeze_vit,
        freeze_query_features_encoder=config.freeze_query_features_encoder,
        use_cell_type_embedding=config.use_cell_type_embedding
    )

    model.to(device)

    # Determine patch size based on model
    if base_model_name == "vinid/plip":
        patch_size = 224
    elif base_model_name == "openai/clip-vit-large-patch14-336":
        patch_size = 336
    else:
        raise ValueError(f"Model {base_model_name} not supported.")

    processor = model.patch_processor

    # Load checkpoint
    if config.checkpoint_path:
        print(f"Loading checkpoint from {config.checkpoint_path}")
        checkpoint = torch.load(config.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully")
    else:
        raise ValueError("No checkpoint path provided. Please specify a checkpoint path.")

    # Load inference data
    print(f"Loading inference data from {config.inference_data_path}")
    if config.inference_data_path.endswith('.csv'):
        inference_data = pd.read_csv(config.inference_data_path)
    elif config.inference_data_path.endswith('.feather'):
        inference_data = pd.read_feather(config.inference_data_path)
    else:
        raise ValueError("Inference data must be a CSV or Feather file")
    
    print(f"Loaded {len(inference_data)} samples for inference")


    # Setup dataset and dataloader
    hdf5_base_dir = config.hdf5_base_dir
    rle_map_path = config.rle_map_path

    if config.model_version == "simple_downsample":
        dataset = HPADatasetDownsample(inference_data, None, data_split="test", target_size=patch_size, processor=processor)
    else:
        dataset = HPADatasetMIL_url(inference_data, rle_map_path, hdf5_base_dir, data_split="test", patch_size=patch_size, processor=processor)

    # Add config to dataset for accessing save_logits parameter
    dataset.config = config

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=False,
        prefetch_factor=8,
        persistent_workers=True
    )

    # Run inference
    results, metrics, metrics_df = run_model_inference(model, dataloader, device, config.output_dir)
    
    # Save results
    save_name = config.save_name if config.save_name else datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(config.output_dir, f"inference_results_{save_name}.csv")
    results.to_csv(output_file, index=False)
    print(f"Inference results saved to {output_file}")
    
    # Save metrics if available
    if metrics:
        with open(os.path.join(config.output_dir, f'metrics_{save_name}.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        metrics_df.to_csv(os.path.join(config.output_dir, f'metrics_{save_name}.csv'), index=False)
        print(f"Metrics saved to {config.output_dir}/metrics_{save_name}.csv")

    # Generate visualizations if requested
    if config.generate_visualizations:
        generate_visualizations(results, config.output_dir, save_name)
        print(f"Visualizations saved to {config.output_dir}")


def run_model_inference(model, dataloader, device, output_dir):
    model.eval()
    
    # Lists to store all outputs and metadata
    all_outputs = {
        'intensity': [],
        'location': [],
        'quantity': [],
        'tissue': [],
        'malignancy': []
    }
    
    all_predicted_classes = {
        'intensity': [],
        'location': [],
        'quantity': [],
        'tissue': [],
        'malignancy': []
    }
    
    all_ground_truth = {
        'intensity': [],
        'location': [],
        'quantity': [],
        'tissue': [],
        'malignancy': []
    }
    
    all_image_urls = []
    all_names = []
    all_metadata = []
    all_snomed_text = []
    all_snomed_code = []
    
    # Set deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            if batch is None:
                continue
                
            patches, metadata, query_input, caption_output, snomed_text, snomed_code, image_url, staining_intensity, staining_location, staining_quantity, malignancy, tissue_one_hot, cell_type_one_hot = batch
            
            # Extract names from metadata
            names = [meta.get('name', 'unknown') for meta in metadata]
            
            # Move data to device
            patches_device = [patch.to(device) for patch in patches]
            staining_intensity = staining_intensity.to(device)
            staining_location = staining_location.to(device)
            staining_quantity = staining_quantity.to(device)
            tissue_one_hot = tissue_one_hot.to(device)
            malignancy = malignancy.to(device)
            
            # Process each sample in the batch individually to ensure consistency
            for i in range(len(names)):
                try:
                    # Debug the shape
                    patch = patches_device[i]
                    
                    # Ensure patch has the right shape (N, 3, H, W)
                    if len(patch.shape) != 4:
                        print(f"Warning: Patch shape is {patch.shape}, expected 4D tensor")
                        continue
                    
                    # Mixed precision inference
                    with torch.cuda.amp.autocast():
                        intensity_out, location_out, quantity_out, tissue_out, malignancy_out, A_raw = model(
                            [patch], [query_input[i]], 
                            cell_type_one_hot[i:i+1].to(device) if cell_type_one_hot is not None else None, 
                            phase="test"
                        )
                    
                    # Store outputs
                    all_outputs['intensity'].append(intensity_out.cpu().numpy()[0])
                    all_outputs['location'].append(location_out.cpu().numpy()[0])
                    all_outputs['quantity'].append(quantity_out.cpu().numpy()[0])
                    all_outputs['tissue'].append(tissue_out.cpu().numpy()[0])
                    all_outputs['malignancy'].append(malignancy_out.cpu().numpy()[0])
                    
                    # Store predicted classes
                    all_predicted_classes['intensity'].append(torch.argmax(intensity_out, dim=1).cpu().numpy()[0])
                    all_predicted_classes['location'].append(torch.argmax(location_out, dim=1).cpu().numpy()[0])
                    all_predicted_classes['quantity'].append(torch.argmax(quantity_out, dim=1).cpu().numpy()[0])
                    all_predicted_classes['tissue'].append(torch.argmax(tissue_out, dim=1).cpu().numpy()[0])
                    all_predicted_classes['malignancy'].append(torch.argmax(malignancy_out, dim=1).cpu().numpy()[0])
                    
                    # Store ground truth if available
                    all_ground_truth['intensity'].append(staining_intensity[i].cpu().numpy())
                    all_ground_truth['location'].append(staining_location[i].cpu().numpy())
                    all_ground_truth['quantity'].append(staining_quantity[i].cpu().numpy())
                    all_ground_truth['tissue'].append(tissue_one_hot[i].cpu().numpy())
                    all_ground_truth['malignancy'].append(malignancy[i].cpu().numpy())
                    
                    # Store metadata
                    all_image_urls.append(image_url[i])
                    all_names.append(names[i])
                    all_metadata.append(metadata[i])
                    all_snomed_text.append(snomed_text[i])
                    all_snomed_code.append(snomed_code[i])
                except Exception as e:
                    print(f"Error processing sample {names[i]}: {e}")
                    print(f"Patch shape: {patches[i].shape}")
                    continue
    
    # Create results dataframe with metadata
    results_df = pd.DataFrame({
        'name': all_names,
        'image_url': all_image_urls,
    })
    
    # Add all metadata fields
    for i, meta in enumerate(all_metadata):
        for key, value in meta.items():
            if key not in results_df.columns:
                results_df[key] = [None] * len(all_metadata)
            results_df.at[i, key] = value
    
    # Add SNOMED information
    results_df['snomed_text'] = all_snomed_text
    results_df['snomed_code'] = all_snomed_code
    
    # Map numeric predictions to human-readable labels
    intensity_labels = ['negative', 'weak', 'moderate', 'strong']
    location_labels = ['none', 'cytoplasmic/membranous', 'nuclear', 'cytoplasmic/membranous,nuclear']
    quantity_labels = ['none', '<25%', '25%-75%', '>75%']
    malignancy_labels = ['normal', 'cancer']
    
    # Load tissue labels
    tissue_labels = ['adipose', 'adrenal gland', 'appendix', 'bone marrow', 'breast',
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
                    'urothelial', 'vagina']
    
    # Add human-readable labels only (no class indices)
    results_df['intensity_pred'] = [intensity_labels[i] for i in all_predicted_classes['intensity']]
    results_df['location_pred'] = [location_labels[i] for i in all_predicted_classes['location']]
    results_df['quantity_pred'] = [quantity_labels[i] for i in all_predicted_classes['quantity']]
    results_df['tissue_pred'] = [tissue_labels[i] if i < len(tissue_labels) else f"unknown_{i}" for i in all_predicted_classes['tissue']]
    results_df['malignancy_pred'] = [malignancy_labels[i] for i in all_predicted_classes['malignancy']]
    
    # Add raw logits if requested
    if hasattr(dataloader.dataset, 'config') and dataloader.dataset.config.save_logits:
        for task in ['intensity', 'location', 'quantity', 'tissue', 'malignancy']:
            for i, logits in enumerate(all_outputs[task]):
                for j, logit in enumerate(logits):
                    results_df.at[i, f'{task}_logit_{j}'] = logit
    
    # Add ground truth labels if available (no class indices)
    if all(len(gt) > 0 for gt in all_ground_truth.values()):
        # Convert one-hot encoded ground truth to class indices
        for task in ['intensity', 'location', 'quantity', 'tissue', 'malignancy']:
            gt_indices = [np.argmax(gt) for gt in all_ground_truth[task]]
            
            # Add human-readable ground truth labels only
            if task == 'intensity':
                results_df[f'{task}_true'] = [intensity_labels[i] for i in gt_indices]
            elif task == 'location':
                results_df[f'{task}_true'] = [location_labels[i] for i in gt_indices]
            elif task == 'quantity':
                results_df[f'{task}_true'] = [quantity_labels[i] for i in gt_indices]
            elif task == 'tissue':
                results_df[f'{task}_true'] = [tissue_labels[i] if i < len(tissue_labels) else f"unknown_{i}" for i in gt_indices]
            elif task == 'malignancy':
                results_df[f'{task}_true'] = [malignancy_labels[i] for i in gt_indices]
    
    # Calculate and print metrics if ground truth is available
    metrics = {}
    metrics_df = pd.DataFrame()
    
    if all(len(gt) > 0 for gt in all_ground_truth.values()):
        print("\nPerformance Metrics:")
        print("=" * 50)
        
        for task in ['intensity', 'location', 'quantity', 'tissue', 'malignancy']:
            gt_indices = [np.argmax(gt) for gt in all_ground_truth[task]]
            y_true = np.array(gt_indices)
            y_pred = np.array(all_predicted_classes[task])
            
            # Get unique classes in both true and predicted
            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
            
            # Calculate metrics
            try:
                accuracy = accuracy_score(y_true, y_pred)
                balanced_acc = balanced_accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='weighted', labels=np.unique(y_true), zero_division=0)
                
                # Store metrics
                metrics[task] = {
                    'accuracy': float(accuracy),  # Convert to float for JSON serialization
                    'balanced_accuracy': float(balanced_acc),
                    'f1_score': float(f1),
                    'num_classes_true': int(len(np.unique(y_true))),
                    'num_classes_pred': int(len(np.unique(y_pred))),
                    'num_classes_total': int(len(unique_classes))
                }
                
                # Add to metrics dataframe
                metrics_df = pd.concat([metrics_df, pd.DataFrame({
                    'task': [task],
                    'accuracy': [accuracy],
                    'balanced_accuracy': [balanced_acc],
                    'f1_score': [f1],
                    'num_classes_true': [len(np.unique(y_true))],
                    'num_classes_pred': [len(np.unique(y_pred))],
                    'num_classes_total': [len(unique_classes)]
                })], ignore_index=True)
                
                # Print metrics
                print(f"\n{task.capitalize()} Metrics:")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Balanced Accuracy: {balanced_acc:.4f}")
                print(f"  Weighted F1 Score: {f1:.4f}")
                print(f"  Classes in ground truth: {len(np.unique(y_true))}")
                print(f"  Classes in predictions: {len(np.unique(y_pred))}")
                print(f"  Total unique classes: {len(unique_classes)}")
                
                # Print warning if there are classes in predictions not in ground truth
                if len(np.unique(y_pred)) > len(np.unique(y_true)):
                    extra_classes = set(y_pred) - set(y_true)
                    print(f"  Warning: Predictions contain {len(extra_classes)} classes not in ground truth")
                
            except Exception as e:
                print(f"Error calculating metrics for {task}: {e}")
                metrics[task] = {
                    'error': str(e)
                }
    
    return results_df, metrics, metrics_df


def generate_visualizations(results_df, output_dir, save_name):
    """Generate visualizations from inference results"""
    os.makedirs(os.path.join(output_dir, 'visualizations', save_name), exist_ok=True)
    
    # Extract ground truth and predictions
    tasks = ['intensity', 'location', 'quantity', 'tissue', 'malignancy']
    
    # Check if ground truth is available
    has_ground_truth = all(f'{task}_true' in results_df.columns for task in tasks)
    
    if has_ground_truth:
        # Generate confusion matrices
        for task in tasks:
            # Define ordered class names for each task
            if task == 'intensity':
                class_names = ['negative', 'weak', 'moderate', 'strong']
            elif task == 'location':
                class_names = ['none', 'nuclear', 'cytoplasmic/membranous', 'cytoplasmic/membranous,nuclear']
            elif task == 'quantity':
                class_names = ['none', '<25%', '25%-75%', '>75%']
            elif task == 'malignancy':
                class_names = ['normal', 'cancer']
            elif task == 'tissue':
                # For tissue, handle separately due to large number of classes
                tissue_labels = ['adipose', 'adrenal gland', 'appendix', 'bone marrow', 'breast',
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
                                'urothelial', 'vagina']
                
                # For tissue, we'll create a confusion matrix for the top N most common classes
                # to keep the visualization manageable
                top_n = 15
                
                # Get unique tissue types in the data
                unique_tissues_true = results_df[f'{task}_true'].unique()
                unique_tissues_pred = results_df[f'{task}_pred'].unique()
                unique_tissues = np.unique(np.concatenate([unique_tissues_true, unique_tissues_pred]))
                
                # Count occurrences of each tissue type
                tissue_counts = {}
                for tissue in unique_tissues:
                    count_true = np.sum(results_df[f'{task}_true'] == tissue)
                    count_pred = np.sum(results_df[f'{task}_pred'] == tissue)
                    tissue_counts[tissue] = count_true + count_pred
                
                # Get top N tissues by occurrence
                top_tissues = sorted(tissue_counts.keys(), key=lambda x: tissue_counts[x], reverse=True)[:top_n]
                
                # Filter data to only include top tissues
                mask = (results_df[f'{task}_true'].isin(top_tissues)) & (results_df[f'{task}_pred'].isin(top_tissues))
                filtered_df = results_df[mask]
                
                # Create confusion matrix
                cm = confusion_matrix(
                    filtered_df[f'{task}_true'], 
                    filtered_df[f'{task}_pred'],
                    labels=top_tissues
                )
                
                plt.figure(figsize=(15, 15))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=top_tissues, yticklabels=top_tissues)
                plt.title(f'Confusion Matrix - Top {top_n} {task.capitalize()} Classes')
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.xticks(rotation=90, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'visualizations', save_name, f'{task}_confusion_matrix_{save_name}.png'))
                plt.close()
                continue
            
            # Create mapping from label to index for consistent ordering
            label_to_idx = {label: i for i, label in enumerate(class_names)}
            
            # Convert string labels to indices based on our defined order
            y_true = np.array([label_to_idx[label] for label in results_df[f'{task}_true']])
            y_pred = np.array([label_to_idx[label] for label in results_df[f'{task}_pred']])
            
            # Create confusion matrix with ordered classes
            cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
            
            # Normalize confusion matrix for better visualization
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm_normalized = np.nan_to_num(cm_normalized, nan=0)
            
            # Plot raw counts
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Confusion Matrix (Counts) - {task.capitalize()}')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'visualizations', save_name, f'{task}_confusion_matrix_counts_{save_name}.png'))
            plt.close()
            
            # Plot normalized (percentage)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Confusion Matrix (Normalized) - {task.capitalize()}')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'visualizations', save_name, f'{task}_confusion_matrix_normalized_{save_name}.png'))
            plt.close()


def parse_config():
    parser = argparse.ArgumentParser(description='Run inference with HPA-VLM model')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--inference_data_path', type=str, required=True, help='Path to the inference data CSV')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Directory to save inference results')
    parser.add_argument('--hdf5_base_dir', type=str, required=True, help='Base directory for HDF5 files')
    parser.add_argument('--rle_map_path', type=str, required=True, help='Path to RLE map JSON file')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--generate_visualizations', action='store_true', help='Generate visualizations of results')
    parser.add_argument('--save_name', type=str, default='', help='Suffix for saved files (instead of timestamp)')
    parser.add_argument('--save_logits', action='store_true', help='Save raw logits in the output')
    
    args = parser.parse_args()
    
    # Load config file
    config = configparser.ConfigParser()
    config.read(args.config)
    
    # Convert config values to desired data types
    def convert_value(convert_func, key, default=None):
        try:
            return convert_func(config['DEFAULT'][key])
        except (KeyError, ValueError):
            if default is not None:
                return default
            raise ValueError(f"Error converting {key}. Please check your config file.")
    
    # Create a dictionary to hold the converted values
    converted_config = {
        'base_model_name': config['DEFAULT']['base_model_name'],
        'freeze_vit': convert_value(lambda x: x.lower() == 'true', 'freeze_vit', True),
        'freeze_query_features_encoder': convert_value(lambda x: x.lower() == 'true', 'freeze_query_features_encoder', True),
        'learning_rate': convert_value(float, 'learning_rate', 1e-4),
        'weight_decay': convert_value(float, 'weight_decay', 1e-5),
        'use_cell_type_embedding': convert_value(lambda x: x.lower() == 'true', 'use_cell_type_embedding', False),
        'model_version': config['DEFAULT']['model_version'],
        'inference_data_path': args.inference_data_path,
        'checkpoint_path': args.checkpoint_path,
        'output_dir': args.output_dir,
        'hdf5_base_dir': args.hdf5_base_dir,
        'rle_map_path': args.rle_map_path,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'generate_visualizations': args.generate_visualizations,
        'save_name': args.save_name,
        'save_logits': args.save_logits,
    }
    
    return argparse.Namespace(**converted_config)


if __name__ == "__main__":
    config = parse_config()
    set_seed(1001)  # Set seed for reproducibility
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save configuration
    config_dict = vars(config)
    with open(os.path.join(config.output_dir, 'inference_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    # Run inference
    run_inference(config)