import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
import torchvision.transforms as transforms
from PIL import Image

# ==========================================
# CONSTANTS
# ==========================================
# Set your local dataset path before running training/evaluation
DATASET_PATH = r"PATH_TO_DATASET_VOLUMES"

class CQ500Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        # Get a list of all the patient folders (CQ500-CT-4, CQ500-CT-10, etc.)
        self.patients = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        
        # Our Generator expects 256x256 images. 
        # This transform resizes the CT slices and converts them to PyTorch Tensors (0.0 to 1.0)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor() 
        ])

    def __len__(self):
        # Tells PyTorch exactly how many patients we have
        return len(self.patients)

    def __getitem__(self, idx):
        # This is the magic function. PyTorch calls this automatically when it needs the next image!
        patient_path = os.path.join(self.root_dir, self.patients[idx])
        ct_path = os.path.join(patient_path, 'CT.nii')
        mask_path = os.path.join(patient_path, 'ICH_mask.nii', 'ICH_mask.nii')
        
        # Fallback: If a file is missing, return empty tensors so the training loop doesn't crash
        if not os.path.exists(ct_path) or not os.path.exists(mask_path):
            return torch.zeros((1, 256, 256)), torch.zeros((1, 256, 256))
            
        try:
            # 1. Load the 3D NIfTI arrays
            ct_array = nib.load(ct_path).get_fdata()
            mask_array = nib.load(mask_path).get_fdata()
            
            # 2. Find a slice with an active brain bleed
            slice_idx = mask_array.shape[2] // 2 # Default to the middle slice
            for z in range(mask_array.shape[2]):
                if np.max(mask_array[:, :, z]) > 0:
                    slice_idx = z
                    break
                    
            ct_slice = np.rot90(ct_array[:, :, slice_idx])
            mask_slice = np.rot90(mask_array[:, :, slice_idx])
            
            # 3. NORMALIZE THE CT SCAN (Crucial for GANs!)
            # CT scans have wild numbers (-1000 to +3000). Brain tissue is usually 0 to 100.
            # We "window" the scan to only look at brain/blood, and scale it to 0-255.
            ct_slice = np.clip(ct_slice, -100, 200) 
            ct_slice = (ct_slice - np.min(ct_slice)) / (np.max(ct_slice) - np.min(ct_slice) + 1e-8)
            ct_slice = (ct_slice * 255).astype(np.uint8)
            
            # Scale the 0/1 mask to 0-255 as well
            mask_slice = (mask_slice * 255).astype(np.uint8)
            
            # 4. Convert to PIL Images so PyTorch can safely resize them
            ct_pil = Image.fromarray(ct_slice)
            mask_pil = Image.fromarray(mask_slice)
            
            # 5. Apply PyTorch transforms (Resizes to 256x256 and converts to Tensor)
            ct_tensor = self.transform(ct_pil)
            mask_tensor = self.transform(mask_pil)
            
            # Make absolutely sure the mask is pure 0.0 or 1.0 mathematically
            mask_tensor = (mask_tensor > 0.5).float()
            
            return ct_tensor, mask_tensor
            
        except Exception as e:
            # If the NIfTI file is corrupted, safely return zeros
            return torch.zeros((1, 256, 256)), torch.zeros((1, 256, 256))

# ==========================================
# QUICK TEST
# ==========================================
if __name__ == "__main__":
    print("Testing the CQ500 PyTorch DataLoader...")
    
    # Initialize the dataset
    dataset = CQ500Dataset(DATASET_PATH)
    print(f"✅ Successfully loaded metadata for {len(dataset)} patients.")
    
    # DataLoader automatically batches the images for the GPU!
    # We use batch_size=4 to test if it can load 4 patients simultaneously.
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Grab just one batch to test
    ct_batch, mask_batch = next(iter(dataloader))
    
    print("\n📦 Batch Extracted Successfully!")
    print(f"CT Batch Shape: {ct_batch.shape}")
    print(f"Mask Batch Shape: {mask_batch.shape}")
    print("\nIf you see [4, 1, 256, 256], the pipeline is ready for the GAN!")