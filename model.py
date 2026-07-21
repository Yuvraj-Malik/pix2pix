import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import our custom engine and fuel
from model import UNetGenerator
from dataset import CQ500Dataset, DATASET_PATH

# ==========================================
# STEP 1: Setup and Load the Brain
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Loading GAN on {device}...")

# Initialize a blank Generator
netG = UNetGenerator().to(device)

# Load the trained weights we just saved!
try:
    netG.load_state_dict(torch.load("generator_weights.pth", map_location=device))
    print("✅ Successfully loaded 'generator_weights.pth'")
except FileNotFoundError:
    print("❌ Could not find the weights file! Did train.py finish running?")
    exit()

# Set it to evaluation mode (turns off things like Dropout that we only use during training)
netG.eval()

# ==========================================
# STEP 2: Load some Test Data
# ==========================================
# We load the dataset again, but this time we just want 3 random patients to look at
dataset = CQ500Dataset(DATASET_PATH)
test_loader = DataLoader(dataset, batch_size=3, shuffle=True)

# Grab exactly one batch
real_ct, real_mask = next(iter(test_loader))
real_ct = real_ct.to(device)
real_mask = real_mask.to(device)

# ==========================================
# STEP 3: Ask the AI to Generate Fake Masks!
# ==========================================
print("🧠 AI is generating masks...")
with torch.no_grad(): # We don't need gradients for testing, this saves memory
    fake_mask = netG(real_ct)

# ==========================================
# STEP 4: The Visual Dashboard
# ==========================================
print("🖼️ Opening Dashboard...")

# Move tensors back to CPU and convert to numpy for matplotlib
real_ct = real_ct.cpu().numpy()
real_mask = real_mask.cpu().numpy()
fake_mask = fake_mask.cpu().numpy()

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle("Pix2Pix GAN: AI Intracranial Hemorrhage Generation", fontsize=16)

for i in range(3):
    # 1. Plot the Raw CT Scan
    axes[i, 0].imshow(real_ct[i, 0], cmap='bone')
    axes[i, 0].set_title(f"Patient {i+1} - Raw CT Scan")
    axes[i, 0].axis('off')
    
    # 2. Plot the Ground Truth (What the Doctor drew)
    axes[i, 1].imshow(real_mask[i, 0], cmap='inferno')
    axes[i, 1].set_title("Ground Truth (Real Mask)")
    axes[i, 1].axis('off')
    
    # 3. Plot the Fake Mask (What our AI generated!)
    # We threshold at 0.5 just to make the mask solid black/white for viewing
    ai_drawing = (fake_mask[i, 0] > 0.5).astype(float)
    axes[i, 2].imshow(ai_drawing, cmap='inferno')
    axes[i, 2].set_title("AI Generated Fake Mask", color='green')
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show()