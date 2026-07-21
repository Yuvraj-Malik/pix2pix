import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np

# Import our custom Engine and Fuel
from model import UNetGenerator, PatchDiscriminator
from dataset import CQ500Dataset, DATASET_PATH

# ==========================================
# STEP 1: Hardware Setup
# ==========================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 Cooking with NVIDIA GPU!")
else:
    device = torch.device("cpu")
    print("🐢 Cooking on CPU (Will be slow!)")

# ==========================================
# STEP 2: Hyperparameters & Data Loading
# ==========================================
BATCH_SIZE = 4
NUM_EPOCHS = 100 
LEARNING_RATE = 0.0001 
PIXEL_LAMBDA = 100     
DICE_LAMBDA = 100  

print("\nLoading Dataset...")
dataset = CQ500Dataset(DATASET_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"✅ Loaded {len(dataset)} training samples.")

# ==========================================
# STEP 3: Initialize Models & Optimizers
# ==========================================
netG = UNetGenerator().to(device)
netD = PatchDiscriminator().to(device)

criterion_GAN = nn.BCEWithLogitsLoss()
criterion_Pixel = nn.BCELoss(reduction='none') 

optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# ==========================================
# STEP 4: The Adversarial Training Loop
# ==========================================
print("\n🔥 Starting the GAN... Nuking the Banana Blobs!")

for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    
    epoch_TP, epoch_FP, epoch_FN, epoch_TN = 0, 0, 0, 0
    total_white_pixels = 0
    
    # ----------------------------------------
    # TRAINING PHASE
    # ----------------------------------------
    netG.train() # Make sure Generator is in training mode
    
    for i, (real_ct, real_mask) in enumerate(dataloader):
        real_ct = real_ct.to(device)
        real_mask = real_mask.to(device)
        
        target_real = torch.ones((real_ct.size(0), 1, 31, 31), device=device) * 0.9
        target_fake = torch.zeros((real_ct.size(0), 1, 31, 31), device=device)
        
        # TRAIN DETECTIVE
        optimizerD.zero_grad()
        pred_real = netD(real_ct, real_mask)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        fake_mask = netG(real_ct)
        pred_fake = netD(real_ct, fake_mask.detach()) 
        loss_D_fake = criterion_GAN(pred_fake, target_fake)
        
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optimizerD.step()

        # TRAIN COUNTERFEITER
        optimizerG.zero_grad()
        pred_fake_for_G = netD(real_ct, fake_mask)
        loss_G_GAN = criterion_GAN(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
        
        pixel_weights = torch.where(real_mask == 1.0, 30.0, 1.0)
        raw_bce = criterion_Pixel(fake_mask, real_mask)
        weighted_bce = torch.mean(raw_bce * pixel_weights)
        loss_G_Pixel = weighted_bce * PIXEL_LAMBDA
        
        smooth_loss = 1e-6
        intersection_loss = (fake_mask * real_mask).sum(dim=(2, 3))
        union_loss = fake_mask.sum(dim=(2, 3)) + real_mask.sum(dim=(2, 3))
        dice_loss_val = 1 - ((2. * intersection_loss + smooth_loss) / (union_loss + smooth_loss))
        loss_G_Dice = dice_loss_val.mean() * DICE_LAMBDA
        
        loss_G = loss_G_GAN + loss_G_Pixel + loss_G_Dice
        loss_G.backward()
        optimizerG.step()
        
        with torch.no_grad():
            gt = (real_mask > 0.5)
            ai = (fake_mask > 0.5)
            epoch_TP += (gt & ai).sum().item()
            epoch_FP += (~gt & ai).sum().item()
            epoch_FN += (gt & ~ai).sum().item()
            epoch_TN += (~gt & ~ai).sum().item()
            total_white_pixels += ai.sum().item()
        
    epoch_time = time.time() - start_time
    
    smooth = 1e-8
    epoch_f1_dice = (2 * epoch_TP) / (2 * epoch_TP + epoch_FP + epoch_FN + smooth)
    epoch_iou = epoch_TP / (epoch_TP + epoch_FP + epoch_FN + smooth)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Time: {epoch_time:.0f}s | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f} | Running Dice: {epoch_f1_dice:.4f} | Running IoU: {epoch_iou:.4f}")

# ==========================================
# STEP 5: FINAL MEAN METRICS SCAN
# ==========================================
print("\n=======================================================")
print("🔬 TRAINING FINISHED. RUNNING FINAL MASTER EVALUATION...")
print("=======================================================")

# Freeze the model so it doesn't try to learn during evaluation
netG.eval() 
final_TP, final_FP, final_FN, final_TN = 0, 0, 0, 0

with torch.no_grad():
    for real_ct, real_mask in dataloader:
        real_ct = real_ct.to(device)
        real_mask = real_mask.to(device)
        
        fake_mask = netG(real_ct)
        
        gt = (real_mask > 0.5)
        ai = (fake_mask > 0.5)
        
        final_TP += (gt & ai).sum().item()
        final_FP += (~gt & ai).sum().item()
        final_FN += (gt & ~ai).sum().item()
        final_TN += (~gt & ~ai).sum().item()

smooth = 1e-8
final_dice = (2 * final_TP) / (2 * final_TP + final_FP + final_FN + smooth)
final_iou = final_TP / (final_TP + final_FP + final_FN + smooth)
final_recall = final_TP / (final_TP + final_FN + smooth)
final_precision = final_TP / (final_TP + final_FP + smooth)
final_efficiency = (final_TP + final_TN) / (final_TP + final_TN + final_FP + final_FN + smooth)

print(f"🏆 FINAL MEAN DICE SCORE (F1): {final_dice:.4f}")
print(f"📊 FINAL MEAN IOU:             {final_iou:.4f}")
print(f"🎯 FINAL PRECISION:            {final_precision:.4f}")
print(f"📈 FINAL RECALL (Sensitivity): {final_recall:.4f}")
print(f"⚙️ FINAL PIXEL EFFICIENCY:     {final_efficiency:.4f}")
print("=======================================================\n")

# ==========================================
# STEP 6: Save Weights
# ==========================================
print("🎉 Saving Brain...")
torch.save(netG.state_dict(), "generator_weights.pth")
print("✅ Saved as 'generator_weights.pth'")