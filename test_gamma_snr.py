import torch
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt
import numpy as np

def minSNR_weighting(timesteps, noise_scheduler, gamma):
    alphas = noise_scheduler.alphas_cumprod
    sqrt_alphas = torch.sqrt(alphas)
    sqrt_minus_one_alphas = torch.sqrt(1.0 - alphas)

    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    all_snr = (sqrt_alphas / (sqrt_minus_one_alphas + epsilon)) ** 2
    snr = torch.stack([all_snr[t] for t in timesteps])

    # Clip SNR values to avoid extremely large numbers
    snr = torch.clamp(snr, min=1e-8, max=1e8)

    gamma_over_snr = gamma / snr
    snr_weight = torch.minimum(gamma_over_snr, torch.ones_like(gamma_over_snr)).float()

    # Ensure weights are between 0 and 1
    # snr_weight = torch.clamp(snr_weight, min=0.0, max=1.0)

    return snr_weight

# Set up the noise scheduler
hf_model = "CompVis/stable-diffusion-v1-4"
#hf_model = "bghira/terminus-xl-velocity-training"
noise_scheduler = DDPMScheduler.from_pretrained(hf_model, subfolder="scheduler")

# Generate timesteps
num_timesteps = 1000
timesteps = torch.arange(num_timesteps)

# Define gamma values to test
gamma_values = [0.1, 0.5, 1.0, 2.0, 5.0]

# Calculate minSNR weights for each gamma value
# weights = {gamma: minSNR_weighting(timesteps, noise_scheduler, gamma).numpy() for gamma in gamma_values}
weights = {"alphas": [(1-torch.sqrt(alphas)).numpy() for alphas in noise_scheduler.alphas_cumprod]}

# Create the plot
plt.figure(figsize=(12, 8))
# for gamma, weight in weights.items():
#     plt.plot(timesteps.numpy(), weight, label=f'γ = {gamma}')
plt.plot(timesteps.numpy(), weights["alphas"], label="Alphas")

plt.xlabel('Timestep')
plt.ylabel('minSNR Weight')
plt.title(f'minSNR Weighting for Different γ Values for Model: {hf_model}')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig('minsnr_weighting_plot.png', dpi=300)
plt.close()

print("Plot saved as 'minsnr_weighting_plot.png'")