
import torch

# --------------- Check CUDA ---------------
if not torch.cuda.is_available():
    print("❌ CUDA is not available. This API requires a GPU.")
    raise RuntimeError("CUDA is not available. This API requires a GPU.")

# --------------- Load model ---------------
print("🚀 Loading CSM model onto GPU...")
#generator = load_csm_1b(device="cuda")
print("✅ Model loaded.")