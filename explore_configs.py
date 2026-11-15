import torch
import time
from para import LightweightMBT2018Mean   

 
experimental_configs = [
    {"N": 64,  "M": 96},
    {"N": 64,  "M": 128},
    {"N": 80,  "M": 128},
    {"N": 96,  "M": 160},
    {"N": 100, "M": 150},
    {"N": 128, "M": 192},
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== Starting Architecture Search for Lightweight Codec ===")
print(f"Using device: {device}\n")

results = []

for cfg in experimental_configs:
    N = cfg["N"]
    M = cfg["M"]
    
    try:
        start_time = time.time()
        model = LightweightMBT2018Mean(N=N, M=M).to(device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        x = torch.randn(1, 3, 256, 256).to(device)
        with torch.no_grad():
            out = model(x)
            
        elapsed = time.time() - start_time
        
        params_str = f"{total_params / 1e6:.3f}M"
        results.append({
            "N": N, "M": M, 
            "Params": params_str, 
            "Status": "SUCCESS",
            "Time (s)": round(elapsed, 2)
        })
        print(f"[OK] N={N}, M={M} --> Parameters: {params_str}")
        
    except Exception as e:
        results.append({
            "N": N, "M": M, 
            "Params": "FAILED", 
            "Status": str(e),
            "Time (s)": 0
        })
        print(f"[ERROR] N={N}, M={M} failed with error: {e}")


print("\n" + "="*50)
print(f"{'N':<6} | {'M':<6} | {'Params':<15} | {'Status':<10}")
print("="*50)
for r in results:
    p_val = r["Params"]
    s_val = r["Status"]
    print(f"{r['N']:<6} | {r['M']:<6} | {p_val:<15} | {s_val:<10}")
print("="*50)