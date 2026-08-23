"""Inspect the internal structure of .npz files from archive.zip."""
import zipfile
import numpy as np
import io
import os

archive_path = r"d:\Nexora-hackathon\archive.zip"
output_dir = r"d:\Nexora-hackathon\module1_mri\data\archive_samples"
os.makedirs(output_dir, exist_ok=True)

with zipfile.ZipFile(archive_path, 'r') as zf:
    all_names = zf.namelist()
    print(f"Total .npz files in archive: {len(all_names)}")
    
    # Extract and inspect first 3 files
    for i, name in enumerate(all_names[:3]):
        print(f"\n{'='*70}")
        print(f"File {i+1}: {name}")
        print(f"{'='*70}")
        
        with zf.open(name) as f:
            data = np.load(io.BytesIO(f.read()), allow_pickle=True)
            
            print(f"  Keys (arrays) inside .npz: {list(data.keys())}")
            
            for key in data.keys():
                arr = data[key]
                print(f"\n  Key: '{key}'")
                print(f"    Type: {type(arr)}")
                print(f"    dtype: {arr.dtype}")
                print(f"    shape: {arr.shape}")
                print(f"    ndim: {arr.ndim}")
                
                if arr.ndim == 0:
                    # Scalar or pickled object
                    val = arr.item()
                    if isinstance(val, dict):
                        print(f"    Value (dict): keys = {list(val.keys())}")
                        for k, v in val.items():
                            if isinstance(v, np.ndarray):
                                print(f"      '{k}': ndarray shape={v.shape}, dtype={v.dtype}")
                            elif isinstance(v, (list, tuple)):
                                print(f"      '{k}': {type(v).__name__} len={len(v)}, sample={v[:3] if len(v)>3 else v}")
                            elif isinstance(v, str) and len(v) > 100:
                                print(f"      '{k}': str len={len(v)}, preview='{v[:80]}...'")
                            else:
                                print(f"      '{k}': {v}")
                    else:
                        print(f"    Value: {val}")
                elif arr.ndim >= 2:
                    print(f"    min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
                    if arr.ndim == 3:
                        print(f"    Looks like 3D volume: Z={arr.shape[0]}, Y={arr.shape[1]}, X={arr.shape[2]}")
                    elif arr.ndim == 2:
                        print(f"    Looks like 2D image: H={arr.shape[0]}, W={arr.shape[1]}")
                else:
                    print(f"    First 10 values: {arr[:10]}")
        
        # Save one sample to disk for inspection
        if i == 0:
            with zf.open(name) as f:
                sample_path = os.path.join(output_dir, name)
                with open(sample_path, 'wb') as out_f:
                    out_f.write(f.read())
                print(f"\n  Saved sample to: {sample_path}")

print("\n\nDone inspecting archive.zip contents.")
