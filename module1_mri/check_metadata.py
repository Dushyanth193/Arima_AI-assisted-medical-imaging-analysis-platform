"""Check for any CSV/metadata files inside archive.zip or same directory."""
import zipfile
import os

archive_path = r"d:\Nexora-hackathon\archive.zip"

# Check for non-npz files inside archive
with zipfile.ZipFile(archive_path, 'r') as zf:
    all_names = zf.namelist()
    non_npz = [n for n in all_names if not n.endswith('.npz')]
    print(f"Total entries: {len(all_names)}")
    print(f"Non-.npz entries: {len(non_npz)}")
    for n in non_npz:
        print(f"  {n}")

# Check parent directory for CSV/labels
print("\nFiles in d:\\Nexora-hackathon\\:")
for f in os.listdir(r"d:\Nexora-hackathon"):
    fpath = os.path.join(r"d:\Nexora-hackathon", f)
    if os.path.isfile(fpath):
        print(f"  {f}  ({os.path.getsize(fpath)} bytes)")
    else:
        print(f"  {f}/  (directory)")
