import torch
from pyzstd import ZstdFile
import io

def save_torch_zstd(obj, file_path, compression_level=3):
    """
    Save a PyTorch object to a file using Zstd compression.
    
    :param obj: The PyTorch object to save
    :param file_path: The path to save the compressed file
    :param compression_level: Zstd compression level (1-22, default 3)
    """
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    buffer.seek(0)
    
    with ZstdFile(file_path, mode='wb', level_or_option=compression_level) as f:
        f.write(buffer.getvalue())

def load_torch_zstd(file_path, map_loc):
    """
    Load a PyTorch object from a Zstd compressed file.
    
    :param file_path: The path to the compressed file
    :return: The loaded PyTorch object
    """
    with ZstdFile(file_path, mode='rb') as f:
        buffer = io.BytesIO(f.read())
    
    buffer.seek(0)
    return torch.load(buffer, map_location=map_loc)