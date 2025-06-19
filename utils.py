def cdiv(a, b):
    return (a + b - 1) // b

def format_bytes(bytes, suffix='B'):
    """Format bytes into a human-readable string."""
    if bytes < 0:
        raise ValueError("Bytes cannot be negative")
    elif bytes == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
    i = 0
    while bytes >= 1024 and i < len(units) - 1:
        bytes /= 1024.0
        i += 1
    
    return f"{bytes:.2f} {units[i]}{suffix}"