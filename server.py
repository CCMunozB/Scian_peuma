# server_file_receiver.py
import socket
import os
import hashlib
from datetime import datetime

# Configuration
HOST = '0.0.0.0'
PORT = 65432
STORAGE_DIR = "BIS"
BUFFER_SIZE = 4096  # 4KB chunks

def ensure_storage_dir():
    """Create storage directory if it doesn't exist"""
    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)

def save_file(file_data, original_name):
    """Save received file with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{STORAGE_DIR}/{timestamp}_{original_name}"
    
    try:
        with open(filename, 'wb') as f:
            f.write(file_data)
        
        # Verify file integrity
        received_hash = hashlib.md5(file_data).hexdigest()
        return True, received_hash
    except Exception as e:
        print(f"Error saving file: {e}")
        return False, None

def start_server():
    ensure_storage_dir()
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")
        
        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connection from {addr}")
                try:
                    # First receive file metadata (name and size)
                    metadata = conn.recv(BUFFER_SIZE).decode('utf-8')
                    file_name, file_size = metadata.split('|')
                    file_size = int(file_size)
                    
                    # Send acknowledgment for metadata
                    conn.sendall(b"METADATA_ACK")
                    
                    # Receive file data in chunks
                    file_data = bytearray()
                    while len(file_data) < file_size:
                        chunk = conn.recv(min(BUFFER_SIZE, file_size - len(file_data)))
                        if not chunk:
                            break
                        file_data.extend(chunk)
                    
                    # Verify and save file
                    success, file_hash = save_file(file_data, file_name)
                    if success:
                        # Send success response with hash
                        conn.sendall(f"SUCCESS|{file_hash}".encode('utf-8'))
                    else:
                        conn.sendall(b"ERROR")
                        
                except Exception as e:
                    print(f"Connection error: {e}")

if __name__ == "__main__":
    start_server()