# client_file_sender.py
import socket
import os
import time
import pickle
import hashlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configuration
SERVER_IP = 'IP_DE_TU_DEBIAN'
SERVER_PORT = 00
QUEUE_FILE = 'send_queue.pkl'
MONITOR_DIR = '/home/electroscian/Documents/PEUMA/Scian_peuma/BIS'  # Directory to watch
BUFFER_SIZE = 4096  # 4KB chunks

class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            add_to_queue(event.src_path)
            print(f"New file detected: {event.src_path}")

def load_queue():
    """Load queue from file"""
    try:
        with open(QUEUE_FILE, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return []

def save_queue(queue):
    """Save queue to file"""
    with open(QUEUE_FILE, 'wb') as f:
        pickle.dump(queue, f)

def add_to_queue(filepath):
    """Add file to queue if not already present"""
    queue = load_queue()
    if filepath not in queue:
        queue.append(filepath)
        save_queue(queue)

def send_file(filepath):
    """Send entire file to server with verification"""
    try:
        with open(filepath, 'rb') as f:
            file_data = f.read()
        
        file_name = os.path.basename(filepath)
        file_size = os.path.getsize(filepath)
        file_hash = hashlib.md5(file_data).hexdigest()
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(30.0)  # 30 second timeout
            s.connect((SERVER_IP, SERVER_PORT))
            
            # Send metadata (filename and size)
            metadata = f"{file_name}|{file_size}"
            s.sendall(metadata.encode('utf-8'))
            
            # Wait for metadata acknowledgment
            response = s.recv(BUFFER_SIZE)
            if response != b"METADATA_ACK":
                raise Exception("Metadata acknowledgment failed")
            
            # Send file data
            s.sendall(file_data)
            
            # Get final confirmation
            response = s.recv(BUFFER_SIZE).decode('utf-8')
            if response.startswith("SUCCESS"):
                server_hash = response.split('|')[1]
                if server_hash == file_hash:
                    return True
            return False
            
    except Exception as e:
        print(f"Error sending {filepath}: {e}")
        return False

def process_queue():
    """Process the send queue"""
    queue = load_queue()
    if not queue:
        return
    
    successful_sends = []
    
    for filepath in queue:
        if not os.path.exists(filepath):
            print(f"File not found, removing from queue: {filepath}")
            successful_sends.append(filepath)
            continue
            
        try:
            if send_file(filepath):
                successful_sends.append(filepath)
                print(f"File sent successfully: {filepath}")
            else:
                print(f"Failed to send {filepath}, will retry")
                break  # Stop processing to retry later
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    # Update queue by removing successful sends
    if successful_sends:
        queue = load_queue()
        queue = [f for f in queue if f not in successful_sends]
        save_queue(queue)
        
        # Optionally move or delete sent files
        for sent_file in successful_sends:
            try:
                os.remove(sent_file)
            except:
                pass

def start_monitoring():
    """Start directory monitoring"""
    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, MONITOR_DIR, recursive=False)
    observer.start()
    print(f"Monitoring directory: {MONITOR_DIR}")
    
    try:
        while True:
            process_queue()
            time.sleep(5)  # Process queue every 5 seconds
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    # Add any existing files to queue on startup
    initial_files = [os.path.join(MONITOR_DIR, f) for f in os.listdir(MONITOR_DIR) 
                   if os.path.isfile(os.path.join(MONITOR_DIR, f))]
    
    for file in initial_files:
        add_to_queue(file)
    
    start_monitoring()