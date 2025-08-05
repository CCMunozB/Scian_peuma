# client_queue_monitor.py
import socket
import time
import os
import json
import pickle
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configuración
SERVER_IP = 'IP_DE_TU_DEBIAN'
SERVER_PORT = 000  # Cambia a un puerto adecuado
QUEUE_FILE = 'send_queue.pkl'
MONITOR_DIR = 'path/to/monitor'  # Directorio a monitorear

class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            add_to_queue(event.src_path)
            print(f"Nuevo archivo detectado: {event.src_path}")

def load_queue():
    """Carga la cola desde archivo"""
    try:
        with open(QUEUE_FILE, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return []

def save_queue(queue):
    """Guarda la cola en archivo"""
    with open(QUEUE_FILE, 'wb') as f:
        pickle.dump(queue, f)

def add_to_queue(filepath):
    """Añade un archivo a la cola"""
    queue = load_queue()
    if filepath not in queue:
        queue.append(filepath)
        save_queue(queue)

def process_queue():
    """Procesa la cola de envío"""
    queue = load_queue()
    if not queue:
        return
    
    successful_sends = []
    
    for filepath in queue:
        try:
            with open(filepath, 'r') as f:
                data = f.read()
            
            if send_data(data.encode('utf-8')):
                successful_sends.append(filepath)
                print(f"Archivo {filepath} enviado con éxito")
            else:
                print(f"Error al enviar {filepath}, se reintentará más tarde")
                break  # Detiene el procesamiento para reintentar luego
            
        except Exception as e:
            print(f"Error procesando {filepath}: {e}")
    
    # Actualiza la cola removiendo los exitosos
    if successful_sends:
        queue = load_queue()
        queue = [f for f in queue if f not in successful_sends]
        save_queue(queue)

def send_data(data):
    """Envía datos al servidor y verifica confirmación"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(10.0)  # Timeout de 10 segundos
            s.connect((SERVER_IP, SERVER_PORT))
            s.sendall(data)
            
            # Espera confirmación
            response = s.recv(1024)
            return response == b"ACK"
    except Exception as e:
        print(f"Error en conexión: {e}")
        return False

def start_monitoring():
    """Inicia el monitoreo del directorio"""
    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, MONITOR_DIR, recursive=False)
    observer.start()
    print(f"Monitoreando directorio {MONITOR_DIR}")
    
    try:
        while True:
            process_queue()
            time.sleep(5)  # Procesa la cola cada 5 segundos
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    # Verifica si hay archivos pendientes al iniciar
    initial_files = [os.path.join(MONITOR_DIR, f) for f in os.listdir(MONITOR_DIR) 
                    if os.path.isfile(os.path.join(MONITOR_DIR, f))]
    
    for file in initial_files:
        add_to_queue(file)
    
    start_monitoring()