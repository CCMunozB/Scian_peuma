import socket
import os
import json
from datetime import datetime

# Configuración
HOST = '0.0.0.0'
PORT = 000  # Cambia a un puerto adecuado
STORAGE_DIR = "BIS"

def ensure_storage_dir():
    """Crea el directorio de almacenamiento si no existe"""
    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)

def save_data(data):
    """Guarda los datos en un archivo con timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{STORAGE_DIR}/data_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "data": data.decode('utf-8')
            }, f)
        return True
    except Exception as e:
        print(f"Error guardando datos: {e}")
        return False

def start_server():
    ensure_storage_dir()
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print(f"Servidor escuchando en {HOST}:{PORT}")
        
        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Conexión establecida desde {addr}")
                try:
                    data = conn.recv(1024)
                    if data:
                        if save_data(data):
                            conn.sendall(b"ACK")  # Confirmación de recepción
                        else:
                            conn.sendall(b"ERROR")  # Error en almacenamiento
                except Exception as e:
                    print(f"Error en conexión: {e}")

if __name__ == "__main__":
    start_server()