import serial
import struct
import time
from datetime import datetime
from collections import deque

class BISMonitor:
    def __init__(self, port='/dev/ttyUSB0'):
        self.ser = serial.Serial(
            port=port,
            baudrate=57600,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=0.1
        )
        self.sequence_number = 0
        self.pending_acks = deque()
        self.last_received_seq = -1

    def calculate_checksum(self, data):
        return sum(data) & 0xFFFF

    def send_ack(self, seq):
        """Send ACK for received sequence number"""
        packet = struct.pack('<HHHHH', 
            0xABBA,        # Start
            seq,           # Sequence number
            0,             # Data length
            2,             # ACK directive
            self.calculate_checksum(struct.pack('<HHH', seq, 0, 2))
        )
        self.ser.write(packet)

    def send_nak(self, seq):
        """Send NAK for received sequence number"""
        packet = struct.pack('<HHHHH', 
            0xABBA,        # Start
            seq,           # Sequence number
            0,             # Data length
            3,             # NAK directive
            self.calculate_checksum(struct.pack('<HHH', seq, 0, 3))
        )
        self.ser.write(packet)

    def send_command(self, message_id, routing_id=4, data=b'', max_retries=1000):
        """Send command with ACK/NAK retry logic"""
        for attempt in range(max_retries + 1):
            current_seq = self.sequence_number
            self._send_command_packet(current_seq, message_id, routing_id, data)
            
            # Wait for ACK with timeout
            start_time = time.time()
            while time.time() - start_time < 0.032:  # 31.25ms + buffer
                packet = self.read_packet()
                
                if packet and packet['type'] == 'ACK' and packet['seq'] == current_seq:
                    self.sequence_number = (current_seq + 1) % 65536
                    return True
                elif packet and packet['type'] == 'NAK' and packet['seq'] == current_seq:
                    print("NAK recieved")
                    break  # Will retry
                    
            # Timeout or NAK received
            if attempt < max_retries:
                print(f"Retrying command {message_id} (attempt {attempt+1}, time {time.time() - start_time})")
                time.sleep(60)
                continue
                
        print(f"Failed to send command {message_id} after {max_retries} retries")
        return False

    def _send_command_packet(self, seq, message_id, routing_id, data):
        """Internal method to construct and send a command packet"""
        # Layer 3
        layer3 = struct.pack('<IHH', message_id, 0, len(data)) + data
        
        # Layer 2
        layer2 = struct.pack('<I', routing_id) + layer3
        
        # Layer 1
        start = 0xABBA
        length = len(layer2)
        directive = 1  # L1_DATA_PACKET
        
        layer1_header = struct.pack('<HHHH', start, seq, length, directive)
        checksum_data = struct.pack('<HHH', seq, length, directive) + layer2
        checksum = self.calculate_checksum(checksum_data)
        
        packet = layer1_header + layer2 + struct.pack('<H', checksum)
        self.ser.write(packet)

    def read_packet(self):
        """Read and parse incoming packet"""
        # Check for header
        header = self.ser.read(8)
        if len(header) < 8:
            return None

        start, seq, length, directive = struct.unpack('<HHHH', header)
        if start != 0xABBA:
            return None  # Invalid packet

        # Read remaining data
        payload = self.ser.read(length + 2)  # Data + checksum
        if len(payload) < length + 2:
            return None  # Incomplete packet

        # Verify checksum
        checksum_data = struct.pack('<HHH', seq, length, directive) + payload[:-2]
        if self.calculate_checksum(checksum_data) != struct.unpack('<H', payload[-2:])[0]:
            return None  # Invalid checksum

        # Handle different packet types
        if directive == 2:  # ACK
            return {'type': 'ACK', 'seq': seq}
        elif directive == 3:  # NAK
            return {'type': 'NAK', 'seq': seq}
        elif directive == 1:  # Data packet
            # Send ACK for received data packet
            self.send_ack(seq)
            
            # Process Layer 2/3
            routing_id = struct.unpack('<I', payload[:4])[0]
            message_id, l3_seq, msg_len = struct.unpack('<IHH', payload[4:12])
            msg_data = payload[12:12+msg_len]
            
            return {
                'type': 'data',
                'seq': seq,
                'routing_id': routing_id,
                'message_id': message_id,
                'data': msg_data
            }
        return None

    def parse_data_raw(self, data):
        """Parse M_DATA_RAW message (ID 50)"""
        num_channels, rate = struct.unpack('<HH', data[:4])
        samples = [struct.unpack('<h', data[i:i+2])[0] 
                 for i in range(4, len(data), 2)]
        
        return {
            'channels': num_channels,
            'rate': rate,
            'samples': samples
        }

    def send_raw_eeg(self, rate=128):
        """Send SEND_RAW_EEG command with ACK handling"""
        return self.send_command(111, data=struct.pack('<H', rate))

    def stop_raw_eeg(self):
        """Send STOP_RAW_EEG command with ACK handling"""
        return self.send_command(112)

    def close(self):
        self.ser.close()

# Usage Example
if __name__ == "__main__":
    time.sleep(60)
    # Initialize the device
    monitor = BISMonitor('/dev/ttyUSB0')

    try:
        if monitor.send_raw_eeg(128):
            print("RAW EEG command acknowledged")
            date = datetime.now()
            start_time = time.time()
            string_date = str(date).replace(" ", "_")
            string_date = "BIS/BIS_" + string_date[:-7].replace(":", "") + ".bin"
            while True:
                packet = monitor.read_packet()
                if packet and packet['type'] == 'data':
                    if packet['message_id'] == 50:  # M_DATA_RAW
                        data_pre = monitor.parse_data_raw(packet['data'])
                        
                        with open(string_date, 'ab') as file_to_write:
                            file_to_write.write(packet['data'])
                        
                        if time.time() - start_time >= 7200:
                            date = datetime.now()
                            start_time = time.time()
                            string_date = str(date).replace(" ", "_")
                            string_date = "BIS/BIS_" + string_date[:-7].replace(":", "") + ".bin"
                        
                        
        else:
            print("Failed to start EEG streaming")
            
    except KeyboardInterrupt:
        monitor.stop_raw_eeg()
        monitor.close()
        print("Stopped")