import socket
import struct
import queue
import pulson.singleScan as ss
import pulson.config as plsconfig 
"""
socket: For UDP communication with the Pulson device.
struct: For packing/unpacking binary data.
queue: For managing message/data queues.
pulson.singleScan: For handling scan data.
"""

# commandFormats: Maps each command type to a struct format string for unpacking received binary data.
commandFormats = {
    0x1101: '!HHI', # Acknowledgment for configuration command
    0x1102: '!HHIiiHHHHHHBBBBBBBBII', # Acknowledgment for get config command
    0x1103: '!HHHHI', # Acknowledgment for control command
    0x1104: '!HHI', 
    0x1105: '!HH',
    0x1106: '!HHI',
    0x1107: '!HHHBBI',
    0xF101: '!HHBBHBBHBBBBIBBBBi32sI',
    0xF102: '!HH',
    0xF103: '!HHII',
    0xF105: '!HHI',
    0xF106: '!HHII',
    0xF201: '!HHIIIIIIiihBBBBHIHH'
}

# control command types
SET_CONFIG_MSG_TYPE = 0x1001
GET_CONFIG_MSG_TYPE = 0x1002
CONTROL_MSG_TYPE = 0x1003
SERVER_CONNECT_MSG_TYPE = 0x1004
SERVER_DISCONNECT_MSG_TYPE = 0x1005
SET_FILTER_MSG_TYPE = 0x1006
GET_FILTER_MSG_TYPE = 0x1007
GET_STATUS_MSG_TYPE = 0xF001
REBOOT_MSG_TYPE = 0xF002
OPMODE_MSG_TYPE = 0xF003
SET_SLEEP_MSG_TYPE = 0xF005
GET_SLEEP_MSG_TYPE = 0xF006

TO_CONFIRMATION_MSG_TYPE = 0x0100 # add to command message types to convert to acknowledgement message type

SCANINFO_MSG_TYPE = 0xF201

class PulsonAPI:
    """
    API class for communicating with the Pulson radar device over UDP.
    It sends commands, receives messages, and manages scanned data.
    """
    def __init__(self, IP=plsconfig.PULSON_IP, port=plsconfig.PULSON_PORT):
        """Initialize the PulsonAPI with the given IP and port."""
        self.IP = IP
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  
        self.message_id = 0
        self.node_id = 8
        self.socket.settimeout(2.0)  # Set a timeout for socket operations to avoid blocking indefinitely

        self.ack_dict = {
        }
        #dictionary of acknowledgement queues indexed message type of ack

        self.data_dict = {
        }
        #dictionary of data queues indexed by millisecond of reception

        self.scans = {}  # key: uid, value: singleScan object
        self.completed_scans = []  # store completed scans for datalogging


    def begin(self):
        """Connect to the Pulson device on the initialized IP and port, also sends server connect command."""
        self.socket.connect((self.IP, self.port))
        self.send_server_connect_command()

    def end(self):
        """Send server disconnect command and disconnect from the Pulson device."""
        if self.socket:
            self.send_server_disconnect_command()
            self.socket.close()
            self.socket = None


    def increment_message_id(self):
        """Increment the message ID for the next command, loops if necessary."""
        self.message_id += 1
        if self.message_id > 65535:
            self.message_id = 0


    def check_for_ack(self, message_type):
        """Check for an acknowledgment for a specific message type.
        Parameters:
            message_type (int): The type of message to check for acknowledgment.
            timeout (float): The timeout in seconds to wait for the acknowledgment.
        Returns:
            tuple: A tuple containing the message type and acknowledgment data if received,
                   or (-1, (-1, -1)) if the acknowledgment is not received within the timeout.
            """
        try:
            #print(self.ack_dict)    
            q = self.ack_dict[message_type]
            #print(q)
            try:
                ack = q.get(timeout=5.0)
                #print(ack)
                self.ack_dict.pop(message_type, None)  # Remove the ack queue after processing
                #print("passed")
                return message_type, ack
            except:
                print("error getting queue element")
                return (-1, (-1, -1))
        except queue.Empty:
            print(f"Timeout waiting for ACK for message type: {message_type}")
            return (-1, (-1, -1))


    def get_uid(self, a, b):
        """mixes two integers to create a unique identifier for the reception picosecond and millisecond."""
        return ((a * 73856093) ^ (b * 19349663)) & 0xFFFFFFFFFFFFFFFF
    

    def process_data(self):
        """Return all completed scans and clear the list."""
        output = self.completed_scans.copy()
        self.completed_scans.clear()
        return output

    def receive_loop(self):
        """Continuously receive messages from the Pulson device, and process scans as soon as complete."""
        while True:
            commandResult = self.receive_message()
            if commandResult == -1:
                continue
            message_type, result = commandResult
            if message_type != SCANINFO_MSG_TYPE:
                if message_type in self.ack_dict:
                    self.ack_dict[message_type].put(result)
                continue
            # Scan data message
            numDataPoints = result[15]
            ending = numDataPoints + 18 + 1
            data = result[18:ending]
            millisTime = result[3]
            startTime = result[8]
            endTime = result[9]
            numSamples = result[16]
            messageIndex = result[17]
            numMessages = result[18]
            uid = self.get_uid(millisTime, startTime)
            # Use singleScan object directly
            if uid not in self.scans:
                self.scans[uid] = ss.singleScan(millisTime, numSamples, numMessages)
            self.scans[uid].add_data_list(data, messageIndex)
            # If scan is complete, process and remove
            if all(self.scans[uid].all_data_received):
                all_data_received, millisTime, dataPoints = self.scans[uid].get_data()
                self.completed_scans.append((millisTime, dataPoints))
                del self.scans[uid]

    def receive_message(self):
        """Receive a message from the Pulson device and unpack it into a tuple of (message_type, data)."""
        #print("unpacking1")
        if not self.socket:
            raise RuntimeError("Socket is not connected")
        try:
            response = self.socket.recv(2048) # Receive the first 2 bytes to determine message type
        except:
            return -1
        #print("unpacking")
        result = struct.unpack('!H', response[:2]) 
        #print(result)
        message_type = result[0]
        if message_type != SCANINFO_MSG_TYPE:
            print("recvmsg")
            try:
                result = struct.unpack(commandFormats.get(message_type), response)
            except:
                print("upackf")
                return -1
        else:
            # Special case for the 0xF201 message type (scan data)
            #numDataPoints = struct.unpack('!H', response[42:44])[0]
            #print(numDataPoints)
            dataFormat = commandFormats.get(message_type) + ('i' * 350)
            #print(len(response))
            #print(struct.unpack(dataFormat, response))
            result = struct.unpack(dataFormat, response)
            print("scanms: ", result[3])
        return message_type, result


    def send_config_command(self, start_scan=plsconfig.DEFAULT_START_SCAN, end_scan=plsconfig.DEFAULT_END_SCAN, 
                            base_int_ind=plsconfig.DEFAULT_BASE_INT_INDEX, scan_res=plsconfig.DEFAULT_SCAN_RES, 
                            tx_gain=plsconfig.DEFAULT_TX_GAIN, code_chan=plsconfig.DEFAULT_CODE_CHANNEL, 
                            persist_flag=plsconfig.DEFAULT_PERSIST_FLAG):
        """        Send a configuration command to the Pulson device with the specified parameters.
        Parameters:
            start_scan (int): Start time of the scan in microseconds relatieve to pulse. Min/max value is -+4998998.
            end_scan (int): End time of the scan in microseconds.
            base_int_ind (int): Base integration index (6-15), Log2 of number of integrated pulsles
            scan_res (int): Scan resolution in bins, typically 32, resulting in 1.9 * scan_res microseconds per scan point.
            tx_gain (int): TX gain (0-63), the higher the more gain.
            code_chan (int): Code channel (0-10), for interference mitigation with other devices.
            persist_flag (int): Persist flag (0 or 1), indicating whether to persist the config through reboots.

        Returns:
            Message code of ack
        """
        
        if not self.socket:
            raise RuntimeError("Socket is not connected") #TADAAAA! runtime error
        if(abs(start_scan) > 4998998 or abs(scan_res - 256) > 255):
            raise ValueError("Invalid scan parameters")
        if(base_int_ind < 6 or base_int_ind > 15):
            raise ValueError("Invalid base integration index")
        if(tx_gain < 0 or tx_gain > 63):
            raise ValueError("Invalid TX gain")
        if(code_chan < 0 or code_chan > 10):
            raise ValueError("Invalid code channel")
        if(persist_flag < 0 or persist_flag > 1):
            raise ValueError("Invalid persist flag")
        
        # Set default values for segments and antenna mode, enusres its null basically
        seg_one_samples = 0 #NOT IMPLEMENTED
        seg_two_samples = 0 #NOT IMPLEMENTED
        seg_three_samples = 0 #NOT IMPLEMENTED 
        seg_four_samples = 0 #NOT IMPLEMENTED
        seg_one_int_mult = 0 #NOT IMPLEMENTED
        seg_two_int_mult = 0 #NOT IMPLEMENTED
        seg_three_int_mult = 0 #NOT IMPLEMENTED
        seg_four_int_mult = 0 #NOT IMPLEMENTED
        antenna_mode = 3
        

        # Pack the command as a byte string and send it
        packed_command = struct.pack('!HHIiiHHHHHHBBBBBBBB', SET_CONFIG_MSG_TYPE, self.message_id, self.node_id, start_scan, end_scan,
                                     scan_res, base_int_ind, seg_one_samples, seg_two_samples,
                                     seg_three_samples, seg_four_samples, seg_one_int_mult,
                                     seg_two_int_mult, seg_three_int_mult, seg_four_int_mult,
                                     antenna_mode, tx_gain, code_chan, persist_flag)
        self.socket.sendall(packed_command)

        # Increment message ID for the next command
        self.increment_message_id()

        self.ack_dict[SET_CONFIG_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE] = queue.Queue()
        return SET_CONFIG_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE  # Return the message type for acknowledgment

        
    def send_get_config_command(self):
        """        Send a command to get the current configuration of the Pulson device.
        Returns: 
            Message code of ack"""
        if not self.socket:
            raise RuntimeError("Socket is not connected")
        
        packed_command = struct.pack('!HH', GET_CONFIG_MSG_TYPE, self.message_id)
        self.socket.sendall(packed_command)

        # Increment message ID for the next command
        self.increment_message_id()
        self.ack_dict[GET_CONFIG_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE] = queue.Queue()
        return GET_CONFIG_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE  # Return the message type for acknowledgment
        
    def send_control_command(self, scan_count, scan_int_time):
        """        Send a control command to the Pulson device to start scanning.

        Parameters:
            scan_count (int): Number of scans to perform, must be between 0 and 65535. 
            0 means stop scan, 1 means single scan, and 65,535 means continuous scan.
            scan_int_time (int): Number of microseconds to wait between scans. Time less than required time for scan with result in fastest scan.
            
        Returns:
            Message code of ack
        
        """
        if not self.socket:
            raise RuntimeError("Socket is not connected")
        
        if(scan_count < 0 or scan_count > 65535):
            raise ValueError("Invalid scan count")
        
        packed_command = struct.pack('!HHHHI', CONTROL_MSG_TYPE, self.message_id, scan_count, 0, scan_int_time)
        self.socket.sendall(packed_command)

        # Increment message ID for the next command
        self.increment_message_id()
        self.ack_dict[CONTROL_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE] = queue.Queue()
        return CONTROL_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE  # Return the message type for acknowledgment
        
    def send_server_connect_command(self):
        """Send a command to connect to the Pulson device server.
        Returns:
            Message code of ack"""
        if not self.socket:
            raise RuntimeError("Socket is not connected")
        packed_ip = struct.unpack("!I", socket.inet_aton(self.IP))[0]
        packed_command = struct.pack('!HHIHH', SERVER_CONNECT_MSG_TYPE, self.message_id, packed_ip, self.port, 0x00)

        self.socket.sendall(packed_command)
        # Increment message ID for the next command
        self.increment_message_id()

        self.ack_dict[SERVER_CONNECT_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE] = queue.Queue()
        return SERVER_CONNECT_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE  # Return the message type for acknowledgment
        
    def send_server_disconnect_command(self):
        """     Send a command to disconnect from the Pulson device server.
        
        Returns: Message code of ack
        
        """
        if not self.socket:
            raise RuntimeError("Socket is not connected")
        packed_command = struct.pack('!HH', SERVER_DISCONNECT_MSG_TYPE, self.message_id)

        self.socket.sendall(packed_command)
        # Increment message ID for the next command
        self.increment_message_id()
        self.ack_dict[SERVER_DISCONNECT_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE] = queue.Queue()
        return SERVER_DISCONNECT_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE  # Return the message type for acknowledgment
    
    def send_set_filter_command(self, filter_mask, motion_filter_ind):
        """     Send a command to set the filter on the Pulson device.
        
        Parameters: 
            filter_mask (int): Filter mask, a bitmask where each bit represents a filter.
                               Valid values are 0-15, where each bit corresponds to a filter.
                               0001 = raw
                               0010 = band pass
                               0100 = motion/doppler
                               1000 = detection list
            motion_filter_ind (int): Motion filter index, valid values are 0-3. See datasheet for details

        Returns:
            Message code of ack
        """
        if not self.socket:
            raise RuntimeError("Socket is not connected")
        
        if (filter_mask < 0 or filter_mask > 15):
            raise ValueError("Invalid filter mask")
        if (motion_filter_ind < 0 or motion_filter_ind > 3):
            raise ValueError("Invalid motion filter index")
        
        packed_command = struct.pack('!HHHBB', SET_FILTER_MSG_TYPE, self.message_id, filter_mask, motion_filter_ind, 0)
        self.socket.sendall(packed_command)
        # Increment message ID for the next command
        self.increment_message_id()
        self.ack_dict[SET_FILTER_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE] = queue.Queue()
        return SET_FILTER_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE  # Return the message type for acknowledgment
    
    def send_get_filter_command(self):
        """Send a command to get the current filter settings from the Pulson device.
        Returns:
            Message code of ack"""

        if not self.socket:
            raise RuntimeError("Socket is not connected")
        packed_command = struct.pack('!HH', GET_FILTER_MSG_TYPE, self.message_id)

        self.socket.sendall(packed_command)
        # Increment message ID for the next command
        self.increment_message_id()
        self.ack_dict[GET_FILTER_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE] = queue.Queue()
        return GET_FILTER_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE  # Return the message type for acknowledgment

    
    def send_get_status_command(self):
        """Send a command to get the current status/info of the Pulson device.
        Returns:
            Message code of ack"""

        if not self.socket:
            raise RuntimeError("Socket is not connected")
        packed_command = struct.pack('!HH', GET_STATUS_MSG_TYPE, self.message_id)

        self.socket.sendall(packed_command)
        # Increment message ID for the next command
        self.increment_message_id()
        self.ack_dict[GET_STATUS_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE] = queue.Queue()
        return GET_STATUS_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE  # Return the message type for acknowledgment
    
    def send_reboot_command(self):
        """Send a command to reboot the Pulson device.
        Returns:
            Message code of ack"""

        if not self.socket:
            raise RuntimeError("Socket is not connected")
        packed_command = struct.pack('!HH', REBOOT_MSG_TYPE, self.message_id)
        self.socket.sendall(packed_command)
        # Increment message ID for the next command
        self.increment_message_id()
        self.ack_dict[REBOOT_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE] = queue.Queue()
        return REBOOT_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE  # Return the message type for acknowledgment

    
    def send_opmode_command(self):
        """Send a command to set the operating mode of the Pulson device to normal mode.
        Returns:
            Message code of ack"""

        if not self.socket:
            raise RuntimeError("Socket is not connected")
        packed_command = struct.pack('!HHI', OPMODE_MSG_TYPE, self.message_id, 1)
        self.socket.sendall(packed_command)
        # Increment message ID for the next command

        self.increment_message_id()
        self.ack_dict[OPMODE_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE] = queue.Queue()
        return OPMODE_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE
    
    def send_sleep_command(self, mode):
        """Send a command to set the sleep mode of the Pulson device.
        Parameters:
            mode (int): Sleep mode to set, valid values are:
                        0: Go to ACTIVE mode (normal operation)
                        1: Go to STANDBY_E mode (wake on LAN)
                        2: Go to STANDBY_S mode (wake on serial)
                        3: Go to SLEEP_D mode (wake on GPIO, see datasheet for details)
        
        Returns:
            Message code of ack
        """

        if not self.socket:
            raise RuntimeError("Socket is not connected")
        if mode not in [0, 1, 2, 3, 4]:
            raise ValueError("Invalid sleep mode")
        packed_command = struct.pack('!HHI', SET_SLEEP_MSG_TYPE, self.message_id, mode)
        self.socket.sendall(packed_command)
        # Increment message ID for the next command
        self.increment_message_id()
        self.ack_dict[SET_SLEEP_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE] = queue.Queue()
        return SET_SLEEP_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE
    
    def send_get_sleep_command(self):
        """Send a command to get the current sleep mode of the Pulson device.
        Returns:
            Message code of ack
        """

        if not self.socket:
            raise RuntimeError("Socket is not connected")
        packed_command = struct.pack('!HH', GET_SLEEP_MSG_TYPE, self.message_id)
        self.socket.sendall(packed_command)
        # Increment message ID for the next command
        self.increment_message_id()
        self.ack_dict[GET_SLEEP_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE] = queue.Queue()
        return GET_SLEEP_MSG_TYPE + TO_CONFIRMATION_MSG_TYPE




