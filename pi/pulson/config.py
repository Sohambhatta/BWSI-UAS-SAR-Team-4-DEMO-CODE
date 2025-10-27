# Pulson device network settings
PULSON_IP = "192.168.1.100"  # Replace with our device's actual IP address
PULSON_PORT = 21210          # Default UDP port for Pulson device

# Default scan parameters
DEFAULT_START_SCAN = 300         # Typical start time (device units)
DEFAULT_END_SCAN = 35000         # Typical end time (device units)
DEFAULT_SCAN_RES = 32            # Typical scan resolution (number of points)
DEFAULT_BASE_INT_INDEX = 9       # Integration index (6-15, 9 is common)
DEFAULT_TX_GAIN = 63             # Max transmit gain (0-63)
DEFAULT_CODE_CHANNEL = 5         # Default code channel (0-10)
DEFAULT_PERSIST_FLAG = 0         # 1 = persistent, 0 = non-persistent

# Node ID for the radar device
NODE_ID = 8

# Timeout settings (in seconds)
SOCKET_TIMEOUT = 5

# Logging level
LOG_LEVEL = "INFO"  

# Actual scan parameters, these can be adjusted based on the application
SCAN_START = 300
SCAN_END = 35000
SCAN_COUNT = 400 # Number of scans to perform
SCAN_INTERVAL = 50000  # Interval between scans in microseconds
