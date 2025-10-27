import time
import pulson.pulsonAPI as pls 
import threading
import pulson.config as plsconfig
import datalogger as dl
import signal
import os  # Import os module to check for file-based stop flag

pulson = pls.PulsonAPI("192.168.1.100")
timeout = 5 #seconds
stop_requested = False

def receive_loop():
    while True:
        pulson.receive_loop()

def setup():
    pulson.begin()
    threading.Thread(target=receive_loop, daemon=True).start()  # Start the receive loop in a separate thread
    
    status_success = pulson.check_for_ack(pulson.send_get_status_command())[1][-1]
    if status_success != 0:
        print("statF: ", status_success)
        return
    else:
        print("statG ")
    time.sleep(1)

    config_success = pulson.check_for_ack(pulson.send_config_command(start_scan=plsconfig.SCAN_START, 
                                                                     end_scan=plsconfig.SCAN_END))[1][-1] #configs to receive pulses from a nominal range of approx 10cm to 10m integrating 2^9 pulses per scan
    if config_success != 0:
        print("confF: ", config_success)
        return
    else:
        print("confG ")
    time.sleep(1)

    config_info = pulson.check_for_ack(pulson.send_get_config_command())[1]  # Get the current configuration
    if config_info[-1] != 0:
        print("gconfF", config_info[-1])
        return
    else:
        start_scan = config_info[3]
        scan_end = config_info[4]
        scan_res = config_info[5]
        datalogger = dl.Datalogger()
        datalogger.write_metadata(start_scan, scan_end, scan_res)  # Write metadata to file

    # Calculate expected wait time
    expected_time = plsconfig.SCAN_COUNT * plsconfig.SCAN_INTERVAL / 1000000
    timeout_time = time.time() + expected_time
    control_success = pulson.check_for_ack(pulson.send_control_command(scan_count=plsconfig.SCAN_COUNT, 
                                                                       scan_int_time=plsconfig.SCAN_INTERVAL))[1][-1]
    if control_success != 0:
        print("Control command failed with error code:", control_success)
    else:   
        print("Control command successful, scan started.")
    # Wait until all expected scans are received or timeout
    print(f"Waiting for {plsconfig.SCAN_COUNT} scans or timeout in {2*expected_time:.2f} seconds...")
    stop_flag_path = "/tmp/stop_scan"  # File-based stop flag
    while True:
        try:
            if os.path.exists(stop_flag_path):
                print("File-based stop flag detected. Stopping scan.")
                pulson.send_control_command(scan_count=0, scan_int_time=plsconfig.SCAN_INTERVAL)
                os.remove(stop_flag_path)
                time.sleep(0.5)  # Give some time for the command to be processed
                break
        except Exception as e:
            print(f"Error checking/removing stop flag: {e}")
        # Check if all expected scans have been received
        if len(pulson.completed_scans) >= plsconfig.SCAN_COUNT:
            print("All expected scans received.")
            break
        if time.time() > timeout_time:
            time.sleep(1.0)
            print("Timeout reached before all scans received, stopping scan")
            pulson.send_control_command(scan_count=0, scan_int_time=plsconfig.SCAN_INTERVAL)  # Stop the scan
            time.sleep(0.5)  # Give some time for the command to be processed
            break
        time.sleep(0.2)
    data = pulson.process_data()
    if data is None:
        print("No data received or processing failed.")
        return
    else:
        datalogger.write_scan(data)
        print("Data received and processed successfully.")

def loop():
    return
setup()
