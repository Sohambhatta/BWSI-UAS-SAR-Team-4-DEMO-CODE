import dearpygui.dearpygui as dpg
import paramiko
import time
import os
import shutil
import textwrap
import threading


c = 3e8  # Speed of light in m/s

go = (13, 200, 0, 255)
nogo = (255, 0, 0, 255)

ssh = paramiko.SSHClient()

statGood = False
confGood = False
scanGood = False

config_path = "Team_Four_Repo/pi/pulson/config.py"
main_path = "Team_Four_Repo/pi/main.py"


def add_text_and_autoscroll(text, window_tag):
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    text = f"[{timestamp}] {text}"
    # Wrap text to fit the window width (approx 60 chars, adjust as needed)
    wrapped_lines = textwrap.wrap(text, width=40)
    for line in wrapped_lines:
        dpg.add_text(line, parent=window_tag)
    # Autoscroll to bottom
    dpg.set_y_scroll(window_tag, dpg.get_y_scroll_max(window_tag))


def connect_ssh(host, port, username, password):
    """Connect to an SSH server using Paramiko."""
    try:
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, port=port, username=username, password=password)
        print("SSH connection established")
        add_text_and_autoscroll("SSH connection established", "sent_window")
        return 1
    except Exception as e:
        print(f"Failed to connect: {e}")
        return 0


def send_command(command):
    """Send a command to the SSH server and return the output."""
    try:
        stdin, stdout, stderr = ssh.exec_command(command)
        output = stdout.read().decode("utf-8")
        add_text_and_autoscroll(f"{command}", "sent_window")
        return output
    except Exception as e:
        print(f"Command execution failed: {e}")
        return None


def send_input():
    global ssh
    try:
        command = dpg.get_value("command_input")
        stdin, stdout, stderr = ssh.exec_command(command)
        add_text_and_autoscroll(f"{command}", "sent_window")
        add_text_and_autoscroll(stdout.read().decode("utf-8"), "received_window")
    except Exception as e:
        print(f"Command execution failed: {e}")
        return None


def run_python(script_path):
    global confGood, statGood, scanGood, scanms_seen, scan_progress, scan_progress_max, can_run_python
    if not can_run_python:
        add_text_and_autoscroll(
            "Please fetch config before running scan.", "sent_window"
        )
        return
    confGood = False
    statGood = False
    scanGood = False
    scanms_seen = set()
    scan_progress = 0
    dpg.set_value("scan_progress_bar", 0)
    dpg.set_value("scan_progress_text", f"0 / {scan_progress_max}")
    command = f"python3 {script_path}"
    def run_script():
        global confGood, statGood, scanGood, scanms_seen, scan_progress, scan_progress_max, can_run_python
        try:
            stdin, stdout, stderr = ssh.exec_command(command)
            add_text_and_autoscroll(command, "sent_window")
            exit_thread = False
            while not stdout.channel.exit_status_ready() and not exit_thread:
                while stdout.channel.recv_ready():
                    line = stdout.readline()
                    print(line.strip())
                    if "confG" in line:
                        confGood = True
                    if "statG" in line:
                        statGood = True
                    if "processed successfully" in line:
                        scanGood = True
                        exit_thread = True
                        break
                    if "scanms" in line:
                        parts = line.strip().split(" ")
                        if len(parts) > 1:
                            ms_val = int(parts[2])
                            if ms_val not in scanms_seen:
                                print(ms_val)
                                scanms_seen.add(ms_val)
                                scan_progress = len(scanms_seen)
                                if scan_progress_max > 0:
                                    dpg.set_value(
                                        "scan_progress_bar", scan_progress / scan_progress_max
                                    )
                                    dpg.set_value(
                                        "scan_progress_text",
                                        f"{scan_progress} / {scan_progress_max}",
                                    )
                    else:
                        add_text_and_autoscroll(line.strip(), "received_window")
                time.sleep(0.1)
            # Read any remaining lines after process exits
            for line in stdout.readlines():
                if "processed successfully" in line:
                    scanGood = True
                    exit_thread = True
                    print(line.strip())
                    break
                if "scanms" in line:
                    parts = line.strip().split(" ")
                    if len(parts) > 1:
                        ms_val = int(parts[2])
                        if ms_val not in scanms_seen:
                            print(ms_val)
                            scanms_seen.add(ms_val)
                            scan_progress = len(scanms_seen)
                            if scan_progress_max > 0:
                                dpg.set_value(
                                    "scan_progress_bar", scan_progress / scan_progress_max
                                )
                                dpg.set_value(
                                    "scan_progress_text",
                                    f"{scan_progress} / {scan_progress_max}",
                                )
                    continue
                print(line.strip())
                add_text_and_autoscroll(line.strip(), "received_window")
        except Exception as e:
            print(f"Failed to run script: {e}")
    threading.Thread(target=run_script, daemon=True).start()


def connect_and_cd():
    if connect_ssh("192.168.2.1", 22, "sardemo", "s@Rdemo"):
        pass


def ls_dir():
    try:
        command = f"ls"
        stdin, stdout, stderr = ssh.exec_command(command)
        add_text_and_autoscroll(f"{command}", "sent_window")
        add_text_and_autoscroll(stdout.read().decode("utf-8"), "received_window")
    except Exception as e:
        print(f"Failed to ls")


def update_remote_config(
    scan_duration,
    scan_frequency,
    scan_start=None,
    scan_end=None,
    base_int_index=None,
    remote_path=config_path,
):
    """
    Update SCAN_COUNT, SCAN_INTERVAL, SCAN_START, SCAN_END, and DEFAULT_BASE_INT_INDEX in the remote config.py via SSH.
    """
    try:
        scan_count = int(scan_duration * scan_frequency)
        scan_interval = int(
            1e6 / scan_frequency
        )  # Convert frequency to interval in microseconds
        scan_start = int(scan_start * 2.0 * 1e12 / c)
        scan_end = int(scan_end * 2.0 * 1e12 / c)

        print(scan_start, scan_end)

        python_code = (
            f"import re\n"
            f"path = '{remote_path}'\n"
            f"with open(path) as f:\n"
            f"    lines = f.readlines()\n"
            f"with open(path, 'w') as f:\n"
            f"    for line in lines:\n"
            f"        if line.strip().startswith('SCAN_COUNT'):\n"
            f"            f.write('SCAN_COUNT = {scan_count}  # Updated remotely\\n')\n"
            f"        elif line.strip().startswith('SCAN_INTERVAL'):\n"
            f"            f.write('SCAN_INTERVAL = {scan_interval}  # Updated remotely\\n')\n"
            f"        elif line.strip().startswith('SCAN_START') and {scan_start is not None}:\n"
            f"            f.write('SCAN_START = {scan_start}  # Updated remotely\\n')\n"
            f"        elif line.strip().startswith('SCAN_END') and {scan_end is not None}:\n"
            f"            f.write('SCAN_END = {scan_end}  # Updated remotely\\n')\n"
            f"        elif line.strip().startswith('DEFAULT_BASE_INT_INDEX') and {base_int_index is not None}:\n"
            f"            f.write('DEFAULT_BASE_INT_INDEX = {base_int_index}  # Updated remotely\\n')\n"
            f"        else:\n"
            f"            f.write(line)\n"
        )
        python_code_escaped = python_code.replace('"', '\\"')
        command = f'python3 -c "{python_code_escaped}"'
        ssh.exec_command(command)
        add_text_and_autoscroll("Config.py updated", "sent_window")
        print("Remote config.py updated successfully.")
    except Exception as e:
        print(f"Failed to update remote config.py: {e}")


scan_progress = 0
scan_progress_max = 1
scanms_seen = set()
can_run_python = False


def reset_run_python_flag():
    global can_run_python
    can_run_python = False
    dpg.set_value("scan_progress_bar", 0)
    dpg.set_value("scan_progress_text", "0 / ?")
    global scanms_seen, scan_progress, scan_progress_max
    scanms_seen = set()
    scan_progress = 0
    scan_progress_max = 1


def send_config_callback():
    scan_count = dpg.get_value("scan_count_input")
    scan_interval = dpg.get_value("scan_interval_input")
    scan_start = dpg.get_value("scan_start_input")
    scan_end = dpg.get_value("scan_end_input")
    base_int_index = dpg.get_value("base_int_index_input")
    reset_run_python_flag()
    update_remote_config(scan_count, scan_interval, scan_start, scan_end, base_int_index)


def get_config_callback():
    import re

    global scan_progress_max, can_run_python, scanms_seen, scan_progress
    output = send_command("cat " + config_path)
    if output is None:
        print("Failed to fetch config.py")
        return
    print(output)
    scan_count = scan_interval = scan_start = scan_end = base_int_index = None
    for line in output.splitlines():
        if line.strip().startswith("SCAN_COUNT"):
            m = re.search(r"SCAN_COUNT\s*=\s*(\d+)", line)
            if m:
                scan_count = int(m.group(1))
        elif line.strip().startswith("SCAN_INTERVAL"):
            m = re.search(r"SCAN_INTERVAL\s*=\s*(\d+)", line)
            if m:
                scan_interval = int(m.group(1))
        elif line.strip().startswith("SCAN_START"):
            m = re.search(r"SCAN_START\s*=\s*(\d+)", line)
            if m:
                scan_start = int(m.group(1))
        elif line.strip().startswith("SCAN_END"):
            m = re.search(r"SCAN_END\s*=\s*(\d+)", line)
            if m:
                scan_end = int(m.group(1))
        
        elif line.strip().startswith("DEFAULT_BASE_INT_INDEX"):
            m = re.search(r"BASE_INT_INDEX\s*=\s*(\d+)", line)
            if m:
                base_int_index = int(m.group(1))

    if None in (scan_count, scan_interval, scan_start, scan_end):
        print(scan_count, scan_interval, scan_start, scan_end)
        print("Could not parse all config values.")
        return
    status_lines = [
        f"SCAN_START (ps): {scan_start}",
        f"SCAN_END (ps): {scan_end}",
        f"SCAN_INTERVAL (us): {scan_interval}",
        f"SCAN_COUNT: {scan_count}",
        f"DEFAULT_BASE_INT_INDEX: {base_int_index}",
    ]
    try:
        scan_duration = scan_count * (scan_interval / 1e6)
        freq = 1e6 / scan_interval if scan_interval != 0 else 0
        start_m = scan_start * c / (2.0 * 1e12)
        end_m = scan_end * c / (2.0 * 1e12)
    except Exception as e:
        dpg.set_value("config_status_text", f"Error in calculations: {e}")
        return
    status_lines += [
        f"Derived Scan Duration (s): {scan_duration:.2f}",
        f"Derived Frequency (Hz): {freq:.2f}",
        f"Derived Start (m): {start_m:.2f}",
        f"Derived End (m): {end_m:.2f}",
    ]
    input_duration = dpg.get_value("scan_count_input")
    input_freq = dpg.get_value("scan_interval_input")
    input_start = dpg.get_value("scan_start_input")
    input_end = dpg.get_value("scan_end_input")
    input_base_int_index = dpg.get_value("base_int_index_input")
    def within_10pct(a, b):
        return abs(a - b) <= 0.1 * abs(b) if b != 0 else a == b
    checks = [
        ("Duration", scan_duration, input_duration),
        ("Frequency", freq, input_freq),
        ("Start", start_m, input_start),
        ("End", end_m, input_end),
        ("Base Int Index", base_int_index, input_base_int_index),
    ]
    for label, actual, expected in checks:
        #print(checks)
        if within_10pct(actual, expected):
            status_lines.append(f"{label}: OK")
        else:
            status_lines.append(
                f"{label}: OUT OF RANGE (expected {expected}, got {actual:.2f})"
            )

    dpg.set_value("config_status_text", "\n".join(status_lines))
    scan_progress_max = scan_count
    scan_progress = 0
    scanms_seen = set()
    dpg.set_value("scan_progress_bar", 0)
    dpg.set_value("scan_progress_text", f"0 / {scan_progress_max}")
    can_run_python = True


def transfer_and_cleanup_datas():
    """
    Transfer all folders/files from remote DATAS directory to local DATAS directory,
    ensure all data is transferred, then remove the remote DATAS directory.
    """
    try:
        remote_datas = "DATAS"
        local_datas = os.path.join(os.getcwd(), "DATAS")
        sftp = ssh.open_sftp()

        # Ensure local DATAS directory exists
        if not os.path.exists(local_datas):
            os.makedirs(local_datas)

        # List all files/folders in remote DATAS
        try:
            remote_items = sftp.listdir(remote_datas)
        except Exception as e:
            add_text_and_autoscroll(f"Failed to list remote DATAS: {e}", "sent_window")
            sftp.close()
            return

        # Transfer each item
        for item in remote_items:
            remote_path = f"{remote_datas}/{item}"
            local_path = os.path.join(local_datas, item)
            try:
                # Check if it's a directory or file
                stat = sftp.stat(remote_path)
                if stat.st_mode & 0o40000:  # Directory
                    if not os.path.exists(local_path):
                        os.makedirs(local_path)
                    # Recursively transfer directory
                    for dirpath, dirnames, filenames in sftp_walk(sftp, remote_path):
                        rel_dir = os.path.relpath(dirpath, remote_datas)
                        local_dir = os.path.join(local_datas, rel_dir)
                        if not os.path.exists(local_dir):
                            os.makedirs(local_dir)
                        for fname in filenames:
                            rfile = f"{dirpath}/{fname}"
                            lfile = os.path.join(local_dir, fname)
                            sftp.get(rfile, lfile)
                else:  # File
                    sftp.get(remote_path, local_path)
                add_text_and_autoscroll(f"Transferred: {item}", "sent_window")
            except Exception as e:
                add_text_and_autoscroll(
                    f"Failed to transfer {item}: {e}", "sent_window"
                )

        # Verify all files/folders transferred
        remote_items_after = sftp.listdir(remote_datas)
        local_items = os.listdir(local_datas)
        if set(remote_items) - set(local_items):
            add_text_and_autoscroll(
                "Transfer mismatch detected! Aborting remote delete.", "sent_window"
            )
            sftp.close()
            return

        # Remove remote DATAS directory
        try:
            # Remove all files/folders recursively
            remove_remote_dir(sftp, remote_datas)
            add_text_and_autoscroll("Remote DATAS directory removed.", "sent_window")
        except Exception as e:
            add_text_and_autoscroll(
                f"Failed to remove remote DATAS: {e}", "sent_window"
            )

        sftp.close()
    except Exception as e:
        add_text_and_autoscroll(f"Transfer failed: {e}", "sent_window")


def sftp_walk(sftp, remotepath):
    # Generator to walk SFTP directories like os.walk
    import posixpath

    path = remotepath
    files = []
    folders = []
    for f in sftp.listdir_attr(remotepath):
        if f.st_mode & 0o40000:
            folders.append(f.filename)
        else:
            files.append(f.filename)
    yield remotepath, folders, files
    for folder in folders:
        new_path = posixpath.join(remotepath, folder)
        for x in sftp_walk(sftp, new_path):
            yield x


def remove_remote_dir(sftp, remotepath):
    # Recursively remove a remote directory and its contents
    import posixpath

    for entry in sftp.listdir_attr(remotepath):
        remote_item = posixpath.join(remotepath, entry.filename)
        if entry.st_mode & 0o40000:
            remove_remote_dir(sftp, remote_item)
        else:
            sftp.remove(remote_item)
    sftp.rmdir(remotepath)

def set_stop_flag():
    """
    Create the /tmp/stop_scan file on the remote Pi via SSH to signal main.py to stop scanning.
    """
    try:
        # This command creates the stop flag file
        ssh.exec_command("touch /tmp/stop_scan")
        add_text_and_autoscroll("Stop flag set on remote Pi.", "sent_window")
    except Exception as e:
        add_text_and_autoscroll(f"Failed to set stop flag: {e}", "sent_window")

def sync_pi_directory():
    """
    Remove the remote Team_Four_Repo/pi directory and replace it with the local /pi directory and all its contents.
    """
    try:
        remote_pi = "Team_Four_Repo/pi"
        local_pi = os.path.join(os.getcwd(), "pi")
        sftp = ssh.open_sftp()
        # Remove remote pi directory
        try:
            remove_remote_dir(sftp, remote_pi)
            add_text_and_autoscroll("Remote pi directory removed.", "sent_window")
        except Exception as e:
            add_text_and_autoscroll(f"Failed to remove remote pi: {e}", "sent_window")
        # Recreate remote pi directory
        try:
            sftp.mkdir(remote_pi)
        except Exception as e:
            add_text_and_autoscroll(f"Failed to create remote pi: {e}", "sent_window")
        # Upload all files and subfolders from local pi
        for root, dirs, files in os.walk(local_pi):
            rel_path = os.path.relpath(root, local_pi)
            if rel_path == '.':
                remote_root = remote_pi
            else:
                remote_root = f"{remote_pi}/{rel_path.replace('\\', '/')}"
            try:
                sftp.mkdir(remote_root)
            except Exception:
                pass  # Directory may already exist
            for file in files:
                local_file = os.path.join(root, file)
                remote_file = f"{remote_root}/{file}"
                try:
                    sftp.put(local_file, remote_file)
                    add_text_and_autoscroll(f"Uploaded: {remote_file}", "sent_window")
                except Exception as e:
                    add_text_and_autoscroll(f"Failed to upload {remote_file}: {e}", "sent_window")
        sftp.close()
        add_text_and_autoscroll("pi directory sync complete.", "sent_window")
    except Exception as e:
        add_text_and_autoscroll(f"pi sync failed: {e}", "sent_window")

dpg.create_context()


with dpg.font_registry():
    default_font = dpg.add_font("helpers/futuramediumbt.ttf", 16) 
    medium_font = dpg.add_font("helpers/futuramediumbt.ttf", 20) 
    large_font = dpg.add_font("helpers/futuramediumbt.ttf", 28) 
    huge_font = dpg.add_font("helpers/futuramediumbt.ttf", 36) 
    
with dpg.theme(tag="stat_window_theme"):
    with dpg.theme_component(dpg.mvChildWindow):
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, nogo, tag="stat_window_color")

with dpg.theme(tag="conf_window_theme"):
    with dpg.theme_component(dpg.mvChildWindow):
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, nogo, tag="conf_window_color")


with dpg.theme(tag="scan_window_theme"):
    with dpg.theme_component(dpg.mvChildWindow):
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, nogo, tag="scan_window_color")

with dpg.window(label="SSH Connection", width=600, height=800):
    dpg.bind_font(default_font)
    dpg.add_button(label="Connect to SSH", callback=connect_and_cd)
    dpg.add_button(label="List", callback=ls_dir)
    dpg.add_button(label="Sync pi directory", callback=lambda: sync_pi_directory())

    dpg.add_separator()
    dpg.add_input_text(label="Command", tag="command_input")
    dpg.add_button(label="Send Command", callback=send_input)
    dpg.add_separator()

    with dpg.group(horizontal=True):  # Horizontal layout for Sent/Received windows
        # dpg.bind_font(default_font)
        with dpg.child_window(width=280, height=550):
            with dpg.group(horizontal=True):
                dpg.add_text("Sent:")
                dpg.add_button(
                    label="Clear",
                    callback=lambda: dpg.delete_item("sent_window", children_only=True),
                    width=60,
                )
            with dpg.child_window(
                tag="sent_window", width=270, height=500, tracked=True
            ):
                pass
        with dpg.child_window(width=280, height=550):
            with dpg.group(horizontal=True):
                dpg.add_text("Recieved:")
                dpg.add_button(
                    label="Clear",
                    callback=lambda: dpg.delete_item(
                        "received_window", children_only=True
                    ),
                    width=60,
                )
            with dpg.child_window(
                tag="received_window", width=270, height=500, tracked=True
            ):
                pass

with dpg.window(label="Control Panel", width=600, height=800, pos=(600, 0)):
    dpg.bind_font(default_font)
    # Add input fields for config values
    dpg.add_input_int(label="Scan Duration (s)", tag="scan_count_input", default_value=25)
    dpg.add_input_int(label="Scan Frequency (hz)", tag="scan_interval_input", default_value=180)
    dpg.add_input_int(label="Scan Start (m)", tag="scan_start_input", default_value=4)
    dpg.add_input_int(label="Scan End (m)", tag="scan_end_input", default_value=12)
    dpg.add_input_int(label="Base Int Index", tag="base_int_index_input", default_value=10)

    dpg.add_button(label="Send Config", callback=lambda: send_config_callback())
    dpg.add_text(
        """SCAN_START
SCAN_END
SCAN_INTERVAL
SCAN_COUNT
DEFAULT_BASE_INT_INDEX
DERIVED SCANDURATION
DERIVED FREQUENCY
DERIVED START
DERIVED END
DURATION
FREQUENCY
START
END
BASE INT INDEX""", 
                 tag="config_status_text")
    dpg.add_button(label="Get Config", callback=lambda: get_config_callback())
    dpg.add_button(label="Fire Radar", callback=lambda: run_python(main_path))
    dpg.add_button(label="Set Stop Flag (End Scan)", callback=lambda: set_stop_flag())

    with dpg.group(horizontal=True):
        with dpg.child_window(tag="stat_window", width=250, height=60):
            dpg.add_text("STAT FAIL", tag="stat_window_text", indent=53)
            dpg.bind_item_theme("stat_window", "stat_window_theme")
            dpg.bind_item_font("stat_window_text", huge_font)
        with dpg.child_window(tag="conf_window", width=250, height=60):
            dpg.add_text("CONF FAIL", tag="conf_window_text", indent=43)
            dpg.bind_item_theme("conf_window", "conf_window_theme")
            dpg.bind_item_font("conf_window_text", huge_font)

    # Progress bar between stat/conf and scan windows
    dpg.add_progress_bar(
        default_value=0, tag="scan_progress_bar", width=510, overlay=""
    )
    dpg.add_text("0 / ?", tag="scan_progress_text", indent=220)

    with dpg.child_window(tag="scan_window", width=510, height=60):
        dpg.add_text("AWAITING SCAN", tag="scan_window_text", indent=43)
        dpg.bind_item_theme("scan_window", "scan_window_theme")
        dpg.bind_item_font("scan_window_text", huge_font)

    # Add transfer button below scan status window
    dpg.add_button(
        label="Transfer & Cleanup DATAS", callback=lambda: transfer_and_cleanup_datas()
    )

    # Reset run_python allowance if config inputs are changed
    def on_config_input_change(sender, app_data, user_data):
        reset_run_python_flag()

    dpg.set_item_callback("scan_count_input", on_config_input_change)
    dpg.set_item_callback("scan_interval_input", on_config_input_change)
    dpg.set_item_callback("scan_start_input", on_config_input_change)
    dpg.set_item_callback("scan_end_input", on_config_input_change)
    dpg.set_item_callback("base_int_index_input", on_config_input_change)

dpg.create_viewport(title="Simple Radar Station", width=1215, height=800)
dpg.setup_dearpygui()
dpg.show_viewport()


while dpg.is_dearpygui_running():
    if statGood:
        dpg.set_value("stat_window_text", "STAT GOOD")
        dpg.set_value("stat_window_color", go)
    else:
        dpg.set_value("stat_window_text", "STAT FAIL")
        dpg.set_value("stat_window_color", nogo)
    if confGood:
        dpg.set_value("conf_window_text", "CONF GOOD")
        dpg.set_value("conf_window_color", go)
    else:
        dpg.set_value("conf_window_text", "CONF FAIL")
        dpg.set_value("conf_window_color", nogo)
    if scanGood:
        dpg.set_value("scan_window_text", "SCAN COMPLETE")
        dpg.set_value("scan_window_color", go)
    else:
        dpg.set_value("scan_window_text", "AWAITING SCAN")
        dpg.set_value("scan_window_color", nogo)
    dpg.render_dearpygui_frame()

dpg.destroy_context()