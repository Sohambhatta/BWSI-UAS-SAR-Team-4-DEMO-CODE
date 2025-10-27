import numpy as np

max_samples_per_message = 350

class singleScan:
    """Class to handle a single scan of data from the Pulson device, as it comes in over multiple messages which may be in the wrong order.
    Assumes that each message is filled. """

    def __init__(self, millis_time, num_samples, num_messages):
        """ Initializes the singleScan object with the given parameters.
        Parameters:
            millis_time (int): Time since boot in milliseconds for the scan.
            num_samples (int): Total Number of samples in the scan.
            num_messages (int): Total Number of messages expected for this scan.
        """

        self.num_samples = num_samples
        self.millis_time = millis_time
        self.all_data_received = [False] * num_messages
        self.data = np.zeros(num_samples, dtype=np.int32)  # Use NumPy array for efficiency

    def add_data_list(self, data_list, msg_index):

        """ Adds a list of data to the scan at the specified message index.
        Parameters:
            data_list (list): List of data to be added, should contain amplitude values in order
            msg_index (int): Index of the message to which the data belongs.
        """

        startingIndex = msg_index * max_samples_per_message
        endIndex = min(startingIndex + len(data_list), self.num_samples)
        self.data[startingIndex:endIndex] = data_list[:endIndex-startingIndex]
        self.all_data_received[msg_index] = True
            
    
    def get_data(self):
        """Returns whether all data has been received, the millis time of the scan, and the data collected,
        Data collected is a list of amplitudes, decoded into range by preconfigured range bins."""

        return all(self.all_data_received), self.millis_time, self.data

