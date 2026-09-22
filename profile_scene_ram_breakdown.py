#####################################################################
#
# Python code to profile the RAM breakdown of scene elements in a 
# Helios simulation - this is intended to provide information to aid
# splitting up the overall scene so it can fit into normal RAM
# requirements (ideally sub 100GB)
#
# Written by N. Origo (22/09/2026)
#
#####################################################################

import numpy as np
import psutil
import time

class LogRAM:

    def __init__(self, pid, logtime=5):
        """
        Class to log memory usage over the course a process

        Parameters:
        -----------
        pid     : int
                  The PID of the process to track.
        logtime : int
                  The number of seconds to wait before querying.
        """
        ram = self.get_ram_use(pid)
        self.start_time = time.time()
        ram_profile = []
        while ram!=0:
            ram = self.get_ram_use(pid)
            ram_profile.append(ram)
            time.sleep(logtime)
        
        self.end_time = time.time()
        self.max_ram = np.max(ram_profile)
        self.ram_profile = np.array(ram_profile)


    def get_ram_use(self, pid):
        """
        Gets the RAM usage based on the PID, returns 0 if the process
        ended.
        """
        try:
            ram = psutil.Process(pid).memory_info().rss / (1024**2)
        except psutil.NoSuchProcess:
            ram = 0
        return ram
