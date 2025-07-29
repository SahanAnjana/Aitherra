import scipy.io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np

# Load the data
data = scipy.io.loadmat('686200107191515.mat', simplify_cells=True)

# Print available variables for debugging
print("Available variables:", [k for k in data.keys() if not k.startswith('__')])

# Function to safely get data with error handling
def get_data(key):
    if key in data:
        item = data[key]
        # Handle structured array
        if isinstance(item, dict) and 'data' in item:
            return np.array(item['data']).squeeze()
        # Handle regular array
        elif isinstance(item, (np.ndarray, list)):
            return np.array(item).squeeze()
        # Handle scalar values
        else:
            return np.array([item])
    else:
        print(f"Warning: {key} not found in data")
        return np.array([])

# Get all available data
alt = get_data('ALT')
vrtg = get_data('VRTG')
cas = get_data('CAS')
tas = get_data('TAS')
mach = get_data('MACH')
ptch = get_data('PTCH')
roll = get_data('ROLL')
hdgs = get_data('HDGS')
long = get_data('LONG')

# Get engine parameters
n1 = [get_data(f'N1_{i}') for i in range(1, 5)]
egt = [get_data(f'EGT_{i}') for i in range(1, 5)]

# Print data shapes for debugging
print(f"Altitude data shape: {alt.shape if hasattr(alt, 'shape') else 'scalar'}")
print(f"Sample altitude values: {alt[:5] if len(alt) > 0 else 'No data'}")

if len(alt) > 0:
    # Create time vector
    start_time = datetime(2020, 1, 4, 11, 17, 24)  # From filename
    timestamps = [start_time + timedelta(seconds=float(i)) for i in range(len(alt))]
    
    # Create a figure with subplots
    plt.figure(figsize=(18, 24))
    plt.suptitle(f'Flight Data Analysis - {start_time}', fontsize=16, y=1.02)

    # 1. Altitude and Vertical Speed
    plt.subplot(5, 1, 1)
    plt.plot(timestamps, alt, 'b-', label='Altitude (ft)')
    plt.ylabel('Altitude (ft)')
    plt.grid(True)
    plt.legend(loc='upper left')

    if len(vrtg) > 0 and len(vrtg) == len(alt):
        ax2 = plt.gca().twinx()
        ax2.plot(timestamps, vrtg, 'r-', label='Vertical Speed (ft/min)')
        ax2.set_ylabel('Vertical Speed (ft/min)', color='r')
        ax2.legend(loc='upper right')
    plt.title('Altitude and Vertical Speed')

    # 2. Airspeed and Mach
    plt.subplot(5, 1, 2)
    if len(cas) > 0 and len(cas) == len(alt):
        plt.plot(timestamps, cas, 'b-', label='Calibrated Airspeed (kts)')
    if len(tas) > 0 and len(tas) == len(alt):
        plt.plot(timestamps, tas, 'g-', label='True Airspeed (kts)')
    plt.ylabel('Speed (kts)')
    plt.grid(True)
    plt.legend()

    if len(mach) > 0 and len(mach) == len(alt):
        ax2 = plt.gca().twinx()
        ax2.plot(timestamps, mach, 'r-', label='Mach')
        ax2.set_ylabel('Mach', color='r')
        ax2.legend()
    plt.title('Airspeed and Mach Number')

    # 3. Engine Parameters (N1)
    plt.subplot(5, 1, 3)
    for i, n1_data in enumerate(n1, 1):
        if len(n1_data) > 0 and len(n1_data) == len(alt):
            plt.plot(timestamps, n1_data, label=f'Engine {i} N1 (%)')
    plt.ylabel('N1 (% of max)')
    plt.grid(True)
    plt.legend()
    plt.title('Engine Fan Speed (N1)')

    # 4. Engine Parameters (EGT)
    plt.subplot(5, 1, 4)
    for i, egt_data in enumerate(egt, 1):
        if len(egt_data) > 0 and len(egt_data) == len(alt):
            plt.plot(timestamps, egt_data, label=f'Engine {i} EGT (C)')
    plt.ylabel('EGT (C)')
    plt.grid(True)
    plt.legend()
    plt.title('Exhaust Gas Temperature (EGT)')

    # 5. Flight Controls
    plt.subplot(5, 1, 5)
    if len(ptch) > 0 and len(ptch) == len(alt):
        plt.plot(timestamps, ptch, 'b-', label='Pitch (deg)')
    if len(roll) > 0 and len(roll) == len(alt):
        plt.plot(timestamps, roll, 'r-', label='Roll (deg)')
    if len(hdgs) > 0 and len(hdgs) == len(alt):
        plt.plot(timestamps, hdgs, 'g-', label='Heading (deg)')
    plt.ylabel('Degrees')
    plt.grid(True)
    plt.legend()
    plt.title('Aircraft Attitude')

    # Format x-axis
    for i in range(1, 6):
        try:
            plt.subplot(5, 1, i)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.gcf().autofmt_xdate()
            if i < 5:  # Don't show xlabel for top 4 subplots
                plt.tick_params(labelbottom=False)
            else:
                plt.xlabel('Time (UTC)')
        except Exception as e:
            print(f"Error formatting subplot {i}: {e}")
            continue

    plt.tight_layout()
    
    # 3D plot if we have position data
    if len(long) > 0 and len(alt) > 0 and len(long) == len(alt):
        try:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(long, np.ones_like(long), alt, 'b-')  # Using dummy y-coordinate
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude (dummy)')
            ax.set_zlabel('Altitude (ft)')
            ax.set_title('3D Flight Path (Top-Down View)')
            plt.tight_layout()
        except Exception as e:
            print(f"Could not create 3D plot: {e}")
    
    plt.show()
else:
    print("No valid altitude data found for plotting")