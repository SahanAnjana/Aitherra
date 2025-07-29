# Flight Data Visualization

This project provides tools for visualizing and analyzing flight data from MATLAB (.mat) files. It processes flight parameters and generates interactive visualizations to help analyze aircraft performance and flight characteristics.

## Features

- Load and parse flight data from MATLAB .mat files
- Visualize multiple flight parameters including:
  - Altitude and vertical speed
  - Airspeed (Calibrated and True) and Mach number
  - Engine parameters (N1 and EGT for all engines)
  - Flight controls (Pitch, Roll, Heading)
  - 3D flight path visualization
- Handle various data formats and structures
- Robust error handling for missing or malformed data

## Requirements

- Python 3.7+
- Required Python packages:
  - NumPy
  - SciPy
  - Matplotlib
  - datetime

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SahanAnjana/Aitherra.git
   cd Aitherra
   ```

2. Install the required packages:
   ```bash
   pip install numpy scipy matplotlib
   ```

## Usage

1. Place your MATLAB .mat file in the project directory
2. Update the filename in `main.py` if needed (default: '686200107191515.mat')
3. Run the script:
   ```bash
   python main.py
   ```

## Output

The script will generate several plots showing:
1. Altitude and vertical speed over time
2. Airspeed (CAS/TAS) and Mach number
3. Engine N1 parameters for all engines
4. Exhaust Gas Temperature (EGT) for all engines
5. Aircraft attitude (Pitch, Roll, Heading)
6. A 3D flight path visualization (if position data is available)

## Data Sources

This project can be used with flight data from NASA's DASHlink repository:
- [Boeing 737 Flight Data 1](https://c3.ndc.nasa.gov/dashlink/resources/663/)
- [Boeing 737 Flight Data 2](https://c3.ndc.nasa.gov/dashlink/resources/664/)

These datasets contain real flight data from Boeing 737 aircraft, including flight parameters, engine data, and system status information.

## Data Format

The script expects MATLAB .mat files containing flight data with the following parameters (case-sensitive):
- `ALT`: Altitude data
- `VRTG`: Vertical speed
- `CAS`: Calibrated Airspeed
- `TAS`: True Airspeed
- `MACH`: Mach number
- `PTCH`: Pitch angle
- `ROLL`: Roll angle
- `HDGS`: Heading
- `LONG`: Longitude (for 3D path)
- `N1_1` to `N1_4`: Engine N1 parameters
- `EGT_1` to `EGT_4`: Exhaust Gas Temperature

## Acknowledgements

- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)