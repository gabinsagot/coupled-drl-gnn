# Lit les fichiers Efforts.txt et en retire les valeurs de portance et de traînée en régime établi
"""Par exemple, entrée :
CompteurTemps	Temps	Cx0	Cy0	
1	0	-4.10108	1.19247	
2	0.01	-3.16285	1.23261	
3	0.02	-0.404688	-0.980231	
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def read_lift_drag(file_path):
    """
    Reads lift and drag data from a file of type 'Efforts.txt'.

    Args:
        file_path (str): Path to the 'Efforts.txt' file.
    Returns:
        tuple: A tuple containing three numpy arrays (times, cx0_values, cy0_values) over the whole simulation.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    times = []
    cx0_values = []
    cy0_values = []

    for line in lines[1:]:  # Skip header
        parts = line.split()
        if len(parts) >= 4:
            times.append(float(parts[1]))
            cx0_values.append(float(parts[2]))
            cy0_values.append(float(parts[3]))

    return np.array(times), np.array(cx0_values), np.array(cy0_values)

def readall_lift_drag(plot=False):
    """
    Reads lift and drag data from all 'Efforts.txt' files in the current directory.

    Args:
        plot (bool): Whether to plot the results.
    Returns:
        dict: A dictionary containing the lift and drag data for each simulation, whose keys are the simulation names.
    """
    data = {}
    directory = os.getcwd()
    for name in os.listdir(directory):
        dir_path = os.path.join(directory, name)
        if os.path.isdir(dir_path) and 'Efforts.txt' in os.listdir(dir_path):
            file_path = os.path.join(dir_path, 'Efforts.txt')
            with open(file_path, 'r') as f:
                print(f"Reading '{name}'")
            with open(file_path, 'r') as file:
                lines = file.readlines()

            times = []
            cx0_values = []
            cy0_values = []

            for line in lines[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 4:
                    times.append(float(parts[1]))
                    cx0_values.append(float(parts[2]))
                    cy0_values.append(float(parts[3]))

            data[name] = [np.array(times), np.array(cx0_values), np.array(cy0_values)]

    if plot:
        plotall_lift_drag(data)

    #print(data)
    return data


def plot_lift_drag(times, cx0_values, cy0_values):

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(times, cx0_values, label='Cx0 (Drag Coefficient)', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Cx0')
    plt.title('Drag Coefficient over Time')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(times, cy0_values, label='Cy0 (Lift Coefficient)', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Cy0')
    plt.title('Lift Coefficient over Time')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

def plotall_lift_drag(data):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)

    for key in data:
        times, cx0_values, cy0_values = data[key]
        plt.plot(times, cx0_values, label=key)

    plt.xlabel('Time (s)')
    plt.ylabel('Cx0')
    plt.title('Drag Coefficient over Time')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    for key in data:
        times, cx0_values, cy0_values = data[key]
        plt.plot(times, cy0_values, label=key)

    
    plt.xlabel('Time (s)')
    plt.ylabel('Cy0')
    plt.title('Lift Coefficient over Time')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

def avg_lift_drag(data, 
                  start_time=None, 
                  end_time=None, 
                  plot = False):
    
    """
    Computes the average lift and drag over [start_time, end_time] from a dictionnary data.

    Args:
        data : tuple[np.ndarray, np.ndarray, np.ndarray], Tuple of dimension 3 containing respectively time, Cx0 and Cy0 data.
        start_time (float): Start time for averaging.
        end_time (float): End time for averaging.
        plot (bool): Whether to plot the results.

    """
    end_time = max((data[0])) if end_time is None else end_time
    start_time = end_time/2.0 if start_time is None else start_time

    avg_data = {}
 
    times, cx0_values, cy0_values = data
    if start_time is not None :
        cx0_values = cx0_values[(times >= start_time) & (times <= end_time)]
        cy0_values = cy0_values[(times >= start_time) & (times <= end_time)]

    if (max(cy0_values) - min(cy0_values)) > 0.1:
        
        # Filtrer les hautes fréquences dans Cy0 pour ne garder que les oscillations principales
        #cy0_values = moving_average(cy0_values, n=70)
        #print(f"Estimating average data.")

        # Trouver les maxima locaux
        period_indices, properties = find_peaks(cy0_values, width = 13, distance=20)
            
        period_times = start_time + period_indices * (times[1] - times[0])
        #print(f"Periods at times: {period_times}")

        period_averages_cx0 = []
        period_averages_cy0 = []
        if len(period_indices) > 4:
            start_idx = period_indices[-3]
            end_idx = period_indices[-1]

            #print(f"Using average over periods between {start_time+start_idx*(times[1]-times[0])}s and {start_time+end_idx*(times[1]-times[0])}s for Cy0.")

            period_averages_cx0 = np.mean(cx0_values[start_idx:end_idx+1])
            period_averages_cy0 = np.mean(cy0_values[start_idx:end_idx+1])
            avg_cx0 = np.mean(period_averages_cx0)
            avg_cy0 = np.mean(period_averages_cy0)

        elif len(period_indices) > 1:
            start_idx = period_indices[-len(period_indices)]
            end_idx = period_indices[-1]

            #print(f"Using average over periods between {start_time+start_idx*(times[1]-times[0])}s and {start_time+end_idx*(times[1]-times[0])}s for Cy0.")

            period_averages_cx0 = np.mean(cx0_values[start_idx:end_idx+1])
            period_averages_cy0 = np.mean(cy0_values[start_idx:end_idx+1])
            avg_cx0 = np.mean(period_averages_cx0)
            avg_cy0 = np.mean(period_averages_cy0)
        
        elif len(period_indices) == 1:
                #print(f"Warning : Only one maxima found for Cy0, using basic average instead.")
                avg_cx0 = np.mean(cx0_values)
                avg_cy0 = np.mean(cy0_values)
                avg_data[key] = (round(avg_cx0,5), round(avg_cy0,5))
        
        else:
            #print(f"Warning : No maxima found for Cy0.")
            None

    else :
        #print(f"Using basic average between times {start_time} and {end_time} for Cy0.")
        avg_cx0 = np.mean(cx0_values)
        avg_cy0 = np.mean(cy0_values)

    avg_data = (round(avg_cx0,5), round(avg_cy0,5))
    #print("Average drag & lift: ", avg_data)

    if plot:
        avg_cy0s = []
        avg_cx0s = []
        angles = []
        for key in avg_data:
            angle = key.split("_")[-2].split("deg")[0]
            angles.append(float(angle))
            avg_cx0, avg_cy0 = avg_data[key]
            avg_cx0s.append(avg_cx0)
            avg_cy0s.append(avg_cy0)

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 2, 1)
        plt.scatter(angles, avg_cx0s)
        plt.xlabel('Angle (°)')
        plt.ylabel('Cx0')
        plt.title('Drag Coefficient for different inclinations')
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.scatter(angles, avg_cy0s)
        plt.xlabel('Angle (°)')
        plt.ylabel('Cy0')
        plt.title('Lift Coefficient for different inclinations')
        plt.grid()

        plt.subplot(2, 2, 3)
        plt.scatter(angles, np.array(avg_cy0s)/np.array(avg_cx0s))
        plt.xlabel('Angle (°)')
        plt.ylabel('Cy0/Cx0')
        plt.title('Lift to drag ratio')
        plt.grid()

        plt.subplot(2, 2, 4)
        plt.scatter(-np.array(avg_cx0s), np.array(avg_cy0s))
        plt.xlabel('|Cx0|')
        plt.ylabel('Cy0')
        plt.title('Lift vs Drag Coefficient')
        plt.grid()

        plt.tight_layout()
        plt.show()

    return avg_data

def moving_average(list, n=3):
    
    ret = np.cumsum(list, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret2 = np.concatenate((list[:-1+n//2],ret[n - 1:] / n,list[-n//2:]), axis=0)
    return ret2


