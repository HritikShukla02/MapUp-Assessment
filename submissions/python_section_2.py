import pandas as pd
import numpy as np
import math
from datetime import datetime

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here

    # Write your logic here
    # Creating a placeholder output matrix
    values= np.zeros([df['id_start'].nunique(), df['id_start'].nunique()])
    matrix = pd.DataFrame(index=df['id_start'].unique(), columns=df['id_start'].unique(), df= values)

    # Filling the matrix with given direct distances
    for _, row in df.iterrows():
        id_start, id_end, distance = row['id_start'], row['id_end'], row['distance']
        matrix.loc[id_start, id_end] = distance
        matrix.loc[id_end, id_start] = distance 

    # Calculating distances and filling teh matrix
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):  # Only loop through upper triangle of the matrix
            if matrix.iloc[i, j] == 0 and i != j:  # Skip diagonal and update only unfilled (0) values
                
                # Check if a direct distance exists in the dataframe
                dist = df[(df['id_start'] == matrix.index[i]) & (df['id_end'] == matrix.columns[j])]['distance'].values
                
                if len(dist) > 0:  # If a direct distance is available
                    dist = dist[0]  # Extract the scalar value
                    matrix.iloc[i, j] = dist  # Set the direct distance
                    matrix.iloc[j, i] = dist  # Ensure symmetry
                else:
                    # If no direct distance, find the shortest route through an intermediate point
                    cumulative_dist = float('inf')  # Initialize with a large value to find the minimum
                    for k in range(matrix.shape[0]):  # Check all possible intermediate points
                        if matrix.iloc[i, k] > 0 and matrix.iloc[k, j] > 0:  # Both segments must exist
                            cumulative_dist = min(cumulative_dist, matrix.iloc[i, k] + matrix.iloc[k, j])
                    
                    if cumulative_dist < float('inf'):  # If we found a valid route
                        matrix.iloc[i, j] = cumulative_dist
                        matrix.iloc[j, i] = cumulative_dist  # Ensure symmetry


    return matrix


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here


    rows = []

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if i != j and df.iloc[i,j]:
                rows.append((df.index[i], df.columns[j], df.iloc[i,j]))

    result = pd.DataFrame(columns=['id_start', 'id_end', 'distance'], data=rows)

    return result
    


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here


    # Create a dataframe wirh the averages for all values of id_start: 
    avg_df = df.groupby(by='id_start', as_index=False).agg({'distance':pd.Series.mean})

    # Get the average value for reference_id:
    ref_avg = avg_df[avg_df['id_start'] == reference_id]['distance']

    # Decide on the range of distance values in result dataframe:
    ref_floor = math.floor(ref_avg - ref_avg*0.1)
    ref_ceil = math.ceil(ref_avg + ref_avg*0.1)

    # Get the results df with all IDs whose average distance lies within decided range average distance values:
    result_df = avg_df[(ref_floor <= avg_df['distance']) & (avg_df['distance'] <= ref_ceil)]

    # Sort the results df by distances:
    result_df.sort_values(by='distance', inplace=True)

    return result_df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here

    # Define coefficients:
    COEF_MOTO = 0.8
    COEF_CAR = 1.2
    COEF_RV = 1.5
    COEF_BUS = 2.2
    COEF_TRUCK = 3.6

    # Calculate toll rates
    df['moto'] = df['distance'] * COEF_MOTO
    df['car'] = df['distance'] * COEF_CAR
    df['rv'] = df['distance'] * COEF_RV
    df['bus'] = df['distance'] * COEF_BUS
    df['truck'] = df['distance'] * COEF_TRUCK

    # Drop distance column
    df.drop(columns=['distance'], axis=1, inplace=True)

    return df




def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here

    
    # Defining timestamps:
    time_00_00_00 = datetime.strptime('00:00:00', '%H:%M:%S').time()
    time_10_00_00 = datetime.strptime('10:00:00', '%H:%M:%S').time()
    time_18_00_00 = datetime.strptime('18:00:00', '%H:%M:%S').time()

    # Creating time interval dataframe to be added for each row of input df:
    time_range_data = [['Monday','00:00:00','Friday', '10:00:00'],
                       ['Monday','10:00:00','Friday', '18:00:00'],
                       ['Monday','18:00:00','Friday', '23:59:59'],
                       ['Saturday','00:00:00','Sunday', '23:59:59'],]
    
    time_range_df = pd.DataFrame(data=time_range_data, columns=['startDay', 'startTime', 'endDay', 'endTime'])

    # Updating data type of startTime and endTime from string to time:
    time_range_df['startTime'] = pd.to_datetime(time_range_df['startTime'], format='%H:%M:%S').dt.time
    time_range_df['endTime'] = pd.to_datetime(time_range_df['endTime'], format='%H:%M:%S').dt.time

    # Adding this time range df to input df for each row:
    time_df = pd.DataFrame(columns=['id_start', 'id_end', 'startDay', 'startTime', 'endDay', 'endTime'])
    for i, row in df.iterrows():
        temp_df1 = pd.DataFrame(data=[row[['id_start','id_end']].values]*4, columns=['id_start','id_end'])
        temp_df1['id_start'] = temp_df1['id_start'].astype(int)
        temp_df1['id_end'] = temp_df1['id_end'].astype(int)

        row_df = pd.concat([temp_df1, time_range_df], axis=1)
        time_df = pd.concat([time_df, row_df])

    
    result_df = df.merge(time_df, on=['id_start', 'id_end'])
    result_df = result_df[['id_start', 'id_end', 'distance','startDay', 'startTime', 'endDay', 'endTime', 'moto', 'car', 'rv', 'bus', 'truck']]



    # Updating the toll values based on the day and time interval:
    for i in range(len(result_df)):
        if result_df.loc[i,'startDay'] == 'Monday':
            st_time = result_df.loc[i,'startTime']

            if (st_time == time_00_00_00) or (st_time == time_18_00_00):
                values = result_df.iloc[i,7:] * 0.8
                result_df.iloc[i,7:] = [round(val,2) for val in values]
                
            elif st_time == time_10_00_00:
                values = result_df.iloc[i,7:] * 1.2
                result_df.iloc[i,7:] = [round(val,2) for val in values]

        elif result_df.loc[i,'startDay'] == 'Saturday':
            values = result_df.iloc[i,7:] * 0.7
            result_df.iloc[i,7:] = [round(val,2) for val in values]
    

    return result_df

