from typing import Dict, List

import pandas as pd
from datetime import timedelta, datetime
import numpy as np
import polyline
import math
import re

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    rev_list = []

    # Check if length of list is completely divisible by n to determine the number of times the loop should run:
    if len(lst) % n == 0:
        div = len(lst) // n
    else:
        div = (len(lst) // n) + 1


    for i in range(div):
        #  Creating sublist with n elements:
        try:
            sub_list = lst[i*n:(i*n)+n]
        except:
            sub_list = lst[i*n:]

        # Storing sublist in reverse order:
        for j in range(len(sub_list)):
            rev_list.append(sub_list[-(j+1)])

    lst = rev_list
    return lst




def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    unsorted_dict = {}
    for item in lst:
        if len(item) in unsorted_dict:
            unsorted_dict[len(item)].append(item)
        else:
            unsorted_dict[len(item)] = [item]

    myKeys = sorted(unsorted_dict.keys())


    dict = {i : unsorted_dict[i] for i in myKeys}
    return dict




def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here

    # Empty Dict
    dict = {}

    # Creating an inner function to parse dictionary recursively:
    def _flatten(sub_dict, parent_key=''):

        # Iterating over dict items
        for key, value in sub_dict.items():
            # Creating new key in the asked format
            new_key = f"{parent_key}{sep}{key}" if parent_key else key

            # Checking if current value is again a dict, if so calling _flatten in recursion
            if isinstance(value, Dict):
                _flatten(value, new_key)
            
            # Checking if current value is a list, if so calling _flatten in recursion with index appended to new key

            elif isinstance(value, List):
                for i, item in enumerate(value):
                    _flatten(item, f"{new_key}[{i}]")

            # If value is not a dict or list then directly appending it to dict
            else:
                dict[new_key] = value

    _flatten(nested_dict)
    return dict




def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    permutations = []
    
    for i in range(len(nums)):
        temp_list = nums.copy()
        val = temp_list[i]
        temp_list.pop(i)
        for j in range(len(nums)):
            temp_list.insert(j, val)
            if temp_list in permutations:
                temp_list.pop(j)
                

            else:
                perm = temp_list.copy()
                permutations.append(perm)
                temp_list.pop(j)
    return permutations



def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    formats = ["\d{2}-\d{2}-\d{4}", "\d{2}/\d{2}/\d{4}", "\d{4}.\d{2}.\d{2}"]

    dates = []
    
    for fmt in formats:
        matches = re.findall(fmt, text)
        dates.extend(matches)
    return dates



def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    def radian(val):
        return val*math.pi/ 180

    def haversine_distance(lat1, long1, lat2, long2):
        diff_lat = radian(lat1-lat2)
        diff_long = radian(long1-long2)
        
        lat1 = radian(lat1)
        lat2 = radian(lat2)

        a = pow(math.sin(diff_lat/2), 2) + pow(math.sin(diff_long/2), 2) * math.cos(lat1) * math.cos(lat2)

        radius = 6371

        distance = 2*radius*math.asin(math.sqrt(a))

        return distance
    

    list_coordinates = polyline.decode(polyline_str)

    list_coord_dist = []
    for i in range(len(list_coordinates)):
        if i == 0:
            item = (list_coordinates[i][0], list_coordinates[i][1], 0)
            list_coord_dist.append(item)
        else:
            curr_coord = list_coordinates[i]
            prev_coord = list_coordinates[i-1]
            dist = haversine_distance(curr_coord[0],curr_coord[1], prev_coord[0], prev_coord[1])
            item = (curr_coord[0], curr_coord[1], dist)
            list_coord_dist.append(item)

    df = pd.DataFrame(data=list_coord_dist, columns=['latitude', 'longitude', 'distance'])
    
    return df



def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here

    #converting list to np.array
    mat = np.array(matrix)
    print(mat)
    # Rotating matrix by 90 degrees
    
    mat_t = mat.T

    rotated = mat_t[:, ::-1]
    print(rotated)

    # calculating sum:
    sum_matrix = np.zeros(rotated.shape)
    for row_index, row in enumerate(rotated):
        for col_index, element in enumerate(row):
            sum_matrix[row_index, col_index] = np.sum(rotated[row_index,:]) + np.sum(rotated[:, col_index]) - (2*element)

    # Converting matrix back to list
    sum_matrix = sum_matrix.tolist()

    return sum_matrix




def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    WEEK_SECS = 604800
    result = []

    
    # Calculates total days elapsed in a run
    def days_counter(startDay:str, endDay:str, startTime='00:00:00', endTime='00:00:00'):

        days_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        # converting input to title case
        startDay = startDay.title()
        endDay = endDay.title()

        # converting string to timestamp
        startTime = datetime.strptime(startTime, '%H:%M:%S')
        endTime = datetime.strptime(endTime, '%H:%M:%S')

        # Calculating day count based on indices of startDay and endDay 
        start_idx = days_list.index(startDay)
        end_idx = days_list.index(endDay)
        
        day_count = end_idx - start_idx
        if day_count < 0:
            day_count+=7
        elif day_count == 0:
            if endTime < startTime:
                day_count = 7
        
        # Return total days elapsed
        return day_count
   

    # Calculates total seconds between startTime and endTime based on number of Days
    def calc_secs(startTime, endTime, day_count=0):

        # Convert string to time stamp
        startDay_secs = datetime.strptime(startTime, '%H:%M:%S')
        endDay_secs = datetime.strptime(endTime, '%H:%M:%S')

        # placeholders
        day_start_time = datetime.strptime('00:00:00', '%H:%M:%S')
        day_end_time = day_start_time + timedelta(days=1)

        # calculating seconds
        day_secs = 86400
        if day_count == 0:
            total_seconds= (endDay_secs - startDay_secs).total_seconds() + 1
        elif day_count == 1:
            total_seconds =  (day_end_time - startDay_secs).total_seconds() + (endDay_secs - day_start_time).total_seconds() + 1
        else:
            total_seconds = (day_end_time - startDay_secs).total_seconds() + (day_count-2) * day_secs + (endDay_secs - day_start_time).total_seconds() + 1

        # return total seconds in a run
        return total_seconds

    # Checks if current iteration is in continuation with the previous one: 
    def next_iter_check(prev_endDay, prev_endTime, curr_startDay, curr_startTime):
        day_diff = days_counter(prev_endDay, curr_startDay)
        time_diff = calc_secs(prev_endTime, curr_startTime)
        return (day_diff == 1 and time_diff <= 1) or (day_diff == 0 and time_diff <= 1)
    

    # Main logic:
    '''
    In ervery group
    count and add seconds for every consecutive run.
    If it comes out to be more than total seconds in a week,
    then append True to results list for that group 
    else apend False.
    Return the results list as pd.Series.
    '''

    for (id, id_2), group in df.groupby(['id', 'id_2']):
        
        total_secs = 0
        temp_list = []
        data = pd.DataFrame(group)
        data.reset_index(inplace=True, drop=True)

        for ind in range(len(data)):

            if ind == 0:
                days = days_counter(data.loc[ind,'startDay'], data.loc[ind,'endDay'], data.loc[ind,'startTime'], data.loc[ind,'endTime']) 
                total_secs += calc_secs(data.loc[ind, 'startTime'], data.loc[ind, 'endTime'], day_count=days)

            elif next_iter_check(data.loc[ind - 1,'endDay'], data.loc[ind - 1,'endTime'], data.loc[ind, 'startDay'], data.loc[ind, 'startTime']):
                i = ind
                while i< len(data) and next_iter_check(data.loc[i - 1,'endDay'], data.loc[i,'endTime'], data.loc[i,'startDay'], data.loc[i,'startTime']):
                    days =  days_counter(data.loc[i,'startDay'], data.loc[i,'endDay'],data.loc[i,'startTime'], data.loc[i,'endTime'])
                    total_secs += calc_secs(data.loc[i,'startTime'], data.loc[i,'endTime'], day_count=days) + 1
                    i += 1


            if total_secs >= WEEK_SECS:
                temp_list.append(True)
            else:
                temp_list.append(False)

            total_secs = 0
                    
        if True in temp_list:
            result.append(True)
        else:
            result.append(False)
    
    return pd.Series(result)

        
            

