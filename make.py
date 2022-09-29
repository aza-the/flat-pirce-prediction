import csv
import json
import re
import os

import numpy as np

FOLDER_PATH = 'files'
JSON_OF_RAW_DATA_FILE_NAME = 'ready.json'
CSV_FILE_NAME = './files/flats.csv'
CSV_VECTORIZED_FILE_NAME = './files/flats_v.csv'


def pre_main():
    """ 
        Reads the JSON_OF_RAW_DATA_FILE_NAME and makes a csv file from it. 
    """
    
    try:
        os.mkdir(FOLDER_PATH)
    except Exception:
        pass

    flats = {}

    with open(JSON_OF_RAW_DATA_FILE_NAME) as file:
        flats = json.load(file)

    print(len(flats))

    result = []

    kitchens = []
    for key in flats:
        try:
            unit = flats[key]['Кухня']
            unit = re.findall(r'\d+', unit)[0]
            kitchens.append(unit)
        except Exception:
            pass

    constructeds = []
    for key in flats:
        try:
            unit = flats[key]['Построен']
            unit = re.findall(r'\d+', unit)[0]
            constructeds.append(unit)
        except Exception:
            pass 

    kitchens = np.array(kitchens, dtype=float)
    constructeds = np.array(constructeds, dtype=int)

    kit_mean = int(kitchens.mean())
    con_mean = int(constructeds.mean())

    print(f'kit_mean: {kit_mean}')
    print(f'con_mean: {con_mean}')

    for key in flats:

        district = flats[key]['district']

        price = int(flats[key]['price'])

        try:
            metro_list = list(flats[key]['metro'])
            metro_name = ''
            metro_time = 0
            metro_get_type = ''
            idx = -1
            for i in metro_list:
                idx = flats[key]['metro'][i].find('пешком')
                if(idx != -1):
                    metro_name = i
                    break

            if idx != -1:
                metro_time = flats[key]['metro'][metro_name]
                metro_time = re.findall(r'\d+', metro_time)[0]
                metro_get_type = 'пешком'
            else:
                for i in metro_list:
                    idx = flats[key]['metro'][i].find('транспорте')
                    if(idx != -1):
                        metro_name = i
                        break
                if idx != -1:
                    metro_time = flats[key]['metro'][metro_name]
                    metro_time = re.findall(r'\d+', metro_time)[0]
                    metro_get_type = 'транспорт'
                else:
                    raise Exception
        except Exception:
            continue

        try:
            size = flats[key]['Общая']
            size = re.findall(r'\d+', size)[0]
        except Exception:
            continue

        try:
            kitchen = flats[key]['Кухня'] 
            kitchen = re.findall(r'\d+', kitchen)[0] 
        except Exception:
            kitchen = kit_mean      

        floor = flats[key]['Этаж']
        floor = re.findall(r'\d+', floor)[0]

        floors = flats[key]['Этаж']
        floors = re.findall(r'\d+', floors)[1]

        try:
            constructed = int(flats[key]['Построен'])
        except Exception:
            constructed = con_mean

        try:
            fix = flats[key]['Ремонт']
        except Exception:
            fix = 'None'

        try:
            type_of_building = flats[key]['Тип дома']
        except Exception:
            type_of_building = 'None'

        try:
            type_of_walls = flats[key]['Тип перекрытий']
        except Exception:
            type_of_walls = 'None'


        result.append([
            price,  
            district, 
            metro_name, 
            metro_time, 
            metro_get_type,
            size, 
            kitchen, 
            floor, 
            floors, 
            constructed,
            fix,
            type_of_building,
            type_of_walls,
        ])

    with open(CSV_FILE_NAME, "w", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                'price',  
                'district', 
                'metro_name', 
                'metro_time', 
                'metro_get_type',
                'size', 
                'kitchen', 
                'floor', 
                'floors', 
                'constructed',
                'fix',
                'type_of_building',
                'type_of_walls',
            ]
        )
        for flat in result:
            writer.writerow(flat)

def dict_maker(idx: int, readed: list) -> dict:
    """
        Gets list with special index and makes a dictionary from it.
    """

    new_dict = {}
    count = 0
    for r in readed:
        if r[idx] not in new_dict.keys():
            new_dict[r[idx]] = count
            count += 1
    return new_dict

def make():
    """ 
        Reads the CSV_FILE_NAME and makes a csv file 
        with replacing all str values with nums from it. 
    """

    readed = []
    with open(CSV_FILE_NAME) as file:
        reader = csv.reader(file)
        for row in reader:
            readed.append(np.array(row))

    distrcit_dict         = dict_maker(1, readed)
    metro_name_dict       = dict_maker(2, readed)
    metro_get_type_dict   = dict_maker(4, readed)
    fix_dict              = dict_maker(10, readed)
    type_of_building_dict = dict_maker(11, readed)
    type_of_walls_dict    = dict_maker(12, readed)

    file_names = [
        './files/distrcit_dict.json',
        './files/metro_name_dict.json', 
        './files/metro_get_type_dict.json',
        './files/fix_dict.json', 
        './files/type_of_building_dict.json', 
        './files/type_of_walls_dict.json',
    ]
    files = [
        distrcit_dict,
        metro_name_dict, 
        metro_get_type_dict,
        fix_dict, 
        type_of_building_dict, 
        type_of_walls_dict,
    ]
    count = 0
    for i in file_names:
        with open(i, "w") as file:
            json.dump(files[count], file, indent=4, ensure_ascii=False)
        count += 1

    for i in range(1, len(readed)):
        readed[i][1] = distrcit_dict[readed[i][1]]
        readed[i][2] = metro_name_dict[readed[i][2]]
        readed[i][4] = metro_get_type_dict[readed[i][4]]
        readed[i][10] = fix_dict[readed[i][10]]
        readed[i][11] = type_of_building_dict[readed[i][11]]
        readed[i][12] = type_of_walls_dict[readed[i][12]]

    with open(CSV_VECTORIZED_FILE_NAME, "w") as file:
        writer = csv.writer(file)
        for i in readed:
            writer.writerow(i)

