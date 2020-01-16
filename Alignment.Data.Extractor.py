import numpy as np
import os, argparse

def Data_Generate(data_Path):
    #Feature load
    with open("Phoneme_Feature.txt", "r", encoding='UTF8') as f:
        readLines = f.readlines()

    #Phoneme list and feature dict
    phoneme_List = []
    for readLine in readLines[1:]:
        phoneme_List.append(readLine.strip().split("\t")[0])

    path_Dict = Load_Path_Dict(data_Path)

    data_List = [['Word', 'Talker', 'Phoneme', 'Min', 'Max']]
    for (word, talker), path in path_Dict.items():
        data_List.extend(Load_Data(word, talker, path))
        
    data_List = ['\t'.join(['{}'.format(x) for x in data]) for data in data_List if data[2] in phoneme_List]
    with open('Alignment_Data.txt', 'w') as f:
        f.write('\n'.join(data_List))

def Load_Path_Dict(path):
    path_Dict = {}
    for root, _, files in os.walk(path):
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext.upper() != '.TextGrid'.upper():
                continue
            elif not os.path.exists(os.path.join(root, '{}{}'.format(name, '.wav')).replace('\\', '/')):
                continue
            word, talker = name.upper().split('_')
            path_Dict[word, talker] = os.path.join(root, filename).replace('\\', '/')
        
    return path_Dict

def Load_Data(word, talker, path):
    with open(path, 'r') as f:
        lines = f.readlines()

    data_List = []

    is_Phoneme_Info = False    
    phoneme = None
    xMin = None
    xMax = None
    for line in lines:
        line = line.strip()
        if line.startswith('intervals: size'):
            is_Phoneme_Info = True
            continue
        if is_Phoneme_Info:
            if line.startswith('intervals'):
                if not phoneme is None and phoneme != 'sil' and phoneme != '':
                    data_List.append([word, talker, phoneme[:2], xMin, xMax])
                phoneme = None
                xMin = None
                xMax = None
            elif line.startswith('xmin = '):
                xMin = float(line[7:])
            elif line.startswith('xmax = '):
                xMax = float(line[7:])
            elif line.startswith('text = '):
                phoneme = line[8:-1]

            if line == 'item [2]:':
                if not phoneme is None and phoneme != 'sil':
                    data_List.append([word, talker, phoneme[:2], xMin, xMax])
                break

    return data_List

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-d', '--path', required=True)
    argument_Dict = vars(argParser.parse_args())

    Data_Generate(argument_Dict['path'])
