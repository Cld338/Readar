def loadJson(Dir):
    import json
    """Json 파일에 저장된 데이터 불러오기"""
    with open(Dir, 'r') as file:
        ls = json.load(file)
    return ls

def saveJson(Dir, data):
    import json
    """데이터를 Json 파일에 저장하기"""
    with open(Dir, 'w', encoding="utf-8") as file:
        json.dump(data, file, indent="\t")



def createDirectory(directory):
    import os
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def filesInFolder(Dir, extention=0):
    if extention:
        return [file.replace(f".{extention}", "") for file in os.listdir(Dir) if file.endswith(f".{extention}")]
    else:
        return [file for file in os.listdir(Dir)]
    

def log(text):
    import datetime, os
    currDir = os.path.dirname(os.path.realpath(__file__))
    with open(f'{currDir}/log.txt', 'a') as file:
        file.write(f"{datetime.datetime.now()} - {text}\n")
import os
currDir = os.path.dirname(os.path.realpath(__file__))