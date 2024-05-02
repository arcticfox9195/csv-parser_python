import csv
from datetime import datetime, date, time
from urllib.parse import urlparse
import re
import random
import time

def isEmpty(s):
    if s == '':
        return True
    else:
        return False

def isValidDatetime(dt):
    if isinstance(dt, int) == True: return False
    if "T" in dt:
        arr = dt.split('T')
        if len(arr) == 2:   #分割成date跟time分別確認
            if isValidDate(arr[0]) == True & isValidTime(arr[1]) == True:
                return True
            else:
                return False
        else:
            return False
    elif " " in dt:
        arr = dt.split(' ')
        if len(arr) == 2:
            if isValidDate(arr[0]) == True & isValidTime(arr[1]) == True:
                return True
            else:
                return False
        elif len(arr) == 4: #考慮若前面的date是用空格為分隔符
            tempDate = arr[0] + ' ' + arr[1] + ' ' + arr[2]
            if isValidDate(tempDate) == True & isValidTime(arr[3]) == True:
                return True
            else:
                return False
        else:
            return False
    else:
        return False
    
def isValidDate(d):
    if isinstance(d, int) == True: return False
    d = d.strip()
    if "/" in d: #先用if elif區分不同分隔符的判斷
        try:
            time.strptime(d, "%Y/%m/%d")
            return True
        except: #用三層try依序確認三種格式 全部都不符合代表不為date則return false
            try:
                time.strptime(d, "%d/%m/%Y")
                return True
            except:
                try:
                    time.strptime(d, "%m/%d/%Y")
                    return True
                except:
                    try:
                        time.strptime(d, "%y/%m/%d")
                        return True
                    except:
                        try:
                            time.strptime(d, "%d/%m/%y")
                            return True
                        except:
                            try:
                                time.strptime(d, "%m/%d/%y")
                                return True
                            except:
                                return False

    elif "-" in d:
        try:
            time.strptime(d, "%Y-%m-%d")
            return True
        except:
            try:
                time.strptime(d, "%d-%m-%Y")
                return True
            except:
                try:
                    time.strptime(d, "%m-%d-%Y")
                    return True
                except:
                    try:
                        time.strptime(d, "%y-%m-%d")
                        return True
                    except:
                        try:
                            time.strptime(d, "%d-%m-%y")
                            return True
                        except:
                            try:
                                time.strptime(d, "%m-%d-%y")
                                return True
                            except:
                                return False
                            
    elif " " in d:
        try:
            time.strptime(d, "%Y %m %d")
            return True
        except:
            try:
                time.strptime(d, "%d %m %Y")
                return True
            except:
                try:
                    time.strptime(d, "%m %d %Y")
                    return True
                except:
                    try:
                        time.strptime(d, "%y %m %d")
                        return True
                    except:
                        try:
                            time.strptime(d, "%d %m %y")
                            return True
                        except:
                            try:
                                time.strptime(d, "%m %d %y")
                                return True
                            except:
                                return False
                            
    else:
        try:
            time.strptime(d, "%Y年%m月%d日")
            return True
        except:
            try:
                time.strptime(d, "%Y년%m월%d일")
                return True
            except:
                try:
                    time.strptime(d, "%y年%m月%d日")
                    return True
                except:
                    try:
                        time.strptime(d, "%y년%m월%d일")
                        return True
                    except:
                        return False

def isValidTime(t):
    try:
        if len(t) > 6: datetime.strptime(t, "%H:%M:%S")
        else: datetime.strptime(t, "%H:%M")
        return True
    except:
        return False

def isValidUrl(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])  # 檢查是否存在 scheme 和 netloc
    except ValueError:
        return False
    
def isValidEmail(email):
    if isinstance(email, int) == True: return False
    if re.match(r"[^@]+@[^@]+\.[^@]+", email): return True  
    return False  

def isPercentage(perNum):
    if isinstance(perNum, int) == True: return False
    if perNum.endswith("%"):
        numPart = perNum[:-1]
        return numPart.isdigit()
    
    return False

def isCurrency(value):
    if isinstance(value, int) == True: return False
    value = value.strip()
    if value.startswith("$"):
        numPart = value[1:]
        return numPart.isdigit()
    
    return False

def isNumType(string):
    if isinstance(string, int) == True: return True
    isNum1 = True
    isNum2 = True
    tempFlag = False    # 是否已存在小數點

    string = string.strip()
    if string.isnumeric() == True: return True
    #print(string)
    if len(string) <= 0: return False
    if string[0] == "+": string = string.strip("+")    # 去掉正負號
    elif string[0] == "-": string = string.strip("-")

    if "," in string: string = string.split(",")    # 以逗號或點分隔
    elif "." in string: string = string.split(".")
    else: isNum1 = False

    # case 1 (以千為單位)
    if string[0] == "" or len(string[0]) > 3: isNum1 = False
    if string[0].isnumeric() == False: isNum1 = False

    for i in range(1, len(string)):
        if i == len(string) - 1 and "." not in string[i]:
            if string[i].isnumeric() == False:
                isNum1 = False
                break

        elif len(string[i]) == 3 and string[i].isnumeric() == True: continue
        elif "." in string[i]:
            if tempFlag == True: isNum1 = False

            temp = string[i].split(".")
            for j in range(2):
                if len(temp[j]) == 3 and temp[j].isnumeric() == True: continue
                elif i == len(string) - 1 and j == 1:
                    if temp[j].isnumeric() == False: 
                        isNum1 = False
                        break
                    
        else: 
            isNum1 = False
            break

    if isNum1 == True: return True

    # case 2
    if len(string) > 2: isNum2 = False
    if string[0].isnumeric() == False: isNum2 = False
    
    if "e+" in string[len(string)-1]: string = string[len(string)-1].split("e+")
    elif "e-" in string[len(string)-1]: string = string[len(string)-1].split("e-")
    elif "e" in string[len(string)-1]: string = string[len(string)-1].split("e")

    for i in string:
        for j in i:
            if j.isnumeric() == True: continue
            else:
                isNum2 = False
                break

    if "," not in string and "." not in string: isNum2 = True

    if "e+" in string: string = string[len(string)-1].split("e+")
    elif "e-" in string: string = string[len(string)-1].split("e-")
    elif "e" in string: string = string[len(string)-1].split("e")

    for i in string:
        if i.isnumeric() == True: continue
        else:
            isNum2 = False
            break

    if isNum2 == True: return True
    return False

def isAlphanumeric(str1): 
    str2 = str1.strip()
    after = str2.split(" ")
    #print(after)

    if len(after) == 3:    # 第一種 alternative
        for i in after[0]:
            if ord(i) >= ord("0") and ord(i) <= ord("9"): continue
            else: return False
            
        for i in after[1]:
            if ord(i) >= ord("a") and ord(i) <= ord("z"): continue
            elif ord(i) >= ord("A") and ord(i) <= ord("Z"): continue
            else: return False

        for i in after[2]:
            if ord(i) >= ord("0") and ord(i) <= ord("9"): continue
            elif ord(i) >= ord("a") and ord(i) <= ord("z"): continue
            elif ord(i) >= ord("A") and ord(i) <= ord("Z"): continue
            elif ord(i) == ord(" ") or ord(i) == ord(".") or ord(i) == ord("!") or ord(i) == ord("?"): continue
            elif ord(i) == ord("(") or ord(i) == ord(")") or ord(i) == ord("[") or ord(i) == ord("]") or ord(i) == ord("{") or ord(i) == ord("}"): continue
            else: return False

    elif len(after) == 2:    # 第二種 alternative
        for i in after[0]:
            if ord(i) >= ord("0") and ord(i) <= ord("9"): continue
            elif ord(i) >= ord("a") and ord(i) <= ord("z"): continue
            elif ord(i) >= ord("A") and ord(i) <= ord("Z"): continue
            else: return False

        for i in after[1]:
            if ord(i) >= ord("0") and ord(i) <= ord("9"): continue
            elif ord(i) >= ord("a") and ord(i) <= ord("z"): continue
            elif ord(i) >= ord("A") and ord(i) <= ord("Z"): continue
            elif ord(i) == ord(" ") or ord(i) == ord(".") or ord(i) == ord("!") or ord(i) == ord("?"): continue
            elif ord(i) == ord("(") or ord(i) == ord(")") or ord(i) == ord("[") or ord(i) == ord("]") or ord(i) == ord("{") or ord(i) == ord("}"): continue
            else: return False

    else: return False

    return True

def makeType(csvList):
    typeArray = []
    typeNumArray = []

    for i in csvList: 
        subArray = []
        subNum = []

        for j in i:            
            if isEmpty(j) == True:
                subArray.append("empty")
                subNum.append(1)

            elif isValidDatetime(j) == True: 
                subArray.append("datetime")
                subNum.append(2)
            
            elif isValidDate(j) == True: 
                subArray.append("date")
                subNum.append(3)
            
            elif isValidTime(j) == True: 
                subArray.append("time")
                subNum.append(4)

            elif isValidUrl(j) == True: 
                subArray.append("url")
                subNum.append(5)

            elif isValidEmail(j) == True: 
                subArray.append("email")
                subNum.append(6)

            elif isPercentage(j) == True: 
                subArray.append("percentage")
                subNum.append(7)

            elif isCurrency(j) == True: 
                subArray.append("currency")
                subNum.append(8)

            elif isNumType(j) == True: 
                subArray.append("num type")
                subNum.append(9)

            elif j.strip() == "N/A" or j.strip() == "n/a": 
                subArray.append("n/a type")
                subNum.append(10)

            elif isAlphanumeric(j) == True: 
                subArray.append("alphanumeric")
                subNum.append(11)

            else: 
                subArray.append("no type")
                subNum.append(12)
        
        typeArray.append(subArray)
        typeNumArray.append(subNum)

    return typeNumArray

def process_table(inputList):
    choice = random.choice(['merge', 'delete'])     # delete: 0, merge: 1
    
    if choice == 'delete': 
        row = random.randint(0, len(inputList)-1)
        column = random.randint(0, len(inputList[row])-1)
        inputList[row].pop(column)
        inputList[row].append('')
        choice_num = 0

    else:
        row = random.randint(0, len(inputList)-2)
        column = random.randint(0, len(inputList[row])-2)
        inputList[row][column] += inputList[row][column+1]
        inputList[row].pop(column + 1)
        inputList[row].append('')
        choice_num = 1
    
    return inputList, row, column, choice_num

def csvParsing():
    inputList = read()
    processList, row, column, choice = process_table(inputList)
    type = makeType(processList)

    print(processList)
    #print(type)
    return processList, type, row, column, choice

def read():
    with open("csv.csv", newline = "") as csvfile:
        lines = csv.reader(csvfile, delimiter = ";")
        
        inputList = []    # 形式: [[a, b, c], [d, e]]

        for i in lines:
            sub = []
            for j in i: sub.append(j)  
            inputList.append(sub)
        
        return inputList
    
def count():
    inputList = read()
    return len(inputList), len(inputList[0])

if __name__ == '__main__':
    process_list, type, row, column, choice = csvParsing()
