import csv
from datetime import datetime, date, time
from urllib.parse import urlparse
import re
import random
import time

def isValidDatetime(dt):
    try:
        if len(dt) == 19:
            if "/" in dt:
                datetime.strptime(dt, "%Y/%m/%d %H:%M:%S")
            elif "-" in dt:
                datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
            else:
                return False
        elif len(dt) == 17:
            if "/" in dt:
                datetime.strptime(dt, "%m/%d/%y %H:%M:%S")
            elif "-" in dt:
                datetime.strptime(dt, "%m-%d-%y %H:%M:%S")
            else:
                return False   
        elif len(dt) == 16:
            if "/" in dt:
                datetime.strptime(dt, "%Y/%m/%d %H:%M")
            elif "-" in dt:
                datetime.strptime(dt, "%Y-%m-%d %H:%M")
            else:
                return False
        elif len(dt) == 14:
            if "/" in dt:
                datetime.strptime(dt, "%m/%d/%y %H:%M")
            elif "-" in dt:
                datetime.strptime(dt, "%m-%d-%y %H:%M")
            else:
                return False  
        else:
            return False
        return True
    except:
        return False
    
def isValidDate(d):
    try:
        if len(d) ==10:
            if "/" in d:
                datetime.strptime(d, "%Y/%m/%d")
            elif "-" in d:
                datetime.strptime(d, "%Y-%m-%d")
            elif " " in d:
                datetime.strptime(d, "%Y %m %d")
            else:
                return False
        else:
            if "/" in d:
                datetime.strptime(d, "%m/%d/%y")
            elif "-" in d:
                datetime.strptime(d, "%m-%d-%y")
            elif " " in d:
                datetime.strptime(d, "%m %d %y")
            else:
                return False
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
    if re.match(r"[^@]+@[^@]+\.[^@]+", email): return True  
    return False  

def isPercentage(perNum):
    if perNum.endswith("%"):
        numPart = perNum[:-1]
        return numPart.isdigit()
    
    return False

def isCurrency(value):
    if value.startswith("$"):
        numPart = value[1:]
        return numPart.isdigit()
    
    return False

def isNumType(string):
    isNum1 = True
    isNum2 = True
    tempFlag = False    # 是否已存在小數點

    if string.isnumeric() == True: return True

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

    if "e+" in string[1]: string = string[1].split("e+")
    elif "e-" in string[1]: string = string[1].split("e-")
    elif "e" in string[1]: string = string[1].split("e")

    for i in string:
        for j in i:
            if j.isnumeric() == True: continue
            else:
                isNum2 = False
                break

    if "," not in string and "." not in string: isNum2 = True

    if "e+" in string: string = string[1].split("e+")
    elif "e-" in string: string = string[1].split("e-")
    elif "e" in string: string = string[1].split("e")

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

def patternScore(csvList):
    sc = 0
    for i in csvList: listLen.append(len(i))    # 記錄每一列有幾個
    #print(listLen)

    numExist = []    # 記錄有幾種長度
    numExist.append(listLen[0])

    for i in listLen:
        for j in numExist:
            if i == j: break
            elif j == numExist[len(numExist)-1]: numExist.append(i)
        sc += ((i-1)/i)

    sc /= len(numExist)
    return sc 

def typeScore(csvList):
    #print(csvList)

    for i in csvList: 
        #print(i)
        subArray = []

        for j in i:
            #print(j)
            
            if isValidDatetime(j) == True: 
                typeMatrix.append(1)
                subArray.append("datetime")
            
            elif isValidDate(j) == True: 
                typeMatrix.append(1)
                subArray.append("date")
            
            elif isValidTime(j) == True: 
                typeMatrix.append(1)
                subArray.append("time")

            elif isValidUrl(j) == True: 
                typeMatrix.append(1)
                subArray.append("url")

            elif isValidEmail(j) == True: 
                typeMatrix.append(1)
                subArray.append("email")

            elif isPercentage(j) == True: 
                typeMatrix.append(1)
                subArray.append("percentage")

            elif isCurrency(j) == True: 
                typeMatrix.append(1)
                subArray.append("currency")

            elif isNumType(j) == True: 
                typeMatrix.append(1)
                subArray.append("num type")

            elif j.strip() == "N/A" or j.strip() == "n/a": 
                typeMatrix.append(1)
                subArray.append("n/a Type")

            elif isAlphanumeric(j) == True: 
                typeMatrix.append(1)
                subArray.append("alphanumeric")

            else: 
                typeMatrix.append(0)
                subArray.append("no type")
        
        #print(subArray)
        typeArray.append(subArray)
        #print(typeArray)

    print(typeArray)
    #print(typeMatrix)
    totalCell = len(typeMatrix)
    #print(totalCell)
    typeCell = 0    # "1" 個數

    for i in typeMatrix: 
        if i == 1: typeCell += 1
    #print(typeCell)

    if typeCell == 0: return 10 ** (-10)
    else: return typeCell / totalCell

def addDelimiter(csvList):
    #print(typeArray)
    averLen = 0
    for i in listLen: averLen += i
    averLen /= len(listLen)
    currentRow = 0
    #print(averLen, listLen)

    for i in listLen:
        #print(i)
        if i < averLen:
            for j in range(2):
                randomPick = random.randint(0, len(listLen) - 1)
                #print(randomPick)

                if abs(i - listLen[randomPick]) < 6:
                    currentPos = 0
                    #print(listLen[randomPick])
                    #print(i, typeArray[randomPick])

                    for k in typeArray[randomPick]: 
                        #print(typeArray[randomPick])
                        if currentPos < len(typeArray[currentRow]) and k != typeArray[currentRow][currentPos]:
                            #print(typeArray[currentRow][currentPos])
                            str1 = csvList[randomPick][currentPos]
                            str2 = csvList[currentRow][currentPos]
                            subStr1 = str2[0:len(str1)]
                            subStr2 = str2[len(str1):]
                            #print(str1, str2)
                            #print(subStr1, subStr2)

                            if k == "datetime":
                                if isinstance(subStr1, (datetime, date, time)) == True:
                                    csvList[currentRow][currentPos] = subStr1
                                    csvList[currentRow].insert(currentPos + 1, subStr2)

                            elif k == "url":
                                if isValidUrl(subStr1) == True:
                                    csvList[currentRow][currentPos] = subStr1
                                    csvList[currentRow].insert(currentPos + 1, subStr2)

                            elif k == "email":
                                if isValidEmail(subStr1) == True:
                                    csvList[currentRow][currentPos] = subStr1
                                    csvList[currentRow].insert(currentPos + 1, subStr2)

                            elif k == "percentage":
                                if isPercentage(subStr1) == True:
                                    #print(csvList[currentRow], currentPos, subStr1)
                                    csvList[currentRow][currentPos] = subStr1
                                    csvList[currentRow].insert(currentPos + 1, subStr2)
                                    #print(j, currentRow)
                                    #print(csvList)

                            elif k == "currency":     
                                if isCurrency(subStr1) == True:
                                    csvList[currentRow][currentPos] = subStr1
                                    csvList[currentRow].insert(currentPos + 1, subStr2)

                            elif k == "num type":                               
                                if isNumType(subStr1) == True:
                                    csvList[currentRow][currentPos] = subStr1
                                    csvList[currentRow].insert(currentPos + 1, subStr2)

                            elif k == "n/a type":                               
                                if subStr1 == "N/A" or subStr1 == "n/a":
                                    csvList[currentRow][currentPos] = subStr1
                                    csvList[currentRow].insert(currentPos + 1, subStr2)

                            elif k == "alphanumeric":                               
                                if isAlphanumeric(subStr1) == True:
                                    csvList[currentRow][currentPos] = subStr1
                                    csvList[currentRow].insert(currentPos + 1, subStr2)

                        currentPos += 1
        currentRow += 1                           
    #print(csvList)
    return csvList 
            
def addNewline(csvList):
    averLen = 0
    for i in listLen: averLen += i
    averLen /= len(listLen)
    currentRow = 0
    #print(averLen, len(listLen))
    #print(csvList)

    for i in listLen:
        if i > averLen:
            for j in range(2):
                randomPick = random.randint(0,len(listLen) - 1)
                #print(randomPick)
                #print(listLen[randomPick], i)

                if i / listLen[randomPick] > 1.5:
                    currentPos = 0
                    
                    for k in typeArray[randomPick]:
                        #print(k, typeArray[currentRow][currentPos])
                        #print(randomPick, currentPos)

                        if currentPos < len(typeArray[currentRow]) and k != typeArray[randomPick][currentPos]:
                            str1 = csvList[randomPick][currentPos]
                            str2 = csvList[currentRow][currentPos]
                            subStr1 = str2[0:len(str1)]
                            subStr2 = str2[len(str1):]
                            subList1 = csvList[currentRow][:currentPos-1]
                            subList2 = csvList[currentRow][currentPos+1:]
                            #print(csvList[randomPick][currentPos])
                            #print(str1, str2)
                            #print(subStr1, subStr2)
                            #print(subList1, subList2)

                            if k == "datetime":
                                if isinstance(subStr1, (datetime, date, time)) == True:
                                    subList1.append(subStr1)
                                    subList2.insert(0, subStr2)
                                    csvList.replace(currentRow, subList1)
                                    csvList.insert(currentRow + 1, subList2)

                            elif k == "url":
                                if isValidUrl(subStr1) == True:
                                    subList1.append(subStr1)
                                    subList2.insert(0, subStr2)
                                    csvList.replace(currentRow, subList1)
                                    csvList.insert(currentRow + 1, subList2)

                            elif k == "email":
                                if isValidEmail(subStr1) == True:
                                    subList1.append(subStr1)
                                    subList2.insert(0, subStr2)
                                    csvList.replace(currentRow, subList1)
                                    csvList.insert(currentRow + 1, subList2)

                            elif k == "percentage":
                                if isPercentage(subStr1) == True:
                                    subList1.append(subStr1)
                                    subList2.insert(0, subStr2)
                                    csvList.replace(currentRow, subList1)
                                    csvList.insert(currentRow + 1, subList2)

                            elif k == "currency":                               
                                if isCurrency(subStr1) == True:
                                    subList1.append(subStr1)
                                    subList2.insert(0, subStr2)
                                    csvList.replace(currentRow, subList1)
                                    csvList.insert(currentRow + 1, subList2)

                            elif k == "numeric":                               
                                if subStr1.isnumeric() == True:
                                    subList1.append(subStr1)
                                    subList2.insert(0, subStr2)
                                    csvList.replace(currentRow, subList1)
                                    csvList.insert(currentRow + 1, subList2)

                            elif k == "num type":                               
                                if isNumType(subStr1) == True:
                                    subList1.append(subStr1)
                                    subList2.insert(0, subStr2)
                                    csvList.replace(currentRow, subList1)
                                    csvList.insert(currentRow + 1, subList2)

                            elif k == "n/a type":                               
                                if subStr1 == "N/A" or subStr1 == "n/a":
                                    subList1.append(subStr1)
                                    subList2.insert(0, subStr2)
                                    csvList.replace(currentRow, subList1)
                                    csvList.insert(currentRow + 1, subList2)

                            elif k == "alphanumeric":                               
                                if isAlphanumeric(subStr1) == True:
                                    subList1.append(subStr1)
                                    subList2.insert(0, subStr2)
                                    csvList.replace(currentRow, subList1)
                                    csvList.insert(currentRow + 1, subList2)

                        currentPos += 1
        currentRow += 1                           
    return csvList                                        
                            
# main
with open("csv.csv", newline = "") as csvfile:
    lines = csv.reader(csvfile, delimiter = ";")
    
    inputList = []    # 形式 : [[a, b, c], [d, e]]

    for i in lines:
        sub = []
        for j in i: sub.append(j)  
        inputList.append(sub)
    #print(inputList) 

    listLen = []
    typeMatrix = []    # 0, 1 矩陣
    typeArray = []

    ps = patternScore(inputList)
    #print(ps)

    ts = typeScore(inputList)
    #print(ts)

    qs = ps * ts
    #print(qs)

    result1 = addDelimiter(inputList)
    #print(result1)

    result2 = addNewline(result1)
    #print(result2)
