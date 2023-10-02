import csv
from datetime import datetime, date, time
from urllib.parse import urlparse
import re
import random
import time

def isValidDatetime(dt):
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

    string = string.strip()
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
    
    """if "e+" in string[1]: string = string[1].split("e+")
    elif "e-" in string[1]: string = string[1].split("e-")
    elif "e" in string[1]: string = string[1].split("e")"""

    for i in string:
        for j in i:
            if j.isnumeric() == True: continue
            else:
                isNum2 = False
                break

    if "," not in string and "." not in string: isNum2 = True

    """if "e+" in string: string = string[1].split("e+")
    elif "e-" in string: string = string[1].split("e-")
    elif "e" in string: string = string[1].split("e")"""

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
                subArray.append("n/a type")

            elif isAlphanumeric(j) == True: 
                typeMatrix.append(1)
                subArray.append("alphanumeric")

            else: 
                typeMatrix.append(0)
                subArray.append("no type")
        
        #print(subArray)
        typeArray.append(subArray)
        #print(typeArray)

    #print(typeArray)
    #print(typeMatrix)
    totalCell = len(typeMatrix)
    #print(totalCell)
    typeCell = 0    # "1" 個數

    for i in typeMatrix: 
        if i == 1: typeCell += 1
    #print(typeCell)

    if typeCell == 0: return 10 ** (-10)
    else: return typeCell / totalCell

def findCorrectFormat(csvList):
    print("目前測試的 csv file 共有 " + str(len(csvList)) + " 行")
    q = int(input("以 q 個 Row 為一組: "))

    # 先用if else判斷要分成幾組
    if len(csvList) % q != 0: groupNum = int(len(csvList) // q + 1)
    else: groupNum = int(len(csvList) / q)

    tempRecord = []                 # 暫時紀錄分組過程以選出每組的正確格式
    firstTimeGroupingResult = []    # 紀錄首次分組後每組選出的正確格式

    for i in range(0, groupNum):            # 外迴圈每圈代表一組
        for j in range(0,q):                # 第一個內迴圈負責將該組q個row送入暫用array
            if i * q + j < len(csvList):    # 確保最後一組不會因index超出csv file的row 數而報錯
                tempRecord.append(typeArray[i * q + j])

        maxCount = 0    # 紀錄最多出現次數
        #print(tempRecord) 檢查每一組data是否正確
        for j in range(0, len(tempRecord)):    # 第二個內迴圈負責選出該組內出現最多次的格式
            tempCount = tempRecord.count(tempRecord[j])
            if tempCount > maxCount:
                maxCount = tempCount
                maxAppearFormat = tempRecord[j]
        #print(max_appear_format) 檢查選出的格式是否正確
        firstTimeGroupingResult.append(maxAppearFormat)
        tempRecord.clear()    # 進下一組前記得清空

    maxCount = 0
    for i in range(0, groupNum):    # 從第一次找出的格式中進行第二次選擇 同樣選出出現最多的格式
        tempCount = firstTimeGroupingResult.count(firstTimeGroupingResult[i])
        if tempCount > maxCount:
            maxCount = tempCount
            resultFormat = firstTimeGroupingResult[i]

    return resultFormat    # 回傳最終選出的row格式

def typeCheck(str):
    if isValidDatetime(str) == True: return "datetime"
    elif isValidDate(str) == True: return "date"
    elif isValidTime(str) == True: return "time"
    elif isValidUrl(str) == True: return "url"
    elif isValidEmail(str) == True: return "email"
    elif isPercentage(str) == True: return "percentage"
    elif isCurrency(str) == True: return "currency"
    elif isNumType(str) == True: return "num type"
    elif str.strip() == "N/A" or str.strip() == "n/a": return "n/a type"
    elif isAlphanumeric(str) == True: return "alphanumeric"
    else: return "no type"

def addDelimiter():
    rowIndex = 0
    for i in typeArray:
        if len(i) < correctLen:
            formatIndex = 0
            columnIndex = 0
            for j in i:
                if j != correctFormat[formatIndex]: 
                    for k in range(1, len(inputList[rowIndex][columnIndex])):
                        frontType = typeCheck(inputList[rowIndex][columnIndex][:k])
                        endType = typeCheck(inputList[rowIndex][columnIndex][k:])
                        #print(inputList[rowIndex][columnIndex][:k], inputList[rowIndex][columnIndex][k:])
                        #print(frontType, endType)
                        if frontType == correctFormat[formatIndex] and endType == correctFormat[formatIndex+1]:
                            frontStr = inputList[rowIndex][columnIndex][:k]
                            endStr = inputList[rowIndex][columnIndex][k:]
                            inputList[rowIndex][columnIndex] = frontStr
                            inputList[rowIndex].insert(columnIndex + 1, endStr)
                            typeArray[rowIndex][columnIndex] = frontType
                            typeArray[rowIndex].insert(columnIndex + 1, endType)
                            formatIndex += 1
                            columnIndex += 1
                            break
                formatIndex += 1
                columnIndex += 1
        rowIndex += 1
    #print(inputList)

def addNewline():
    same = True
    rowIndex = 0
    tmpFront = []
    tmpEnd = []

    for i in typeArray:
        if len(i) > 1.5 * correctLen: 
            for j in range(correctLen):
                if i[j] == correctFormat[j]: 
                    tmpFront.append(inputList[rowIndex][j])
                    continue
                else:
                    same = False
                    break

            for j in range(correctLen, len(inputList[rowIndex])): tmpEnd.append(inputList[rowIndex][j])

            if same == True:
                inputList[rowIndex] = tmpFront
                rowIndex += 1
                inputList.insert(rowIndex, tmpEnd)

        rowIndex += 1

    print(inputList)
            

"""
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
    print(csvList)
    #return csvList 
"""
"""           
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
""" 
def addNull():
    for i in typeArray:
        #print(i)
        minus = len(correctFormat) - len(i)
        if minus > 0:    # 如果正確type的row長度大於跑到的row則做檢查
            index = 0    # index for correctFormat
            for j in i:
                #print(j)
                isFound = False
                temp = 0
                if j != correctFormat[index]:    # 如果跑到的row的type和correctFormat的type不一樣，就從correctFormat往後找找看有沒有一樣的
                    for k in range(1, len(correctFormat) - index):
                        if j == correctFormat[index + k]:   # 找相同的type
                            isFound = True
                            temp = k    # 找到後記錄相同的type在後 temp個位置
                            break
                    
                    if isFound == False: continue
                    
                    for l in range(k): i.insert(index + l, "n/a type")  # 補temp個 n/a type讓當前的type對應到正確格式的type
                    
                    index += temp   # index跳至已經修復完的位置
                else:
                    index += 1  # 若type相同，檢查下一個type

    #print(typeArray)

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
    #print(typeArray)
    qs = ps * ts
    #print(qs)

    correctFormat = findCorrectFormat(inputList)
    correctLen = len(correctFormat)
    #print(correctFormat)

    addNewline()
    addDelimiter()

    addNull()

    #result1 = addDelimiter(inputList)
    #print(result1)

    #result2 = addNewline(result1)
    #print(result2)
