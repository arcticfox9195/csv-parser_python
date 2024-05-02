from csv_parsing import isValidDatetime, isValidDate, isValidTime, isValidUrl, isValidEmail, isPercentage, isCurrency, isNumType, isAlphanumeric

def check(input):
    if isValidDatetime(input) == True: return True
    if isValidDate(input) == True: return True
    if isValidTime(input) == True: return True
    if isValidUrl(input) == True: return True
    if isValidEmail(input) == True: return True
    if isPercentage(input) == True: return True
    if isCurrency(input) == True: return True
    if isNumType(input) == True: return True
    if isAlphanumeric(input) == True: return True

def repair(table, row, column, choice):
    if choice == 0: 
        table[row].insert(column, '')
        table[row].pop()
    else:
        find = False
        record = ''
        for index in range(1, len(table[row][column])):
            if check(table[row][column][:index]) == True and check(table[row][column][index:]) == True:
                find = True
                record = index
                break

        if find == True:
            table[row].insert(column+1, table[row][column][record:])
            table[row][column] = table[row][column][:record]
            table[row].pop()
        else: 
            table[row].insert(column + 1, '')
            table[row].pop()

    return table
