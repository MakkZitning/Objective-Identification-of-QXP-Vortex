debug1 = open(r'C:\Users\maizh\OneDrive\meteo_data\Data of my Thesis\vortex database\original data\debug\2017-tst1.txt', 'r', encoding='utf-8')
debug2 = open(r'C:\Users\maizh\OneDrive\meteo_data\Data of my Thesis\vortex database\original data\debug\2017-tst2.txt', 'r', encoding='utf-8')
lst1 = debug1.readlines()
lst2 = debug2.readlines()

difference = open(r'C:\Users\maizh\OneDrive\meteo_data\Data of my Thesis\vortex database\original data\debug\comparision.txt', 'w')
for i in range(min(len(lst1), len(lst2))):
    if lst1[i] != lst2[i]:
        if (('now after changing' not in lst1[i]) and ('now after changing' not in lst2[i])) \
                and (('now before' not in lst1[i]) and ('now before' not in lst2[i]))\
                and (('pjf before' not in lst1[i]) and ('pjf before' not in lst2[i]))\
                and (('now without flag after' not in lst1[i]) and ('now without flag after' not in lst2[i]))\
                and (('now without flag before' not in lst1[i]) and ('now without flag before' not in lst2[i]))\
                and (('past before' not in lst1[i]) and ('past before' not in lst2[i])):
            print(i, '\n', lst1[i].replace('\n', ''), '\n', lst2[i].replace('\n', ''), '\n', file=difference)
