import re
with open('entrada_tokenizador.txt','r',encoding='utf8') as f:
    text = f.read()

#( ). ,‘“?¿!¡ ...; :
general = "(\.{3})"
general1 = "(/w?)|([\.|,|'|?|¿|!|¡|;|\:|\"|%]){1}|(\w*)"
numbers = "([0-9]+[,\.][0-9]+)"
date ="(\d{2}[/|-]\d{2}[/|-]\d{4})|(\d{1,2}[/|-]\d{1,2})"
time = "(\d{1,2}[:]\d{2})"
date_s = "(\d{1,2}\s(?:de){1}\s(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre){1}\s(?:de){1}\s\d{2,4})"
web = "((?:http://)?(?:www\.)?(?:\S+[\.])+(?:com|es|co|org){1}(?:[/]\S*)*)"
hastag = "([@#]\S+)"
#total = "|".join([numbers, date, time, date_s, web, hastag, general1, general])
total = "|".join([web, hastag, date_s, date, time, numbers, general, general1])
pattern=re.compile (total,re.I|re.U)

fiiter=pattern.finditer(text)
for i in fiiter:
    if i.group(1):
        print (i.group(1))
    if i.group(2):
        print (i.group(2))
    if i.group(3):
        print (i.group(3))
    if i.group(4):
        print (i.group(4))
    if i.group(5):
        print (i.group(5))
    if i.group(6):
        print (i.group(6))
    if i.group(7):
        print (i.group(7))
    if i.group(8):
        print (i.group(8))
    if i.group(9):
        print (i.group(9))
    if i.group(10):
        print (i.group(10))
    if i.group(11):
        print (i.group(11))

