# -*- coding: utf-8 -*-
import re
with open('entrada_tokenizador.txt','r',encoding='utf8') as f:
    text = f.read()

#( ). ,‘“?¿!¡ ...; :
general = "(\.{3})"
general1 = "(/w?)|([\.|,|'|?|¿|!|¡|;|\:|\"|%]){1}|(\w*)"
compuesto = "(\w+-\w+)"
numbers = "([0-9]+[,\.][0-9]+)"
date ="(\d{2}[/|-]\d{2}[/|-]\d{4})|(\d{1,2}[/|-]\d{1,2})"
time = "(\d{1,2}[:]\d{2})"
date_s = "(\d{1,2}\s(?:de){1}\s(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre){1}\s(?:de){1}\s\d{2,4})"
web = "((?:http://)?(?:www\.)?(?:\S+[\.])+(?:com|es|co|org){1}(?:[/]\S*)*)"
hastag = "([@#]\S+)"
#total = "|".join([numbers, date, time, date_s, web, hastag, general1, general])
total = "|".join([web, hastag, date_s, date, time, numbers, compuesto, general, general1])
pattern=re.compile (total,re.I|re.U)

fiiter=pattern.finditer(text)
out = ""
for i in fiiter:
    for j in range(1,13):
        if i.group(j):
            print (i.group(j))
            out = out + i.group(j) + "\n"
print(out)

f= open("salida_tokenizador_P1.txt","w+")
f.write(out)
f.close()