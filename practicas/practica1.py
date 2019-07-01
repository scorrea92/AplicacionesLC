#Practica1 Sebastian Correa E


#Funciones
def problex(C,w):
	"Esta funcion devuelve probabilidades léxicas P(C|w) y P(w|C) "
	if (w in dic2):
		if(C in dic2[w][1]):
			return dic2[w][1][C] / dic2[w][0]
		else:
			return 0.0
	elif(C in dic2):
		if(w in dic2[C][1]):
			return dic2[C][1][w] / dic1[w]
		else:
			return 0.0		
	else:
		print("La palabra es desconocida")
		return None


#Punto 1
cadena ="El/DT	perro/N	come/V	carne/N	de/P	la/DT	carnicería/N		y/C	de/P	la/DT	nevera/N	y/C	canta/V	el/DT	la/N	la/N	la/N	./Fp"

cadenaList = cadena.split()
palabras = []
categorias = []
completo = []

for i in cadenaList:
	aux = i.split("/")
	palabras.append(aux[0])
	categorias.append(aux[1])
	completo.append(aux)

catNoSorted = categorias
categorias = sorted(categorias)
dic1 = {i:categorias.count(i) for i in categorias} #dic de categorias con su frecuencia
print(dic1)
print("")
############

#Punto2
palabras = [item.lower() for item in palabras] #transformo todas las palabras a lowcase
palabras = sorted(palabras) #ordeno las palabras
dic2 = {i:palabras.count(i) for i in palabras} #dic de palabras con su frecuencia

#Transformo la data de este dic de int a list con las frecuencias, para poder agregar el resto de info.
for i in dic2:
	dic2[i] = [dic2[i],[]]
	for j in completo:
		if(j[0] == i):
			dic2[i][1].append(j[1])
	dic2[i][1] = {k:dic2[i][1].count(k) for k in dic2[i][1]}

print(dic2) #dic cada palabra, muestra su frecuencia, y una lista de sus categorias morfosintacticas con su respectiva frecuencia
print("")

#Punto3
bigramas = ["<S> "+catNoSorted[0]]
for i in range(len(catNoSorted)-1):
	bigramas.append(catNoSorted[i]+" "+catNoSorted[i+1])
bigramas.append("</S> "+catNoSorted[-1])

dic3 = {i:bigramas.count(i) for i in bigramas} #dic de bigramas con su frecuencia
print(dic3)
print("")

#Punto4
print(problex("DT","la"))
print(problex("N","la"))
print(problex("la","DT"))
print(problex("la","N"))
print(problex("soledad","N"))











