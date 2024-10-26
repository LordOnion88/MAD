#Regresion lineal

## Lectura del archivo cartera 
## Revisar que el archivo cartera se encuentre en documentos, 
## o la ruta en donde se encuentra direccionado R

getwd()

## Copiar los archivos descargados en la ruta que surge de correr el 
## comando anterior
#Ruta de las bases de datos
setwd('~/Desktop/mercadeo2016mac/0VARIOSHUGO/basesdatos/')

if (!require('corrplot')) install.packages('corrplot')
if (!require('lmtest')) install.packages('lmtest')
if (!require('MASS')) install.packages('MASS')
if (!require('leaps')) install.packages('leaps')

## Revisar que el archivo MAOSBASE se encuentre en documentos, 
## o la ruta en donde se encuentra direccionado R

#Instalar la libreria para cargar el archivo de excel.
install.packages("readxl")
library("readxl")

#Archivo con 186 días de Venta 

maobase<-read_excel("MAOSBASE.xlsx")
## Revisar la estructura del objeto que se tiene y los diferentes 
## tipos de datos mediante el comando str
str(maobase)

#El principal producto de Mao son tazones llenos de arroz, verduras y carne 
#hechos a pedido del cliente. El archivo  muestra las ventas unitarias diarias 
#del precio del Bowl/ tazón,  Bowl, refrescos y cerveza

#La variable dependiente (o de respuesta) es Bowls número de tazones. 
#Las demás variables son posibles predictoras (variables independientes) e incluyen

#1-precio del Bowl/ tazón (Bowl Price)
#2-venta de refrescos diarias (Soda)
#3-venta de cervezas diarias (Beer)

### Carga del paquete psych
install.packages("psych")
library(psych)

## Resumen general de las variables del objeto maobase
summary(maobase)

## Uso de la función describe que realiza múltiples descriptivos
describe(maobase)

## Carga de paquetes
library(corrplot)

matrizcor<-cor(maobase)
corrplot(matrizcor)

#Matriz de correlaciones con mayor detalle.
corrplot(matrizcor, type="upper", method="circle", addCoef.col = "black")

#CREANDO MODELOS BASICOS

library('lmtest')
library(zoo)
#un modelo con variables elegidas (preespecificado)
modelo0<- lm(Bowls ~ `Bowl Price`, data=maobase)
# un modelo con todas las variables (preespecificado)
modelo1<- lm(Bowls ~.,data=maobase)

#pruebas de hip?tesis y dos m?tricas
summary(modelo0)
summary(modelo1)

#Primero, presentaremos 4 gráficos de diagnóstico de residuos:

dwtest(modelo0)
dwtest(modelo1)

#obtengo una muestra
set.seed(920203) #se deja alguna semilla para que el muestreo sea replicable
#aqu? se define el tama?o de la muestra, en este caso entrenamiento tendr? el 80% de los casos
sample <- sample.int(nrow(maobase), floor(.8*nrow(maobase)))
maobase.train <- maobase[sample, ]
maobase.test <- maobase[-sample, ]

str(maobase.train)
str(maobase.test)

#un modelo con variables elegidas para el entrenamiento (preespecificado)
modelot0<- lm(Bowls ~ `Bowl Price`, data=maobase.train)
# un modelo con todas las variables para el entrenamiento (preespecificado)
modelot1<- lm(Bowls ~.,data=maobase.train)

#pruebas de hipótesis y dos métricas para el dataset de entrenamiento
summary(modelot0)
summary(modelot1)

#Intervalos de confianza
confint(modelot0, level=0.95)
confint(modelot1, level=0.95)

#Primero, presentaremos 4 gráficos de diagnóstico de residuos:
#layout(matrix(c(1,2,3,4),2,2)) # opcional 4 graficos/pagina
plot(modelot0)
plot(modelot1)

#Primero, presentaremos 4 gráficos de diagnóstico de residuos:

dwtest(modelot0)
dwtest(modelot1)

#Veamos como funciona eso en los modelos que ya hemos creado.
#obteniendo el AIC
AIC(modelot0)
AIC(modelot1)

#Ahora vamos a hacer predicciones basadas en cada uno de nuestros modelos, 
#y compararlas a partir de la raíz cuadrática media del error (RMSE)

#PREDICCIONES

pred0<-predict(modelot0, maobase.test, se.fit=TRUE)

pred1<-predict(modelot1, maobase.test, se.fit=TRUE)

RMSE0<-sqrt(mean((pred0$fit-maobase.test$Bowls)^2))
RMSE1<-sqrt(mean((pred1$fit-maobase.test$Bowls)^2))

RMSE0
RMSE1

#Feature selection
##stepwise
modelostep<- step(modelot1,direction="both")

summary(modelostep)

# Best subsets
library(MASS)
library(leaps)

#nbest (numero de modelos por cada k) es por defecto 1 y nvmax es por defecto 8 (k maximo)
modelsub<-regsubsets(Bowls~.,data=maobase.train, nbest=1, nvmax=6, method = "exhaustive") 
summary(modelsub)

#Obtener r2 ajustado
summary(modelsub)$adjr2
#Obtener el cp
summary(modelsub)$cp
#Obtener el BIC (medida similar al AIC)
summary(modelsub)$bic

#Miremos como se desempeña en la base de validación:
  
#Desempeño en la base de validación

predstep<-predict(modelostep, maobase.test, se.fit=TRUE)
RMSEstep<-sqrt(mean((predstep$fit-maobase.test$Bowls)^2))

RMSE0
RMSE1
RMSEstep







