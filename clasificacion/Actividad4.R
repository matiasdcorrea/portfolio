########################################################################
#                                                                        
# Actividad 4: Arboles de decision. 
# Fuente:
# https://www.indec.gob.ar/indec/web/Institucional-Indec-BasesDeDatos-2
# Alumnos: Correa, Matías; Esperanza, Marcelo Fabián; Ríos, Alejandro.
#
########################################################################
#
# Utilizamos como dataset los datos contenidos en la Encuesta Nacional 
# de Factores de Riesgo del año 2018, provisto por el INDEC 
# (Instituto Nacional de Estadística y Censos).
#  
########################################################################

#### CONFIGURACION Y BIBLIOTECAS ####

rm(list = ls()) # Para limpiar la memoria
options(scipen = 6) # Para evitar notación científica

#### CARGA DE DATOS ####

##### Importamos las biliotecas #####

library(readr) #read_csv
library(randomForest) #turneRF()  randomForest()
library(caret) #createDataPartition()
library(ranger) #ranger()
library(rpart) #rpart()
library(rattle) #fancyRpartPlot() es otra opción
library(tidyr)  #expand_grid()
library(rpart.plot) #rpart.plot()
library(pROC) #plot.roc
library(ROCR) #roc
library(dplyr) #recode()

datos <- read_delim("Datos/encuesta.csv", 
                    delim = ";", escape_double = FALSE, trim_ws = TRUE)

nrow(datos) # Cantidad de casos.
ncol(datos) # Cantidad de variables

# Numero de casos con null mediante iteracion con Sapply.
sapply(datos, function(x) sum(is.na(x)))

# Quitamos los casos con datos na
encuesta.salud <- na.omit(datos)

# Cantidad de casos luego de sacar los nulos.
nrow(encuesta.salud)

# Comprobamos el porcentaje de observaciones del nuevo dataframe respecto del original
nrow(encuesta.salud) / nrow(datos)

# Convertimos en factor algunas variables
encuesta.salud <- encuesta.salud %>% mutate(hipertension = as.factor(
                                                              recode(hipertension,
                                                                        "SI" = "Si",
                                                                        "NO" = "No")),
                                            AnsiedadDepresion = as.factor(
                                                                  recode(AnsiedadDepresion,
                                                                        "MUCHO" = "Mucho")),
                                            FumaFrecuencia = as.factor(
                                                                recode(FumaFrecuencia,
                                                                        "NingunDia" = "Ningun dia",
                                                                        "AlgunosDias" = "Algunos dias",
                                                                        "TodosLosDias" = "Todos los dias")),
                                            SalEnCoccion = as.factor(SalEnCoccion),
                                            Colesterol = as.factor(Colesterol),
                                            consumoRegAlcoholRiesgoso = as.factor(
                                                                          recode(consumoRegAlcoholRiesgoso,
                                                                                  "SI" = "Si",
                                                                                  "NO" = "No")),
                                            prevalencia_diabetes = as.factor(
                                                                    recode(prevalencia_diabetes,
                                                                                  "SI" = "Si",
                                                                                  "NO" = "No")),
                                            SalDespCoccion = as.factor(
                                                                recode(SalDespCoccion,
                                                                        "RaraVez" = "Rara vez")))
# Renombramos las variables para mayor legibilidad
names(encuesta.salud) <- c("Ansiedad_Depresion",
                           "Actividad_Intensa",
                           "Actividad_Moderada",
                           "Caminata",
                           "Sentado",
                           "Fuma_Frecuencia",
                           "Hipertension",
                           "IMC",
                           "Sal_En_Coccion",
                           "Sal_Desp_Coccion",
                           "Promedio_Frut_Ver_Diario",
                           "Colesterol",
                           "Cons_Reg_Alcohol_Riesgoso",
                           "Prevalencia_Diabetes")

# Categorizamos la variable IMC
imc.intervalos <- cut(encuesta.salud$IMC, 
                     c(-Inf, 18.5, 25, 30, 35, 40, Inf), left=F)

encuesta.salud.clasificada <- mutate(encuesta.salud, 
                                     IMC = imc.intervalos)

levels(encuesta.salud.clasificada$IMC) <- c("Peso bajo", "Peso Normal", "Sobrepeso", 
                                            "Obesidad grado 1", "Obesidad grado 2", 
                                            "Obesidad grado 3")

# Eliminamos los valores No sabe - No contesta
diabetes.sin.nsnc <- which(encuesta.salud.clasificada$Prevalencia_Diabetes
                           %in% "NSNC")

encuesta.salud.clasificada <- encuesta.salud.clasificada[-diabetes.sin.nsnc,]

# Comprobamos el porcentaje de observaciones del nuevo dataframe respecto del original
nrow(encuesta.salud.clasificada) / nrow(datos)

##### Definimos datos para entrenamiento y datos para test ######

set.seed(2022) # Número inicial a partir del cuál comenzará a generar una secuencia aleatoria.
# Tomamos un 80% para el entrenamiento y un 20% para el test
split <- createDataPartition(encuesta.salud.clasificada$Colesterol, p=0.8, list=FALSE) #List = Si el resultado devolverá una lista o una matriz. 
train<- encuesta.salud.clasificada[split,]
test<-  encuesta.salud.clasificada[-split,]

#########################################################
##### Modelado de árbol de desición aplicando RPART #####
#########################################################

###### Genero un modelo 1 con rpart #####

tree <- rpart(Colesterol ~., data = train, method="class") # Indicamos que deseamos un arbol de clasificación
rpart.plot(tree) # Graficamos el árbol.
x11()
fancyRpartPlot(tree,type = 2) # Gráfico del árbol


###### Importancia de las variables ######

qplot(x = names(tree$variable.importance), y=tree$variable.importance,
      xlab="Variable", ylab="Importancia", main="rpart - Importancia de las variables")

# Revisamos si hay desbalanceo de las clases
library(plotly)
qplot(encuesta.salud.clasificada$Colesterol, ylab="Observaciones", xlab="Intervalos")

# El 69.25% de los casos no tienen colesterol
# El 30.75% tienen colesterol
prop.table(table(encuesta.salud.clasificada$Colesterol))


###### Predicción del primer modelo ######

prediccion.rpart <- predict(tree, newdata=test, type="class")
# Precision: 70.03%
# Sensibilidad: 16.47%
# Especificidad: 93.8%
confusionMatrix(prediccion.rpart, test$Colesterol, positive = "Si")

##### Tratamiento de clases desbalanceadas #####
library(ROSE)

# Cantidad de casos para cada clase
table(train$Colesterol)

###### Sobre-muestreo ######
# Aumentamos la cantidad de "Si" creando copias de algunos casos
# seleccionados aleatoriamente
# N = cantidad de "No" * 2
over <- ovun.sample(Colesterol ~., data = train, method = "over", N = 19602)$data

# Igual numero de clases
table(over$Colesterol)

# Comprobamos el nuevo modelo
tree.over <- rpart(Colesterol ~., data = over, method="class")

x11()
fancyRpartPlot(tree.over,type = 2)

prediccion.over <- predict(tree.over, newdata = test, type="class")
# Empeoramos la precision (de 70.03% a 61.07%), la especificidad empeoro (de 93.8% a 60.2%)
# pero mejoramos la sensibilidad (de 16.47% a 63.02%)
confusionMatrix(prediccion.over, test$Colesterol, positive = "Si")

###### Sub-Muestreo ######
# Reducimos la cantidad de "No" de forma aleatoria, hasta igualar a los "Si"
# N = cantidad de "Si" * 2
under <- ovun.sample(Colesterol ~., data = train, method = "under", N = 8704)$data

# Igual numero de clases
table(under$Colesterol)

# Comprobamos el nuevo modelo
tree.under <- rpart(Colesterol ~., data = under, method="class")

x11()
fancyRpartPlot(tree.under,type = 2)

prediccion.under <- predict(tree.under, newdata = test, type="class")
# Mejoro la precision (de 61.07% a 62.06%), la especificidad mejoro (de 60.2% a 62.29%)
# pero empeoro la sensibilidad (de 63.02% a 61.55%)
confusionMatrix(prediccion.under, test$Colesterol, positive = "Si")


##### Ambos (Sobre y Sub-Muestreo) #####
# Se reduce la cantidad de "No" y se aumenta la de "Si"
# N = cantidad de "Si" + cantidad de "No"
ambos <- ovun.sample(Colesterol ~., data = train, method = "both", 
                     p = 0.5,
                     seed = 111,
                     N = 14153)$data

# Distinto numero de clases pero mucho mas balanceado
table(ambos$Colesterol)

# Porcentaje de cada clase
prop.table(table(ambos$Colesterol))

# Comprobamos el nuevo modelo
tree.ambos <- rpart(Colesterol ~., data = ambos, method="class")

x11()
fancyRpartPlot(tree.ambos,type = 2)

prediccion.ambos <- predict(tree.ambos, newdata = test, type="class")
# Empeoro la precision (de 62.06% a 61.07%), la especificidad empeoro (de 62.29% a 60.2%)
# pero mejoramos la sensibilidad (de 61.55% a 63.02%)
confusionMatrix(prediccion.ambos, test$Colesterol, positive = "Si")

##### Muestreo Sintetico con ROSE #####
# N puede ser cualquier numero, nosotros elegimos 20000
rose <- ROSE(Colesterol ~., data = train, seed = 222, N = 20000)$data

# Distinto numero de clases pero mucho mas balanceado
table(rose$Colesterol)

# Porcentaje de cada clase
prop.table(table(rose$Colesterol))

# Resumen
summary(ambos)
summary(rose)

# Comprobamos el nuevo modelo
tree.rose <- rpart(Colesterol ~., data = rose, method="class")

x11()
fancyRpartPlot(tree.rose,type = 2)

prediccion.rose <- predict(tree.rose, newdata = test, type="class")
# La precision, sensibilidad y especificidad se mantienen iguales
confusionMatrix(prediccion.rose, test$Colesterol, positive = "Si")

#### Resumen de los modelos ####

##### Sin tratamiento de desbalanceo -----
# Precision: 70.03%
# Sensibilidad: 16.47%
# Especificidad: 93.8%

##### Con Sobre-Muestreo
# Precision: 61.07%
# Sensibilidad: 63.02%
# Especificidad: 60.2%

##### Con Sub-Muestreo
# Precision: 62.06%
# Sensibilidad: 61.55%
# Especificidad: 62.29%

##### Ambos (Sobre y Sub-Muestreo)
# Precision: 61.07%
# Sensibilidad: 63.02%
# Especificidad: 60.2%

##### Muestreo Sintetico con ROSE
# Precision: 61.07%
# Sensibilidad: 63.02%
# Especificidad: 60.2%

###################################
# IMPLEMENTACION DE RANDOM FOREST #
###################################

# ***** Aplicación de Random Forest *****

##### MODELO 1 #####
# Genero el primer modelo seteando número de árboles igual a 700 y número de variables igual a 2.
rf1 <- randomForest( Colesterol ~.
                                  , data= train       # datos para entrenar 
                                  , ntree= 700    # cantidad de arboles
                                  , mtry=2        # cantidad de variables
                                  , replace = T            # muestras con reemplazo
                                  , importance=T        # para poder mostrar la importancia de cada var
                                  , class = NULL)


print(rf1)  #OOB estimate of  error rate: 29.98%

# Se estudia el error OOB (out-of-bag error): es el error promedio para cada observacion
# usando predicciones de los arboles que no contienen a dicha observacion.

## Predicción del modeloRF1 ##
rf1.predict <- predict(rf1, newdata=test, type="class")

# Precision: 70.12%
# Sensibilidad: 12.51%
# Especificidad: 95.67%
confusionMatrix(rf1.predict, test$Colesterol, positive = "Si")          


## Gráfico del error OOB en modeloRF1 ##
qplot(y=rf1$err.rate[,1], main="randomForest, Error out-of-bag, 700 arboles, 2 variables", 
      ylab="Error OOB", xlab="Cantidad de arboles")


##### MODELO 2 #####
# Genero el primer modelo seteando número de árboles igual a 300 y número de variables igual a 2.
rf2 <- randomForest(Colesterol ~ ., data=train, ntree=300, mtry = 2)
print(rf2) #        OOB estimate of  error rate: 30.01% aumenta

## Predicción del modeloRF2 ##
rf2.predict <- predict(rf2, newdata=test, type="class")

# Precision: 70.17%
# Sensibilidad: 12.05%
# Especificidad: 95.96%
confusionMatrix(rf2.predict, test$Colesterol, positive = "Si")          


## Gráfico del error OOB en modeloRF2 ##
qplot(y=rf2$err.rate[,1], main="randomForest, Error out-of-bag, 200 arboles, 2 variables", 
      ylab="Error OOB", xlab="Cantidad de arboles")


##### MODELO 3 #####
# Creamos un modelo con el dataset con las clases balanceadas
rf3 <- randomForest(Colesterol ~ ., data=ambos, ntree=300, mtry = 4)

print(rf3) # OOB error: 13.63%

rf3.predict <- predict(rf3, newdata=test, type="class")

# Precision: 62.68%
# Sensibilidad: 46.64%
# Especificidad: 69.8%
confusionMatrix(rf3.predict, test$Colesterol, positive = "Si")

## Gráfico del error OOB en modeloRF3 ##
qplot(y=rf3$err.rate[,1], main="randomForest, Error out-of-bag, 300 arboles, 4 variables", 
      ylab="Error OOB", xlab="Cantidad de arboles")


###### IMPORTANCIA DE LAS VARIABLES modeloRF1 ######

#La gráfica Mean Decrease Accuracy expresa cuánta 
#precisión pierde el modelo al excluir cada variable
#Cuanto más sufre la precisión, más importante es la variable 
#para la clasificación exitosa
#La disminución media del coeficiente de Gini es una medida de cómo cada variable contribuye 
#a la homogeneidad de los nodos y hojas en el bosque aleatorio resultante. 
#Cuanto mayor sea el valor de la precisión de disminución media o la puntuación de Gini
#de disminución media, mayor será la importa
x11()
varImpPlot(rf1, sort = T, n.var = 13 , main = 'Top 13 importancia de las variables')

### Seleccionamos las 4 variables más importantes ###

rf4 <- randomForest( Colesterol ~ Prevalencia_Diabetes 
                                  + Hipertension + Ansiedad_Depresion 
                                  + IMC 
                                  , data= train       # datos para entrenar 
                                  , ntree= 300    # cantidad de arboles
                                  , mtry=2        # cantidad de variables
                                  , replace = T            # muestras con reemplazo
                                  , importance=T        # para poder mostrar la importancia de cada var
                                  , class = NULL
)
#OOB estimate of  error rate: 29.58%

print(rf4)
prediccion.rf4 <- predict(rf4, newdata=test, type="class")

# Precision: 70.34%
# Sensibilidad: 14.81%
# Especificidad: 94.98%
confusionMatrix(prediccion.rf4, test$Colesterol, positive = "Si")

######################################
# UTILIZACION DE RANGER Y CURVAS ROC #                              
######################################

# Ranger es una implementacion rapida de bosques aleatorios para R, 
# útil cuando la cantidad de variables es grande.
# Pruebo con implementacion ranger, parametros por defecto (500 arboles, 2 variables)
# Error OOB de aproximadamente  29.76 % 
# Varía entre ejecución y ejecución.

ranger1 <- ranger(formula = Colesterol ~ Prevalencia_Diabetes 
                  + Hipertension + Ansiedad_Depresion 
                  + IMC,
                  data=train)
ranger1 # equivalente a : print(modeloRanger1)
prediccion.ranger1 <- predict(ranger1, data=test)

# Precision: 70.34%
# Sensibilidad: 14.81%
# Especificidad: 94.98%
confusionMatrix(prediccion.ranger1$predictions, test$Colesterol, positive = "Si")

# Se observan unos resultados practicamente similares a los anteriores, 
#la ganancia se ve reflejada en los tiempos de ejecucion que son menores.

# Creamos un segundo modelo tomando mas arboles
ranger2 <- ranger(
  formula   = Colesterol ~ Prevalencia_Diabetes 
              + Hipertension + Ansiedad_Depresion 
              + IMC,
  data      = train, 
  num.trees = 1000,
  mtry      = 2,
  max.depth = 20,
  seed      = 1234,
  importance= 'impurity',
  probability= TRUE
)    

ranger2 # OOB error: 19.81%

roc_train <- roc(train$Colesterol, ranger2$predictions[,2], percent= TRUE,
                 auc= T, CI= T, plot= T )

plot.roc(roc_train, legacy.axes = T, print.thres = "best", print.auc = T)

print(roc_train)


#################
# Visualización #
#################

figura_1 <- ggplot(encuesta.salud.clasificada, aes(x = Prevalencia_Diabetes, fill = Colesterol)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3) +
  labs(title="Injerencia de la prevalencia de diabetes en el Colesterol")

figura_1

figura_2 <- ggplot(encuesta.salud.clasificada, aes(x = Hipertension, fill = Colesterol)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3) +
  labs(title="Injerencia de la hipertension en el Colesterol")

figura_2
