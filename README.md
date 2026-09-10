# Tarificación de Seguro de Gastos Médicos mediante GLM Tweedie

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-4051B5?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Área](https://img.shields.io/badge/Área-Tarificación%20Actuarial-1B3A6B?style=flat)

Proyecto actuarial desarrollado en **Python** para estimar el costo esperado de una póliza de gastos médicos mediante un **Modelo Lineal Generalizado (GLM) Tweedie con enlace logarítmico**.

El modelo permite estimar una prima pura individual a partir de características del asegurado y del contrato, y convertir los coeficientes estimados en relatividades de riesgo interpretables.

El desarrollo matemático completo del GLM y de la distribución Tweedie se encuentra en el documento teórico incluido en el repositorio.

---

## 1. Problema actuarial

En seguros de salud, el costo de siniestros presenta características que dificultan el uso de modelos lineales convencionales:

- una proporción importante de asegurados no presenta siniestros durante el periodo;
- cuando existe un siniestro, el costo es positivo y continuo;
- la distribución presenta asimetría;
- y la varianza aumenta con el nivel esperado del costo.

El problema consiste en estimar el **costo esperado por asegurado** y cuantificar cómo diferentes características modifican el nivel de riesgo.

El objetivo del proyecto es construir un modelo de tarificación capaz de:

- incorporar múltiples factores de riesgo;
- ajustar por exposición;
- obtener relatividades multiplicativas;
- generar predicciones individuales;
- evaluar estabilidad fuera de la muestra de entrenamiento;
- y transformar los resultados del modelo en una estructura tarifaria demostrativa.

---

## 2. Metodología

Se implementa un **GLM Tweedie** con enlace logarítmico.

El flujo de modelación es:

1. Cargar y explorar la base de pólizas.
2. Analizar la distribución del costo de siniestros.
3. Preparar y transformar las variables explicativas.
4. Construir categorías actuariales para variables como edad e IMC.
5. Definir las categorías de referencia.
6. Ajustar un GLM Tweedie.
7. Incorporar la exposición mediante un `offset` logarítmico.
8. Transformar los coeficientes del modelo en relatividades de riesgo.
9. Calcular intervalos de confianza.
10. Evaluar el modelo mediante validación cruzada.
11. Comparar resultados observados y predichos.
12. Construir una tabla tarifaria demostrativa.

### Especificación principal

| Elemento | Configuración |
|---|---|
| Familia | Tweedie |
| Parámetro de potencia | `p = 1.5` |
| Función de enlace | Logarítmica |
| Ajuste por exposición | `log(exposure)` como offset |
| Validación | 5-fold cross-validation |
| Métrica principal | Tweedie Deviance |

El parámetro `p = 1.5` se mantiene fijo dentro de esta implementación y debe interpretarse como una decisión de modelación del proyecto, no como una estimación óptima universal para cualquier cartera.

### Factores considerados

El modelo incorpora variables relacionadas con:

- edad;
- índice de masa corporal;
- condición de fumador;
- deducible;
- región;
- sexo;
- plan;
- y exposición.

---

## 3. Datos utilizados

El proyecto utiliza `gastos medicos.csv`, una base sintética de aproximadamente **10,000 pólizas** construida para representar características habituales de una cartera de gastos médicos.

La variable objetivo presenta una proporción considerable de observaciones con costo igual a cero, aproximadamente **85.1%** de la muestra.

Esto permite trabajar con una estructura de datos compatible con el objetivo demostrativo del modelo Tweedie: combinar una masa en cero con costos positivos continuos.

Las variables son utilizadas exclusivamente con fines académicos y de portafolio y no corresponden a información personal real de asegurados.

---

## 4. Herramientas

| Herramienta | Aplicación |
|---|---|
| **Python** | Desarrollo completo del proceso de modelación |
| **pandas** | Preparación y transformación de datos |
| **NumPy** | Cálculos numéricos |
| **statsmodels** | Estimación e interpretación del GLM |
| **scikit-learn** | Validación cruzada y evaluación predictiva |
| **Matplotlib / Seaborn** | Exploración y visualización de resultados |
| **Jupyter Notebook** | Desarrollo reproducible del análisis |

---

## 5. Resultados

El modelo permite transformar los coeficientes del GLM en relatividades de riesgo respecto de categorías de referencia.

### Principales relatividades estimadas

| Factor | Relatividad aproximada |
|---|---:|
| Obesidad vs. IMC normal | 2.084 |
| Edad 65+ vs. 36–45 | 1.976 |
| Sobrepeso vs. IMC normal | 1.622 |
| Edad 56–65 vs. 36–45 | 1.568 |
| Fumador vs. no fumador | 1.427 |
| CDMX vs. Centro | 1.391 |
| Deducible bajo vs. medio | 1.355 |
| Masculino vs. femenino | 1.262 |
| Deducible alto vs. medio | 0.790 |
| Edad 18–25 vs. 36–45 | 0.550 |

Las relatividades permiten interpretar los efectos del modelo de forma multiplicativa manteniendo constantes los demás factores incluidos.

### Validación

La validación cruzada de cinco particiones produce:

| Métrica | Resultado |
|---|---:|
| Tweedie Deviance media | 191.26 |
| Desviación estándar entre folds | 6.90 |
| Coeficiente de variación | 3.61% |
| Tweedie Deviance en prueba | 182.33 |
| Costo medio observado en prueba | $1,463 |
| Costo medio predicho en prueba | $1,592 |

La dispersión relativamente baja de la métrica entre los folds muestra que el desempeño obtenido es razonablemente estable entre las particiones utilizadas.

Esto no constituye, por sí mismo, evidencia suficiente para descartar sobreajuste o garantizar desempeño en una cartera distinta.

### Ejemplo de estructura tarifaria

Partiendo de una prima base aproximada de **$681**, el proyecto genera combinaciones ilustrativas por grupo de edad y condición de fumador.

| Edad | No fumador | Fumador |
|---|---:|---:|
| 18–25 | $472 | $674 |
| 26–35 | $612 | $873 |
| 36–45 | $859 | $1,226 |
| 46–55 | $1,055 | $1,505 |
| 56–65 | $1,348 | $1,922 |
| 65+ | $1,698 | $2,422 |

Esta tabla demuestra cómo las relatividades estimadas pueden convertirse en factores para una estructura tarifaria. No representa una tarifa comercial lista para implementación.

---

## 6. Aprendizajes y limitaciones

### Aprendizajes

El proyecto permite demostrar:

- preparación de datos para tarificación actuarial;
- construcción e interpretación de un GLM;
- utilización de la familia Tweedie;
- ajuste por exposición mediante offset;
- creación de variables categóricas;
- selección de categorías de referencia;
- transformación de coeficientes en relatividades;
- interpretación de intervalos de confianza;
- validación cruzada;
- evaluación de desempeño fuera de muestra;
- y conversión de resultados estadísticos en una estructura tarifaria interpretable.

### Limitaciones

Entre las principales limitaciones del proyecto se encuentran:

- utiliza información sintética;
- el parámetro Tweedie `p = 1.5` se fija previamente y no se estima mediante un procedimiento específico de optimización;
- la selección de variables y categorías responde al diseño del proyecto;
- no se modelan frecuencia y severidad por separado;
- no se incorpora selección de variables mediante un proceso actuarial completo;
- no se analizan interacciones de manera exhaustiva;
- la validación se realiza sobre la misma población generadora del dataset sintético;
- no se incorporan tendencias de inflación médica;
- no se incorporan gastos, comisiones, margen de utilidad, costo de capital ni reaseguro;
- y el resultado no constituye una tarifa comercial ni regulatoria.

Para utilizar un modelo similar en producción sería necesario complementar el análisis con criterios actuariales, de negocio, regulatorios, de gobernanza y validación independiente.

---

## 7. Contenido del repositorio

```text
tarificacion-seguro-gastos-medicos-glm/
├── Proyecto03_GLM_Gastos_Medicos.ipynb   # Desarrollo y estimación del GLM
├── Proyecto_3_teoria.pdf                  # Documentación matemática y actuarial
├── gastos medicos.csv                     # Dataset utilizado
├── LICENSE
└── README.md                              # Descripción ejecutiva del proyecto
```

---

## 8. Documentación técnica

El desarrollo matemático y conceptual del modelo se encuentra en:

**[`Proyecto_3_teoria.pdf`](./Proyecto_3_teoria.pdf)**

El notebook:

**[`Proyecto03_GLM_Gastos_Medicos.ipynb`](./Proyecto03_GLM_Gastos_Medicos.ipynb)**

contiene la implementación completa, la construcción de variables, el ajuste del GLM, la validación y el análisis de resultados.

---

## 9. Cómo ejecutar el proyecto

### Requisitos

- Python 3;
- Jupyter Notebook, JupyterLab, VS Code o un entorno compatible con `.ipynb`.

### Dependencias

```bash
pip install pandas numpy statsmodels scikit-learn matplotlib seaborn
```

### Ejecución

1. Clonar o descargar el repositorio.
2. Mantener `gastos medicos.csv` en el mismo directorio que el notebook.
3. Abrir `Proyecto03_GLM_Gastos_Medicos.ipynb`.
4. Ejecutar las celdas secuencialmente.
5. Revisar el análisis exploratorio.
6. Revisar los coeficientes y relatividades obtenidas.
7. Analizar los resultados de validación cruzada.
8. Revisar las predicciones y la estructura tarifaria generada.

---

## Autor

**Emiliano Guillén Medina**  
Licenciatura en Actuaría  
[GitHub](https://github.com/EmGM112002) · [LinkedIn](https://www.linkedin.com/in/emgm11)
