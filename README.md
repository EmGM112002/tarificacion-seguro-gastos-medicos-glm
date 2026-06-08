# Portafolio Actuarial — Proyecto 03
## Modelo GLM para Tarifación de Seguros de Gastos Médicos

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-2E6DB4?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![LaTeX](https://img.shields.io/badge/LaTeX-008080?style=flat&logo=latex&logoColor=white)
![Área](https://img.shields.io/badge/Área-Pricing%20No--Vida-1B3A6B?style=flat)
![Ramo](https://img.shields.io/badge/Ramo-Gastos%20Médicos-2E6DB4?style=flat)

---

## ¿De qué trata este proyecto?

Pipeline completo de tarifación actuarial que estima la **prima pura** de
cada asegurado usando un **GLM Tweedie con liga logarítmica**, produciendo
relatividades de riesgo interpretables para el equipo de pricing.

El modelo es multiplicativo: la prima de cada perfil es el producto de una
prima base por factores (relatividades) que ajustan según edad, IMC, condición
de fumador, deducible, región, sexo y tipo de plan.

$$\mu_i = e_i \cdot \underbrace{e^{\beta_0}}_{\text{prima base}} \cdot \underbrace{e^{\beta_{\text{edad}}}}_{\text{factor edad}} \cdot \underbrace{e^{\beta_{\text{fumador}}}}_{\text{factor fumador}} \cdots$$

| Característica | Detalle |
|---|---|
| Distribución | Tweedie ($p = 1.5$, compuesta Poisson-Gamma) |
| Liga | Logarítmica (modelo multiplicativo) |
| Offset | $\ln(\text{exposición})$ para normalizar por vigencia |
| Validación | 5-fold cross-validation con devianza Tweedie |
| Dataset | 10,000 pólizas sintéticas, 85.1% sin siniestros |

---

## Estructura del repositorio

```
03-Python-GLM/
│
├── gastos_medicos.csv                              # 10,000 pólizas sintéticas
├── Proyecto03_GLM_Gastos_Medicos.ipynb             # Notebook ejecutable
├── Proyecto03_Python_GLM_Gastos_Medicos.pdf        # Documento teórico
└── README.md
```

---

## Secciones del notebook

El notebook está organizado en 9 secciones que siguen el flujo natural de un
proyecto de pricing:

| # | Sección | Contenido |
|---|---|---|
| 1 | Carga y exploración inicial | Lectura del CSV, tipos de datos, estadística descriptiva |
| 2 | Análisis Exploratorio (EDA) | Distribución de prima pura, boxplots por categoría, relación edad–costo con justificación de liga log |
| 3 | Feature Engineering | Grupos quinquenales de edad, categorías OMS de IMC, codificación de fumador |
| 4 | Ajuste del GLM Tweedie | Configuración de familia Tweedie, liga log, Treatment() con categorías de referencia, offset de exposición |
| 5 | Tabla de Relatividades | Extracción de coeficientes, cálculo de relatividades con IC al 95%, gráfico de factores significativos |
| 6 | Validación Cruzada | Pipeline para prevenir data leakage, scorer personalizado con devianza Tweedie, 5-fold CV |
| 7 | Predicciones y residuos | Predicción sobre test set, análisis de residuos |
| 8 | Tabla de tarifas | Prima base por perfil combinando relatividades |
| 9 | Conclusiones | Hallazgos principales, limitaciones y extensiones |

---

## Conceptos clave implementados

### Distribución Tweedie ($p = 1.5$)

$$Y = \sum_{j=1}^{N} X_j \quad \text{donde} \quad N \sim \text{Poisson}(\lambda), \quad X_j \sim \text{Gamma}(\alpha, \beta)$$

Modela la prima pura sin separar frecuencia y severidad. El parámetro
$p = 1.5$ se elige como punto medio del rango actuarial $(1, 2)$ y es
consistente con la literatura empírica para seguros de gastos médicos
(Ohlsson & Johansson, 2010).

### Liga logarítmica y modelo multiplicativo

La liga log garantiza primas positivas y convierte el modelo aditivo en
multiplicativo: cada factor **multiplica** la prima base en lugar de sumarle
una cantidad fija.

$$\mu_i = e^{\beta_0} \cdot e^{\beta_1 x_{1i}} \cdot e^{\beta_2 x_{2i}} \cdots$$

### Codificación categórica con Treatment()

Cada variable categórica se codifica con $k-1$ dummies usando `Treatment()`
de Patsy para elegir explícitamente la categoría de referencia con sentido
actuarial:

| Variable | Referencia | Justificación |
|---|---|---|
| `grupo_imc` | Normal | IMC saludable como punto de comparación natural |
| `fumador` | No | Perfil de menor riesgo base |
| `deducible` | Medio | Nivel intermedio como punto neutro |
| `grupo_edad` | 36-45 | Edad media de la cartera |
| `region` | Centro | Región de riesgo promedio |
| `sexo` | F | Convención |

### Offset de exposición

El offset $\ln(e_i)$ normaliza por la vigencia de cada póliza, convirtiendo
el modelo en un análisis de riesgo por unidad de tiempo (prima pura anualizada):

$$\ln(\mu_i) = \ln(e_i) + \beta_0 + \sum_k \beta_k x_{ki}$$

### Intervalo de confianza de las relatividades

Cada relatividad se acompaña de su IC al 95% transformado con exponencial.
Si el intervalo contiene el 1, el factor no es estadísticamente significativo
y no debería incorporarse a la tarifa.

### Validación cruzada con Pipeline

El Pipeline encadena preprocesamiento + modelo para prevenir data leakage
durante la validación cruzada. El scorer personalizado usa `make_scorer`
con `mean_tweedie_deviance(power=1.5)` y `greater_is_better=False`.

---

## Relatividades principales

| Factor | β̂ | Relatividad | IC 95% | Interpretación |
|---|---|---|---|---|
| Obesidad (vs Normal) | 0.734 | 2.084 | [1.70, 2.56] | +108% por enfermedades crónicas |
| 65+ (vs 36-45) | 0.681 | 1.976 | [1.54, 2.54] | Efecto acelerado de la edad |
| Sobrepeso (vs Normal) | 0.484 | 1.622 | [1.32, 2.00] | +62% por riesgo incrementado |
| 56-65 (vs 36-45) | 0.450 | 1.568 | [1.22, 2.02] | Morbilidad creciente |
| Fumador (Sí vs No) | 0.355 | 1.427 | [1.20, 1.70] | +43% sobre la prima base |
| CDMX (vs Centro) | 0.330 | 1.391 | [1.11, 1.75] | Mayor costo hospitalario |
| Deducible Bajo (vs Medio) | 0.304 | 1.355 | [1.13, 1.62] | Riesgo moral inverso |
| Masculino (vs Femenino) | 0.233 | 1.262 | [1.08, 1.47] | Diferencial por género |
| Deducible Alto (vs Medio) | −0.236 | 0.790 | [0.65, 0.95] | −21% efecto protector |
| 18-25 (vs 36-45) | −0.599 | 0.550 | [0.41, 0.75] | Grupo más joven y sano |

### Validación cruzada (5-fold)

| Métrica | Valor |
|---|---|
| Devianza media | 191.26 |
| Desviación estándar | 6.90 |
| Coeficiente de variación | 3.61% |
| Devianza en test set | 182.33 |
| Prima real media (test) | $1,463 |
| Prima predicha media (test) | $1,592 |

### Tabla de primas (perfil base, exposición 1 año)

Prima base (categoría de referencia): **$681**

| Grupo edad | No fumador | Fumador |
|---|---|---|
| 18-25 | $472 | $674 |
| 26-35 | $612 | $873 |
| 36-45 | $859 | $1,226 |
| 46-55 | $1,055 | $1,505 |
| 56-65 | $1,348 | $1,922 |
| 65+ | $1,698 | $2,422 |

---

## Cómo reproducir

**Requisitos:** Python 3.8+ con `pandas`, `numpy`, `matplotlib`, `seaborn`,
`statsmodels`, `scikit-learn`, `scipy`.

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn scipy
```

1. Abrir `Proyecto03_GLM_Gastos_Medicos.ipynb` en Jupyter, Colab o VS Code.
2. Asegurarse de que `gastos_medicos.csv` esté accesible (ajustar la ruta de carga si es necesario).
3. Ejecutar todas las celdas en orden secuencial.

> **Nota:** el notebook fue desarrollado en Google Colab. Si se ejecuta
> localmente, modificar la celda de carga para apuntar al CSV local en lugar
> de Google Drive.

---

## Herramientas utilizadas

| Librería | Uso |
|---|---|
| `pandas` | Manipulación del dataset |
| `matplotlib` / `seaborn` | Visualización (EDA, relatividades, residuos) |
| `statsmodels` | Ajuste del GLM Tweedie y tabla de coeficientes |
| `scikit-learn` | Pipeline, OneHotEncoder, TweedieRegressor, cross-validation |
| `scipy` | Pruebas estadísticas |

---

## Conclusiones

1. **El GLM Tweedie con liga log** es apropiado para este dataset: la masa
   en cero (85.1%) y la asimetría positiva de la prima pura son consistentes
   con una distribución compuesta Poisson-Gamma.

2. **Los factores más importantes** por magnitud de relatividad son el grupo
   de edad (efecto creciente acelerado, hasta 1.98× en 65+), el grupo de IMC
   con obesidad (2.08×), la condición de fumador (1.43×) y el tipo de
   deducible (efecto protector del alto: 0.79×).

3. **La validación cruzada 5-fold** muestra baja variabilidad entre folds,
   indicando que el modelo es estable y no presenta sobreajuste significativo.

4. **La tabla de tarifas** generada es directamente utilizable por el equipo
   comercial para cotizar nuevos asegurados.

### Limitaciones y extensiones

- El parámetro `power` de la Tweedie se fijó en 1.5; en producción se estima
  mediante máxima verosimilitud perfilada (búsqueda en rejilla de devianza).
- Se pueden explorar interacciones (fumador × edad, IMC × edad) y efectos no
  lineales con splines.
- Un modelo de dos partes (frecuencia + severidad por separado) puede mejorar
  la interpretabilidad cuando ambos componentes tienen drivers distintos.

---

## Referencias

- Ohlsson, E., & Johansson, B. (2010). *Non-Life Insurance Pricing with Generalized Linear Models*. Springer.
- Nelder, J. A., & Wedderburn, R. W. M. (1972). *Generalized Linear Models*. Journal of the Royal Statistical Society.
- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall.
- James, G., et al. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR.

---
