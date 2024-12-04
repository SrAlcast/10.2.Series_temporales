# 📊 Análisis y Modelado de Series Temporales

## 📖 Descripción

Este proyecto está diseñado para practicar el análisis y modelado de series temporales utilizando el modelo SARIMAX. Se enfoca en identificar patrones como tendencias y estacionalidad, verificar la estacionaridad de las series, y construir modelos predictivos para ayudar en la toma de decisiones estratégicas basadas en datos históricos.

El análisis se lleva a cabo sobre una serie temporal relacionada con ventas mensuales de una empresa ficticia, abarcando un periodo de cinco años.

## 🗂️ Estructura del Proyecto

```plaintext
├── data/                # Datos crudos y procesados
├── notebooks/           # Notebooks de Jupyter con el análisis
├── src/                 # Scripts de procesamiento y modelado
├── results/             # Gráficos y archivos de resultados
├── README.md            # Descripción del proyecto
```

## 🛠️ Instalación y Requisitos

Este proyecto utiliza Python 3.8 y requiere las siguientes bibliotecas:

- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [statsmodels](https://www.statsmodels.org/)
- [scipy](https://scipy.org/)

Recomendamos usar un entorno virtual para instalar las dependencias.

## 📊 Resultados y Conclusiones

- **Estacionaridad**: Aplicamos transformaciones y pruebas estadísticas para lograr series estacionarias, como el test de Dickey-Fuller aumentado.
- **Modelado**: Ajustamos un modelo SARIMAX que captura tanto componentes estacionales como no estacionales.
- **Predicción**: Las predicciones de los próximos 12 meses fueron consistentes con las tendencias históricas observadas, logrando una buena calidad predictiva.
- **Insights**:
  - Las ventas presentan un claro patrón estacional anual.
  - El modelo es útil para prever periodos de alta demanda, mejorando la planificación estratégica.

## 🔄 Próximos Pasos

- Explorar el uso de datos externos (e.g., clima, campañas de marketing) para mejorar el modelo.
- Implementar un pipeline de automatización para actualizar el modelo con nuevos datos.
- Probar otros enfoques avanzados como modelos basados en redes neuronales.

