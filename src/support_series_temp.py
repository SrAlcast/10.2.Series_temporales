# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Visualizaciones
# -----------------------------------------------------------------------
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


# Analisis Exploratorio Series Temporales
# -----------------------------------------------------------------------
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Modelo Series Temporales
# -----------------------------------------------------------------------
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from itertools import product
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Otros
# -----------------------------------------------------------------------
from tqdm import tqdm



def exploracion_grafica(dataframe,columna_variable):
    decomposition = seasonal_decompose(dataframe[columna_variable], model='additive', period=12)
    # Crear figura y subplots
    fig, axes = plt.subplots(4, 1, figsize= (20, 15), sharex=True)

    # Serie original
    axes[0].plot(dataframe[columna_variable], color="blue", linewidth=2)
    axes[0].set_title("Serie Original", fontsize=14)
    axes[0].grid(visible=True, linestyle="--", alpha=0.6)

    # Tendencia
    axes[1].plot(decomposition.trend, color="orange", linewidth=2)
    axes[1].set_title("Tendencia", fontsize=14)
    axes[1].grid(visible=True, linestyle="--", alpha=0.6)

    # Estacionalidad
    axes[2].plot(decomposition.seasonal, color="green", linewidth=2)
    axes[2].set_title("Estacionalidad", fontsize=14)
    axes[2].grid(visible=True, linestyle="--", alpha=0.6)

    # Ruido
    axes[3].plot(decomposition.resid, color="red", linewidth=2)
    axes[3].set_title("Ruido (Residuo)", fontsize=14)
    axes[3].grid(visible=True, linestyle="--", alpha=0.6)

    # Ajustar diseño
    plt.suptitle("Descomposición de la Serie Temporal", fontsize=16, y=0.95)
    plt.xlabel("Fecha", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def exploracion_datos(data,columna,frecuencia=None):
    """
    Exploración completa de una serie temporal.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos.
        columna (str): Nombre de la columna de la serie temporal.
        frecuencia (int): Frecuencia para la descomposición (por ejemplo, 12 para datos mensuales).
    
    Returns:
        dict: Resultados del análisis con claves de Tendencia, Estacionalidad, Residuo y prueba ADF.
    """
    resultados = {}
    
    # 1. Información básica
    print("Resumen estadístico:")
    display(data[columna].describe())
    print("\nPrimeros datos:")
    display(data.head())
    print("\nÚltimos datos:")
    display(data.tail())
    
    # 2. Gráfico de la serie temporal
    plt.figure(figsize=(10, 6))
    data[columna].plot(title=f"Serie Temporal - {columna}")
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.grid()
    plt.show()
    
    # 3. Histogramas y distribución
    plt.figure(figsize=(10, 6))
    data[columna].hist(bins=30, alpha=0.7)
    plt.title("Distribución de la Serie Temporal")
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.grid()
    plt.show()
    
    return resultados

def calcular_lags_recomendados(data):
    """
    Calcula dos opciones recomendadas de lags para gráficos ACF y PACF.
    
    Parámetros:
        data (pd.Series o array-like): Serie temporal o datos.
        
    Retorno:
        dict: Un diccionario con dos opciones:
              - "sqrt": Número de lags basado en la raíz cuadrada del tamaño de la muestra.
              - "n4": Número de lags basado en un cuarto del tamaño de la muestra.
    """
    N = len(data)  # Número de observaciones
    
    # Opción 1: Raíz cuadrada del tamaño de la muestra
    lags_sqrt = int(np.sqrt(N))
    
    # Opción 2: Un cuarto del tamaño de la muestra
    lags_n4 = int(N / 4)
    
    return {
        "sqrt": lags_sqrt,
        "n4": lags_n4
    }

def graficar_autocorrelacion(data, lags=40):
    """
    Genera un gráfico de la función de autocorrelación (ACF).
    
    Parámetros:
        data (pd.Series o array-like): Serie temporal o datos para calcular la autocorrelación.
        lags (int): Número de retardos (lags) a incluir en el gráfico.
        titulo (str): Título del gráfico.
    """
    plt.figure(figsize=(10, 6))
    plot_acf(data, lags=lags)
    plt.title("Función de Autocorrelación (ACF)")
    plt.xlabel("*lags* (Lags)")
    plt.ylabel("Correlación")
    plt.grid()
    plt.show()

def graficar_autocorrelacion_parcial(data, lags=24):
    """
    Genera un gráfico de la función de autocorrelación parcial (PACF).
    
    Parámetros:
        data (pd.Series o array-like): Serie temporal o datos para calcular la autocorrelación parcial.
        lags (int): Número de retardos (lags) a incluir en el gráfico.
        metodo (str): Método para calcular el PACF ('ywm', 'ols', etc.).
        titulo (str): Título del gráfico.
    """
    plt.figure(figsize=(10, 6))
    plot_pacf(data, lags=lags, method="ywm")
    plt.title("Función de Autocorrelación Parcial (PACF)")
    plt.xlabel("*lags* (Lags)")
    plt.ylabel("Correlación Parcial")
    plt.grid()
    plt.show()

def aplicar_adf_test(serie, alpha=0.05):
    """
    Aplica el test de Dickey-Fuller aumentado (ADF) a una serie temporal.
    
    Parámetros:
        serie (pd.Series): Serie temporal sobre la que se aplica el test.
        alpha (float): Nivel de significancia para interpretar los resultados. Por defecto, 0.05.
        
    Retorna:
        dict: Un diccionario con los resultados principales del test:
              - Estadístico del test
              - Valor p (p-value)
              - Lags utilizados
              - Número de observaciones
              - Valor crítico para diferentes niveles
              - Resultado de estacionariedad
    """
    # Aplicar el test ADF
    resultado = adfuller(serie, autolag='AIC')
    estadistico, p_valor, lags, n_obs, valores_criticos, _ = resultado
    
    # Interpretación del resultado
    estacionariedad = p_valor < alpha
    
    # Imprimir resultados principales
    print("Resultados del Test de Dickey-Fuller Aumentado (ADF):")
    print(f"Estadístico del Test: {estadistico:.4f}")
    print(f"Valor p: {p_valor:.4f}")
    print(f"Número de Lags: {lags}")
    print(f"Número de Observaciones: {n_obs}")
    print("Valores Críticos:")
    for nivel, valor_critico in valores_criticos.items():
        print(f"  {nivel}: {valor_critico:.4f}")
    print(f"\n¿Es la serie estacionaria? {'Sí' if estacionariedad else 'No'} (Nivel de significancia: {alpha})")
    
    # Retornar resultados como diccionario
    return {
        "Estadístico del Test": estadistico,
        "Valor p": p_valor,
        "Lags Utilizados": lags,
        "Número de Observaciones": n_obs,
        "Valores Críticos": valores_criticos,
        "Estacionariedad": estacionariedad}

def graficar_acf_pacf(serie, lags=30):
    """
    Genera los gráficos de ACF (Autocorrelation Function) y PACF (Partial Autocorrelation Function).
    
    Parámetros:
        serie (pd.Series o array-like): Serie temporal sobre la que se generan los gráficos.
        lags (int): Número de retardos (lags) a incluir en los gráficos.
    """
    # Crear figura
    plt.figure(figsize=(12, 6))
    
    # Gráfico ACF
    plt.subplot(1, 2, 1)
    plot_acf(serie, lags=lags, ax=plt.gca())
    plt.title("Gráfico ACF")
    plt.xlabel("Lags")
    plt.ylabel("Correlación")
    
    # Gráfico PACF
    plt.subplot(1, 2, 2)
    plot_pacf(serie, lags=lags, method="ywm", ax=plt.gca())
    plt.title("Gráfico PACF")
    plt.xlabel("Lags")
    plt.ylabel("Correlación Parcial")
    
    # Mostrar gráficos
    plt.tight_layout()
    plt.show()

def entrenar_sarimax(df, columna, train_ratio=0.8, seasonal_order=(1, 1, 1, 12), max_p=3, max_q=10):
    """
    Crea y entrena el modelo SARIMAX con la mejor combinación de parámetros p y q.
    
    Parámetros:
        df (pd.DataFrame): DataFrame que contiene la serie temporal.
        columna (str): Nombre de la columna de interés.
        train_ratio (float): Proporción de datos para entrenamiento (el resto para prueba).
        seasonal_order (tuple): Orden estacional (P, D, Q, s).
        max_p (int): Máximo valor para p en el modelo.
        max_q (int): Máximo valor para q en el modelo.
        
    Retorna:
        dict: Contiene el modelo entrenado, los conjuntos de entrenamiento y prueba, 
              los valores óptimos de p y q, y el RMSE.
    """
    ps = range(max_p + 1)
    qs = range(max_q + 1)
    combinaciones = list(product(ps, qs))
    
    # Dividir los datos en entrenamiento y prueba
    train_size = int(len(df) * train_ratio)
    train, test = df[columna][:train_size], df[columna][train_size:]
    
    resultados = {"p": [], "q": [], "rmse": []}
    
    # Ajustar el modelo para diferentes combinaciones de p y q
    for p, q in combinaciones:
        try:
            modelo_prueba = SARIMAX(train, 
                                    order=(p, 1, q),  # ARIMA(p, d, q)
                                    seasonal_order=seasonal_order).fit(disp=False)
            pred_test = modelo_prueba.predict(start=train_size, end=len(df)-1)
            rmse_valor = np.sqrt(mean_squared_error(test, pred_test))
            resultados["p"].append(p)
            resultados["q"].append(q)
            resultados["rmse"].append(rmse_valor)
        except:
            # Manejo de errores si el modelo no converge
            continue
    
    # Convertir resultados a DataFrame y seleccionar el mejor modelo
    resultados_df = pd.DataFrame(resultados).sort_values("rmse", ascending=True)
    mejor_modelo = resultados_df.iloc[0]
    p_value, q_value = int(mejor_modelo["p"]), int(mejor_modelo["q"])
    
    # Ajustar el mejor modelo al conjunto de entrenamiento
    modelo = SARIMAX(train, 
                     order=(p_value, 1, q_value),  # ARIMA(p, d, q)
                     seasonal_order=seasonal_order).fit(disp=False)
    
    return {"p": p_value,
            "q": q_value,
            "rmse": mejor_modelo["rmse"]}


def evaluar_ruido_blanco(modelo, train, test, df, columna):
    """
    Evalúa si los residuales del modelo cumplen las condiciones de ruido blanco.
    
    Parámetros:
        modelo: Modelo SARIMAX entrenado.
        train (pd.Series): Conjunto de entrenamiento.
        test (pd.Series): Conjunto de prueba.
        df (pd.DataFrame): DataFrame original para referencias.
        columna (str): Nombre de la columna de interés.
    
    Retorna:
        dict: Contiene los residuales, el RMSE y las predicciones futuras.
    """
    # Predicciones en el conjunto de prueba
    predicciones = modelo.predict(start=len(train), end=len(train) + len(test) - 1)
    
    # Visualización de predicciones vs valores reales
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[columna], label='Real')
    plt.plot(df.index[len(train):], predicciones, label='Predicciones', linestyle='--')
    plt.title('Predicciones SARIMAX sobre los datos de prueba')
    plt.legend()
    plt.show()
    
    # Evaluación del modelo
    rmse_error = np.sqrt(mean_squared_error(test, predicciones))
    print(f'Error cuadrático medio (RMSE): {rmse_error:.4f}')
    
    # Analizar residuales
    residuales = modelo.resid
    plt.figure(figsize=(10, 6))
    plt.plot(residuales, label="Residuales")
    plt.axhline(0, linestyle="--", color="red")
    plt.title("Residuales del Modelo SARIMAX")
    plt.legend()
    plt.show()
    
    # Predicciones futuras (12 períodos fuera del rango actual)
    predicciones_futuras = modelo.predict(start=len(train) + len(test), end=len(train) + len(test) + 11)
    print("Predicciones futuras:")
    print(predicciones_futuras)
    
    return {"residuales": residuales,
            "rmse": rmse_error,
            "predicciones_futuras": predicciones_futuras}

def predecir_futuro(modelo, data, columna, meses=12):
    """
    Genera predicciones para un número dado de meses futuros usando un modelo SARIMAX entrenado.
    
    Parámetros:
        modelo: Modelo SARIMAX ya entrenado.
        data (pd.DataFrame): DataFrame original para referencia.
        columna (str): Nombre de la columna con los datos originales.
        meses (int): Número de meses a predecir.
    
    Retorna:
        pd.Series: Predicciones futuras para los próximos meses.
    """
    # Predicción para los próximos meses
    inicio_prediccion = len(data)
    fin_prediccion = inicio_prediccion + meses - 1
    predicciones_futuras = modelo.predict(start=inicio_prediccion, end=fin_prediccion)
    
    # Crear índice de tiempo para las predicciones futuras
    indice_futuro = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(1), periods=meses, freq='M')
    predicciones_futuras.index = indice_futuro
    
    # Visualizar las predicciones futuras
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data[columna], label='Datos Originales')
    plt.plot(predicciones_futuras.index, predicciones_futuras, label='Predicciones Futuras', linestyle='--', color='red')
    plt.title('Predicciones para los Próximos Meses')
    plt.xlabel('Fecha')
    plt.ylabel('Valores')
    plt.legend()
    plt.grid()
    plt.show()
    
    return display(predicciones_futuras)
    











class TimeSeriesAnalysis:
    def __init__(self, dataframe, temporal_column, value_column):
        """
        Inicializa el objeto TimeSeriesAnalysis.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            El DataFrame que contiene los datos de la serie temporal.
        temporal_column : str
            Nombre de la columna con las fechas o tiempo.
        value_column : str
            Nombre de la columna con los valores de la serie temporal.
        """
        self.data = dataframe.copy()
        self.temporal_column = temporal_column
        self.value_column = value_column

        # Asegurar que la columna temporal es de tipo datetime
        self.data[self.temporal_column] = pd.to_datetime(self.data[self.temporal_column])
        self.data.set_index(self.temporal_column, inplace=True)
    
    def exploracion_datos(self):
        """
        Realiza una exploración básica de los datos.
        """
        print(f"El número de filas es {self.data.shape[0]} y el número de columnas es {self.data.shape[1]}")
        print("\n----------\n")
        
        if self.data.duplicated().sum() > 0:
            print(f"En este conjunto de datos tenemos {self.data.duplicated().sum()} valores duplicados")
        else:
            print("No hay duplicados")
        
        print("\n----------\n")
        if self.data.isnull().sum().sum() > 0:
            print("Las columnas con valores nulos y sus porcentajes son:")
            nulos = self.data.isnull().sum()
            display((nulos[nulos > 0] / self.data.shape[0]) * 100)
        else:
            print("No hay valores nulos")
        
        print("\n----------\n")
        print("Estadísticas de las variables numéricas:")
        display(self.data.describe().T)
    
    def comprobar_serie_continua(self):
        """
        Comprueba si la serie temporal es continua.
        """
        fecha_completa = pd.date_range(start=self.data.index.min(), end=self.data.index.max(), freq="MS")
        mes_anio_actual = self.data.index.to_period("M")
        mes_anio_completo = fecha_completa.to_period("M")
        meses_faltantes = mes_anio_completo.difference(mes_anio_actual)

        if len(meses_faltantes) == 0:
            print("La serie temporal es continua, no faltan meses.")
        else:
            print("La serie temporal NO es continua.")
            print("Meses-Años faltantes:", meses_faltantes)
    
    def graficar_serie(self):
        """
        Grafica la serie temporal original.
        """
        fig = px.line(
            self.data,
            x=self.data.index,
            y=self.value_column,
            title="Serie Temporal Original",
            labels={self.temporal_column: "Fecha", self.value_column: "Valores"}
        )
        fig.update_layout(template="plotly_white", xaxis_title="Fecha", yaxis_title="Valores")
        fig.show()
    
    def graficar_media_movil(self, window=30):
        """
        Grafica la media móvil de la serie temporal.
        
        Parameters:
        -----------
        window : int
            Tamaño de la ventana para calcular la media móvil.
        """
        self.data["rolling_window"] = self.data[self.value_column].rolling(window=window).mean()
        fig = px.line(
            self.data,
            x=self.data.index,
            y=[self.value_column, "rolling_window"],
            title="Evolución con Media Móvil",
            labels={self.temporal_column: "Fecha", self.value_column: "Valores"}
        )
        fig.data[0].update(name="Valores Originales")
        fig.data[1].update(name=f"Media Móvil ({window} días)", line=dict(color="red"))
        fig.update_layout(template="plotly_white", xaxis_title="Fecha", yaxis_title="Valores")
        fig.show()
    
    def detectar_estacionalidad(self, figsize = (12, 10)):
        """
        Detecta visualmente si la serie temporal tiene un componente estacional.
        """
        decomposition = seasonal_decompose(self.data[self.value_column], model='additive', period=12)
        
        # Crear figura y subplots
        fig, axes = plt.subplots(4, 1, figsize= figsize, sharex=True)
        
        # Serie original
        axes[0].plot(self.data[self.value_column], color="blue", linewidth=2)
        axes[0].set_title("Serie Original", fontsize=14)
        axes[0].grid(visible=True, linestyle="--", alpha=0.6)
        
        # Tendencia
        axes[1].plot(decomposition.trend, color="orange", linewidth=2)
        axes[1].set_title("Tendencia", fontsize=14)
        axes[1].grid(visible=True, linestyle="--", alpha=0.6)
        
        # Estacionalidad
        axes[2].plot(decomposition.seasonal, color="green", linewidth=2)
        axes[2].set_title("Estacionalidad", fontsize=14)
        axes[2].grid(visible=True, linestyle="--", alpha=0.6)
        
        # Ruido
        axes[3].plot(decomposition.resid, color="red", linewidth=2)
        axes[3].set_title("Ruido (Residuo)", fontsize=14)
        axes[3].grid(visible=True, linestyle="--", alpha=0.6)
        
        # Ajustar diseño
        plt.suptitle("Descomposición de la Serie Temporal", fontsize=16, y=0.95)
        plt.xlabel("Fecha", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    
    def graficar_acf_pacf(self, lags=40):
        """
        Grafica las funciones de autocorrelación (ACF) y autocorrelación parcial (PACF).
        
        Parameters:
        -----------
        lags : int
            Número de rezagos a graficar.
        """
        plt.figure(figsize=(12, 10))
        plot_acf(self.data[self.value_column].dropna(), lags=lags)
        plt.title("Función de Autocorrelación (ACF)")
        plt.grid()
        plt.show()
        
        plt.figure(figsize=(12, 10))
        plot_pacf(self.data[self.value_column].dropna(), lags=lags, method="ywm")
        plt.title("Función de Autocorrelación Parcial (PACF)")
        plt.grid()
        plt.show()
    
    def prueba_estacionariedad(self):
        """
        Aplica la prueba de Dickey-Fuller aumentada para verificar estacionariedad.
        """
        result = adfuller(self.data[self.value_column].dropna())
        print("ADF Statistic:", result[0])
        print("p-value:", result[1])
        print("Valores Críticos:")
        for key, value in result[4].items():
            print(f"{key}: {value}")
        if result[1] < 0.05:
            print("Rechazamos la hipótesis nula. La serie es estacionaria.")
        else:
            print("No podemos rechazar la hipótesis nula. La serie NO es estacionaria.")


class SARIMAModel:
    def __init__(self):
        self.best_model = None
        self.best_params = None

    def generar_parametros(self, p_range, q_range, seasonal_order_ranges):
        """
        Genera combinaciones de parámetros SARIMA de forma automática.

        Args:
            p_range (range): Rango de valores para el parámetro p.
            q_range (range): Rango de valores para el parámetro q.
            seasonal_order_ranges (tuple of ranges): Rango de valores para los parámetros estacionales (P, D, Q, S).

        Returns:
            list of tuples: Lista con combinaciones en formato (p, q, (P, D, Q, S)).
        """
        P_range, D_range, Q_range, S_range = seasonal_order_ranges

        parametros = [
            (p, q, (P, D, Q, S))
            for p, q, (P, D, Q, S) in product(
                p_range, q_range, product(P_range, D_range, Q_range, S_range)
            )
        ]

        return parametros

    def evaluar_modelos(self, y_train, y_test, parametros, diferenciacion, df_length, variable):
        """
        Evalúa combinaciones de parámetros SARIMA, devuelve un DataFrame con los resultados,
        y genera una visualización de las predicciones comparadas con los valores reales.

        Args:
            y_train (pd.Series): Serie temporal de entrenamiento.
            y_test (pd.Series): Serie temporal de prueba.
            parametros (list of tuples): Lista de combinaciones de parámetros en formato [(p, q, (P, D, Q, S)), ...].
            diferenciacion (int): Valor para el parámetro `d` de diferenciación.
            df_length (int): Longitud total del dataset para calcular los índices de predicción.

        Returns:
            pd.DataFrame: DataFrame con las combinaciones de parámetros y los errores RMSE.
        """
        results = []

        for p, q, seasonal_order in tqdm(parametros):
            try:
                # Crear y entrenar el modelo SARIMAX
                modelo_sarima = SARIMAX(
                    y_train,
                    order=(p, diferenciacion, q),
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)


                start_index = len(y_train)
                end_index = df_length - 1
                pred_test = modelo_sarima.predict(start=start_index, end=end_index)
                pred_test = pd.Series(pred_test, index=y_test.index)  # Convertir a Serie de pandas

                # Calcular RMSE para el conjunto de prueba
                error = np.sqrt(mean_squared_error(y_test, pred_test))
                results.append({"p": p, "q": q, "seasonal_order": seasonal_order, "RMSE": error})

                # Guardar el mejor modelo
                if self.best_model is None or error < self.best_model["RMSE"]:
                    self.best_model = {
                        "modelo": modelo_sarima,
                        "RMSE": error,
                        "pred_test": pred_test,
                    }
                    self.best_params = {"p": p, "q": q, "seasonal_order": seasonal_order}

            except Exception as e:
                # Manejar errores durante el ajuste
                results.append({"p": p, "q": q, "seasonal_order": seasonal_order, "RMSE": None})

        # Convertir los resultados a un DataFrame
        results_df = pd.DataFrame(results)

        # Visualizar las predicciones del mejor modelo
        self._visualizar_predicciones_test(y_test, variable)
        return results_df


    def _visualizar_predicciones_test(self, y_test, variable):
        """
        Visualiza las predicciones del mejor modelo SARIMA comparando
        los valores reales y predicciones del conjunto de prueba, incluyendo
        el intervalo de confianza.

        Args:
            y_test (pd.Series): Serie temporal de prueba.
            variable (str): Nombre de la variable objetivo.
        """
        if self.best_model is None:
            raise ValueError("No se ha ajustado ningún modelo aún. Llama a 'evaluar_modelos' primero.")

        # Obtener las predicciones y el intervalo de confianza
        modelo = self.best_model["modelo"]
        forecast = modelo.get_forecast(steps=len(y_test))
        pred_test = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # Crear la figura
        plt.figure(figsize=(14, 7))

        # Graficar valores reales
        sns.lineplot(x=y_test.index, y=y_test[variable], label="Valores Reales", color="blue", linewidth=2)

        # Graficar predicciones
        sns.lineplot(x=y_test.index, y=pred_test, label="Predicciones SARIMA", color="red", linestyle="--", linewidth=2)

        # Graficar intervalo de confianza
        plt.fill_between(
            y_test.index,
            conf_int.iloc[:, 0],  # Límite inferior
            conf_int.iloc[:, 1],  # Límite superior
            color="pink",
            alpha=0.3,
            label="Intervalo de Confianza",
        )

        # Personalización
        plt.title("Comparación de Predicciones vs Valores Reales (Conjunto de Prueba)", fontsize=16)
        plt.xlabel("Fecha", fontsize=14)
        plt.ylabel("Valores", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

