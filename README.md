# 💱 Dólar Paralelo Bolivia — Alertas con ARIMA

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-4.x-092E20?style=flat&logo=django&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14-3498DB?style=flat)
![ARIMA](https://img.shields.io/badge/Modelo-ARIMA-orange?style=flat)
![Status](https://img.shields.io/badge/Estado-En_Uso-brightgreen)

Sistema de alertas para el mercado del dólar paralelo boliviano. Usa un modelo ARIMA entrenado con datos históricos desde abril 2023 para generar proyecciones a 2 semanas y alertas automáticas cuando el tipo de cambio supera o rompe bandas estadísticas.

> **En uso real en [Kapitalya](https://kapitalya.com).** Los datos publicados aquí son sintéticos — los datos reales fueron anonimizados antes de subir el código.

---

## 🎯 Por Qué ARIMA para el Dólar Paralelo

El mercado paralelo boliviano tiene características que lo hacen modelable con series de tiempo:

1. **Autocorrelación significativa**: el tipo de cambio de hoy predice el de mañana mejor que la media histórica (test ADF confirma estacionariedad en primeras diferencias)
2. **No hay suficientes datos para LSTM**: con ~400 observaciones diarias, los modelos de deep learning no generalizan bien
3. **Interpretabilidad**: Kapitalya necesita explicar las alertas a clientes no técnicos — ARIMA permite explicar "el modelo detecta que el TC está acelerando por encima de su tendencia"

---

## 📐 Metodología

```
Datos históricos TC paralelo
       ↓
Test ADF + KPSS → ¿Es estacionaria?
       ↓
ACF / PACF → Candidatos p, q
       ↓
Grid search AIC → ARIMA(p,d,q) óptimo
       ↓
Forecast 14 días con IC 80% y 95%
       ↓
Lógica de alertas:
  🔴 ALERTA ALTA:  TC > banda superior IC 80%
  🟡 ALERTA MEDIA: TC > media proyectada + 0.5σ
  🟢 NORMAL:       TC dentro de bandas
```

---

## 🚨 Lógica de Alertas

El sistema emite 3 tipos de alerta evaluadas diariamente:

| Tipo | Condición | Acción recomendada |
|---|---|---|
| 🔴 **Banda superior** | TC actual > P90 proyección | Considerar compra inmediata de USD |
| 🔴 **Banda inferior** | TC actual < P10 proyección | Posible corrección — esperar |
| 🟡 **Volatilidad alta** | σ_7d > 2 × σ_histórica | Monitorear diariamente |

---

## 📁 Estructura

```
dolar-paralelo-bolivia-arima/
├── src/
│   ├── modelo_arima.py        # Entrenamiento y forecast
│   ├── alertas.py             # Lógica de alertas
│   └── anonimizar_datos.py    # Script de anonimización ← ver abajo
├── datos/
│   └── datos_sinteticos.csv   # Datos sintéticos para demo
├── notebooks/
│   └── exploracion_arima.ipynb
├── requirements.txt
└── README.md
```

---

## 🔒 Privacidad de Datos

Los datos reales del tipo de cambio paralelo provienen de Kapitalya y no pueden publicarse directamente. El script `src/anonimizar_datos.py` aplica las siguientes transformaciones antes de publicar:

```python
# 1. Desplazamiento temporal aleatorio (±30 días)
# 2. Ruido gaussiano calibrado (mantiene autocorrelación, altera valores absolutos)
# 3. Normalización a índice base 100 (oculta valores absolutos reales)
# 4. Eliminación de columnas de identificación
```

Los datos sintéticos en `datos/datos_sinteticos.csv` preservan las propiedades estadísticas reales (autocorrelación, volatilidad, tendencia) sin exponer valores del cliente.

---

## ⚙️ Instalación

```bash
git clone https://github.com/Serius69/dolar-paralelo-bolivia-arima
cd dolar-paralelo-bolivia-arima

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Con datos sintéticos de demo
python src/modelo_arima.py --datos datos/datos_sinteticos.csv

# Para anonimizar tus propios datos antes de subir
python src/anonimizar_datos.py --input mis_datos_reales.csv --output datos_anonimizados.csv
```

---

## 🔗 Proyectos Relacionados

- [🇧🇴 Inflación Bolivia — SARIMA](https://github.com/Serius69/inflacion-bolivia-series-tiempo)
- [₿ Bitcoin LATAM](https://github.com/Serius69/bitcoin-latam-analisis)
- [💰 Dashboard Finanzas Personales BO](https://github.com/Serius69/dashboard-finanzas-personales-bo)

---

## 👤 Autor

**Sergio** — Data Scientist en Finanzas | [github.com/Serius69](https://github.com/Serius69)
