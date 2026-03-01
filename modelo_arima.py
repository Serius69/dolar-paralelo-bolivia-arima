"""
src/modelo_arima.py
─────────────────────────────────────────────────────────────────────────────
Pipeline de producción del modelo ARIMA para el tipo de cambio paralelo
boliviano. Entrena, valida, genera forecast y evalúa alertas.

Uso:
    python src/modelo_arima.py --datos datos/datos_sinteticos.csv
    python src/modelo_arima.py --datos datos/datos_sinteticos.csv --horizonte 14
    python src/modelo_arima.py --demo   # genera datos sintéticos y corre todo
"""

import argparse
import itertools
import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings('ignore')

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
TC_OFICIAL   = 6.96       # Tipo de cambio oficial fijo BCB
HORIZONTE    = 14         # Días hábiles de forecast
ALPHA_80     = 0.20       # Para IC 80%
ALPHA_95     = 0.05       # Para IC 95%
MODELS_DIR   = 'modelos'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs('img', exist_ok=True)

# ─── UMBRALES DE ALERTA ───────────────────────────────────────────────────────
UMBRAL_VOL = 2.0   # Múltiplo de volatilidad histórica para alerta de vol.


# ─── FUNCIONES ────────────────────────────────────────────────────────────────

def cargar_serie(ruta: str) -> pd.Series:
    """Carga CSV y devuelve serie temporal con frecuencia de días hábiles."""
    df = pd.read_csv(ruta, parse_dates=['fecha'])
    df = df.sort_values('fecha').reset_index(drop=True)
    col = 'tc_paralelo' if 'tc_paralelo' in df.columns else 'indice_tc_paralelo'
    serie = df.set_index('fecha')[col].asfreq('B')
    serie = serie.interpolate(method='time')
    print(f'✅ Serie cargada: {len(serie)} observaciones '
          f'({serie.index[0].date()} → {serie.index[-1].date()})')
    return serie


def es_estacionaria(serie: pd.Series) -> tuple[bool, int]:
    """
    Determina si la serie es estacionaria con ADF + KPSS.
    Retorna (es_estacionaria, orden_diferenciacion).
    """
    adf_p  = adfuller(serie.dropna(), autolag='AIC')[1]
    kpss_p = kpss(serie.dropna(), regression='c', nlags='auto')[1]

    if adf_p < 0.05 and kpss_p > 0.05:
        return True, 0

    # Probar con primera diferencia
    diff1  = serie.diff().dropna()
    adf_p1 = adfuller(diff1, autolag='AIC')[1]
    return False, 1 if adf_p1 < 0.05 else 2


def seleccionar_arima(serie: pd.Series,
                       max_p: int = 4, max_d: int = 2,
                       max_q: int = 4) -> tuple:
    """Grid search ARIMA(p,d,q) minimizando AIC."""
    resultados = []
    combinaciones = [
        (p, d, q)
        for p, d, q in itertools.product(range(max_p+1), range(max_d+1), range(max_q+1))
        if not (p == 0 and q == 0)
    ]
    print(f'⚙️  Evaluando {len(combinaciones)} combinaciones ARIMA(p,d,q)...')

    for p, d, q in combinaciones:
        try:
            m = SARIMAX(serie, order=(p, d, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
            f = m.fit(disp=False)
            resultados.append({'orden': (p, d, q), 'aic': f.aic, 'bic': f.bic})
        except Exception:
            pass

    df_res = pd.DataFrame(resultados).sort_values('aic').reset_index(drop=True)
    mejor = tuple(df_res.iloc[0]['orden'])
    print(f'🏆 Mejor orden: ARIMA{mejor}  (AIC: {df_res.iloc[0]["aic"]:.2f})')
    return mejor, df_res


def ajustar_modelo(serie: pd.Series, orden: tuple) -> object:
    """Ajusta el modelo ARIMA y valida los residuos."""
    print(f'🔧 Ajustando ARIMA{orden}...')
    modelo = SARIMAX(serie, order=orden,
                     enforce_stationarity=False,
                     enforce_invertibility=False).fit(disp=False)

    residuos = modelo.resid
    lb_p = acorr_ljungbox(residuos.dropna(), lags=[10],
                           return_df=True)['lb_pvalue'].values[0]

    print(f'   AIC: {modelo.aic:.2f}  |  BIC: {modelo.bic:.2f}')
    print(f'   Ljung-Box p={lb_p:.4f} '
          f'{"✅ sin autocorrelación residual" if lb_p > 0.05 else "⚠️ revisar modelo"}')
    return modelo


def generar_forecast(modelo, horizonte: int = HORIZONTE) -> dict:
    """Genera forecast con intervalos de confianza 80% y 95%."""
    fc     = modelo.get_forecast(steps=horizonte)
    mean   = fc.predicted_mean
    ci80   = fc.conf_int(alpha=ALPHA_80)
    ci95   = fc.conf_int(alpha=ALPHA_95)

    fechas = pd.bdate_range(
        start=modelo.fittedvalues.index[-1] + pd.Timedelta(days=1),
        periods=horizonte
    )
    mean.index = ci80.index = ci95.index = fechas

    return {'mean': mean, 'ci80': ci80, 'ci95': ci95}


def evaluar_alertas(serie: pd.Series, forecast: dict) -> dict:
    """
    Evalúa nivel de alerta basado en TC actual vs bandas del forecast.

    Niveles:
        ALTO:  TC fuera de IC 80% del forecast
        MEDIO: Volatilidad reciente > UMBRAL_VOL × histórica
        NORMAL: dentro de bandas
    """
    tc_actual       = serie.dropna().iloc[-1]
    banda_sup       = forecast['ci80'].iloc[0, 1]
    banda_inf       = forecast['ci80'].iloc[0, 0]
    vol_7d          = serie.dropna().tail(7).pct_change().std()
    vol_historica   = serie.dropna().pct_change().std()
    ratio_vol       = vol_7d / vol_historica if vol_historica > 0 else 1

    if tc_actual > banda_sup:
        nivel = 'ALTO'
        msg   = f'TC ({tc_actual:.4f}) > banda superior IC80% ({banda_sup:.4f})'
        emoji = '🔴'
    elif tc_actual < banda_inf:
        nivel = 'ALTO'
        msg   = f'TC ({tc_actual:.4f}) < banda inferior IC80% ({banda_inf:.4f})'
        emoji = '🔴'
    elif ratio_vol > UMBRAL_VOL:
        nivel = 'MEDIO'
        msg   = f'Volatilidad 7d = {ratio_vol:.1f}x la histórica'
        emoji = '🟡'
    else:
        nivel = 'NORMAL'
        msg   = 'TC dentro de bandas esperadas'
        emoji = '🟢'

    return {
        'nivel':          nivel,
        'mensaje':        msg,
        'emoji':          emoji,
        'tc_actual':      tc_actual,
        'banda_superior': banda_sup,
        'banda_inferior': banda_inf,
        'ratio_vol':      round(ratio_vol, 2),
    }


def graficar_forecast(serie: pd.Series, modelo, forecast: dict,
                       alerta: dict, orden: tuple,
                       dias_historico: int = 60) -> None:
    """Genera gráfica de forecast con bandas de alerta."""
    serie_rec = serie.dropna().tail(dias_historico)
    fitted    = modelo.fittedvalues.tail(dias_historico)
    fc_mean   = forecast['mean']
    fc_ci80   = forecast['ci80']
    fc_ci95   = forecast['ci95']

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.fill_between(serie_rec.index, serie_rec.values, serie_rec.min() * 0.98,
                     alpha=0.08, color='#E74C3C')
    ax.plot(serie_rec.index, serie_rec.values, color='#E74C3C', lw=2.5,
             label='TC Paralelo (histórico)')
    ax.plot(fitted.index, fitted.values, color='#7F8C8D', lw=1.5,
             linestyle='--', alpha=0.7, label='Ajuste in-sample')
    ax.axvline(serie.dropna().index[-1], color='gray', linestyle=':', lw=2)

    ax.fill_between(fc_ci95.index, fc_ci95.iloc[:, 0], fc_ci95.iloc[:, 1],
                     alpha=0.10, color='#2E86C1', label='IC 95%')
    ax.fill_between(fc_ci80.index, fc_ci80.iloc[:, 0], fc_ci80.iloc[:, 1],
                     alpha=0.22, color='#2E86C1', label='IC 80%')
    ax.plot(fc_mean.index, fc_mean.values, color='#2E86C1', lw=2.5,
             marker='D', ms=5, label=f'Forecast ARIMA{orden}')

    color_alerta = {'NORMAL': '#27AE60', 'MEDIO': '#F39C12', 'ALTO': '#E74C3C'}[alerta['nivel']]
    ax.scatter([serie.dropna().index[-1]], [alerta['tc_actual']],
                color=color_alerta, s=220, zorder=6,
                label=f'{alerta["emoji"]} {alerta["nivel"]}: {alerta["mensaje"]}')

    ax.set_title(f'Dólar Paralelo Bolivia — Forecast {len(fc_mean)} días hábiles\n'
                  f'ARIMA{orden}', fontweight='bold')
    ax.set_ylabel('TC Paralelo')
    ax.legend(fontsize=9, loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    plt.tight_layout()
    ruta_img = 'img/forecast_dolar_paralelo.png'
    plt.savefig(ruta_img, dpi=150, bbox_inches='tight')
    print(f'💾 Gráfica guardada: {ruta_img}')
    plt.show()


def imprimir_reporte(serie: pd.Series, forecast: dict,
                      alerta: dict, orden: tuple) -> None:
    """Imprime reporte ejecutivo en consola."""
    fc_mean = forecast['mean']
    fc_ci80 = forecast['ci80']

    print('\n' + '='*60)
    print('REPORTE — DÓLAR PARALELO BOLIVIA')
    print('='*60)
    print(f'Datos:   {len(serie.dropna())} observaciones')
    print(f'Período: {serie.dropna().index[0].date()} → {serie.dropna().index[-1].date()}')
    print(f'Modelo:  ARIMA{orden}')
    print(f'\n{alerta["emoji"]} ALERTA: {alerta["nivel"]}')
    print(f'   {alerta["mensaje"]}')
    print(f'\nForecast próximos {len(fc_mean)} días hábiles:')
    print(f'  {"Fecha":<14} {"Forecast":>10} {"IC80% Lo":>10} {"IC80% Hi":>10}')
    print(f'  {"-"*46}')
    for i, (fecha, val) in enumerate(fc_mean.items()):
        lo = fc_ci80.iloc[i, 0]
        hi = fc_ci80.iloc[i, 1]
        print(f'  {str(fecha.date()):<14} {val:>10.4f} {lo:>10.4f} {hi:>10.4f}')
    print('='*60)


def guardar_modelo(modelo, orden: tuple, serie: pd.Series) -> None:
    """Serializa el modelo entrenado."""
    ruta = os.path.join(MODELS_DIR, 'arima_dolar_paralelo.pkl')
    joblib.dump({
        'modelo': modelo,
        'orden':  orden,
        'ultima_fecha': serie.dropna().index[-1],
        'aic':    modelo.aic,
    }, ruta)
    print(f'💾 Modelo guardado: {ruta}')


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run(ruta_datos: str, horizonte: int = HORIZONTE) -> None:
    """Ejecuta el pipeline completo."""
    print('\n' + '='*60)
    print('PIPELINE ARIMA — DÓLAR PARALELO BOLIVIA')
    print('='*60)

    serie = cargar_serie(ruta_datos)
    orden, _ = seleccionar_arima(serie.dropna())
    modelo   = ajustar_modelo(serie.dropna(), orden)
    forecast = generar_forecast(modelo, horizonte)
    alerta   = evaluar_alertas(serie, forecast)
    imprimir_reporte(serie, forecast, alerta, orden)
    graficar_forecast(serie, modelo, forecast, alerta, orden)
    guardar_modelo(modelo, orden, serie)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Modelo ARIMA — Dólar Paralelo Bolivia')
    parser.add_argument('--datos',     default=None,
                        help='Ruta al CSV de datos (tc_paralelo o indice_tc_paralelo)')
    parser.add_argument('--horizonte', default=HORIZONTE, type=int,
                        help=f'Días hábiles de forecast (default: {HORIZONTE})')
    parser.add_argument('--demo',      action='store_true',
                        help='Generar datos sintéticos y ejecutar pipeline completo')
    args = parser.parse_args()

    if args.demo or args.datos is None:
        # Importar generador de datos sintéticos
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from anonimizar_datos import generar_datos_sinteticos
        os.makedirs('datos', exist_ok=True)
        df_sint = generar_datos_sinteticos(n_dias=400)
        ruta    = 'datos/datos_sinteticos.csv'
        df_sint.to_csv(ruta, index=False)
        print(f'✅ Datos sintéticos generados: {ruta}')
        run(ruta, args.horizonte)
    else:
        run(args.datos, args.horizonte)
