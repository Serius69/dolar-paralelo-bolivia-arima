"""
src/anonimizar_datos.py
─────────────────────────────────────────────────────────────────────────────
Anonimiza datos reales del tipo de cambio paralelo boliviano antes de
publicar en GitHub. Preserva propiedades estadísticas clave (autocorrelación,
volatilidad relativa, tendencia) sin exponer valores absolutos reales.

Transformaciones aplicadas:
  1. Desplazamiento temporal aleatorio (±30 días)
  2. Ruido gaussiano calibrado al 2% del valor
  3. Normalización a índice base 100
  4. Eliminación de columnas sensibles

Uso:
    python src/anonimizar_datos.py --input datos_reales.csv --output datos_anonimizados.csv
    python src/anonimizar_datos.py --demo   # Genera datos sintéticos de ejemplo
"""

import argparse
import numpy as np
import pandas as pd
from datetime import timedelta
import os


# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
SEED                = 42
RUIDO_PCT           = 0.02      # 2% de ruido sobre el valor
MAX_DESPLAZ_DIAS    = 30        # Desplazamiento temporal máximo
COLUMNAS_SENSIBLES  = [         # Columnas a eliminar si existen
    'usuario', 'fuente', 'ip', 'cliente', 'empresa',
    'source', 'user', 'id_cliente', 'nombre'
]


# ─── FUNCIONES ────────────────────────────────────────────────────────────────

def cargar_datos(ruta: str, col_fecha: str = 'fecha',
                  col_tc: str = 'tc_paralelo') -> pd.DataFrame:
    """Carga CSV y valida columnas mínimas requeridas."""
    df = pd.read_csv(ruta, parse_dates=[col_fecha])
    df = df.rename(columns={col_fecha: 'fecha', col_tc: 'tc_paralelo'})
    df = df.sort_values('fecha').reset_index(drop=True)

    print(f'✅ Cargados {len(df)} registros')
    print(f'   Período: {df["fecha"].min().date()} → {df["fecha"].max().date()}')
    print(f'   TC rango: {df["tc_paralelo"].min():.4f} — {df["tc_paralelo"].max():.4f}')
    return df


def desplazar_fechas(df: pd.DataFrame, semilla: int = SEED) -> pd.DataFrame:
    """Desplaza todas las fechas por un número aleatorio de días."""
    np.random.seed(semilla)
    dias = np.random.randint(-MAX_DESPLAZ_DIAS, MAX_DESPLAZ_DIAS)
    df = df.copy()
    df['fecha'] = df['fecha'] + timedelta(days=int(dias))
    print(f'   Fechas desplazadas: {dias:+d} días')
    return df


def agregar_ruido(df: pd.DataFrame, col: str = 'tc_paralelo',
                   pct: float = RUIDO_PCT, semilla: int = SEED) -> pd.DataFrame:
    """
    Agrega ruido gaussiano calibrado al valor.
    El ruido es proporcional al valor para mantener la escala relativa.
    Se usa semilla para que sea reproducible.
    """
    np.random.seed(semilla + 1)
    df = df.copy()
    sigma = df[col].mean() * pct
    ruido = np.random.normal(0, sigma, len(df))
    df[col] = (df[col] + ruido).clip(lower=df[col].min() * 0.8)
    print(f'   Ruido gaussiano aplicado: σ = {sigma:.4f} ({pct*100:.0f}% del valor medio)')
    return df


def normalizar_a_indice(df: pd.DataFrame, col: str = 'tc_paralelo',
                          base: float = 100.0) -> pd.DataFrame:
    """
    Normaliza la serie a un índice base 100.
    Oculta el valor absoluto real pero preserva variaciones relativas.
    """
    df = df.copy()
    primer_valor = df[col].iloc[0]
    df[col] = df[col] / primer_valor * base
    df = df.rename(columns={col: 'indice_tc_paralelo'})
    print(f'   Normalizado a base {base} (valor original inicial: {primer_valor:.4f})')
    return df


def eliminar_columnas_sensibles(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas que podrían identificar la fuente o clientes."""
    cols_presentes = [c for c in COLUMNAS_SENSIBLES if c in df.columns]
    if cols_presentes:
        df = df.drop(columns=cols_presentes)
        print(f'   Columnas eliminadas: {cols_presentes}')
    return df


def verificar_propiedades(df_original: pd.DataFrame, df_anon: pd.DataFrame,
                           col_orig: str = 'tc_paralelo',
                           col_anon: str = 'indice_tc_paralelo') -> None:
    """
    Verifica que las propiedades estadísticas clave se preservaron.
    Imprime comparación.
    """
    from scipy.stats import pearsonr

    orig = df_original[col_orig].values
    anon = df_anon[col_anon].values

    # Retornos (captura autocorrelación y volatilidad relativa)
    ret_orig = np.diff(orig) / orig[:-1]
    ret_anon = np.diff(anon) / anon[:-1]

    corr, _ = pearsonr(ret_orig, ret_anon)

    print(f'\n   📊 Verificación de propiedades estadísticas:')
    print(f'   Correlación retornos orig vs anon:  {corr:.4f} (>0.95 = bien preservado)')
    print(f'   Volatilidad original:               {ret_orig.std()*100:.4f}%')
    print(f'   Volatilidad anonimizada:            {ret_anon.std()*100:.4f}%')
    print(f'   Autocorrelación lag-1 original:     {pd.Series(orig).autocorr(1):.4f}')
    print(f'   Autocorrelación lag-1 anonimizada:  {pd.Series(anon).autocorr(1):.4f}')


def generar_datos_sinteticos(n_dias: int = 400, semilla: int = SEED) -> pd.DataFrame:
    """
    Genera una serie sintética de TC paralelo con propiedades realistas.
    Útil para probar el pipeline sin datos reales.
    """
    np.random.seed(semilla)
    fechas = pd.date_range('2023-04-01', periods=n_dias, freq='B')  # Días hábiles

    # Proceso ARIMA(1,1,1) sintético
    tc = [7.20]  # Valor inicial aproximado al período
    errores = np.random.normal(0, 0.025, n_dias)
    for i in range(1, n_dias):
        # Tendencia alcista leve + AR(1) + ruido
        cambio = 0.003 + 0.35 * (tc[-1] - tc[-2] if len(tc) > 1 else 0) + errores[i]
        tc.append(tc[-1] + cambio)

    df = pd.DataFrame({
        'fecha':       fechas[:len(tc)],
        'tc_paralelo': np.array(tc[:len(fechas)])
    })
    return df


# ─── PIPELINE COMPLETO ────────────────────────────────────────────────────────

def anonimizar(ruta_entrada: str, ruta_salida: str,
                col_fecha: str = 'fecha', col_tc: str = 'tc_paralelo') -> pd.DataFrame:
    """Pipeline completo de anonimización."""
    print('\n' + '='*55)
    print('PIPELINE DE ANONIMIZACIÓN')
    print('='*55)

    df_orig = cargar_datos(ruta_entrada, col_fecha, col_tc)

    print('\n⚙️  Aplicando transformaciones:')
    df = desplazar_fechas(df_orig)
    df = agregar_ruido(df)
    df = normalizar_a_indice(df)
    df = eliminar_columnas_sensibles(df)

    verificar_propiedades(df_orig, df)

    os.makedirs(os.path.dirname(ruta_salida) if os.path.dirname(ruta_salida) else '.', exist_ok=True)
    df.to_csv(ruta_salida, index=False)
    print(f'\n✅ Datos anonimizados guardados en: {ruta_salida}')
    print(f'   Registros: {len(df)} | Columnas: {list(df.columns)}')
    return df


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anonimizador de datos TC paralelo Bolivia')
    parser.add_argument('--input',     default=None, help='CSV de entrada con datos reales')
    parser.add_argument('--output',    default='datos/datos_anonimizados.csv',
                        help='CSV de salida anonimizado')
    parser.add_argument('--col-fecha', default='fecha',      help='Columna de fecha')
    parser.add_argument('--col-tc',    default='tc_paralelo', help='Columna del TC')
    parser.add_argument('--demo',      action='store_true',
                        help='Generar datos sintéticos de demo sin datos reales')
    args = parser.parse_args()

    if args.demo or args.input is None:
        print('📊 Generando datos sintéticos de demostración...')
        df_sint = generar_datos_sinteticos(n_dias=400)
        os.makedirs('datos', exist_ok=True)
        ruta_sint = 'datos/datos_sinteticos.csv'
        df_sint.to_csv(ruta_sint, index=False)
        print(f'✅ Datos sintéticos guardados en {ruta_sint}')
        print(f'   {len(df_sint)} días hábiles | '
              f'{df_sint["fecha"].min().date()} → {df_sint["fecha"].max().date()}')
        print(f'\n💡 Para anonimizar tus datos reales:')
        print(f'   python src/anonimizar_datos.py --input mis_datos.csv --output datos/anonimizados.csv')
    else:
        anonimizar(args.input, args.output, args.col_fecha, args.col_tc)
