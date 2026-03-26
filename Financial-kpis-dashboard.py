# ============================================================
# PROYECTO 3: Dashboard de KPIs Financieros

# Autor: Víctor Manuel Bancayán Capuñay

# Dataset: Financial Sample — Kaggle

# Objetivo: Construir un P&L simplificado, calcular márgenes,
#            detectar productos/segmentos más rentables y
#            analizar tendencias de ingresos en el tiempo
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
from pathlib import Path

# Directorio base = carpeta donde vive este script
BASE_DIR = Path(__file__).parent

warnings.filterwarnings('ignore')

# Configuración visual
plt.rcParams['figure.figsize'] = (13, 6)
sns.set_style("whitegrid")
sns.set_palette("Set2")

# ── Carga del dataset ────────────────────────────────────────────────────


df = pd.read_csv(BASE_DIR / 'Financials.csv', thousands=',')

print(f"Shape: {df.shape}")
print(f"\nColumnas:\n{df.columns.tolist()}")
print(f"\nTipos de datos:\n{df.dtypes}")
print(f"\nVista previa:\n{df.head(3).to_string()}")





# ============================================================
# PASO 2: Limpieza de datos financieros
#
# Los datasets financieros tienen problemas muy típicos:
# → Valores con símbolos: "$1,200.50" → necesitas 1200.50
# → Valores negativos entre paréntesis: "(500)" → -500
# → Columnas con espacios al inicio/final: "  Sales"
# → Fechas mal parseadas
# ============================================================


df_clean = df.copy()


# ── 2.1 Limpiar nombres de columnas ──────────────────────────
df_clean.columns = (df_clean.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(' ', '_')
                    .str.replace(r'[^a-z0-9_]', '', regex=True))

print("Columnas limpias:")
print(df_clean.columns.tolist())


# ── 2.2 Función para limpiar columnas monetarias ─────────────
def limpiar_moneda(serie):
    """
    Convierte strings tipo '$1,200.50' o '(500.00)' a float.
    Los paréntesis en contabilidad significan número negativo.
    """
    return (serie.astype(str)
                 .str.replace(r'[\$,\s]', '', regex=True)   # quita $, comas, espacios
                 .str.replace(r'^\((.+)\)$', r'-\1', regex=True)  # (500) → -500
                 .pipe(pd.to_numeric, errors='coerce'))      # convierte a número



# Identificar columnas numéricas que llegaron como texto
# (aplica la función solo a las que tienen $ o comas)
cols_monetarias = ['gross_sales', 'discounts', 'sales', 'cogs',
                   'profit', 'sale_price', 'manufacturing_price']

for col in cols_monetarias:
    if col in df_clean.columns:
        df_clean[col] = limpiar_moneda(df_clean[col])
        print(f"✅ {col} → dtype: {df_clean[col].dtype} | "
              f"min: {df_clean[col].min():.0f} | max: {df_clean[col].max():.0f}")



# ── 2.3 Parsear fechas ────────────────────────────────────────
# El dataset tiene columna 'date' — la convertimos a datetime
if 'date' in df_clean.columns:
    df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
    df_clean['año']  = df_clean['date'].dt.year
    df_clean['mes']  = df_clean['date'].dt.month
    df_clean['mes_nombre'] = df_clean['date'].dt.strftime('%b')   # Ene, Feb, etc.
    df_clean['trimestre']  = df_clean['date'].dt.quarter.map(
        {1:'Q1', 2:'Q2', 3:'Q3', 4:'Q4'})
    print(f"\nRango de fechas: {df_clean['date'].min()} → {df_clean['date'].max()}")



# ── 2.4 Reporte de nulos post-limpieza ───────────────────────
nulos = df_clean.isnull().sum()
print(f"\nNulos después de limpieza:\n{nulos[nulos > 0]}")
print(f"\nShape final: {df_clean.shape}")




# ============================================================
# PASO 3: P&L simplificado y KPIs
#
# P&L = Profit & Loss statement (Estado de Resultados)
#      Es el documento financiero más importante de una empresa.
#
# Estructura básica:
#
#   Ventas Brutas
# - Descuentos
# = Ventas Netas          ← "Sales" en el dataset
# - Costo de Ventas       ← COGS (Cost of Goods Sold)
# = Utilidad Bruta
# ÷ Ventas Netas
# = Margen Bruto (%)      ← el KPI más importante en finanzas
# ============================================================

# ── 3.1 Verificar que las columnas base existen ──────────────
cols_necesarias = ['gross_sales', 'discounts', 'sales', 'cogs', 'profit']
for col in cols_necesarias:
    if col not in df_clean.columns:
        print(f"⚠️  Columna '{col}' no encontrada — revisa el nombre en tu dataset")

# ── 3.2 Calcular KPIs por fila ───────────────────────────────
# Margen bruto: qué % de cada venta es ganancia después del costo
df_clean['margen_bruto_pct'] = np.where(
    df_clean['sales'] != 0,
    (df_clean['profit'] / df_clean['sales'] * 100),
    np.nan
).round(2)

# Tasa de descuento: qué % de las ventas brutas se va en descuentos
df_clean['tasa_descuento_pct'] = np.where(
    df_clean['gross_sales'] != 0,
    (df_clean['discounts'].abs() / df_clean['gross_sales'] * 100),
    np.nan
).round(2)

# Ratio costo/venta: qué tan eficiente es la operación
# Menor es mejor — por encima de 80% hay problema de costos
df_clean['ratio_cogs_ventas'] = np.where(
    df_clean['sales'] != 0,
    (df_clean['cogs'] / df_clean['sales'] * 100),
    np.nan
).round(2)

# Clasificación de rentabilidad por transacción
def clasificar_margen(margen):
    if pd.isna(margen):     return 'Sin datos'
    if margen >= 30:        return '🟢 Alta (>30%)'
    if margen >= 15:        return '🟡 Media (15-30%)'
    if margen >= 0:         return '🟠 Baja (0-15%)'
    return                         '🔴 Negativa (<0%)'

df_clean['rentabilidad_cat'] = df_clean['margen_bruto_pct'].apply(clasificar_margen)

# ── 3.3 P&L agregado (visión ejecutiva) ──────────────────────
pl_total = {
    'Ventas Brutas':    df_clean['gross_sales'].sum(),
    'Descuentos':      -df_clean['discounts'].abs().sum(),
    'Ventas Netas':     df_clean['sales'].sum(),
    'Costo de Ventas': -df_clean['cogs'].sum(),
    'Utilidad Bruta':   df_clean['profit'].sum(),
}

print("\n" + "=" * 45)
print("   ESTADO DE RESULTADOS — VISIÓN GLOBAL")
print("=" * 45)
for concepto, valor in pl_total.items():
    signo = "  " if valor >= 0 else ""
    print(f"  {concepto:<25} ${valor:>12,.0f}")
print("-" * 45)
margen_global = (pl_total['Utilidad Bruta'] / pl_total['Ventas Netas'] * 100)
print(f"  {'Margen Bruto Global':<25} {margen_global:>12.1f}%")
print("=" * 45)

print(f"\nEstadísticas de margen bruto por transacción:")
print(df_clean['margen_bruto_pct'].describe().round(2))





# ============================================================
# PASO 4: Visualizaciones financieras
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Dashboard de KPIs Financieros — Análisis de Rentabilidad',
             fontsize=15, fontweight='bold')



# ── Gráfico 1: Ventas y utilidad por mes (tendencia) ─────────
# Agrupamos por año-mes para ver la tendencia temporal
if 'date' in df_clean.columns:
    tend = df_clean.groupby(df_clean['date'].dt.to_period('M')).agg(
        ventas=('sales', 'sum'),
        utilidad=('profit', 'sum')
    ).reset_index()
    tend['date'] = tend['date'].astype(str)

    x = range(len(tend))
    axes[0,0].bar(x, tend['ventas'], color='#3498DB', alpha=0.7, label='Ventas')
    axes[0,0].bar(x, tend['utilidad'], color='#2ECC71', alpha=0.9, label='Utilidad')
    axes[0,0].set_title('Ventas vs. Utilidad Mensual', fontweight='bold')
    axes[0,0].set_xticks(x[::2])                         # cada 2 meses para no saturar
    axes[0,0].set_xticklabels(tend['date'].iloc[::2], rotation=45, fontsize=8)
    axes[0,0].set_ylabel('USD')
    axes[0,0].legend()
    # Formatear eje Y en millones
    axes[0,0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda val, _: f'${val/1e6:.1f}M'))



# ── Gráfico 2: Margen bruto por producto ─────────────────────

col_producto = 'product'   # ajusta si el nombre es diferente
if col_producto in df_clean.columns:
    margen_prod = (df_clean.groupby(col_producto)['margen_bruto_pct']
                   .mean().sort_values(ascending=False))
    colores = ['#2ECC71' if v >= 15 else '#E74C3C' for v in margen_prod.values]
    margen_prod.plot(kind='bar', ax=axes[0,1], color=colores, edgecolor='white')
    axes[0,1].set_title('Margen Bruto Promedio por Producto (%)', fontweight='bold')
    axes[0,1].set_ylabel('Margen Bruto (%)')
    axes[0,1].axhline(y=15, color='orange', linestyle='--',
                      linewidth=1.5, label='Benchmark 15%')
    axes[0,1].legend(fontsize=9)
    axes[0,1].tick_params(axis='x', rotation=30)



# ── Gráfico 3: Ventas por segmento (donut) ───────────────────
col_segmento = 'segment'
if col_segmento in df_clean.columns:
    ventas_seg = df_clean.groupby(col_segmento)['sales'].sum().sort_values(ascending=False)
    wedges, texts, autotexts = axes[0,2].pie(
        ventas_seg.values,
        labels=ventas_seg.index,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.82,
        colors=sns.color_palette("Set2", len(ventas_seg))
    )
    # Hacerlo donut
    centro = plt.Circle((0,0), 0.60, fc='white')
    axes[0,2].add_patch(centro)
    axes[0,2].set_title('Participación en Ventas por Segmento', fontweight='bold')



# ── Gráfico 4: Tasa de descuento por producto ────────────────
if col_producto in df_clean.columns:
    desc_prod = (df_clean.groupby(col_producto)['tasa_descuento_pct']
                 .mean().sort_values(ascending=False))
    desc_prod.plot(kind='barh', ax=axes[1,0], color='#E67E22', edgecolor='white')
    axes[1,0].set_title('Tasa de Descuento Promedio por Producto (%)',
                         fontweight='bold')
    axes[1,0].set_xlabel('% Descuento sobre Ventas Brutas')
    axes[1,0].axvline(x=desc_prod.mean(), color='red', linestyle='--',
                      linewidth=1.5, label=f'Promedio: {desc_prod.mean():.1f}%')
    axes[1,0].legend(fontsize=9)



# ── Gráfico 5: Rentabilidad por país ─────────────────────────
col_pais = 'country'
if col_pais in df_clean.columns:
    rent_pais = df_clean.groupby(col_pais).agg(
        utilidad=('profit', 'sum'),
        ventas=('sales', 'sum')
    )
    rent_pais['margen'] = (rent_pais['utilidad'] / rent_pais['ventas'] * 100).round(1)
    rent_pais['margen'].sort_values().plot(
        kind='barh', ax=axes[1,1],
        color=['#E74C3C' if v < 0 else '#2ECC71' for v in rent_pais['margen'].sort_values()],
        edgecolor='white'
    )
    axes[1,1].set_title('Margen Bruto por País (%)', fontweight='bold')
    axes[1,1].set_xlabel('Margen Bruto (%)')
    axes[1,1].axvline(x=0, color='black', linewidth=0.8)



# ── Gráfico 6: Distribución de márgenes (histograma) ─────────
axes[1,2].hist(df_clean['margen_bruto_pct'].dropna(),
               bins=40, color='#9B59B6', edgecolor='white', alpha=0.85)
axes[1,2].axvline(x=0,  color='red',    linestyle='--', linewidth=1.5, label='0%')
axes[1,2].axvline(x=15, color='orange', linestyle='--', linewidth=1.5, label='15% benchmark')
axes[1,2].axvline(x=df_clean['margen_bruto_pct'].mean(),
                  color='green', linestyle='-', linewidth=2,
                  label=f"Media: {df_clean['margen_bruto_pct'].mean():.1f}%")
axes[1,2].set_title('Distribución de Márgenes Brutos', fontweight='bold')
axes[1,2].set_xlabel('Margen Bruto (%)')
axes[1,2].legend(fontsize=9)

plt.tight_layout()
plt.savefig(BASE_DIR / 'eda_kpis_financieros.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico guardado como 'eda_kpis_financieros.png'")





# ============================================================
# PASO 5: Análisis trimestral — crecimiento QoQ y YoY
#
# QoQ = Quarter over Quarter (vs trimestre anterior)
# YoY = Year over Year (vs mismo trimestre del año anterior)
# ============================================================



if 'trimestre' in df_clean.columns and 'año' in df_clean.columns:

    trim = df_clean.groupby(['año', 'trimestre']).agg(
        ventas=('sales', 'sum'),
        utilidad=('profit', 'sum'),
        cogs=('cogs', 'sum')
    ).reset_index()

    trim['margen_trim'] = (trim['utilidad'] / trim['ventas'] * 100).round(2)
    trim['periodo'] = trim['año'].astype(str) + ' ' + trim['trimestre']



    # ── Crecimiento QoQ de ventas ─────────────────────────────
    # pct_change() calcula la variación % respecto a la fila anterior
    trim['ventas_qoq_pct'] = trim['ventas'].pct_change() * 100



    # ── Crecimiento YoY (comparar vs mismo Q del año anterior) ─
    trim['ventas_yoy_pct'] = trim.groupby('trimestre')['ventas'].pct_change() * 100

    print("Análisis Trimestral:")
    print(trim[['periodo', 'ventas', 'utilidad', 'margen_trim',
                'ventas_qoq_pct', 'ventas_yoy_pct']].to_string(index=False))



    # Exportar esta tabla también — va directo al README
    trim.to_csv(BASE_DIR / 'financials_trimestral.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ Tabla trimestral exportada en: {BASE_DIR / 'financials_trimestral.csv'}")






    # ============================================================
# PASO 6: Exportar todo para Power BI
# ============================================================


# Dataset principal limpio con todos los KPIs calculados
df_clean.to_csv('financials_clean.csv', index=False, encoding='utf-8-sig')


# Resumen por producto
resumen_producto = df_clean.groupby('product').agg(
    ventas_total=('sales', 'sum'),
    utilidad_total=('profit', 'sum'),
    cogs_total=('cogs', 'sum'),
    descuentos_total=('discounts', lambda x: x.abs().sum()),
    margen_promedio=('margen_bruto_pct', 'mean'),
    tasa_descuento_prom=('tasa_descuento_pct', 'mean'),
    transacciones=('sales', 'count')
).round(2).reset_index()
resumen_producto.to_csv('financials_por_producto.csv', index=False, encoding='utf-8-sig')



# Resumen por segmento y país
resumen_seg_pais = df_clean.groupby(['segment', 'country']).agg(
    ventas=('sales', 'sum'),
    utilidad=('profit', 'sum'),
    margen=('margen_bruto_pct', 'mean')
).round(2).reset_index()
resumen_seg_pais.to_csv('financials_seg_pais.csv', index=False, encoding='utf-8-sig')



# KPI cards globales
ventas_netas   = df_clean['sales'].sum()
utilidad_total = df_clean['profit'].sum()
margen_global  = utilidad_total / ventas_netas * 100

kpis_fin = pd.DataFrame({
    'KPI': ['Ventas Brutas (USD)', 'Descuentos (USD)', 'Ventas Netas (USD)',
            'Costo de Ventas (USD)', 'Utilidad Bruta (USD)',
            'Margen Bruto (%)', 'Tasa de Descuento Promedio (%)'],
    'Valor': [
        round(df_clean['gross_sales'].sum(), 0),
        round(df_clean['discounts'].abs().sum(), 0),
        round(ventas_netas, 0),
        round(df_clean['cogs'].sum(), 0),
        round(utilidad_total, 0),
        round(margen_global, 2),
        round(df_clean['tasa_descuento_pct'].mean(), 2)
    ]
})
kpis_fin.to_csv('financials_kpis_summary.csv', index=False, encoding='utf-8-sig')

print("✅ Archivos exportados:")
print("   → financials_clean.csv")
print("   → financials_por_producto.csv")
print("   → financials_seg_pais.csv")
print("   → financials_trimestral.csv")
print("   → financials_kpis_summary.csv")