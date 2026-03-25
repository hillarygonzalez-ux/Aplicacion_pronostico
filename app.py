from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from prophet import Prophet
import os

app = Flask(__name__)

# ─────────────────────────────────────────────
# VALIDAR CSV
# ─────────────────────────────────────────────
def archivo_valido(filename):
    return filename.endswith('.csv')


# ─────────────────────────────────────────────
# PROMEDIO MOVIL  —  model / fit / predict
# ─────────────────────────────────────────────
def modelo_promedio_movil(serie, fechas, fecha_fin, n):
    """
    Paso 1 – model  : configuración de la ventana n
    Paso 2 – fit    : cálculo del rolling sobre el histórico
    Paso 3 – predict: extensión hacia fecha_fin repitiendo el último valor
    """
    # model
    ventana = n

    # fit  (valores históricos)
    pm_historico = serie.rolling(window=ventana).mean()

    # predict  (proyección futura)
    ultimo_valor = pm_historico.dropna().iloc[-1]
    fechas_futuras = pd.date_range(
        start=fechas.max() + pd.Timedelta(days=1),
        end=fecha_fin,
        freq='D'
    )
    pm_futuro = pd.Series([ultimo_valor] * len(fechas_futuras), index=fechas_futuras)

    fechas_completas = list(fechas) + list(fechas_futuras)
    valores_completos = list(pm_historico) + list(pm_futuro)

    return pm_historico, fechas_completas, valores_completos


# ─────────────────────────────────────────────
# SUAVIZACIÓN EXPONENCIAL SIMPLE  —  model / fit / predict
# ─────────────────────────────────────────────
def modelo_ses(serie, fechas, fecha_fin):
    """
    Paso 1 – model  : SimpleExpSmoothing(serie)
    Paso 2 – fit    : .fit()
    Paso 3 – predict: .forecast(periodos_futuros)
    """
    # model
    modelo = SimpleExpSmoothing(serie)

    # fit
    fit = modelo.fit(optimized=True)

    # predict  (proyección futura)
    fechas_futuras = pd.date_range(
        start=fechas.max() + pd.Timedelta(days=1),
        end=fecha_fin,
        freq='D'
    )
    periodos_futuros = max(len(fechas_futuras), 1)
    ses_futuro = fit.forecast(periodos_futuros)

    ses_historico = fit.fittedvalues

    fechas_completas = list(fechas) + list(fechas_futuras)
    valores_completos = list(ses_historico) + list(ses_futuro)

    return ses_historico, fechas_completas, valores_completos


# ─────────────────────────────────────────────
# PROPHET  —  model / fit / predict
# ─────────────────────────────────────────────
def modelo_prophet(df_producto, fecha_fin):
    """
    Paso 1 – model  : Prophet()
    Paso 2 – fit    : .fit(df_prophet)
    Paso 3 – predict: .predict(future)
    """
    df_prophet = df_producto.rename(columns={df_producto.columns[0]: "ds",
                                             df_producto.columns[1]: "y"})
    df_prophet = df_prophet.dropna()

    # model
    modelo = Prophet(daily_seasonality=False, weekly_seasonality=False)

    # fit
    modelo.fit(df_prophet)

    # predict
    dias = (pd.to_datetime(fecha_fin) - df_prophet["ds"].max()).days
    dias = max(dias, 1)
    future = modelo.make_future_dataframe(periods=dias)
    forecast = modelo.predict(future)

    return forecast


# ─────────────────────────────────────────────
# MEDIDAS DE ERROR
# ─────────────────────────────────────────────
def medidas_error(real, pron):
    df = pd.DataFrame({"real": real.values, "pron": list(pron)[:len(real)]}).dropna()
    df = df[df["real"] != 0]

    if len(df) == 0:
        return 0, 0, 0

    mae  = abs(df["real"] - df["pron"]).mean()
    mse  = ((df["real"] - df["pron"]) ** 2).mean()
    mape = (abs((df["real"] - df["pron"]) / df["real"])).mean() * 100

    return round(mae, 2), round(mse, 2), round(mape, 2)


# ─────────────────────────────────────────────
# RUTA PRINCIPAL
# ─────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():

    resultados = {}
    tabla      = {}
    error      = None

    if request.method == "POST":

        file = request.files.get("file")

        if not file or not archivo_valido(file.filename):
            return render_template("pronostico.html", error="Por favor sube un archivo CSV válido.")

        try:
            df = pd.read_csv(file)
        except Exception as e:
            return render_template("pronostico.html", error=f"Error leyendo el archivo: {e}")

        if "Fecha" not in df.columns:
            return render_template("pronostico.html", error="El CSV debe tener una columna llamada 'Fecha'.")

        df["Fecha"] = pd.to_datetime(df["Fecha"])

        try:
            n        = int(request.form["n"])
            metodo   = request.form["metodo"]
            fecha_fin = pd.to_datetime(request.form["fecha_fin"])
        except Exception:
            return render_template("pronostico.html", error="Verifica que todos los campos del formulario estén completos.")

        if fecha_fin <= df["Fecha"].max():
            return render_template("pronostico.html",
                                   error="La fecha de proyección debe ser posterior a la última fecha del CSV.")

        # Excluir columna Fecha de forma robusta
        productos = [col for col in df.columns if col != "Fecha"]

        os.makedirs("static", exist_ok=True)

        for p in productos:

            serie  = df[p].reset_index(drop=True)
            fechas = df["Fecha"].reset_index(drop=True)

            # ── Promedio Móvil ──────────────────────────────────
            pm_hist, pm_fechas_c, pm_vals_c = modelo_promedio_movil(serie, fechas, fecha_fin, n)
            mae_pm, mse_pm, mape_pm = medidas_error(serie, pm_hist)

            # ── SES ────────────────────────────────────────────
            ses_hist, ses_fechas_c, ses_vals_c = modelo_ses(serie, fechas, fecha_fin)
            mae_ses, mse_ses, mape_ses = medidas_error(serie, ses_hist)

            # ── Prophet ────────────────────────────────────────
            try:
                forecast = modelo_prophet(df[["Fecha", p]].copy(), fecha_fin)
                mae_p, mse_p, mape_p = medidas_error(serie, forecast["yhat"][:len(serie)])
                prophet_ok = True
            except Exception as e:
                print(f"Error Prophet ({p}): {e}")
                forecast   = None
                mae_p, mse_p, mape_p = 0, 0, 0
                prophet_ok = False

            # ── Errores según método seleccionado ──────────────
            if metodo == "promedio":
                resultados[p] = {"mae": mae_pm,  "mse": mse_pm,  "mape": mape_pm}
            elif metodo == "ses":
                resultados[p] = {"mae": mae_ses, "mse": mse_ses, "mape": mape_ses}
            else:
                resultados[p] = {"mae": mae_p,   "mse": mse_p,   "mape": mape_p}

            # ── Gráfica ────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(8, 4))

            # Serie real
            ax.plot(fechas, serie, marker='o', linewidth=2,
                    color='#2c3e50', label="Real", zorder=3)

            # Línea vertical que separa histórico de proyección
            ax.axvline(x=fechas.max(), color='gray', linestyle='--',
                       linewidth=1, alpha=0.7, label="Inicio proyección")

            # Proyección del método seleccionado
            if metodo == "promedio":
                ax.plot(pm_fechas_c, pm_vals_c, linestyle='--', linewidth=2,
                        color='#e67e22', marker=None, label=f"Prom. Móvil (n={n})")

            elif metodo == "ses":
                ax.plot(ses_fechas_c, ses_vals_c, linestyle='--', linewidth=2,
                        color='#27ae60', marker=None, label="SES")

            else:
                if prophet_ok and forecast is not None:
                    ax.plot(forecast["ds"], forecast["yhat"],
                            linestyle='--', linewidth=2,
                            color='#8e44ad', label="Prophet")
                    ax.fill_between(forecast["ds"],
                                    forecast["yhat_lower"],
                                    forecast["yhat_upper"],
                                    alpha=0.15, color='#8e44ad', label="Intervalo confianza")

            ax.set_title(f"Pronóstico — {p}", fontsize=13, fontweight='bold')
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Ventas")
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(fontsize=9)

            plt.tight_layout()
            ruta = f"static/{p}.png"
            plt.savefig(ruta, dpi=120)
            plt.close()

            # ── Tabla comparativa ──────────────────────────────
            tabla[p] = {
                "Promedio": round(mape_pm, 2),
                "SES":      round(mape_ses, 2),
                "Prophet":  round(mape_p, 2),
            }

    return render_template("pronostico.html",
                           resultados=resultados,
                           tabla=tabla,
                           error=error)


if __name__ == "__main__":
    app.run(debug=True)