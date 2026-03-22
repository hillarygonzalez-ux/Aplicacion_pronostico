from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from prophet import Prophet
import os

app = Flask(__name__)

# Validar archivo
def archivo_valido(filename):
    return filename.endswith('.csv')

# Promedio móvil
def promedio_movil(serie, n):
    return serie.rolling(window=n).mean()

# Medidas de error
def medidas_error(real, pron):
    df = pd.DataFrame({"real": real, "pron": pron})
    df = df.dropna()

    mae = (abs(df["real"] - df["pron"])).mean()
    mse = ((df["real"] - df["pron"])**2).mean()
    mape = (abs((df["real"] - df["pron"]) / df["real"])).mean()*100

    return round(mae,2), round(mse,2), round(mape,2)

@app.route("/", methods=["GET","POST"])
def index():

    resultados = {}
    tabla_modelos = {}
    error = None

    if request.method == "POST":

        file = request.files["file"]

        if not archivo_valido(file.filename):
            return render_template("pronostico.html", error="Solo archivos CSV")

        df = pd.read_csv(file)

        if "Fecha" not in df.columns:
            return render_template("pronostico.html", error="Debe tener columna Fecha")

        df["Fecha"] = pd.to_datetime(df["Fecha"])

        n = int(request.form["n"])
        metodo = request.form["metodo"]
        fecha_fin = request.form["fecha_fin"]

        productos = df.columns[1:]

        for p in productos:

            serie = df[p]

            # ---------------- PROMEDIO MOVIL ----------------
            pm = promedio_movil(serie, n)
            mae_pm, mse_pm, mape_pm = medidas_error(serie, pm)

            # ---------------- SES ----------------
            modelo_ses = SimpleExpSmoothing(serie).fit()
            ses = modelo_ses.fittedvalues
            mae_ses, mse_ses, mape_ses = medidas_error(serie, ses)

            # ---------------- PROPHET ----------------
            df_prophet = df[["Fecha", p]].rename(columns={"Fecha":"ds", p:"y"})
            modelo_p = Prophet()
            modelo_p.fit(df_prophet)

            # calcular periodos futuros
            dias = (pd.to_datetime(fecha_fin) - df["Fecha"].max()).days
            future = modelo_p.make_future_dataframe(periods=dias)
            forecast = modelo_p.predict(future)

            mae_p, mse_p, mape_p = medidas_error(df_prophet["y"], forecast["yhat"][:len(df)])

            # ---------------- SELECCION DE METODO ----------------
            if metodo == "promedio":
                pron = pm
            elif metodo == "ses":
                pron = ses
            else:
                pron = forecast["yhat"]

            # ---------------- GRAFICA ----------------
            plt.figure()
            plt.plot(df["Fecha"], serie, label="Real")

            if metodo == "prophet":
                plt.plot(forecast["ds"], forecast["yhat"], label="Prophet")
            else:
                plt.plot(df["Fecha"], pron, label=metodo)

            plt.legend()
            ruta = f"static/{p}.png"
            plt.savefig(ruta)
            plt.close()

            # resultados principales
            resultados[p] = {
                "mae": mae_pm,
                "mse": mse_pm,
                "mape": mape_pm
            }

            # tabla comparativa
            tabla_modelos[p] = {
                "Promedio": mape_pm,
                "SES": mape_ses,
                "Prophet": mape_p
            }

    return render_template("pronostico.html", resultados=resultados, tabla=tabla_modelos, error=error)

if __name__ == "__main__":
    app.run(debug=True)