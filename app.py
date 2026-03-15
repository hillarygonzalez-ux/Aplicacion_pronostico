from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# FUNCION PROMEDIO MOVIL
def promedio_movil(serie, n):  
    pronostico = serie.rolling(window=n).mean()
    return pronostico

# FUNCION MEDIDAS DE ERROR
def medidas_error(real, pron):
    df = pd.DataFrame({
        "real": real,
        "pron": pron
    })
    df = df.dropna()
    mae = (abs(df["real"] - df["pron"])).mean()
    mse = ((df["real"] - df["pron"])**2).mean()
    mape = (abs((df["real"] - df["pron"]) / df["real"])).mean()*100
    return round(mae,2), round(mse,2), round(mape,2)

# PAGINA PRINCIPAL
@app.route("/", methods=["GET","POST"])
def index():

    resultados = {}

    if request.method == "POST":

        # valor de N
        n = int(request.form["n"])

        # archivo cargado
        file = request.files["file"]

        # leer CSV
        df = pd.read_csv(file)

        productos = df.columns[1:]  # asumiendo que la primera columna es fecha o ID

        for p in productos:

            pron = promedio_movil(df[p], n)

            mae, mse, mape = medidas_error(df[p], pron)

            resultados[p] = {
                "mae": mae,
                "mse": mse,
                "mape": mape,
                "pronostico": round(pron.iloc[-1],2)
            }

            # GRAFICO
            plt.figure()
            plt.plot(df[p], label="Real")
            plt.plot(pron, label="Promedio Movil")
            plt.title(f"Producto {p}")
            plt.legend()
            ruta = f"static/{p}.png"
            plt.savefig(ruta)
            plt.close()

    return render_template("pronostico.html", resultados=resultados)

# EJECUTAR APP
if __name__ == "__main__":
    app.run(debug=True)