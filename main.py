from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import shap

app = FastAPI(title="API Alerta Temprana - Prevención del Abandono Escolar")

# 1. Cargar el modelo entrenado y la lista de columnas
modelo = joblib.load('modelo_desercion_xgboost.pkl')
columnas = joblib.load('columnas_modelo.pkl')
explainer = shap.TreeExplainer(modelo)

# 2. Definir la estructura del cuerpo del request (JSON)
class EstudianteData(BaseModel):
    promedio_secundaria_o_prepa: float
    promedio_acumulado_actual: float
    promedio_parcial_actual: float
    materias_reprobadas_historico: int
    materias_reprobadas_actual: int
    materias_inscritas: int
    creditos_acumulados_porcentaje: float
    porcentaje_asistencia: float
    interacciones_lms_mes: int
    dias_inactividad_plataforma: int
    retrasos_pago_acumulados: int
    dias_promedio_mora: int
    es_becado: int
    porcentaje_beca: int
    ingreso_familiar_estimado: float
    trabaja_actualmente: int
    edad_ingreso: int
    tiempo_traslado_minutos: int
    cambio_residencia: int
    nivel_estudios_padres: int

@app.get("/")
def inicio():
    return {"status": "API activa", "modelo": "XGBoost Abandono Escolar"}

@app.post("/predict")
def predecir_riesgo(datos: EstudianteData):
    try:
        # Convertir datos a DataFrame ordenado
        df_input = pd.DataFrame([datos.dict()])[columnas]
        
        # Predicción de probabilidad
        probabilidad = float(modelo.predict_proba(df_input)[0][1] * 100)
        
        # Determinar nivel de riesgo
        if probabilidad >= 70:
            nivel_riesgo = "Alto"
        elif probabilidad >= 40:
            nivel_riesgo = "Medio"
        else:
            nivel_riesgo = "Bajo"
            
        # Explicabilidad con SHAP
        shap_values = explainer(df_input)
        impactos = pd.DataFrame({
            'variable': columnas,
            'impacto': shap_values.values[0]
        }).sort_values(by='impacto', ascending=False)
        
        # Seleccionar las 3 características que más incrementan el riesgo
        factores_clave = impactos[impactos['impacto'] > 0]['variable'].head(3).tolist()
        if not factores_clave:
            factores_clave = ["Sin factores de riesgo significativos"]

        return {
            "probabilidad_abandono": round(probabilidad, 2),
            "nivel_riesgo": nivel_riesgo,
            "factores_detectados": factores_clave
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))