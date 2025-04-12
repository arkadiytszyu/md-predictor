import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title("📊 Оптимизация MD в зависимости от приоритета")

# === Попытка загрузки моделей ===
try:
    model_c_output = joblib.load("model_c_output.pkl")
    model_cp2o5 = joblib.load("model_cp2o5.pkl")
    model_cmgo = joblib.load("model_cmgo.pkl")
except Exception as e:
    st.error(f"❌ Не удалось загрузить модели: {e}")
    st.stop()

# === Функция recommend_md (встроена прямо здесь) ===
def recommend_md(fraction_str, f_p2o5, f_mgo, feed, prioritet,
                 model_output, model_cp2o5, model_cmgo):
    fraction_map = {'20-40': 0, '40-80': 1, '80-130': 2}
    if fraction_str not in fraction_map:
        raise ValueError(f"Неизвестная фракция: {fraction_str}")

    fraction = fraction_map[fraction_str]
    md_range = range(15, 56)
    results = []

    for i, md in enumerate(md_range, start=1):
        row = pd.DataFrame([{
            'Fraction': fraction,
            'MD': md,
            'F_P2O5_%': f_p2o5,
            'F_MgO_%': f_mgo
        }])

        row['Fraction_str'] = fraction_str
        c_output = model_output.predict(row[['Fraction_str', 'MD', 'F_P2O5_%', 'F_MgO_%']])[0]
        row['C_output_pred'] = c_output

        row_extended = row[['Fraction', 'MD', 'F_P2O5_%', 'F_MgO_%', 'C_output_pred']]
        c_p2o5 = model_cp2o5.predict(row_extended)[0]

        row_extended['C_P2O5_%_pred'] = c_p2o5
        c_mgo = model_cmgo.predict(row_extended)[0]

        concentrate = c_output * feed
        tails = feed - concentrate

        c_p2o5_t = c_p2o5 * concentrate / 100
        c_mgo_t = c_mgo * concentrate / 100
        f_p2o5_t = f_p2o5 * feed / 100
        f_mgo_t = f_mgo * feed / 100
        t_p2o5_t = f_p2o5_t - c_p2o5_t
        t_mgo_t = f_mgo_t - c_mgo_t
        t_p2o5 = t_p2o5_t / tails * 100 if tails != 0 else 0
        t_mgo = t_mgo_t / tails * 100 if tails != 0 else 0
        extraction = c_p2o5_t / f_p2o5_t * 100 if f_p2o5_t != 0 else 0

        results.append({
            '№': i,
            'MD': md,
            'Fraction': fraction_str,
            'Feed': feed,
            'C_output': c_output,
            'C_P2O5_%': c_p2o5,
            'C_MgO_%': c_mgo,
            'Concentrate': concentrate,
            'Tails': tails,
            'C_P2O5_t': c_p2o5_t,
            'C_MgO_t': c_mgo_t,
            'T_P2O5_%': t_p2o5,
            'T_MgO_%': t_mgo,
            'T_P2O5_t': t_p2o5_t,
            'T_MgO_t': t_mgo_t,
            'F_P2O5_%': f_p2o5,
            'F_MgO_%': f_mgo,
            'F_P2O5_t': f_p2o5_t,
            'F_MgO_t': f_mgo_t,
            'Extraction': extraction
        })

    df_results = pd.DataFrame(results)

    if prioritet == 'P2O5':
        df_sorted = df_results.sort_values(by=['C_P2O5_%', 'C_MgO_%', 'C_output'], ascending=[False, True, False])
    elif prioritet == 'MgO':
        df_sorted = df_results.sort_values(by=['C_MgO_%', 'C_P2O5_%', 'C_output'], ascending=[True, False, False])
    elif prioritet == 'Output':
        df_sorted = df_results.sort_values(by=['C_output', 'C_P2O5_%', 'C_MgO_%'], ascending=[False, False, True])
    else:
        raise ValueError("Prioritet должен быть 'P2O5', 'MgO' или 'Output'")

    top5_df = df_sorted.head(5).reset_index(drop=True)
    full_df_sorted = df_results.sort_values(by='MD').reset_index(drop=True)

    return top5_df, full_df_sorted

# === Streamlit-интерфейс ===
fraction_str = st.selectbox("Выберите фракцию:", ['20-40', '40-80', '80-130'])
f_p2o5 = st.number_input("F_P2O5_%", value=24.5, step=0.1)
f_mgo = st.number_input("F_MgO_%", value=4.1, step=0.1)
feed = st.number_input("Feed (тонн)", value=250, step=1)
prioritet = st.radio("Приоритет оптимизации:", ['P2O5', 'MgO', 'Output'])

if st.button("🔍 Рассчитать"):
    try:
        top5_df, full_df = recommend_md(
            fraction_str=fraction_str,
            f_p2o5=f_p2o5,
            f_mgo=f_mgo,
            feed=feed,
            prioritet=prioritet,
            model_output=model_c_output,
            model_cp2o5=model_cp2o5,
            model_cmgo=model_cmgo
        )

        st.subheader("🔝 Топ-5 рекомендаций")
        st.dataframe(top5_df, use_container_width=True)

        st.subheader("📋 Полная таблица (MD 15–55)")
        st.dataframe(full_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Ошибка: {e}")

