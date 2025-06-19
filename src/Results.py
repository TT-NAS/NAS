import os
import json
import pandas as pd

path = './output'

# Read
df_data = []
idx = 1
for carpeta in sorted(os.listdir(path)):
    carpeta_path = os.path.join(path, carpeta)
    if os.path.isdir(carpeta_path):
        json_path = os.path.join(carpeta_path, 'model.json')
        if os.path.isfile(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)

            search_time     = data.get('search_time')
            stop_reason     = data.get('stop_reason')
            stop_gen        = data.get('stop_gen')
            predicted_iou   = data.get('predicted_iou')
            training_time   = data.get('training_time')
            last_epoch      = data.get('last_epoch')
            training_iou    = data.get('training_iou')
            validation_iou  = data.get('validation_iou')

            diff_p_t = abs(predicted_iou - training_iou)
            diff_p_v = abs(predicted_iou - validation_iou)
            

            # Add data
            df_data.append({
            'N°': idx,
            #'T. búsqueda': search_time,
            #'Criterio': stop_reason,
            'Gen. final': stop_gen,
            'T. entrenamiento': training_time,
            #'Época fin': last_epoch,
            'IOU train': training_iou,
            'IOU pred.': predicted_iou,
            'IOU valid.': validation_iou,
            'IOU (p-t)': diff_p_t,
            'IOU (p-v)': diff_p_v
        })
            idx += 1

# DataFrame
df = pd.DataFrame(df_data)

# Save latex
tabla_latex = df.to_latex(
    index=False,
    float_format="%.3f",
    caption="Resultados de los modelos encontrados",
    label="tab:resultados_busqueda",
    longtable=False
)

output_path = 'tabla_resultados.txt'
with open(output_path, 'w') as f:
    f.write(tabla_latex)
