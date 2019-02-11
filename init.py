from SplitDataSet import *
from seleccion import *


raw_data = datasets.load_wine()
data_train, data_test, label_train, label_test = SplitDataSet(raw_data.data, raw_data.target)
print(len(data_train),' samples in training data\n',
      len(data_test),' samples in test data\n', )
print("---------------------------------------------------")



namel_labels = ['C1','C2','C3']
(df_results, dataSet_Params) = batch_classify(data_train, label_train, data_test, label_test,namel_labels)
df_results_sort=df_results.sort_values(by='MCC', ascending=False)

print(df_results_sort)
df_results_sort.to_csv('save_Files/Resultados_by_Clasificador.csv')

import json
with open('save_Files/Better_Params.json', 'w') as outfile:
    json.dump(dataSet_Params, outfile)