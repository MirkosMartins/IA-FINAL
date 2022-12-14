import pandas as pd
import streamlit as st #importando o streamlit

st.title('Floresta Randomica para AVC')

dados = pd.read_csv('stroke.csv',sep=';')
dados = dados.drop(columns=['id'])#apaga a coluna id
dados.head()
nomes_colunas = dados.columns.to_list()#coloca os nomes das colunas em uma lista
nomes_colunas = nomes_colunas[:len(nomes_colunas)-1]#retiro o 'stroke'
#separando as features das classes
features = dados[nomes_colunas]
classes = dados['stroke']

#componente para converter valores object para inteiro/float
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
features['gender'] = encoder.fit_transform(pd.DataFrame(features['gender']))
features['ever_married'] = encoder.fit_transform(pd.DataFrame(features['ever_married']))
features['work_type'] = encoder.fit_transform(pd.DataFrame(features['work_type']))
features['Residence_type'] = encoder.fit_transform(pd.DataFrame(features['Residence_type']))
features['smoking_status'] = encoder.fit_transform(pd.DataFrame(features['smoking_status']))
features = features.fillna(0)


#dividindo os dados entre treino e teste
from sklearn.model_selection import train_test_split

features_treino,features_teste,classes_treino,classes_teste = train_test_split(features,
                                                                               classes,
                                                                               test_size=0.4,
                                                                               random_state=2)

from sklearn.ensemble import RandomForestClassifier #importa o codigo para gerar florestas randomicas
#criando a floresta
floresta = RandomForestClassifier(n_estimators=1000) #constroi a floresta
#treinar a floresta
floresta.fit(features_treino,classes_treino)

#testar quanto a floresta acerta
predicoes = floresta.predict(features_teste)

st.write('Floresta construida')

f1 = st.number_input('Digite o genero (0=feminino, 1=masculino)',min_value=0,max_value=1)
f2 = st.number_input('Digite a idade',min_value=1, max_value=200)
f3 = st.number_input('Digite se eh hipertenso: (0=nao, 1=sim)',min_value=0,max_value=1)
f4 = st.number_input('Digite a f4')
f5 = st.number_input('Digite a f5')
f6 = st.number_input('Digite a f6')
f7 = st.number_input('Digite a f7')
f8 = st.number_input('Digite a f8')
f9 = st.number_input('Digite a f9')
f10 = st.number_input('Digite a f10')

if st.button('O sujeito tera AVC?'):
  resposta = floresta.predict([[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10]])#fara a predicao
  if resposta == 0:
    st.write('O sujeito nao tera AVC!')
  else:
    st.write('O sujeito tera AVC!')

