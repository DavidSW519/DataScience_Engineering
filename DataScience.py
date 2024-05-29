# %%
import pandas as pd
import numpy as np

# %%
class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def preprocess(self,option = 'all'):
        if option =='all':
            self.data = self.data.dropna()
            self.data = self.data.drop_duplicates()
            self.data = self.data.reset_index(drop=True)
        elif option == 'drop_na':
            self.data = self.data.dropna()
            self.data = self.data.reset_index(drop=True)
        elif option == 'drop_duplicates':
            self.data = self.data.drop_duplicates()
            self.data = self.data.reset_index(drop=True)
        return self.data

# %%
class Rolling:
    def __init__(self, data,vobs,vdes,col_periodo):
        self.data = data
        self.col_periodo = col_periodo
        self.vobs = vobs
        self.vdes = vdes

        # Crear Catalogo de Ventanas de Observación
        cat_ventanas = data[[col_periodo]].drop_duplicates().sort_values([col_periodo]).reset_index(drop=True).reset_index()
        cat_ventanas.rename(columns = {'index':'Ventana'},inplace=True)
        cat_ventanas['Ventana'] = cat_ventanas['Ventana'] + 1
        self.cat_ventanas = cat_ventanas

        # Unir el cátalogo con la data
        self.data_v = data.merge(cat_ventanas,on=col_periodo,how='left')

        # Generamos la ventana inicial y venta final
        self.v_i = self.data_v['Ventana'].min() + vobs
        self.v_f = self.data_v['Ventana'].max() - vdes +1 

        minimo = self.data_v['Ventana'].min()
        maximo = self.data_v['Ventana'].max()

        print(f'Periodos: {self.v_i}-{self.v_f}')
        print(f'Ventanas Disponibles: {minimo}-{maximo}')

    def calculate_X(self,var,um = [],sufijo='x_'):
        data = self.data_v
        v_i = self.v_i
        v_f = self.v_f
        vobs = self.vobs

        X = pd.DataFrame()
        # De la ventana inicial a la final hacemos el bucle que haga las agrupaciones necesariias
        for v in range(v_i,v_f+1):
            X_i = data[data['Ventana'] < v ].copy() #Periodo menor a la ventana a predecir
            ventanas_disp = list(X_i[['Ventana']].sort_values('Ventana',ascending=False)['Ventana'].unique())
            ventanas_disp = ventanas_disp[:vobs]
            X_i = X_i[X_i['Ventana'].isin(ventanas_disp)]  #Nos quedaremos solo con la información de las ventanas a predecir

            funcs = [np.min,np.mean,np.median,np.max,np.std]
            nombres = ['min','media','mediana','max','desv']
            expr = {f'{v}_{n}':(v,f) for v in var for f,n in zip(funcs,nombres)} #funciones de agregación por cada una de las variables predictoras

            if len(um) == 0:  # Si no hay um, entonces se agrupa por el total
                X_i['Grupo'] = 1
                X_i = X_i.groupby('Grupo').agg(**expr).reset_index()
                X_i.drop(columns='Grupo',inplace=True)
            else:  #En otro caso se agrupa por la unidad muestral
                X_i = X_i.groupby(um).agg(**expr).reset_index()

            X_i.insert(0,'Ventana',v) #Insertamos la ventana correspondiente
            X = pd.concat([X,X_i],axis=0) # Agrupamos la info de todas las ventanas

        varc = X.columns.tolist() 
        varc = [x for x in varc if x not in ['Ventana']+um]
        varx = [f'{sufijo}_{v}' for v in varc]
        self.features = varx
        X.rename(columns=dict(zip(varc,varx)),inplace=True)  #Renombramos los variaboles predictoras para que tenga el sufijo 'x_' o el que se especifico

        return X

    def calculate_y(self,tgt,um = [],func='mean'):
        data = self.data_v
        v_i = self.v_i
        v_f = self.v_f
        vdes = self.vdes

        y = pd.DataFrame()

        for v in range(v_i,v_f+1):
            y_i = data[data['Ventana'] >= v ].copy()
            ventanas_disp = list(y_i[['Ventana']].sort_values('Ventana',ascending=True)['Ventana'].unique())
            ventanas_disp = ventanas_disp[:vdes]
            y_i = y_i[y_i['Ventana'].isin(ventanas_disp)]

            
            expr = {f'{v}_{n}':(v,f) for v in [tgt] for f,n in zip([func],[tgt])}

            if len(um) == 0:
                y_i['Grupo'] = 1
                y_i = y_i.groupby('Grupo').agg(**expr).reset_index()
                y_i.drop(columns='Grupo',inplace=True)
            else:
                y_i = y_i.groupby(um).agg(**expr).reset_index()
            y_i.rename(columns={f'{tgt}_{tgt}':tgt},inplace=True)

            y_i.insert(0,'Ventana',v)
            y = pd.concat([y,y_i],axis=0)
            
        return y
    
    def calculate_TAD(self, var, tgt, um, func_y='mean'):
        X = self.calculate_X(var, um)
        y = self.calculate_y(tgt, um, func_y)
        tad = X.merge(y, on=['Ventana'] + um, how='left')
        return tad


