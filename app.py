
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import *
from sklearn.neighbors import NearestNeighbors
import streamlit as st
from statistics import stdev, mean
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score




# ------------------------------------------------------------------------- COMMENT DONNER UN POIDS DIFFÉRENT AUX FEATURES ? ENLEVER DES FEATURES? ??




st.set_page_config(page_title="Cordages de tennis", page_icon=":tennis", layout="wide")



@st.cache_data
def importdata():
    url= 'https://raw.githubusercontent.com/cmvs3/tennis/main/bd_poly51fast.xlsx'
    
    df = pd.read_excel(url)

    df.info()
    df.isnull().sum()

    df = df.drop(['Ave Perp. Force (lbs)'], axis = 1)
    df = df.dropna()



    # report = ProfileReport(df, title='Original Data')
    # report.to_file("profiling_report.html")

    df = df[~df["Stretch at 51 lbs (%)"].isin([88.2])]

    df = df[~df["Energy Return (%)"].isin([35.5])]

    df.index = pd.RangeIndex(len(df.index))
    
    return df


# def modelpredict(dfpred = df):
    
#     return pred

def color_survived(val):
    color = 'aqua' if val=="0" else 'green' if val=="2" else 'yellow'
    return f'background-color: {color}'

def color_font(val):
    color = 'red' if val==1 else 'green' if val==2 else "blue" if val==3 else "yellow" if val==4 else "purple" if val==5 else 'blue'
    return color


if __name__ == "__main__":
    
    df = importdata()
    

    st.title(" :tennis: Recommandations de cordages de tennis ")
    
    tab1, tab2, tab3 = st.tabs(["Recommandations", "Segmentation", "Données"])
    
    with tab1:
    

        st.sidebar.header("👉 Faire une sélection")
        cordage = st.sidebar.selectbox(label="Cordage:", options=df["STRING"].unique() )
        NbResultats = st.sidebar.selectbox(label="Nombre de cordages similaires:", options=[1,2,3])
        NbResultats += 1
        
        st.sidebar.text("")
        st.sidebar.text("")
        st.sidebar.text("")
        st.sidebar.text("")
        st.sidebar.markdown("**Données sources**")
        st.sidebar.markdown("Type de cordage : Polyester")
        st.sidebar.markdown("Tension de référence : 51 lbs")
        st.sidebar.markdown("Vitesse de référence : Rapide")
        st.sidebar.text("")
        st.sidebar.text("")
       

        
        st.sidebar.link_button("Tennis Warehouse University", "https://twu.tennis-warehouse.com/learning_center/index.php")
    


        
        # fig = px.line(df, title="Ventes par trimestre" )
        # fig.update_layout(title_x=0.4)
        # st.plotly_chart(fig, use_container_width=True, height = 200) 
        
        
        
    
        
        # KMEANS TOUTES COLONNES AVEC NORMALISATION -----------
        scale = StandardScaler()
        df.iloc[:,4:] = scale.fit_transform(df.iloc[:,4:])

        # KNN AVEC NORMALISATION --------------------------------------------------
            
        dataknn = df.iloc[:,4:]


        neigh = NearestNeighbors(n_neighbors=NbResultats)
        neigh.fit(dataknn)
        
        
        LineID = np.int64(df.loc[df['STRING'] == cordage].index)

            
        val1 = dataknn.loc[int(LineID)][0]
        val2 = dataknn.loc[int(LineID)][1]
        val3 = dataknn.loc[int(LineID)][2]
        val4 = dataknn.loc[int(LineID)][3]
        val5 = dataknn.loc[int(LineID)][4]
        val6 = dataknn.loc[int(LineID)][5]
        val7 = dataknn.loc[int(LineID)][6]
        val8 = dataknn.loc[int(LineID)][7]
        val9 = dataknn.loc[int(LineID)][8]
        val10 = dataknn.loc[int(LineID)][9]
        val11 = dataknn.loc[int(LineID)][10]
        val12 = dataknn.loc[int(LineID)][11]
        val13 = dataknn.loc[int(LineID)][12]
        val14 = dataknn.loc[int(LineID)][13]
        val15 = dataknn.loc[int(LineID)][14]
        val16 = dataknn.loc[int(LineID)][15]
        val17 = dataknn.loc[int(LineID)][16]


        Resultat = np.array(neigh.kneighbors([[val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13, val14, val15, val16, val17]]))
        
        df_resultat = pd.DataFrame(Resultat[1])
        
        
        dataknn = scale.inverse_transform(dataknn)

        np.set_printoptions(suppress=True)
        
        df.iloc[:,4:] = scale.inverse_transform(df.iloc[:,4:])
        
        df = df.drop('Ref. Ten. (lbs)', axis=1)
        df = df.drop('Swing Speed', axis=1)
        df = df.drop('Material', axis=1)

        
       
        
      
        
        st.subheader("  👉 VOTRE SÉLECTION")
        st.dataframe(df.loc[[int(df_resultat[0])]],hide_index=True)
        
    
       
            
        st.subheader(" 🏆 VOS RECOMMANDATIONS")
        for x in range(1,NbResultats):
        
            st.dataframe(df.loc[[int(df_resultat[x])]].style.set_properties(**{"color": color_font(x)}), hide_index=True)
            
            # STOCKER JUSQUA 10 VALEURS DANS UNE VAR
            if x==1:    
            
                rec1var8 = df.loc[int(df_resultat[x])][8]
                rec1var13 = df.loc[int(df_resultat[x])][13]
                rec1var14 = df.loc[int(df_resultat[x])][14]
                rec1var17 = df.loc[int(df_resultat[x])][17]
                
                
            elif x==2:
                
         
                rec2var8 = df.loc[int(df_resultat[x])][8]
                rec2var13 = df.loc[int(df_resultat[x])][13]
                rec2var14 = df.loc[int(df_resultat[x])][14]
                rec2var17 = df.loc[int(df_resultat[x])][17]
                
            elif x==3:
                
                rec3var8 = df.loc[int(df_resultat[x])][8]
                rec3var13 = df.loc[int(df_resultat[x])][13]
                rec3var14 = df.loc[int(df_resultat[x])][14]
                rec3var17 = df.loc[int(df_resultat[x])][17]
                

                
    #fig10, ax = plt.subplots()
    
    #--------------
    #ax.errorbar(x=df.loc[int(df_resultat[0])][1], y=0, xerr=asymmetric_error, fmt="o")
    #fig10.set_size_inches(10,0.5)
    #st.pyplot(fig10)
    #---------------
    
    
        st.text("")
        st.text("")
        st.text("")

        fig10, ax = plt.subplots()
        n, bins, patches = ax.hist(np.array(df.iloc[:,14:15]), 100, density=True)
        
        sigma = stdev(np.array(df.iloc[:,14:15]).flatten())
        mu = mean(np.array(df.iloc[:,14:15]).flatten())
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
        ax.plot(bins, y*1000, '--', color="lightgrey")


        # two points
        #ax[1].plot([105, 110], [200, 210], '-ro', label='line & marker')
        plt.plot([df.loc[int(df_resultat[0])][14]], [190], 'gd', label='marker only', color="black")
        plt.plot([rec1var14], [190], 'gd', label='marker only', color="red")
        
        if NbResultats>=3:
            plt.plot([rec2var14], [190], 'gd', label='marker only', color="green")
            
        if NbResultats>=4:
            plt.plot([rec3var14], [190], 'gd', label='marker only', color="blue")
        
        plt.plot([75, 95], [190, 190], label='no marker - default line', color="dimgrey")
        
       
        
        
        ax.set(title='Energy return (%)')
        ax.title.set_size(5)
        
        plt.yticks([])
        ax.xaxis.set_tick_params(labelsize=4)
        #ax.xaxis.set_tick_params(colors='lightgrey')
        

        plt.gca().set(frame_on=False)
        fig10.set_size_inches(10,0.2)
        
        st.pyplot(fig10)
        
        
        
        
        
        
        fig11, ax = plt.subplots()
        
        n, bins, patches = ax.hist(np.array(df.iloc[:,17:18]), 10, density=True)
        
        sigma = stdev(np.array(df.iloc[:,17:18]).flatten())
        mu = mean(np.array(df.iloc[:,17:18]).flatten())
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
        ax.plot(bins, y*500, '--', color="lightgrey")

        # two points
        #ax[1].plot([105, 110], [200, 210], '-ro', label='line & marker')
        plt.plot([df.loc[int(df_resultat[0])][17]], [190], 'gd', label='marker only', color="black")
        plt.plot([rec1var17], [190], 'gd', label='marker only', color="red")
        
        if NbResultats>=3:
            plt.plot([rec2var17], [190], 'gd', label='marker only', color="green")
            
        if NbResultats>=4:
            plt.plot([rec3var17], [190], 'gd', label='marker only', color="blue")
            
        plt.plot([0, 12], [190, 190], label='no marker - default line', color="dimgrey")
        
        ax.set(title='Spin potential')
        ax.title.set_size(5)
        
        plt.yticks([])
        ax.xaxis.set_tick_params(labelsize=4)
        

        plt.gca().set(frame_on=False)
        fig11.set_size_inches(10,0.2)
        
        st.pyplot(fig11)
    
    
    
    
    
    
        fig12, ax = plt.subplots()
        
        n, bins, patches = ax.hist(np.array(df.iloc[:,8:9]), 100, density=True)
        
        sigma = stdev(np.array(df.iloc[:,8:9]).flatten())
        mu = mean(np.array(df.iloc[:,8:9]).flatten())
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
        ax.plot(bins, y*8000, '--', color="lightgrey")

        # two points
        #ax[1].plot([105, 110], [200, 210], '-ro', label='line & marker')
        plt.plot([df.loc[int(df_resultat[0])][8]], [190], 'gd', label='marker only', color="black")
        plt.plot([rec1var8], [190], 'gd', label='marker only', color="red")
        
        if NbResultats>=3:
            plt.plot([rec2var8], [190], 'gd', label='marker only', color="green")
            
        if NbResultats>=4:
            plt.plot([rec3var8], [190], 'gd', label='marker only', color="blue")
        
        plt.plot([125, 300], [190, 190], label='no marker - default line', color="dimgrey")
        
        ax.set(title='Stiffness (lb/in)')
        ax.title.set_size(5)
        
        plt.yticks([])
        ax.xaxis.set_tick_params(labelsize=4)
        

        plt.gca().set(frame_on=False)
        fig12.set_size_inches(10,0.2)
        
        st.pyplot(fig12)
    
    
    
    
    
    
    
        
        fig13, ax = plt.subplots()
        
        n, bins, patches = ax.hist(np.array(df.iloc[:,13:14]), 100, density=True)
        
        sigma = stdev(np.array(df.iloc[:,13:14]).flatten())
        mu = mean(np.array(df.iloc[:,13:14]).flatten())
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
        ax.plot(bins, y*4000, '--', color="lightgrey")

        # two points
        #ax[1].plot([105, 110], [200, 210], '-ro', label='line & marker')
        plt.plot([df.loc[int(df_resultat[0])][13]], [190], 'gd', label='marker only', color="black")
        plt.plot([rec1var13], [190], 'gd', label='marker only', color="red")
        
        
        if NbResultats>=3:
            plt.plot([rec2var13], [190], 'gd', label='marker only', color="green")
        
        if NbResultats>=4:
            plt.plot([rec3var13], [190], 'gd', label='marker only', color="blue")
        
        plt.plot([20, 70], [190, 190], label='no marker - default line', color="dimgrey")
        
        ax.set(title='Tension loss (%)')
        ax.title.set_size(5)
        
        plt.yticks([])
        ax.xaxis.set_tick_params(labelsize=4)
        

        plt.gca().set(frame_on=False)
        fig13.set_size_inches(10,0.2)
        
        st.pyplot(fig13)
    
    
    
    #fig_bc1 = plt.barh(
    #        x=[df.loc[int(df_resultat[0])][1],
    #        y=["Gauge nominal (mm)"],
    #        width=10)

   
        
    
    
    #KMEANS -------------------------------------------------------------------------
    
    with tab2:
        
        scale = StandardScaler()
        df.iloc[:,4:] = scale.fit_transform(df.iloc[:,4:])

        kmeans = KMeans(n_clusters=3)
        fit = kmeans.fit(df.iloc[:,4:])

        df.iloc[:,4:] = scale.inverse_transform(df.iloc[:,4:])

        result = pd.concat([df.reset_index(), pd.DataFrame(fit.labels_).reset_index() ], axis = 1)
        result = result.rename(columns={0: "Cluster"})
        result["Cie"] = result["STRING"].str.split(" ").str[0]
        result.to_excel("output_concat.xlsx")

        result["Cluster"] = result["Cluster"].astype(str)

        # result = result[result["Cluster"].isin(["3"])]
        # result = result[result["Cie"].isin(["Babolat", "Solinco", "Yonex"])]
        


        fig = px.scatter_3d(result, x='Spin Potential', z='Total Loss (lbs)', y='Stiffness (lb/in)', symbol='Cluster', color='Cluster', hover_name="STRING", opacity=0.6, color_discrete_map= {'0': 'aqua','1': 'yellow', '2': 'green'})

        # specify trace names and symbols in a dict
        # symbols = {'0': 'square',
        #            '1':'circle',
        #            '2':'diamond'}

        # # set all symbols in fig
        # for i, d in enumerate(fig.data):
        #     fig.data[i].marker.symbol = symbols[fig.data[i].name.split(', ')[1]]

        fig.update_layout(
        autosize=False,
        height=800,
        )
        
    
        
        fig2 = go.Figure(data=[go.Scatter3d(x=[dataknn[int(LineID)][16]], y=[dataknn[int(LineID)][7]], z=[dataknn[int(LineID)][11]],
                                    mode='markers', marker_size=30, marker_symbol="cross", marker_color="black",name=cordage)])
        fig2.add_traces(fig.data)
        
        fig2.update_layout(
        autosize=False,
        height=800,
        showlegend=True,
        scene=Scene(
            xaxis=XAxis(title='Spin Potential'),
            yaxis=YAxis(title='Stiffness (lb/in)'),
            zaxis=ZAxis(title='Total Loss (lbs)')
        )
        )

        
        st.text("")
        st.text("")
        st.subheader(" REGROUPEMENT DE TOUS LES CORDAGES EN 3 GROUPES DISTINCTS")
       
        
        
        
        
        result_group = result.groupby(["Cluster"]).agg({'Stiffness (lb/in)':np.mean,'Total Loss (lbs)':np.mean, "Spin Potential":np.mean, "STRING":np.count_nonzero}) \
        .reset_index().sort_values('Cluster').rename(columns={'Stiffness (lb/in)': 'Stiffness (lb/in) average', 'Total Loss (lbs)': 'Total Loss (lbs) average', 'Spin Potential': 'Spin Potential average', 'STRING': 'Number of strings'})

        result_group['Interpretation'] = np.where(
        result_group['Stiffness (lb/in) average'] < 180, "plus doux pour le bras, plus de perte de tension", np.where(
        result_group['Spin Potential average'] < 5, "moins de spin, plus raide pour le bras ", "plus de spin, moins de perte de tension")) 


        st.dataframe(result_group.style.applymap(color_survived, subset=['Cluster']), hide_index=True)
        
    
        st.plotly_chart(fig2, use_container_width=True)
        


    # MOYENNES DES CLUSTER ------------
    
    with tab3:

        result.groupby(["Cluster"]).agg({'Stiffness (lb/in)':np.mean,'Total Loss (lbs)':np.mean, "Spin Potential":np.mean, "STRING":np.count_nonzero}) \
        .reset_index().sort_values('Stiffness (lb/in)').to_excel("output.xlsx")
        
        result = result.drop('index', axis=1)
        
        first_column = result.pop('Cluster') 
        result.insert(0, 'Cluster', first_column) 
        result = result.style.applymap(color_survived, subset=['Cluster'])
        st.dataframe(result, hide_index=True)
        
        st.text("Données sources: Tennis Warehouse University")
        st.text("")
        st.text("")
        st.image("https://raw.githubusercontent.com/cmvs3/tennis/main/vs3logo2.jpg")
        
        
        
            



    # col1, col2, col3 = st.columns(3)

    # NbTrim = col1.number_input(":green[**Nombre de trimestre(s) à prédire**]",min_value=2, max_value=5)

    # TenFut = col2.radio(":green[**Tendance future**]", ["Stabilisation", "Continuation"])

    # VarFut = col3.radio(":green[**Variation future**]", ["Min", "Max"])
    
    
   
    # pred = modelpredict(NbTrim, VarFut, TenFut)
    # df2 = df["Montant"].copy()
    # df_pred = pd.concat([df2, pred])
    # df_pred = pd.DataFrame(df_pred)
    # df_pred["valeur"] = np.where(df_pred.index > np.datetime64("2023-12-01"), "Pred", "Actuel")
    # fig = px.line(df_pred, color = df_pred["valeur"], color_discrete_sequence=["black", "green", "pink", "skyblue"], title="Ventes par trimestre avec prédiction(s)") 
    # fig.update_layout(title_x=0.35, xaxis_title="", yaxis_title="Ventes (en $)")
    # fig.update_xaxes(tickfont=dict(family='Rockwell', color='black', size=14))
    # fig.update_yaxes(tickfont=dict(family='Rockwell', color='black', size=14))
    # st.plotly_chart(fig, use_container_width=True, height = 200) 
    
    # df_pred