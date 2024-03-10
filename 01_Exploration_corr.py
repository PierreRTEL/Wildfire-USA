# <span style="color:#B22222;font-size:3em">Wildfires in USA</span>

# # Import des librairies et du dataset

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import datetime


# <span style="color:#a61c00;font-size:2em"><strong>===========================================================================</strong></span>  
# <span style="color:#a61c00;font-size:2em"><strong>I. Préparation des datasets</strong></span>
# <span style="color:#a61c00;font-size:2em"><strong>===========================================================================</strong></span>

# <span style="color:#cc4125;font-size:2em">--- Dataset Kaggle ---</span>

fires_orig = pd.read_csv('FPA_FOD_20170508.Fires_IMPORT.csv', sep=';')


# autre méthode avec parse_dates
# fires_orig = pd.read_csv('FPA_FOD_20170508.Fires_IMPORT.csv', sep=';', parse_dates=[['FIRE_YEAR','DISCOVERY_DOY']], date_format='%Y %j', keep_date_col=True)


# # Création d'une copie du dataset

fires = fires_orig.copy()


# # Quelques statistiques

# ## Type de variables et nombre de valeurs non nulles

fires.info(verbose=True, memory_usage=True, show_counts=True)


# ## Statistiques des colonnes numériques

fires.describe()


# ## Résumé des principales caractéristiques de chaque colonne

def summary(df):

    table = pd.DataFrame(
        index=df.columns,
        columns=['type_info', '%_missing_values', 'nb_unique_values', 'list_unique_values', "mean_or_mode", "flag"])
    table.loc[:, 'type_info'] = df.dtypes.values
    table.loc[:, '%_missing_values'] = df.isna().sum().values / len(df)
    table.loc[:, 'nb_unique_values'] = df.nunique().values

    def get_list_unique_values(colonne):
        if colonne.nunique() < 6:
            return colonne.unique()
        else:
            return "Too many categories..." if colonne.dtypes == "O" else "Too many values..."

    def get_mean_mode(colonne):
        return colonne.mode()[0] if colonne.dtypes == "O" else colonne.mean()

    def alerts(colonne, thresh_na = 0.25, thresh_balance = 0.8):
        if (colonne.isna().sum()/len(df)) > thresh_na:
            return "Too many missing values ! "
        elif colonne.value_counts(normalize=True).values[0] > thresh_balance:
            return "It's imbalanced !"
        else:
            return "Nothing to report"

    table.loc[:, 'list_unique_values'] = df.apply(get_list_unique_values)
    table.loc[:, 'mean_or_mode'] = df.apply(get_mean_mode)
    table.loc[:, 'flag'] = df.apply(alerts)

    return table

# summary(fires)


# # Suppression des colonnes

# ## Colonnes majoritairement vides

# Les colonnes suivantes ont un taux de valeurs manquantes élevé ( > 40 %) et ne sont pas nécessairement pertinentes pour répondre à la problématique.

cols_empty = [
    'ICS_209_INCIDENT_NUMBER', 
    'ICS_209_NAME', 
    'MTBS_ID', 
    'MTBS_FIRE_NAME', 
    'COMPLEX_NAME',
    'LOCAL_FIRE_REPORT_ID', 
    'LOCAL_INCIDENT_ID', 
    'FIRE_CODE', 
    'LOCAL_INCIDENT_ID', 
    'FIRE_NAME'
]


fires = fires.drop(cols_empty, axis=1)


# ## Colonnes non pertinentes

# Les colonnes suivantes n'ont pas d'intérêt quant à la problématique ou présentent trop de valeurs impropres à l'utilisation. 

cols_to_drop = [
    'OBJECTID',
    'SOURCE_SYSTEM_TYPE',
    'SOURCE_SYSTEM',
    'NWCG_REPORTING_AGENCY',
    'NWCG_REPORTING_UNIT_ID',
    'NWCG_REPORTING_UNIT_NAME',
    'SOURCE_REPORTING_UNIT',
    'SOURCE_REPORTING_UNIT_NAME',
    'COUNTY',
    'FIPS_CODE',
    'FIPS_NAME'
]


fires = fires.drop(cols_to_drop, axis=1)


# # fires.info(verbose=True, memory_usage=True, show_counts=True)


# # Colonnes d'ID : doublons, nettoyage

# ## Lignes entières

fires.duplicated().sum()


# Il n'y a pas de lignes entières en doublon dans le jeu de données. 

# ## Identifiant fonctionnel FPA_ID

# fires.loc[fires['FPA_ID'].duplicated(keep=False)].sort_values(by='FPA_ID')


# Il y a des doublons d'ID fonctionnels.  
# On remarque qu'on pourrait utiliser le FPA_ID pour déduire l'année du feu, puisque l'ID comporte parfois l'année. Toutefois, comme on ne sait pas où se trouve l'erreur, on décide de supprimer ces lignes, vu leur petit nombre.

# Avant suppression des doublons FPA_ID
print('Avant suppression :')
print(fires['FPA_ID'].info(verbose=True, memory_usage=True, show_counts=True), '\n\n============================\n')

# Suppression
fires = fires.drop_duplicates('FPA_ID')

# Après suppression des doublons FPA_ID
print('Après suppression :')
print(fires['FPA_ID'].info(verbose=True, memory_usage=True, show_counts=True))


# Les doublons ont bien été supprimés.

# ## Identifiant générique FOD_ID

# fires.loc[fires['FOD_ID'].duplicated(keep=False)].sort_values(by='FOD_ID')


# Il n'y a pas de doublons d'ID techniques.

# ## Suppression des espaces en tête et fin d'ID

# set(fires['FPA_ID'])


# On constate qu'il y a des espaces en fin d'ID.

# Suppression des espaces en tête et fin de chaîne
fires['FPA_ID'] = fires['FPA_ID'].str.strip()


# set(fires['FPA_ID'])


fires_fpa_set = set(fires['FPA_ID'])
print(f"Nombres d'ID fonctionnels uniques dans le dataset 'fires' : {len(fires_fpa_set)}")


# # Changement de type
# 
# Comme les valeurs sont en nombre restreint dans certaines colonnes, on modifie le type de certaines colonnes d'object à category afin de gagner de l'espace mémoire.  
# De même, on transforme le type de certaines colonnes numériques en un type plus léger.

# fires.columns


# Colonnes catégorielles
# si besoin, jeter un coup d'oeil à la documentation pd.Categorical()
fires[['STAT_CAUSE_DESCR', 'FIRE_SIZE_CLASS', 'OWNER_DESCR', 'STATE']] = \
    fires[['STAT_CAUSE_DESCR', 'FIRE_SIZE_CLASS', 'OWNER_DESCR', 'STATE']].astype('category')


# Colonnes catégorielles 
fires[['STAT_CAUSE_CODE', 'OWNER_CODE']] = fires[['STAT_CAUSE_CODE', 'OWNER_CODE']].astype('uint8')

# Colonnes numériques
fires[['FIRE_YEAR', 'DISCOVERY_DOY']] = fires[['FIRE_YEAR', 'DISCOVERY_DOY']].astype('uint16')


# fires.info()


# # Renommage de colonnes  
# Par souci de praticité et de temps, on raccourcit certains noms de colonne.

fires.rename(
    {
        'FIRE_YEAR':'DISC_YEAR',
        'DISCOVERY_DATE':'DISC_DATE',
        'DISCOVERY_DOY':'DISC_DOY',
        'DISCOVERY_TIME':'DISC_TIME',
        'STAT_CAUSE_CODE':'CAUSE_CODE',
        'STAT_CAUSE_DESCR':'CAUSE_DESCR',
        'FIRE_SIZE':'SIZE',
        'FIRE_SIZE_CLASS':'CLASS',
        'LATITUDE':'LAT',
        'LONGITUDE':'LON'
    }, 
    axis=1, inplace=True)


# fires.columns


# # Recalage et renommage des colonnes "XX_DATE"
# 
# Les deux colonnes DISC_DATE et CONT_DATE sont en fait des sortes de compteurs de jour, dont la plage correspond à la période temporelle étudiée en jours.

# calcul de la durée temporelle entre les dates de début de feu
# fires[['DISC_DATE']].max() - fires[['DISC_DATE']].min()


# calcul de la durée temporelle entre les dates de fin de feu
# fires[['CONT_DATE']].max() - fires[['CONT_DATE']].min()


# On renomme les deux colonnes DISC_DATE et CONT_DATE pour mettre en évidence l'aspect "compteur de jours".

fires = fires.rename({'DISC_DATE':'DISC_DAYS', 'CONT_DATE':'CONT_DAYS'}, axis=1)


# fires.columns


# fires[['DISC_DAYS', 'CONT_DAYS']].agg(['min', 'max'])


# On recale ces compteurs à 0 : la référence est alors le premier jour du dataset, à savoir le 01/01/1992.

min_days = fires['DISC_DAYS'].min()

# Recalage de la colonne de compteur de la date de début de feu
fires['DISC_DAYS'] = fires['DISC_DAYS'] - min_days

# Recalage de la colonne de compteur de la date de fin de feu
# Attention : utiliser le même minimum, même s'ils sont identiques
fires['CONT_DAYS'] = fires['CONT_DAYS'] - min_days


# min_days


# Vérification du recalage
# fires[['DISC_DAYS', 'CONT_DAYS']].agg(['min', 'max'])


# # Enrichissement du dataset

# ## Colonne "DUR_DAYS" de durée de feu
# 
# On crée une colonne de durée de feu en jours. Malheureusement, il manque près de la moitié des valeurs dans la colonne "CONT_DATE", compteur de jours qui marque la fin du feu. 
# Cela nous permet notamment de créer la colonne "YEAR" pour la date de maîtrise du feu.
# 
# <span style="color:red">ATTENTION : il s'agit de la partie entière de la durée en jours. Cela signifie qu'un feu de 2 h aura une durée en jours de 0 ou encore qu'un feu de 26 h aura une durée de 1 jour.</span> 

# Durée de feu en jour
fires['DUR_DAYS'] = fires['CONT_DAYS'] - fires['DISC_DAYS']
# fires[['CONT_DAYS','DISC_DAYS','DUR_DAYS']].head()


# fires.loc[fires['CONT_DAYS'].isna(),['CONT_DAYS','DISC_DAYS','DUR_DAYS']].head()


# ## Nouvelles colonnes "DISC_DATE" et "CONT_DATE"
# 
# On crée une colonne de date de début du feu et une colonne de date de fin de feu.

# Création de la date de début du feu
fires['DISC_DATE'] = \
    pd.to_datetime(fires['DISC_YEAR'].astype('str') + fires['DISC_DOY'].astype('str')
                   , format='%Y%j'
                   , errors='coerce')


fires.head()


fires.info()



# fires['DUR_DAYS'].max()


# Création de la date de fin du feu
fires['CONT_DATE'] = \
    fires.loc[fires['DUR_DAYS'].notna()]['DISC_DATE'] + pd.to_timedelta(fires['DUR_DAYS'], unit='D')


# fires.head()


# fires.loc[fires['CONT_DATE'].isna()].head()


# ## Colonne "CONT_YEAR" pour l'année de maîtrise du feu
# 
# Par souci d'homogénéité, on crée une colonne "CONT_YEAR" afin d'avoir l'année de fin du feu, pour les lignes disposant de l'information de la date de feu. 

# Création de la colonne de l'année de fin de feu
fires['CONT_YEAR'] = fires['CONT_DATE'].dt.year
# fires[['CONT_YEAR']].head()


# fires[fires['CONT_YEAR'].isna()].head()


# ### Colonne "DISC_MONTH" pour le mois de la découverte du feu
# 
# On crée une colonne "DISC_MONTH" afin d'étudier la saisonnalité des feux en fonction du mois de l'année.

fires['DISC_MONTH'] = fires['DISC_DATE'].dt.month


# ## Colonnes "HOUR" et "MINUTE" pour les horaires de départ et de fin de feu
# 
# On crée une colonne pour l'heure et une pour les minutes des horaires de départ et de fin de feu pour une utilisation potentielle plus tard dans l'imputing ou les analyses.  
# On supprime les deux colonnes de départ.

# Récupération de l'heure de début de feu
fires['DISC_HOUR'] = fires['DISC_TIME'] // 100
# Récupération des minutes de début de feu
fires['DISC_MIN'] = fires['DISC_TIME'] % 100

# Récupération de l'heure de fin de feu
fires['CONT_HOUR'] = fires['CONT_TIME'] // 100
# Récupération des minutes de fin de feu
fires['CONT_MIN'] = fires['CONT_TIME'] % 100


fires[['DISC_TIME', 'DISC_HOUR', 'DISC_MIN', 'CONT_TIME', 'CONT_HOUR', 'CONT_MIN']].head()


fires[['DISC_TIME', 'DISC_HOUR', 'DISC_MIN', 'CONT_TIME', 'CONT_HOUR', 'CONT_MIN']].info(verbose=True, memory_usage=True, show_counts=True)


# Bien qu'il y ait un peu plus d'horaires de début de feu que de fin de feu, on remarque qu'ils manquent tout de même près de la moitié des horaires. Ceci est plutôt logique car il ne doit pas être toujours aisé de donner avec précision l'heure de départ ou de fin d'un feu. 

# fires.loc[fires['CONT_TIME'].isna(),['CONT_TIME','CONT_HOUR','CONT_MIN']].head()


# Après séparation des heures et minutes, suppression des colonnes initiales
fires.drop(['DISC_TIME','CONT_TIME'], axis=1, inplace=True)


# ## Nouvelles colonnes "DISC_DATETIME" et "CONT_DATETIME"
# 
# On crée deux colonnes datetime pour les dates et horaires de départ de feu et de fin de feu. Cela permettra d'affiner, pour les lignes complètes, la durée du feu.

# Création d'une colonne de datetime de début de feu
fires['DISC_DATETIME'] = \
    fires.loc[fires['DISC_HOUR'].notna()]['DISC_DATE'] + \
    pd.to_timedelta(fires['DISC_HOUR'], unit='h') + pd.to_timedelta(fires['DISC_MIN'], unit='m')


# fires[['DISC_DATETIME', 'DISC_DATE', 'DISC_HOUR', 'DISC_MIN']].head()


# fires.loc[fires['DISC_DATETIME'].isna()].head()


# Création d'une colonne de datetime de début de feu
fires['CONT_DATETIME'] = \
    fires.loc[fires['CONT_HOUR'].notna()]['CONT_DATE'] + \
    pd.to_timedelta(fires['CONT_HOUR'], unit='h') + pd.to_timedelta(fires['CONT_MIN'], unit='m')


# fires[['CONT_DATETIME', 'CONT_DATE', 'CONT_HOUR', 'CONT_MIN']].head()


# fires.loc[fires['CONT_DATETIME'].isna()].head()


# ## Nouvelle colonne "DUR_MIN"
# 
# On crée une colonne de durée de feu en minutes, ce qui enrichira le dataset d'une nouvelle variable.  
# Cette variable servira aussi pour l'impute sur les durées manquantes dans une partie complémentaire à l'étude initiale.

fires['DUR_MIN'] = (fires['CONT_DATETIME'] - fires['DISC_DATETIME']) / pd.Timedelta(minutes=1)


fires[['DUR_MIN', 'CONT_DATETIME', 'DISC_DATETIME']].head()


fires[['DUR_MIN']].info(verbose=True, memory_usage=True, show_counts=True)


# Il aurait fallu à cet endroit tracer un boxplot de la durée en fonction de la classe de feu pour se rendre compte qu'il y avait un problème pour certains feux qui dureraient, prétendument, 10 ans pour certains.  
# Malheureusement, erreur de novice : Thibault est parti du principe que le nettoyage était bon au-dessus, d'autant plus qu'il était précisé sur Kaggle que le dataset était plutôt clean. Il a réalisé trop tard que même un simple calcul comme une différence pouvait mettre au jour une nouvelle incohérence des données.   
# 
# Petit aperçu du problème ci-dessous :  certains feux de classe "petite", A ou B, durent 10 ans. On suppose une erreur dans le fameux compteur de jour, d'autant plus que l'horaire de fin semble lui cohérent avec l'horaire de début (exemple : première ligne, avec un classe A qui aurait duré 1h30, si l'année de fin était bien identique à l'année de début).

fires.loc[fires['DUR_MIN'] > 1E6, [
    'FPA_ID', 'DUR_MIN', 'SIZE', 'CLASS', 
    'DISC_YEAR', 'DISC_DATETIME', 'CONT_YEAR', 'CONT_DATETIME', 
    'CAUSE_DESCR', 'STATE']].sort_values('DUR_MIN', ascending=False)


# fires.iloc[362576]


# fires_duration = fires[['CLASS', 'DUR_MIN']]

# fig = px.box(fires_duration, x='CLASS', y = 'DUR_MIN', color='CLASS',
#              category_orders={"CLASS": ["A", "B", "C", "D", "E", "F", "G"]},
#              width=1000, height=600
#             )

# fig.update_layout(
#     title={
#         'text': "Distribution de la durée (min) par classe de feu",
#         'y':0.95,
#         'x':0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'}, 
#     xaxis={'title':"Classe de feu",'categoryorder':'category ascending'},    
#     yaxis_title="Durée (min)"
# )


# On constate des durées de feu complètement aberrants de plusieurs années. Finalement, les outliers pour les classes F et G ont l'air moins aberrants que ceux des classes A, B, etc...

# fires_duration = fires.loc[fires['DUR_MIN'] < 2E5, ['CLASS', 'DUR_MIN']]

# fig = px.box(fires_duration, x='CLASS', y = 'DUR_MIN', color='CLASS',
#              category_orders={"CLASS": ["A", "B", "C", "D", "E", "F", "G"]},
#              width=1000, height=600
#             )

# fig.update_layout(
#     title={
#         'text': "Distribution de la durée (min) par classe de feu (outliers supérieurs à 2E5 min supprimés)",
#         'y':0.95,
#         'x':0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'}, 
#     xaxis={'title':"Classe de feu",'categoryorder':'category ascending'},    
#     yaxis_title="Durée (min)"
# )


# En éliminant les plus grands outliers, on commence à voir mieux apparaître les boxplots pour les classes les plus grandes. On constate toujours la présence d'outliers pour les classes de petits feux, qui sont pourtant plus élevés que ceux de plus grande classe : ceci confirme le problème.

fires.groupby('CLASS', observed=False)['DUR_MIN'].agg(['min', 'median', 'mean', 'max'])


fires.loc[fires['DUR_MIN'] < 2E5].groupby('CLASS', observed=False)['DUR_MIN'].agg(['min', 'median', 'mean', 'max'])


# ## Nouvelle colonne de cause humaine : "CAUSE_DESCR_HUMAN"
# Il s'agit de créer un booléen qui indique si le feu est d'origine humaine. On exclut donc la foudre ou le cas "indéfini".

fires['CAUSE_DESCR_HUMAN'] = fires['CAUSE_DESCR'].apply(lambda x: 1 if x not in ['Lightning', 'Missing/Undefined'] else 0)


# fires.columns


# On constate bien que la colonne a été créée en fin de dataframe.

# ## Constatations sur la date de fin du feu

# ### Répartition par année de l'absence de la date de fin du feu

endate_nan_year = fires[fires['CONT_DATE'].isna()].groupby('DISC_YEAR')['FOD_ID'].count().astype('int')
endate_nan_year.rename('NaN', inplace=True)
year_total = fires.groupby('DISC_YEAR')['FOD_ID'].count().astype('int')
year_total.rename('Total', inplace=True)

df_endate_nan_year = pd.concat([endate_nan_year, year_total], axis=1)
df_endate_nan_year['ratio'] = df_endate_nan_year.apply(lambda row: round(row['NaN'] / row['Total'],2), axis=1)
# df_endate_nan_year


# On remarque que les valeurs manquantes de la date de fin du feu, réparties par année, ne sont pas localisées sur une seule période, malgré des fluctuations importantes. 

# ### Répartition par Etat de l'absence de la date de fin du feu

endate_nan_state = fires[fires['CONT_DATE'].isna()].groupby('STATE', observed=False)['FOD_ID'].count().astype('int')
endate_nan_state.rename('NaN', inplace=True)
state_total = fires.groupby('STATE', observed=False)['FOD_ID'].count().astype('int')
state_total.rename('Total', inplace=True)

df_endate_nan_state = pd.concat([endate_nan_state, state_total], axis=1)
df_endate_nan_state['ratio'] = df_endate_nan_state.apply(lambda row: round(row['NaN'] / row['Total'],2), axis=1)
# df_endate_nan_state


# On remarque que la proportion de date de fin manquante varie beaucoup selon l'Etat. Cela pourrait affecter certaines analyses.

# ## Création des dictionnaires des catégories des variables catégorielles

# # Méthode pour créer un dictionnaire des propriétaires des terrains touchés
# fires[['CAUSE_CODE','CAUSE_DESCR']].drop_duplicates().sort_values(by='CAUSE_CODE')
# keys = fires[['OWNER_CODE','OWNER_DESCR']].drop_duplicates().sort_values(by='OWNER_CODE')['OWNER_CODE'].values
# values = fires[['OWNER_CODE','OWNER_DESCR']].drop_duplicates().sort_values(by='OWNER_CODE')['OWNER_DESCR'].values
# d_causes = dict(zip(keys, values))
# d_causes


dict_states = {'AK':'Alaska','AL':'Alabama','AZ':'Arizona','AR':'Arkansas','CA':'Californie','CO':'Colorado',
               'CT':'Connecticut','DE':'Delaware','FL':'Floride','GA':'Géorgie','HI':'Hawaï','ID':'Idaho',
               'IL':'Illinois','IN':'Indiana','IA':'Iowa','KS':'Kansas','KY':'Kentucky','LA':'Louisiane',
               'ME':'Maine','MD':'Maryland','MA':'Massachusetts','MI':'Michigan','MN':'Minnesota','MS':'Mississippi',
               'MO':'Missouri','MT':'Montana','NE':'Nebraska','NV':'Nevada','NH':'New Hampshire','NJ':'New Jersey',
               'NM':'Nouveau-Mexique','NY':'New York','NC':'Caroline du Nord','ND':'Dakota du Nord','OH':'Ohio',
               'OK':'Oklahoma','OR':'Oregon','PA':'Pennsylvanie','RI':'Rhode Island','SC':'Caroline du Sud',
               'SD':'Dakota du Sud','TN':'Tennessee','TX':'Texas','UT':'Utah','VT':'Vermont','VA':'Virginie',
               'WA':'Washington','WV':'Virginie-Occidentale','WI':'Wisconsin','WY':'Wyoming','DC':'District de Columbia',
               'AS':'Samoa américaines','GU':'Guam','MP':'Îles Mariannes du Nord','PR':'Porto Rico',
               'UM':'Îles mineures éloignées des États-Unis','VI':'Îles Vierges américaines'}


dict_causes = {1: 'Lightning', 2: 'Equipment Use', 3: 'Smoking', 4: 'Campfire', 5: 'Debris Burning', 6: 'Railroad',
               7: 'Arson', 8: 'Children', 9: 'Miscellaneous', 10: 'Fireworks', 11: 'Powerline', 12: 'Structure',
               13: 'Missing/Undefined'}


list_causes = list(fires['CAUSE_DESCR'].cat.categories)
list_causes.append('Human')
# list_causes
list_colors = ['#EF553B','#FFA15A','#FF6692','#2CA02C','#636EFA','#FECB52','#1616A7',
               '#AB63FA','#19D3F3','#B68100','#778AAE','#862A16','#1CFFCE','rgb(228,26,28)']
dict_causes_colors = dict(zip(list_causes, list_colors))
# dict_causes_colors


dict_owners = {0: 'FOREIGN', 1: 'BLM', 2: 'BIA', 3: 'NPS', 4: 'FWS', 5: 'USFS', 6: 'OTHER FEDERAL', 7: 'STATE',
               8: 'PRIVATE', 9: 'TRIBAL', 10: 'BOR', 11: 'COUNTY', 12: 'MUNICIPAL/LOCAL', 13: 'STATE OR PRIVATE',
               14: 'MISSING/NOT SPECIFIED', 15: 'UNDEFINED FEDERAL'}


fires.to_csv('fires.csv', sep=';', encoding='utf-8', index_label='index')


# <span style="color:#cc4125;font-size:2em">--- Dataset végétation et météo ---</span>

fires_veg_orig = pd.read_csv('all_fires.csv', sep=',')


# Création d'une copie du dataset
fires_veg = fires_veg_orig.copy()
fires_veg.head()


# # Quelques statistiques

# ## Type de variables et nombre de valeurs non nulles

fires_veg.info(verbose=True, memory_usage=True, show_counts=True)


# ## Statistiques des colonnes numériques

fires_veg.describe()


# ## Résumé des principales caractéristiques de chaque colonne

# summary(fires_veg)


# # Gestion des doublons d'ID

# ## Lignes entières

# fires_veg.duplicated().sum()


# Il n'y a pas de lignes entières en doublon dans le jeu de données. 

# ## Identifiant fonctionnel FPA_ID

# On utilise tout d'abord la colonne "is_id_duplicated" qui indique les doublons d'ID FPA.

fires_veg_duplicated = fires_veg.loc[fires_veg['is_id_duplicated'] == True, ['FPA_ID']].sort_values(by='FPA_ID')['FPA_ID'].values
# fires_veg_duplicated


# Il y a 3 doublons repérés dans le dataset. On les supprime par précaution.

print('Avant suppression :')
print(fires_veg['FPA_ID'].info(verbose=True, memory_usage=True, show_counts=True), '\n\n============================\n')

fires_veg = fires_veg.loc[fires_veg['is_id_duplicated'] == False]

print('Avant suppression :')
print(fires_veg['FPA_ID'].info(verbose=True, memory_usage=True, show_counts=True))


# Présence de doublons d'identifiants fonctionnels
fires_veg_duplicated_id = fires_veg.loc[fires_veg['FPA_ID'].duplicated(keep=False)]
# fires_veg_duplicated_id.shape


fires_veg_duplicated_id.sort_values(by='FPA_ID').head(10)


# Il y a encore des doublons au niveau de l'identifiant FPA_ID. Sur quelques exemples, on constate que les lignes sont identiques à part un élément : l'écorégion de niveau 3 (nom, code et surface).  
# Par mesure de précaution et souci de rapidité, on décide de supprimer l'intégralité de ces lignes, vu leur petit nombre. 

print('Avant suppression :')
print(fires_veg['FPA_ID'].info(verbose=True, memory_usage=True, show_counts=True), '\n\n============================\n')

fires_veg_duplicated_id_list = fires_veg_duplicated_id['FPA_ID'].values
fires_veg = fires_veg.loc[~fires_veg['FPA_ID'].isin(fires_veg_duplicated_id_list)]

print('Avant suppression :')
print(fires_veg['FPA_ID'].info(verbose=True, memory_usage=True, show_counts=True))


# ## Identifiant générique clean_id

fires_veg.loc[fires_veg['clean_id'].duplicated(keep=False)].sort_values(by='clean_id')


# Il n'y a pas de doublons d'ID techniques.

# # Nouvelle colonne "CLASS"  
# Afin de faciliter les analyses et les visualisations, on crée une colonne "CLASS" dans ce nouveau dataset, à l'image de ce qui existe dans le dataset de Kaggle.   
# On rappelle les classes de feux :  
# - A : comprise entre 0 et 0.25 acres,
# - B : comprise entre 0,26 et 9,9 acres,
# - C : comprise entre 10,0 et 99,9 acres,
# - D : comprise entre 100 et 299 acres,
# - E : comprise entre 300 et 999 acres,
# - F : comprise entre 1000 et 4999 acres,
# - G : supérieure à 5000 acres
#   
# On rappelle l’équivalence : 1 acre = 0,404686 hectare.

conditions = [
    (fires_veg['FIRE_SIZE'] > 0.0) & (fires_veg['FIRE_SIZE'] < 0.26),
    (fires_veg['FIRE_SIZE'] >= 0.26) & (fires_veg['FIRE_SIZE'] < 10.0),
    (fires_veg['FIRE_SIZE'] >= 10.0) & (fires_veg['FIRE_SIZE'] < 100.0),
    (fires_veg['FIRE_SIZE'] >= 100.0) & (fires_veg['FIRE_SIZE'] < 300.0),
    (fires_veg['FIRE_SIZE'] >= 300.0) & (fires_veg['FIRE_SIZE'] < 1000.0),
    (fires_veg['FIRE_SIZE'] >= 1000.0) & (fires_veg['FIRE_SIZE'] < 5000.0),
    (fires_veg['FIRE_SIZE'] >= 5000.0),
]
choices = ['A','B','C','D','E','F','G']

fires_veg['CLASS'] = np.select(conditions, choices)


fires_veg['CLASS'].unique()


# # Suppression des colonnes

# Pour une prochaine interrogation, plus loin dans le notebook
# Etats présents dans le dataset "végétation et météo"
fires_veg_set = set(fires_veg['STATE'].unique())


# fires_veg.columns


# ## Colonnes majoritairement vides  
# Les colonnes suivantes ont un taux de valeurs manquantes élevé ( > 40 %) et ne sont pas nécessairement pertinentes pour répondre à la problématique.

cols_empty_veg = [
    'ICS_209_INCIDENT_NUMBER', 
    'ICS_209_NAME', 
    'MTBS_ID', 
    'MTBS_FIRE_NAME'
]


# Suppression des colonnes majoritairement vides
fires_veg = fires_veg.drop(cols_empty_veg, axis=1)


# fires_veg.info(verbose=True, memory_usage=True, show_counts=True)


# ## Colonnes non pertinentes  
# Les colonnes suivantes n'ont pas d'intérêt quant à la problématique : identifiants, colonnes en doublon...

cols_to_drop_veg = [
    'clean_id', 
    #'FPA_ID', # conservé pour la jointure des deux datasets
    'is_id_duplicated', # tous les doublons sont déjà supprimés
    #'Wind', 'fm', # conservés car données explicatives
    'STATE', 'LATITUDE', 'LONGITUDE', # colonnes en doublon
    #'DISCOVERY_YEAR', 'DISCOVERY_DOY', 'DISCOVERY_DATE', # conservé pour la vérification de la jointure des deux datasets
    'FIRE_YEAR', 'DISCOVERY_DAY', 'DISCOVERY_MONTH',  # colonnes en doublon
    'STAT_CAUSE_DESCR', 'IGNITION', # colonnes en doublon
    'FIRE_SIZE', 'FIRE_SIZE_m2', 'FIRE_SIZE_ha', # colonnes en doublon
    #'CLASS', # conservé pour les visualisations
    'us_130bps', # code
    #'NBCD_countrywide_biomass_mosaic', 'GROUPVEG', # conservés car données explicatives
    'NA_L3CODE', 'NA_L1CODE' # code
    #'NA_L3NAME', 'NA_L1NAME', 'EcoArea_km2' # conservés car données explicatives
]


# Suppression de colonnes
fires_veg = fires_veg.drop(cols_to_drop_veg, axis=1)


fires_veg.info(verbose=True, memory_usage=True, show_counts=True)


# # Suppression des espaces en tête et fin d'ID

# De la même manière que le dataset initial, on constate qu'il y a des espaces en fin d'ID.

set(fires_veg['FPA_ID'])


fires_veg['FPA_ID'] = fires_veg['FPA_ID'].str.strip()


set(fires_veg['FPA_ID'])





# # Colonne "ECO_AREA_KM2" : correction
# L'aire de la région de niveau 3 semble avoir une corrélation non négligeable avec la classe de feu, d'après le KBest (mutual_info_classif). On décide de la retravailler afin de corriger des erreurs.

# Colonnes présentes
# fires_veg.columns


print(f"Il y a {fires_veg['NA_L3NAME'].nunique()} écorégions de niveau 3.")


# #commenté car lourd en taille de notebook
# for eco in df['ECO_REG_LVL3'].unique():
#     fig = go.Figure(data=[
#         go.Histogram(x=df.loc[(fires_merge['ECO_REG_LVL3']==eco) & (fires_merge['CLASS']=='A'), 
#                                       'ECO_AREA_1000KM2'], name='A'),
#         go.Histogram(x=df.loc[(fires_merge['ECO_REG_LVL3']==eco) & (fires_merge['CLASS']=='B'), 
#                                       'ECO_AREA_1000KM2'], name='B'),
#         go.Histogram(x=df.loc[(fires_merge['ECO_REG_LVL3']==eco) & (fires_merge['CLASS']=='C'), 
#                                       'ECO_AREA_1000KM2'], name='C'),
#         go.Histogram(x=df.loc[(fires_merge['ECO_REG_LVL3']==eco) & (fires_merge['CLASS']=='D'), 
#                                       'ECO_AREA_1000KM2'], name='D'),
#         go.Histogram(x=df.loc[(fires_merge['ECO_REG_LVL3']==eco) & (fires_merge['CLASS']=='E'), 
#                                       'ECO_AREA_1000KM2'], name='E'),
#         go.Histogram(x=df.loc[(fires_merge['ECO_REG_LVL3']==eco) & (fires_merge['CLASS']=='F'), 
#                                       'ECO_AREA_1000KM2'], name='F'),
#         go.Histogram(x=df.loc[(fires_merge['ECO_REG_LVL3']==eco) & (fires_merge['CLASS']=='G'), 
#                                       'ECO_AREA_1000KM2'], name='G')
#     ])

#     # The two histograms are drawn on top of another
#     fig.update_layout(barmode='stack', 
#                       title = f"Distribution de l'aire de l'écorégion {eco}",
#                       xaxis_title_text='Aire (1000 km²)',
#                       yaxis_title_text='Nombre'
#                       )
#     fig.show()


# On constate qu'il y a des erreurs dans les surfaces des écorégions, avec des valeurs ici et là ridiculement faibles... Problèmes de conversion ? De saisie des données ?  
# On décide d'appliquer le maximum de la surface d'une écorégion à tous ces enregistrements associés, sachant que cette valeur est normalement constante (ou tout du moins évolue très peu).

# Récupération du maximum de l'aire de chacune des écorégions de niveau 3 - tentative n°1
eco_areas_dict = fires_veg.groupby(['NA_L3NAME'])['EcoArea_km2'].max().round(0).to_dict()
# eco_areas_dict


# En observant les clés du dictionnaire, on se rend compte qu'il y a une erreur de saisie pour le Chihuahuan (Desert, Deserts). On corrige.

# Correction du doublon de noms de l'écorégion Chihuahuan
fires_veg.loc[fires_veg['NA_L3NAME']=='Chihuahuan Deserts'] =\
    fires_veg.loc[fires_veg['NA_L3NAME']=='Chihuahuan Deserts']\
        .replace('Chihuahuan Deserts', 'Chihuahuan Desert') 


# Récupération du maximum de l'aire de chacune des écorégions de niveau 3 - tentative n°2
eco_areas_dict = fires_veg.groupby(['NA_L3NAME'])['EcoArea_km2'].max().round(0).to_dict()
# eco_areas_dict


# fires_veg.columns


# Avant correction
fires_veg[['NA_L3NAME','EcoArea_km2']].head(20)


# Application de la correction
fires_veg['EcoArea_km2'] = fires_veg['NA_L3NAME'].map(eco_areas_dict)
fires_veg[['NA_L3NAME','EcoArea_km2']].head(20)


# La correction est faite.

# # Colonne "NBCD_FIA_BIOMASS_MOSAIC"
# L'indice de biomasse indique "la vie" sur une parcelle normalisée de terre. On analyse rapidement la distribution de cette colonne.

# # commenté car gourmand à l'affichage
# fig = px.box(fires_veg,
#              x='NBCD_countrywide_biomass_mosaic',
#              y='NA_L1NAME',
#              title = f"Distribution de l'indice de biomasse")
# fig.show()


ratio_null_values_NBCD = fires_veg.loc[fires_veg['NBCD_countrywide_biomass_mosaic'] == 0, 'NBCD_countrywide_biomass_mosaic'].count() / fires_veg.shape[0]
print(f"Ratio toutes classes confondues de valeurs nulles : {np.round(ratio_null_values_NBCD, 2) * 100} %")


# Un quart des valeurs de l'indice de biomasse sont nulles. Deux solutions :   
# 1) on décide de changer toutes les valeurs nulles par une valeur de leur plus proche voisin non nulle ou bien par la médiane
# 2) on n'utilise pas cette colonne.
# 
# Par mesure de sécurité et souci de temps, on décide de ne pas utiliser cette colonne.

# # Changement de type
# 
# On modifie le type de certaines colonnes d'object à category afin de gagner de l'espace mémoire.

# fires_merge.columns


fires_veg[['GROUPVEG', 'NA_L3NAME', 'NA_L1NAME', 'CLASS']] = \
        fires_veg[['GROUPVEG', 'NA_L3NAME', 'NA_L1NAME', 'CLASS']].astype('category')


# fires_merge.info(verbose=True, memory_usage=True, show_counts=True)


# # Renommage de colonnes

# Colonnes présentes
# fires_veg.columns


# Renommage des colonnes
fires_veg= fires_veg.rename(
    {
        'fm':'FUEL_MOISTURE',
        'Wind':'WIND',
        'NBCD_countrywide_biomass_mosaic':'NBCD_FIA_BIOMASS_MOSAIC',
        'GROUPVEG':'VEGETATION',
        'NA_L3NAME':'ECO_REG_LVL3',
        'NA_L1NAME':'ECO_REG_LVL1',
        'EcoArea_km2':'ECO_AREA_KM2'
    }, 
    axis=1)


# Colonnes renommées
# fires_veg.columns


# # Comparaison rapide des FPA_ID dans les deux datasets avant jointure
# Comme la variable FPA_ID va servir de clé de jointure, on analyse les différences de cette variable dans les deux datasets.

fires_fpa_set = set(fires['FPA_ID'])
fires_veg_fpa_set = set(fires_veg['FPA_ID'])

print(f"Nombres d'ID fonctionnels uniques dans le dataset fires : {len(fires_fpa_set)}")
print(f"Nombres d'ID fonctionnels au total dans le dataset fires : {fires.shape[0]}")
print(f"Nombres d'ID fonctionnels uniques dans le dataset fires_veg : {len(fires_veg_fpa_set)}")
print(f"Nombres d'ID fonctionnels au total dans le dataset fires_veg : {fires_veg.shape[0]}")
print(f"On constate qu'il y a {len(fires_fpa_set - fires_veg_fpa_set)} 'FPA_ID' uniques dans le dataset Kaggle qui ne sont pas présentes dans le dataset complémentaire.")
print(f"On constate qu'il y a {len(fires_veg_fpa_set - fires_fpa_set)} 'FPA_ID' uniques dans le dataset complémentaire qui ne sont pas présentes dans le dataset Kaggle.")


# # Des Etats manquants...
# On pousse l'analyse un peu plus loin en comparant les Etats présents dans les deux datasets.

# Etats présents dans le dataset initial
fires_states_set = set(fires['STATE'].unique())

# Etats absents dans le dataset "végétation et météo"
print("Les Etats absents dans le dataset 'végétation et météo' :",fires_states_set - fires_veg_set)


# Distribution des classes de feu pour l'entièreté du dataset : 
print("Distribution des classes de feu :\n", fires['CLASS'].value_counts(), '\n')

# Distribution des classes de feu pour l'Alaska : 
print("Distribution des classes de feu pour l'Alaska :\n", fires.loc[fires['STATE'] == 'AK', 'CLASS'].value_counts())


# Cela pose question : prend-on le risque à cause de la jointure de ne pas considérer les records de l'Alaska, sachant que c'est un des Etats les plus touchés par les feux de grande classe (10 % de la classe G et 5 % de la classe E), ou se passe-t-on des variables "vent", "humidité de la végétation", "écorégion" en ne faisant pas la jointure ?  
# 
# On décide tenter la jointure des deux datasets. 

# # Gestion des valeurs manquantes

fires_veg.isnull().sum()


# On supprime les quelques milliers de lignes où il manque le vent et l'humidité de la matière végétale.

fires_veg = fires_veg.dropna()


fires_veg.isnull().sum()


fires_veg.to_csv('fires_veg.csv', sep=';', encoding='utf-8', index_label='index')


# <span style="color:#a61c00;font-size:2em"><strong>===========================================================================</strong></span>  
# <span style="color:#a61c00;font-size:2em"><strong>II. Visualisation</strong></span>
# <span style="color:#a61c00;font-size:2em"><strong>===========================================================================</strong></span>

# <span style="color:#cc4125;font-size:2em">--- Dataset Kaggle ---</span>

# ## A - Création d'un dataset agrégé pour la surface cumulée de feu selon l'année, la classe de feu, la cause, l'Etat, le critère humain/foudre

fire_size_cumul_hum = fires.loc[fires['CAUSE_DESCR_HUMAN'] == 1].groupby(['STATE','DISC_YEAR','CLASS','CAUSE_DESCR'], observed=False)['SIZE'].sum().reset_index()
fire_size_cumul_lig = fires.loc[fires['CAUSE_DESCR'] == 'Lightning'].groupby(['STATE','DISC_YEAR','CLASS','CAUSE_DESCR'], observed=False)['SIZE'].sum().reset_index()


fire_size_cumul_lig.head(10)


# ## B - Vision Globale des feux au Etats-Unis

# ### Map du nombre de feux par Etat en fonction des années

fires_count = pd.DataFrame(fires.groupby(['STATE', 'DISC_YEAR'], observed=False)['FOD_ID'].count().rename('COUNT')).reset_index()
fires_count.head()


fires_count['STATE_NAME'] = fires_count['STATE'].replace(dict_states)
fires_count[fires_count['STATE_NAME']=='Californie'].head()


map_nb_feu = px.scatter_geo(fires_count, locations="STATE", size='COUNT', size_max = 35,
                     color = 'COUNT', 
                     color_continuous_scale=['rgb(254,217,118)','rgb(254,178,76)','rgb(253,141,60)', 'rgb(252,78,42)',
                                             'rgb(227,26,28)','rgb(189,0,38)', 'rgb(128,0,38)'], 
                     title="Nombre de feux en fonction des Etats et de l'année",
                     animation_frame='DISC_YEAR', 
                     locationmode="USA-states", projection='albers usa',
                     labels = {'STATE':'Code Etat', "STATE_NAME": "Nom de l'état",'COUNT':'Nombre', 'DISC_YEAR':'Année'},
                     hover_data=["STATE_NAME",'DISC_YEAR', 'COUNT'], 
                     height=700, width=1200
                    )
map_nb_feu.update_traces(hovertemplate='Etat : %{customdata[0]}<br>Année : %{customdata[1]}<br>Nombre : %{customdata[2]:.3s}')
for f in map_nb_feu.frames:
    f.data[0].update(hovertemplate='Etat : %{customdata[0]}<br>Année : %{customdata[1]}<br>Nombre : %{customdata[2]:.3s}')
# map_nb_feu.show()


# ### Map de la surface cumulée brûlée par Etat en fonction des années

fires_size = pd.DataFrame(fires.groupby(['STATE', 'DISC_YEAR'], observed=False)['SIZE'].sum().rename('CUMUL_SIZE')).reset_index()
fires_size.head()


fires_size['STATE_NAME'] = fires_size['STATE'].replace(dict_states)
fires_size[fires_size['STATE_NAME']=='Californie'].head()


map_surf_cum = px.scatter_geo(fires_size, locations="STATE", size='CUMUL_SIZE', size_max = 50,
                     color = 'CUMUL_SIZE', 
                     color_continuous_scale=['rgb(254,217,118)','rgb(254,178,76)','rgb(253,141,60)', 'rgb(252,78,42)',
                                             'rgb(227,26,28)','rgb(189,0,38)','rgb(128,0,38)'], 
                     labels = {'STATE':'Code Etat', "STATE_NAME": "Nom de l'état",'CUMUL_SIZE':'Surface cumulée', 'DISC_YEAR':'Année'},
                     animation_frame='DISC_YEAR', locationmode="USA-states", projection='albers usa',
                     title="Surface cumulée brûlée par Etat en fonction des années",
                     hover_data=["STATE_NAME",'DISC_YEAR', 'CUMUL_SIZE'],
                     height=700, width=1200
                    )
map_surf_cum.update_traces(hovertemplate='Etat : %{customdata[0]}<br>Année : %{customdata[1]}<br>Surface cumulée : %{customdata[2]:.3s} acres')
for f in map_surf_cum.frames:
    f.data[0].update(hovertemplate='Etat : %{customdata[0]}<br>Année : %{customdata[1]}<br>Surface cumulée : %{customdata[2]:.3s} acres')
# map_surf_cum.show()


# ### Map des feux en fonction des années

fires_reduced = fires.loc[fires['CLASS']=='E',["LAT", "LON", "DISC_YEAR", "CLASS"]].sort_values(by="DISC_YEAR")
fires_reduced["CLASS"].replace({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}, inplace=True)
fires_reduced.head()


# Density Mapbox
density_map = px.density_mapbox(fires_reduced, lat='LAT', lon='LON', radius=5,
                        center=dict(lat=50, lon=260), zoom=1.9,
                        mapbox_style="open-street-map", 
                        animation_frame='DISC_YEAR', 
                        labels={"DISC_YEAR":"Année"},
                        title = "Evolution temporelle des feux de classe E",
                        height=700, width=1200)
# density_map.show() 


# ### Distribution de la durée des feux

# ## C - Les classes de feux

# ### Distribution des classes de feux

fires_class = fires[['CLASS', 'DISC_YEAR', 'STATE']]


nb_class_year = px.histogram(fires_class, x = 'CLASS', 
                   animation_frame = fires_class['DISC_YEAR'].sort_values(),
                   category_orders=dict(CLASS=["A", "B", "C", "D","E","F","G"]),
                   labels={'DISC_YEAR': 'Année', 'CLASS': 'Classe', 'animation_frame': 'Année'},
                   width=1200, height=700
                  )

nb_class_year.update_layout(
    title="Distribution des classes de feu par année",
    xaxis_title="Classes de feux",
    yaxis_title="Nombre de feux")
# Problème d'auto range sur l'axe des y
nb_class_year.update_yaxes(range=[0, 65000])
# nb_class_year.show()


# ## D - Les causes de feux

# ### Comparaison des incendies naturels, humains et sans cause connue

natural_count = fires.loc[fires['CAUSE_DESCR'] == 'Lightning']
human_count = fires.loc[~fires['CAUSE_DESCR'].isin(['Lightning', 'Missing/Undefined'])]
natural_sum = natural_count.shape[0]
human_sum = human_count.shape[0]
missing_count = fires.loc[fires['CAUSE_DESCR'] == 'Missing/Undefined']
missing_sum = missing_count.shape[0]
# Créer un DataFrame contenant les sommes
data = {'Cause': ['Foudre', 'Facteurs humains', 'Missing/Undefined'], "Nombre d'incendies": [natural_sum, human_sum, missing_sum]}
df = pd.DataFrame(data)


bar_foudre = go.Bar(x=['Foudre'], y=[natural_sum], name='Foudre')
bar_human = go.Bar(x=['Facteurs humains'], y=[human_sum], name='Facteurs humains')
bar_missing = go.Bar(x = ['Missing/Undefined'], y=[missing_sum], name = 'Missing/Undefined')
data = [bar_foudre, bar_human, bar_missing]
cause_3_feu = go.Figure(data=data)
cause_3_feu.update_layout(
    title="Comparaison des incendies naturels, humains et sans cause connue",
    xaxis_title="Cause",
    yaxis_title="Nombre d'incendies",
    height=500, width=1200
)

# cause_3_feu.show()


# ### Distribution des feux par cause dite humaine

# human_fires = fires.loc[fires['CAUSE_DESCR']!= 'Lightning']
# human_fires.loc[:,'CAUSE_DESCR'] = human_fires['CAUSE_DESCR'].cat.remove_categories(['Lightning'])
# fig = px.histogram(human_fires, x='CAUSE_DESCR', color = 'CAUSE_DESCR', 
#                    color_discrete_map=dict_causes_colors,
#                    labels = {'CAUSE_DESCR':'Cause','count':'Nombre'}).update_xaxes(categoryorder='total descending')
# fig.update_layout(
#     title="Distribution des feux par cause dite 'humaine'",
#     xaxis_title="Causes humaines des feux",
#     yaxis_title="Nombre de feux", 
#     height=500, width=1200
# )

# fig.show()


# ### Décompte des feux : foudre VS humain

f_human = fires.loc[:,['DISC_YEAR', 'CAUSE_DESCR']]
f_human.loc[:,'human_cause'] = fires.loc[:,'CAUSE_DESCR'].apply(lambda x: 0 if x=='Lightning' else 1)
cause_count = pd.DataFrame(f_human['CAUSE_DESCR'].value_counts()).reset_index()
cause_count.head()
cause_count.loc[:,'human_cause'] = cause_count.loc[:,'CAUSE_DESCR'].apply(lambda x: 0 if x=='Lightning' else 1)
cause_count = cause_count.sort_values(['human_cause', 'count'], ascending = False)

nb_feu_cause = px.bar(cause_count, x='human_cause', y = 'count', color = cause_count['CAUSE_DESCR'],
             color_discrete_map=dict_causes_colors, labels={'CAUSE_DESCR': 'Cause', 'human_cause': 'Cause humaine'},
             hover_data={'human_cause':False}
            )

nb_feu_cause.update_layout(
    title="Comparaison du nombre d'incendies causés par la foudre et par des facteurs humains",
    xaxis_title="Cause",
    yaxis_title="Nombre d'incendies",
    legend=dict(
        traceorder='normal',  # Change the order to 'normal' for sorting
    ),
    height=600, width=1200
)

nb_feu_cause.update_xaxes(
    type='category', tickvals=[0, 1],  # Specify the positions of the ticks
    ticktext=['Lightning', 'Human_causes'],  # Specify the labels for the ticks
    title='Causes'
)


# ### Comparaison des surfaces cumulées : foudre VS humain

fire_size_cumul = fires.groupby('CAUSE_DESCR', observed=False)['SIZE'].sum().sort_values(ascending=False).reset_index()
fire_size_cumul['human_cause'] = fire_size_cumul['CAUSE_DESCR'].apply(lambda x: 0 if x=='Lightning' else 1)

surf_cum_cause = px.bar(fire_size_cumul, x='human_cause', y = 'SIZE',
             color = 'CAUSE_DESCR', color_discrete_map=dict_causes_colors,
             labels={'CAUSE_DESCR': 'Cause', 'human_cause': 'Cause humaine'},
             hover_data={'human_cause':False}
            )

surf_cum_cause.update_layout(
    title="Comparaison des surfaces cumulées brûlées par la foudre et par des facteurs humains",
    xaxis_title="Cause",
    yaxis_title="Surface cumulée (acres)",
    legend=dict(
        traceorder='normal',  # Change the order to 'normal' for sorting
    ),
    height=600, width=1200
)
surf_cum_cause.update_xaxes(type='category', tickvals=[0, 1],  # Specify the positions of the ticks
        ticktext=['Lightning', 'Human cause'],  # Specify the labels for the ticks
        title='Causes')

# surf_cum_cause.show()


# On constate que la surface cumulée de feux est majoritairement due à des causes naturelles (la foudre). 

# ### Surface cumulée de feu et nombre de feux en fonction de la cause du feu

fires_count = fires[['SIZE', 'CAUSE_DESCR']]
fires_count = fires_count.groupby('CAUSE_DESCR', observed=False)['CAUSE_DESCR'].count().sort_values(ascending=False).rename('COUNT')
fires_size_cumul = fires[['SIZE', 'CAUSE_CODE', 'CAUSE_DESCR']]
fires_size_cumul = fires_size_cumul.groupby('CAUSE_DESCR', observed=False)['SIZE'].sum().sort_values(ascending=False).rename('SIZE_CUMUL')


fires_size_count = pd.concat([fires_size_cumul, fires_count], axis=1)
fires_size_count.index.names = ['CAUSE']
# fires_size_count


nb_vs_surf_cum = go.Figure(
    data=[
        go.Bar(name='Surface cumulée'
               , x=fires_size_count.index, y=fires_size_count['SIZE_CUMUL']
               , yaxis='y', offsetgroup=1),
        go.Bar(name='Nombre de feux'
               , x=fires_size_count.index, y=fires_size_count['COUNT']
               , yaxis='y2', offsetgroup=2)
    ],
    layout={
        'xaxis': {'title': 'Causes', 'showgrid':False},  # Removes X-axis grid lines},
        'yaxis': {'title': 'Surface cumulée (acres)', 'showgrid':False}, # Removes X-axis grid lines},
        'yaxis2': {'title': 'Nombre de feux', 'overlaying': 'y', 'side': 'right', 'showgrid':False}  # Removes X-axis grid lines}
    })

# Change the bar mode
nb_vs_surf_cum.update_layout(
    title={
        'text': 'Mise en perspective du nombre de feux et de la surface cumulée brûlée par cause',
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}, 
    barmode='group', 
    legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.8),
    height=600, width=1200
)
# nb_vs_surf_cum.show()


# ### Distributions des classes de feu en fonction des causes de feu

# ### Distribution de la surface cumulée des feux par cause dite humaine

fires_class = fires[['CLASS', 'DISC_YEAR', 'STATE']]


fire_s_cumul = fires.groupby('CAUSE_DESCR', observed=False)['SIZE'].sum().sort_values(ascending=False).reset_index()


fires_cause = fires.CAUSE_DESCR.value_counts()
labels = ['Lightning','Miscellaneous','Arson','Missing/Undefined',\
          'Equipment Use','Debris Burning','Campfire','Powerline','Railroad','Smoking',\
          'Children','Fireworks','Structure']

# Create a pie chart
pie_surfcum_cause = go.Figure(data=[go.Pie(labels=labels, values=fire_s_cumul['SIZE'], pull=[0.2,0,0.2,0.4,0,0,0,0,0,0,0,0,0],
                             textinfo='percent', hole=0.3)])

# Update layout with title and legend
pie_surfcum_cause.update_layout(title='Proportion des surfaces cumulées brûlées par cause de feu du jeu de données',
                  legend=dict(title='Cause de feu',
                              yanchor="top",y=0.99,xanchor="right",x=1.40), 
                  width=900, height=900)

colors = [dict_causes_colors[label] for label in labels]
pie_surfcum_cause.update_traces(marker=dict(colors=colors))

# Show the plot
# pie_surfcum_cause.show()


# ## E - Les feux par états

# ### Proportion des Etats dans chaque Classe de feu

# ### Nombre de feux par Etat avec répartition des causes

nb_state_cause = px.histogram(fires, x="STATE", color="CAUSE_DESCR", 
                   color_discrete_map=dict_causes_colors, labels={'CAUSE_DESCR': 'Cause', 'STATE': 'State'}).update_xaxes(categoryorder='total descending')
nb_state_cause.update_layout(
    title=dict(text="Nombre de feux par Etat avec répartition des causes"),
    xaxis=dict(title="Etat", tickfont_size=10),
    yaxis=dict(title="Nombre de feux"),
    height=600, width=1200
)   
# nb_state_cause.show()


# On constate que la répartition des causes de feux peut grandement varier d'un Etat à un autre.

# ### Surface cumulée des feux par Etat avec répartition des causes

size_cumul = fires.groupby(['STATE', 'CAUSE_DESCR'], observed=False)['SIZE'].sum().rename('SIZE_CUMUL')
size_cumul = pd.DataFrame(size_cumul).reset_index()


surfcum_state_cause = px.histogram(
    size_cumul, x="STATE", y='SIZE_CUMUL', color="CAUSE_DESCR",
    color_discrete_map=dict_causes_colors, 
    labels={'CAUSE_DESCR': 'Cause', 'STATE': 'State', 'SIZE_CUMUL': 'Surface cumulée'})\
                                                            .update_xaxes(categoryorder='total descending')
surfcum_state_cause.update_layout(
    title=dict(text="Surface cumulée de feu par Etat avec répartition des causes"),
    xaxis=dict(title="Etat", tickfont_size=10),   
    yaxis=dict(title="Surface cumulée (acres)"),
    height=600, width=1200
)
# surfcum_state_cause.show()


# On constate que la majeure partie de la surface cumulée brûlée l'est à cause de la foudre.

# ## F - Evolution temporelle des feux

# ### Nombre de feux par an

# plt.figure(figsize=(12, 6))
# plt.hist(fires['DISC_YEAR'], bins=24, edgecolor='black')
# plt.title("Nombre de feu par an")
# plt.xlabel('Années')
# plt.ylabel("Nombre d'occurences")
# plt.show()


fires_counts_per_year = fires['DISC_YEAR'].value_counts().sort_index()


# fires_counts_per_year.index, fires_counts_per_year.values


# ### Nombre de feu par jour de l'année

val_count = fires['DISC_DOY'].value_counts()


# # Nombre de feux par jour de l'année
# plt.figure(figsize=(10, 6))
# sns.lineplot(x=val_count.index, y=val_count.values, errorbar=None)
# plt.title("Nombre de feux par jour de l'année")
# plt.xlabel('Jour de l'année')
# plt.ylabel("Nombre de feux")
# plt.show()


# ### Evolution temporelle de la superficie totale des feux  

# ### Surface cumulée de feu et nombre de feux sur la durée

fires_count = fires[['DISC_YEAR']]
fires_count = fires_count.groupby('DISC_YEAR')['DISC_YEAR'].count().sort_values(ascending=False).rename('COUNT')
fires_size_cumul = fires[['SIZE', 'DISC_YEAR']]
fires_size_cumul = fires_size_cumul.groupby('DISC_YEAR')['SIZE'].sum().sort_values(ascending=False).rename('SIZE_CUMUL')


fires_size_count = pd.concat([fires_size_cumul, fires_count], axis=1)
fires_size_count.index.names = ['YEAR']
fires_size_count = fires_size_count.sort_index()


# ## G - Saisonnalité des feux

# ### Distribution des classes de feux selon le jour de l'année des feux

# for fire_class in fires['CLASS'].unique().sort_values():
#     class_count = fires.loc[fires['CLASS'] == fire_class]
    
#     fig2 = px.histogram(class_count, x='DISC_DOY')
    
#     fig = go.Figure(data=fig2.data)
    
#     fig.update_layout(
#         title=f"Distribution des feux de classe {fire_class} selon le jour de l'année",
#         yaxis_title="Nombre de feux",
#         xaxis=dict(
#             title="Mois de l'année",
#             tickvals=[15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 350],
#             ticktext=['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Aou', 'Sep', 'Oct', 'Nov', 'Déc']
#         ),
#         barmode='overlay', height=500, width=1200
#     )
    
#     fig.show()


# Classe A > forte saisonnalité en été 
# 
# Classe B > forte saisonnalité au printemps
# 
# Classe C > forte saisonnalité au printemps
# 
# Classe D > forte saisonnalité au printemps puis en été
# 
# Classe E > forte saisonnalité en été puis au printemps
# 
# Classe F > forte saisonnalité en été
# 
# Classe G > forte saisonnalité en été
# 

# ### Distribution de la surface cumulée des feux de classe X selon le jour de l’année

# ### Saisonnalité du nombre de feux du fait de l'Homme

fires_saison_hum = fires.loc[fires['CAUSE_DESCR'] != 'Lightning'].groupby(['DISC_YEAR','DISC_MONTH'])['FOD_ID'].count().rename('COUNT')
fires_saison_hum = pd.DataFrame(fires_saison_hum).reset_index()


fires_saison_hum = pd.pivot(fires_saison_hum, index='DISC_MONTH', columns='DISC_YEAR', values='COUNT')
fires_saison_hum.reset_index()


# fires_saison_hum.columns


hmap_nb_human = px.imshow(fires_saison_hum,
                labels=dict(x="Année", y="Mois", color="Nombre"),
                x = fires_saison_hum.columns,
                y = ['Jan', 'Fév', "Mar", 'Avr', 'Mai', 'Juin', 'Juil', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
               )
hmap_nb_human.update_layout(
    title={
        'text': "Saisonnalité du nombre de feux du fait de l'Homme",
        'font_family': "verdana",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    height=600, width=1000
)
hmap_nb_human.update_xaxes(tickangle=0, tickfont=dict(family="verdana", color='black', size=12), dtick=2, title_text=None)
hmap_nb_human.update_yaxes(tickfont=dict(family="verdana", color='black'), title_text=None)
# hmap_nb_human.show()


# ##### Foudre

fires_saison_nat = fires.loc[fires['CAUSE_DESCR'] == 'Lightning'].groupby(['DISC_YEAR','DISC_MONTH'])['FOD_ID'].count().rename('COUNT')
fires_saison_nat = pd.DataFrame(fires_saison_nat).reset_index()
fires_saison_nat = pd.pivot(fires_saison_nat, index='DISC_MONTH', columns='DISC_YEAR', values='COUNT')
fires_saison_nat.reset_index()


hmap_nb_natural = px.imshow(fires_saison_nat,
                labels=dict(x="Année", y="Mois", color="Nombre"),
                x = fires_saison_nat.columns,
                y = ['Jan', 'Fév', "Mar", 'Avr', 'Mai', 'Juin', 'Juil', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
               )
hmap_nb_natural.update_layout(
    title={
        'text': 'Saisonnalité du nombre de feux du fait de la foudre',
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    height=600, width=1000
)
hmap_nb_natural.update_xaxes(tickangle=0, tickfont=dict(family="verdana", color='black', size=12), dtick=2, title_text=None)
hmap_nb_natural.update_yaxes(tickfont=dict(family="verdana", color='black'), title_text=None)
# hmap_nb_natural.show()


human_count = fires.loc[~fires['CAUSE_DESCR'].isin(['Lightning','Missing/Undefined'])]
fig1 = px.histogram(human_count, x='DISC_DOY', color_discrete_sequence=[dict_causes_colors['Human']])
fig2 = px.histogram(natural_count, x='DISC_DOY',color_discrete_sequence=[dict_causes_colors['Lightning']])

day_human_natural = go.Figure(data = fig1.data + fig2.data)

day_human_natural.update_layout(
    	title={
        	'text': "Distribution selon le jour de l'année des feux : foudre VS causes humaines<br>(sans Missing/Undefined de la catégorie 'Human')",
        	'y':0.95,
        	'x':0.5,
        	'xanchor': 'center',
        	'yanchor': 'top'},
    yaxis_title="Nombre de feux", 
    xaxis=dict(title="Mois de l'année",
               tickvals= [15,  46,  74, 105, 135, 166, 196, 227, 258, 288, 319, 350],
               ticktext = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Aou', 'Sep', 'Oct', 'Nov', 'Déc']),
    barmode='overlay', 
    height=600, width=1000
)

# day_human_natural.show()


# ## H - Géographie

# ### Distribution de la latitude en fonction de la classe de feu

fires_lat = fires[['CLASS', 'LAT']]

lat_class = px.box(fires_lat, x='CLASS', y = 'LAT', color='CLASS',
         	category_orders={"CLASS": ["A", "B", "C", "D", "E", "F", "G"]},
         	height=600, width=1000
)

lat_class.update_layout(
	title={
    	'text': "Distribution de la latitude par classe de feu",
    	'y':0.95,
    	'x':0.5,
    	'xanchor': 'center',
    	'yanchor': 'top'},
	xaxis={'title': "Classe de feu", 'categoryorder':'category ascending'},
	yaxis_title="Latitude (°)",
	showlegend=False
)


# ### Distribution de la longitude en fonction de la classe de feu

fires_lon = fires[['CLASS', 'LON']]

lon_class = px.box(fires_lon, x='CLASS', y = 'LON', color='CLASS',
         	category_orders={"CLASS": ["A", "B", "C", "D", "E", "F", "G"]},
         	height=600, width=1000
)

lon_class.update_layout(
	title={
    	'text': "Distribution de la longitude par classe de feu",
    	'y':0.95,
    	'x':0.5,
    	'xanchor': 'center',
    	'yanchor': 'top'},
	xaxis={'title': "Classe de feu", 'categoryorder':'category ascending'},
	yaxis_title="Longitude (°)",
	showlegend=False
)








# <span style="color:#cc4125;font-size:2em">--- Dataset végétation et météo ---</span>

# ## A - Vent et classe de feu

fires_veg_wind = fires_veg[['CLASS', 'WIND']]

wind_class = px.box(fires_veg_wind, x='CLASS', y='WIND', color='CLASS',
             labels={'WIND': 'Vitesse du vent', 'CLASS': 'Classe de feu'},
             category_orders=dict(CLASS=["A", "B", "C", "D", "E", "F", "G"]))

wind_class.update_layout(
    title={
        'text': "Vitesse du vent (m/s, moyenné sur le mois) en fonction des classes de Feu",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}, 
    xaxis={'title': "Classe de feu", 'categoryorder':'category ascending'},
    yaxis_title='Vitesse du vent moyennée sur le mois (m/s)',
    showlegend=False,
    width=1000,  
    height=600  
)

# wind_class.show()


# ## B - Humidité et classe de feu

fires_veg_moisture = fires_veg[['CLASS', 'FUEL_MOISTURE']]

veg_moist_class = px.box(fires_veg_moisture, x='CLASS', y = 'FUEL_MOISTURE', color='CLASS',
             category_orders={"CLASS": ["A", "B", "C", "D", "E", "F", "G"]},
             labels={'FUEL_MOISTURE': "Taux d'humidité du combustible végétal", 'CLASS': 'Classe de feu'}
            )

veg_moist_class.update_layout(
    title={
        'text': "Distribution de l'humidité du combustible végétal (%) par classe de feu",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}, 
    xaxis={'title':'Classe de feu', 'categoryorder':'category ascending'},
    yaxis_title="Taux d'humidité du combustible végétal (%)",
    showlegend=False,
    width=1000,  
    height=600
)
# veg_moist_class.show()


# ## C - Indice de biomasse et classe de feu

# fires_veg_biomass = fires_veg[['CLASS', 'NBCD_FIA_BIOMASS_MOSAIC']]

# biomass_class = px.box(fires_veg_biomass, x='CLASS', y='NBCD_FIA_BIOMASS_MOSAIC', color='CLASS',
#              category_orders={"CLASS": ["A", "B", "C", "D", "E", "F", "G"]},
#              labels={'NBCD_FIA_BIOMASS_MOSAIC': "Indice de biomasse", 'CLASS': 'Classe de feu'})

# biomass_class.update_layout(
#     title={
#         'text': "Indice de biomasse en fonction des classes de feu",
#         'y':0.95,
#         'x':0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'}, 
#     xaxis={'title': "Classe de feu", 'categoryorder':'category ascending'},
#     yaxis_title="Indice de biomasse",
#     showlegend=False,
#     width=1000,  
#     height=600  
# )

# biomass_class.show()


# ## D - Ecorégion et classe de feu

# fires_veg_reg = fires_veg[['CLASS', 'ECO_REG_LVL1']]

# reg_class = px.histogram(fires_veg_reg, x="CLASS",
#                    color="ECO_REG_LVL1",
#                    barnorm='percent', text_auto='.2f',
#                    category_orders={"CLASS": ["A", "B", "C", "D", "E", "F", "G"]}
#                   )
# reg_class.update_layout(
#     title={
#         'text': "Eco-région de niveau 1 en fonction des classes de feu",
#         'y':0.95,
#         'x':0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'}, 
#     xaxis={'title': "Classes de feu", 'categoryorder':'category ascending'},
#     yaxis_title="Eco-région niveau 1",
#     width=1000,  
#     height=600,
# )


# ## E - Végétation et classe de feu

fires_veg_veg = fires_veg[['CLASS', 'VEGETATION']]

veg_class = px.histogram(fires_veg_veg, x="CLASS",
                   color="VEGETATION",
                   barnorm='percent', text_auto='.2f',
                   category_orders=dict(CLASS=["A", "B", "C", "D", "E", "F", "G"]))
veg_class.update_layout(
    title={
        'text': "Végétation en fonction des classes de feu",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}, 
    xaxis={'title':'Classe de feu', 'categoryorder':'category ascending'},
    yaxis_title="Végétation",
    width=1000,  
    height=600
)

# veg_class.show()




