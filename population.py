import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score


def get_global_pop_at_year(df, yr):
    
    pop_at_year = (df[df['Year'] == yr]['Population (historical)'].sum()) 

    return pop_at_year



def make_global_pop_series(df):
    
   
    df_country = df[df['Entity'] == 'Turkey']
    
    
    pops = [get_global_pop_at_year(df, yr) for yr in df_country['Year']]
    
    d = {'Year' : df_country['Year'], 'Population': pops}
    df_pop = pd.DataFrame(data=d)
    df_pop.reset_index(inplace=True)
   
    
    
    # global_pops = df['Year'].map(lambda yr : get_global_pop_at_year(df, yr))

    return df_pop

def pop_ottoman_empire(main_df, ottoman_countries):
    
    ottoman_countries_1500 = [ctry for ctry in ottoman_countries if ctry not in ['Hungary', 'Syria', 'Iraq', 'Lebanon', 'Egypt', 'Israel', 'Tunisia', 'Libya', 'Algeria', 'Kuwait', 'Saudi Arabia', 'Yemen', 'Cyprus']]
    ottoman_countries_1800 = [ctry for ctry in ottoman_countries if ctry not in ['Hungary', 'Ukraine', 'Montenegro']]
    ottoman_countries_1900 = [ctry for ctry in ottoman_countries if ctry not in ['Hungary', 'Ukraine', 'Montenegro', 'Bosnia and Herzegovina', 'Egypt', 'Algeria', 'Tunisia', 'Cyprus', 'Bulgaria', 'Cyprus', 'Serbia', 'Romania', 'Moldova', 'Greece']]
    
    
    
    ottoman_df = main_df[main_df['Entity'].isin(ottoman_countries)]
    ottoman_df_1500 = main_df[main_df['Entity'].isin(ottoman_countries_1500)]
    ottoman_df_1800 = main_df[main_df['Entity'].isin(ottoman_countries_1800)]
    ottoman_df_1900 = main_df[main_df['Entity'].isin(ottoman_countries_1900)]
    
    print(ottoman_df_1900[ottoman_df_1900['Year'] == 1900])
    
    
    # ottoman_pop_1500 = ottoman_df[ottoman_df['Year'] == 1500]['Population (historical)'].sum()
    ottoman_pop_1500 = ottoman_df_1500[ottoman_df_1500['Year'] == 1500]['Population (historical)'].sum()
    france_pop_1500 = main_df[(main_df['Entity'] == 'France') & (main_df['Year'] == 1500)]['Population (historical)'].values[0]
    germany_pop_1500 = main_df[(main_df['Entity'] == 'Germany') & (main_df['Year'] == 1500)]['Population (historical)'].values[0]
    
    ottoman_pop_1600 = ottoman_df[ottoman_df['Year'] == 1600]['Population (historical)'].sum()
    france_pop_1600 = main_df[(main_df['Entity'] == 'France') & (main_df['Year'] == 1600)]['Population (historical)'].values[0]
    germany_pop_1600 = main_df[(main_df['Entity'] == 'Germany') & (main_df['Year'] == 1600)]['Population (historical)'].values[0]
    
    ottoman_pop_1700 = ottoman_df[ottoman_df['Year'] == 1700]['Population (historical)'].sum()
    france_pop_1700 = main_df[(main_df['Entity'] == 'France') & (main_df['Year'] == 1700)]['Population (historical)'].values[0]
    germany_pop_1700 = main_df[(main_df['Entity'] == 'Germany') & (main_df['Year'] == 1700)]['Population (historical)'].values[0]
    
    ottoman_pop_1800 = ottoman_df_1800[ottoman_df_1800['Year'] == 1800]['Population (historical)'].sum()
    france_pop_1800 = main_df[(main_df['Entity'] == 'France') & (main_df['Year'] == 1800)]['Population (historical)'].values[0]
    germany_pop_1800 = main_df[(main_df['Entity'] == 'Germany') & (main_df['Year'] == 1800)]['Population (historical)'].values[0]
    
    ottoman_pop_1900 = ottoman_df_1900[ottoman_df_1900['Year'] == 1900]['Population (historical)'].sum()
    france_pop_1900 = main_df[(main_df['Entity'] == 'France') & (main_df['Year'] == 1900)]['Population (historical)'].values[0]
    germany_pop_1900 = main_df[(main_df['Entity'] == 'Germany') & (main_df['Year'] == 1900)]['Population (historical)'].values[0]
    
    plt.scatter([1500, 1600, 1700, 1800, 1900], [ottoman_pop_1500, ottoman_pop_1600, ottoman_pop_1700, ottoman_pop_1800, ottoman_pop_1900], marker='x', color='green')
    plt.scatter([1500, 1600, 1700, 1800, 1900], [france_pop_1500, france_pop_1600, france_pop_1700, france_pop_1800, france_pop_1900], marker='x', color='blue')
    plt.scatter([1500, 1600, 1700, 1800, 1900], [germany_pop_1500, germany_pop_1600, germany_pop_1700, germany_pop_1800, germany_pop_1900], marker='x', color='black')
    plt.xlabel('Year')
    plt.ylabel('Population (in hundred thousands)')
    
    plt.legend(["Ottoman Empire" , "France", "Germany"])
    
    plt.show()
    
    
    
    
    return 0



def pop_turkey(main_df):
    
    # df = main_df[(main_df['Entity'] == 'Turkey') & (main_df['Year'] >= -1000) & (main_df['Year'] <= 1700)]
    
    df = main_df[(main_df['Entity'] == 'Turkey') & (main_df['Year'] <= 1700)]
    
    df_modern = main_df[(main_df['Entity'] == 'Turkey') & (main_df['Year'] > 1700)]
    
    
    
    global_population = make_global_pop_series(main_df)
    
    global_pop_old = global_population[global_population['Year'] <= 1700]
    global_pop_modern = global_population[global_population['Year'] > 1700]
    
    years = df['Year']
    population = df['Population (historical)']
    
    years_modern = df_modern['Year']
    population_modern = df_modern['Population (historical)']
    
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.scatter(years, population, marker='x', color='red')
    ax1.scatter(years, global_pop_old['Population'], marker='o', color='green')
    ax1.set(xlabel='Year', ylabel='Population (in hundred thousands)')
    ax1.title.set_text('Population of Anatolia from 1000 BC to 1700 AD')
    
    ax2.scatter(years_modern, population_modern, marker='x', color='red')
    ax2.scatter(years_modern, global_pop_modern['Population'], marker='o', color='green')
    ax2.set(xlabel='Year', ylabel='Population (in hundred thousands)')
    ax2.title.set_text('Population of Anatolia from 1700 AD to 2023 AD')
 

    pop_turkey_0 = df[df['Year'] == 0]['Population (historical)'].values[0]*100000
    pop_turkey_1600 = df[df['Year'] == 1600]['Population (historical)'].values[0]*100000
    print(f"Population of Turkey in 0 is {pop_turkey_0}")
    print(f"Population of Turkey in 1600 is {pop_turkey_1600}")
   
   
 
    plt.show()
    
    
    
    return 0



def create_x_y(df_pop, df_gdp, df_poli_idx, df_military_spending):
    
    
    x_y_dataset = pd.DataFrame()
    
    
    
    years = df_pop[ (df_pop['Year'] >= 1820) & (df_pop['Year'] < 1900)]['Year']
   
    
    france_pop = df_pop[ (df_pop['Year'] >= 1820) & (df_pop['Year'] < 1900)]['France']
    france_pop = france_pop.map(lambda x : x*1000)
    france_gdp = df_gdp[ (df_gdp['Entity'] == 'France') & (df_gdp['Year'] >= 1820) & (df_gdp['Year'] < 1900) ]['Gross domestic product (GDP)']
 
    france_poli_idx = df_poli_idx[ (df_poli_idx['Entity'] == 'France') & (df_poli_idx['Year'] >= 1820) & (df_poli_idx['Year'] < 1900) ]['Liberal component index (best estimate)']
    france_military_spending = df_military_spending[ (df_military_spending['Entity'] == 'France') & (df_military_spending['Year'] >= 1820) & (df_military_spending['Year'] < 1900) ][r"Military expenditure (% of GDP)"]
    print(france_gdp)
    
    x_y_dataset['France Population'] = france_pop
    x_y_dataset.set_index(years, inplace=True)
    
    france_gdp.index = x_y_dataset.index
    france_poli_idx.index = x_y_dataset.index
    france_military_spending.index = x_y_dataset.index
    
    x_y_dataset['France GDP'] = france_gdp
    
    x_y_dataset[r"France Military Spending (% of GDP)"] = france_military_spending
    x_y_dataset['France Political Index'] = france_poli_idx
    
    
    # plt.scatter(x_y_dataset['France Population'], x_y_dataset['France GDP'], marker='x', color='blue')
    # plt.xlabel('Population')
    # plt.ylabel('GDP')
    # plt.title('Population vs GDP in 19th Century France')
    
    # plt.show()
    
    
    
    
    
    
    
    return x_y_dataset


def linear_analysis(dataset):
    
    
    
    X_train = dataset.loc[1820:1859, "France GDP" : r'France Military Spending (% of GDP)']
    X_test = dataset.loc[1860:1899:, "France GDP" : r'France Military Spending (% of GDP)']
    
    y_train = dataset.loc[1820:1859, 'France Political Index']
    y_test = dataset.loc[1860:1899, 'France Political Index']
    
    regr = linear_model.LinearRegression()
    
    regr.fit(X_train, y_train)
    
    y_pred = regr.predict(X_test)
    
    # The coefficients
    print("Coefficients: \n", regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
    
    fig, axs = plt.subplots(1,2, sharey=True)
    
    cols = X_train.columns
    print(dataset.tail(n=10))
    
    for i in range(len(cols)):
        axs[i].scatter(X_train[cols[i]],y_train, label = 'target')
        axs[i].set_xlabel(cols[i])
        axs[i].scatter(X_train[cols[i]],y_pred,color="orange", label = 'predict')
    
    axs[0].set_ylabel("Political Index"); axs[0].legend()
    
    plt.show()
    # Plot outputs
    # plt.scatter(X_test, y_test, color="black")
    # plt.plot(X_test, y_pred, color="blue", linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    
    
    
    return 0


def open_datasheets():
    
        
    maddison_pop = pd.read_excel(r"C:\Users\tasoglum\Desktop\Kaggle\mpd2023_web.xlsx", sheet_name='Population')
    
    maddison_gdp = pd.read_csv(r"C:\Users\tasoglum\Desktop\Kaggle\gdp-maddison-project-database.csv")
    
    poli_idx = pd.read_csv(r"C:\Users\tasoglum\Desktop\Kaggle\liberal-political-institutions-index.csv")
    
    military_spending = pd.read_csv(r"C:\Users\tasoglum\Desktop\Kaggle\military-spending-as-a-share-of-gdp-gmsd.csv")
    
    
    
    og_headers = maddison_pop.columns
    
    maddison_pop.drop([0,1], inplace=True)
    maddison_pop.columns = og_headers
    
    maddison_pop.rename({"Population": "Year"}, axis=1, inplace=True)
    maddison_pop.reset_index(inplace=True)
    
    france_datasheet = create_x_y(maddison_pop, maddison_gdp, poli_idx, military_spending)
    
    linear_analysis(france_datasheet)
    
    
    
    return 0




def main():
    
    main_df = pd.read_csv(r"C:\Users\tasoglum\Desktop\Kaggle\population.csv")
    
    main_df['Population (historical)'] = main_df['Population (historical)']/100000
    # print(main_df[main_df['Entity'] == 'Turkey'])
    main_df = main_df[main_df['Entity'] != 'World']
    main_df.dropna(subset=['Code'], inplace=True)
    
    
    
    open_datasheets()
    
    # pop_series = make_global_pop_series(main_df)
    
 
    # ottoman_countries = ['Bosnia and Herzegovina', 'Turkey', 'Albania', 'Bulgaria', 'North Macedonia', 'Greece', 'Romania', 
    #                      'Hungary', 'Moldova','Montenegro', 'Syria', 'Iraq', 'Kuwait', 'Lebanon', 'Israel', 'Egypt', 'Libya', 
    #                      'Tunisia', 'Algeria', 'Ukraine', 'Yemen', 'Saudi Arabia', 'Cyprus']
    
    # roman_countries = ['Bosnia and Herzegovina', 'Croatia', 'Slovenia', 'Turkey', 'Albania', 'Bulgaria', 'North Macedonia', 
    #                     'Greece', 'Romania', 'Hungary', 'Montenegro', 'Syria', 'Lebanon', 'Israel', 'Egypt', 'Libya', 'Tunisia', 
    #                     'Algeria', 'Morocco', 'Italy', 'Spain', 'Portugal', 'France', 'Belgium', 'United Kingdom', 'Austria',
    #                     'Switzerland', 'Cyprus']
    
    
    
    # non_ottoman_europe = ['France', 'Spain', 'Portugal', 'United Kingdom', 'Germany', 'Poland', 'Austria', 'Italy', 'Czechia', 
    #                       'Slovakia', 'Belgium', 'Netherlands', 'Russia']
    
   
    
    # pop_ottoman_empire(main_df, ottoman_countries)
    
    # turk_df = pop_turkey(main_df)
    
    return 0

if __name__ == "__main__":
    main()