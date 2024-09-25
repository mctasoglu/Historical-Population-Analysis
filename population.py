import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



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



def main():
    
    main_df = pd.read_csv(r"C:\Users\tasoglum\Desktop\Kaggle\population.csv")
    
    main_df['Population (historical)'] = main_df['Population (historical)']/100000
    # print(main_df[main_df['Entity'] == 'Turkey'])
    main_df = main_df[main_df['Entity'] != 'World']
    main_df.dropna(subset=['Code'], inplace=True)
    
    
    
    pop_series = make_global_pop_series(main_df)
    
 
    ottoman_countries = ['Bosnia and Herzegovina', 'Turkey', 'Albania', 'Bulgaria', 'North Macedonia', 'Greece', 'Romania', 
                         'Hungary', 'Moldova','Montenegro', 'Syria', 'Iraq', 'Kuwait', 'Lebanon', 'Israel', 'Egypt', 'Libya', 
                         'Tunisia', 'Algeria', 'Ukraine', 'Yemen', 'Saudi Arabia', 'Cyprus']
    
    roman_countries = ['Bosnia and Herzegovina', 'Croatia', 'Slovenia', 'Turkey', 'Albania', 'Bulgaria', 'North Macedonia', 
                        'Greece', 'Romania', 'Hungary', 'Montenegro', 'Syria', 'Lebanon', 'Israel', 'Egypt', 'Libya', 'Tunisia', 
                        'Algeria', 'Morocco', 'Italy', 'Spain', 'Portugal', 'France', 'Belgium', 'United Kingdom', 'Austria',
                        'Switzerland', 'Cyprus']
    
    
    
    non_ottoman_europe = ['France', 'Spain', 'Portugal', 'United Kingdom', 'Germany', 'Poland', 'Austria', 'Italy', 'Czechia', 
                          'Slovakia', 'Belgium', 'Netherlands', 'Russia']
    
   
    
    pop_ottoman_empire(main_df, ottoman_countries)
    
    # turk_df = pop_turkey(main_df)
    
    return 0

if __name__ == "__main__":
    main()