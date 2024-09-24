import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def pop_turkey(main_df):
    
    df = main_df[(main_df['Entity'] == 'Turkey') & (main_df['Year'] >= -1000) & (main_df['Year'] <= 1700)]
    
    df_modern = main_df[(main_df['Entity'] == 'Turkey') & (main_df['Year'] > 1700)]
    
    years = df['Year']
    population = df['Population (historical)']
    
    years_modern = df_modern['Year']
    population_modern = df_modern['Population (historical)']
    
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.scatter(years, population, marker='x', color='red')
    ax1.set(xlabel='Year', ylabel='Population (in hundred thousands)')
    ax1.title.set_text('Population of Anatolia from 1000 BC to 1700 AD')
    
    ax2.scatter(years_modern, population_modern, marker='x', color='red')
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
    # print(main_df.head())
    
    
    ottoman_countries = ['Bosnia and Herzegovina', 'Turkey', 'Albania', 'Bulgaria', 'North Macedonia', 'Greece', 'Romania', 
                         'Hungary', 'Moldova','Montenegro', 'Syria', 'Iraq', 'Kuwait', 'Lebanon', 'Israel', 'Egypt', 'Libya', 
                         'Tunisia', 'Algeria', 'Ukraine', 'Yemen', 'Saudi Arabia']
    
    roman_countries = ['Bosnia and Herzegovina', 'Croatia', 'Slovenia', 'Turkey', 'Albania', 'Bulgaria', 'North Macedonia', 
                        'Greece', 'Romania', 'Hungary', 'Montenegro', 'Syria', 'Lebanon', 'Israel', 'Egypt', 'Libya', 'Tunisia', 
                        'Algeria', 'Morocco', 'Italy', 'Spain', 'Portugal', 'France', 'Belgium', 'United Kingdom', 'Austria',
                        'Switzerland']
    
    turk_df = pop_turkey(main_df)
    
    return 0

if __name__ == "__main__":
    main()