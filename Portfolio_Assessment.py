#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Necessary imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go
import pandas_datareader as pdr
from datetime import datetime
from plotly.offline import plot


# In[2]:


# required inputs. Tickers can be read in as a text file as well.
Tickers = ['LDVAX', 'BGSAX', 'OPGIX', 'PJFZX', 'PGTAX', 'PSGAX', 'WAMCX']
Start_date = '2018-01-01'
End_date = '2024-08-15'
dates = pd.date_range(Start_date, End_date)
start_date = datetime(2018, 1, 1)
end_date = datetime(2024, 8, 15)
Principle = 24000
Allocations = [0.207, 0.165, 0.071, 0.103, 0.185, 0.103, 0.166]


# In[3]:


def plot_selected(df, columns, start_index, end_index):
    # Plot the desired columns over a desired index range
    plot_data(df.loc[start_index:end_index, columns], title="Selected Data")

def get_data(symbols, dates):
    df = pd.DataFrame(index=dates)  # Creates empty DF with index = dates
    
    spy_added = False  # Flag to check if SPY was added
    
    if 'SPY' not in symbols:  # Uses SPY as a reference, this is unnecessary
        symbols.insert(0, 'SPY')
        spy_added = True  # Set flag to True if SPY was added
    
    for symbol in symbols:
        df_temp = yf.download(symbol, start=dates.min(), end=dates.max())[['Adj Close']]
        df_temp = df_temp.rename(columns={'Adj Close': symbol})  # Rename adj close to symbol name before joining 
        df = df.join(df_temp)  # Joining the empty DF with dates to our data
        if symbol == 'SPY':
            df = df.dropna(subset=["SPY"])  # Drops all NaN values for SPY (reference col)
    
    df = normalize_dollar_data(df)
    return df, spy_added

def normalize_dollar_data(df):
    return df / df.iloc[0, :]

def plot_data(df, top_n):
    # Calculate the performance of each fund
    performance = df.iloc[-1] / df.iloc[0] - 1
    if top_n.lower() in ['a', 'all']:
        top_performers = df.columns
    else:
        top_n = int(top_n)
        top_performers = performance.nlargest(top_n).index
    
    if 'SPY' not in top_performers:
        top_performers = top_performers.insert(0, 'SPY')  # Ensure 'SPY' is always included
    
    df_top = df[top_performers]  # Filter the DataFrame to only include top performers
    
    # Subtract 1 from the performance values
    df_top = df_top - 1
    
    # Create the interactive plot
    fig = go.Figure()

    for col in df_top.columns:
        fig.add_trace(go.Scatter(x=df_top.index, y=df_top[col], mode='lines', name=col))

    fig.update_layout(
        title="Adj Close over Time",
        xaxis_title="Date",
        yaxis_title="Performance",
        hovermode="x unified"
    )

    # Use plotly.offline.plot to create a pop-up window
    plot(fig)
    
    #ax = df_top.plot(title="Adj Close over Time", fontsize=12)
    #ax.set_xlabel("Date")
    #ax.set_ylabel("Performance")
    #plt.show(block=False)

def TR3():
    df, spy_added = get_data(Tickers, dates)
    
    while True:
        top_n = input(f"Enter the number of top performers to display (1-{len(Tickers)}) or 'a'/'all' to display all: ")
        if top_n.lower() in ['a', 'all'] or (top_n.isdigit() and 1 <= int(top_n) <= len(Tickers)):
            break
        else:
            print(f"Invalid input. Please enter a number between 1 and {len(Tickers)}, or 'a'/'all'.")
    
    plot_data(df, top_n)
    
    return df, spy_added


# In[4]:


def Portfolio_Return(dfport, spy_added):
     # Make a copy of the DataFrame to avoid modifying the original
    dfport_copy = dfport.copy()
    
    # Remove SPY if it was not included in the ticker list (it was included as a comparison)
    if spy_added == True:
        dfport_copy.drop(columns=['SPY'], inplace=True)
    
    # Create a new DF that displays the dollar amount for each fund in the ticker list given the 
    # Principle amount and allocation percents
    dfport_a = dfport_copy*Principle*Allocations
    
    # Create a new column that sums the portfolio for each day
    dfport_a["Port. Total"] = dfport_a.sum(axis = 1)
    
    # Create a new column that calculates the daily percent return of the overal portfolio
    dfport_a["Daily Return"] = dfport_a["Port. Total"].diff()/dfport_a["Port. Total"]
    
    # Calculate the cummulitive return of the portfolio
    Cummulitive_Return = (dfport_a.iloc[-1]['Port. Total'] - dfport_a.iloc[0]['Port. Total'])/dfport_a.iloc[0]['Port. Total']
    
    return dfport_a, Cummulitive_Return


# In[5]:


def sharpe(dfport_a):
    Avg_Daily_Return = np.mean(dfport_a["Daily Return"]) # Average daily return
    SD_Daily_Return = np.std(dfport_a["Daily Return"]) # Standard deviation of daily return
    Sharpe_Ratio = Avg_Daily_Return / SD_Daily_Return # Daily sharpe ratio
    A_Sharpe_Ratio = (252**0.5) * Sharpe_Ratio # Anualized sharpe ratio
    return Sharpe_Ratio, A_Sharpe_Ratio


# In[6]:


def normalize_data(df):
    df = df/df.iloc[0, :]
    df -= 1
    return df


# In[7]:


def port_performance_plot(df):
    # Fetch SPY data
    spy = yf.download("SPY", start=start_date, end=end_date)
    dfspy = spy[['Close']].rename(columns={'Close': 'SPY'})

    # Normalize your portfolio data
    df = normalize_data(df)

    # Normalize SPY data
    dfspy = normalize_data(dfspy)
    
    # Retrieving inflation data and calculating the cumulative inflation.
    dfcpi = pdr.get_data_fred('CPIAUCSL', start_date, end_date)
    dfcpi['Cum. Inflation'] = dfcpi['CPIAUCSL'] / dfcpi['CPIAUCSL'].iloc[0] - 1

    # Align the indices
    dfcpi = dfcpi.reindex(df.index).fillna(method='ffill')

    # Create a DataFrame for plotting
    df_combined = pd.DataFrame({
        'Date': df.index,
        'Portfolio': df["Port. Total"],
        'SPY': dfspy["SPY"],
        'Cum. Inflation': dfcpi['Cum. Inflation']
    }).set_index('Date')

    # Create the interactive plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_combined.index, y=df_combined['Portfolio'], mode='lines', name='Portfolio', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df_combined.index, y=df_combined['SPY'], mode='lines', name='SPY', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df_combined.index, y=df_combined['Cum. Inflation'], mode='lines', name='Inflation', line=dict(color='red')))

    fig.update_layout(
        title="M.F. Performance vs. SPY vs. Inflation",
        xaxis_title="Date",
        yaxis_title="Percent Gain",
        hovermode="x unified"
    )

    plot(fig)


# In[ ]:





# In[ ]:





# In[8]:


def run_all():
    # Creating a DF for the input tickers over the input dates. SPY flag to see if SPY was added
    # as a reference, or if it was already a part of the ticker list.
    # Will also plot the portfolio funds against eachother and SPY to see the performance over time
    dfport, spy_added = TR3()
    
    # Returning the DF with additional metrics and the calculated cummulative return
    dfport_a, Cummulitive_Return = Portfolio_Return(dfport, spy_added)
    
    # Calculating the daily and anualized sharpe ratios
    Sharpe_Ratio, A_Sharpe_Ratio = sharpe(dfport_a)
    
    # Plotting the portfolio performance against SPY as a metric, and including cummulative inflation
    port_performance_plot(dfport_a)
    
    print(f'The cummulative return from your portfolio is {round(Cummulitive_Return*100, 2)}%.\nThe Daily Sharpe Ratio is {round(Sharpe_Ratio, 3)}. \nThe anualized sharpe ratio is {round(A_Sharpe_Ratio, 3)}.')


# In[9]:


run_all()

