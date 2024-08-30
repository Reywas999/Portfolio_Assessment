#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Necessary imports
import pandas as pd
import numpy as np
from tkhtmlview import HTMLLabel
import plotly.io as pio
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkcalendar import DateEntry
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go
import pandas_datareader as pdr
from datetime import datetime
from plotly.offline import plot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# In[ ]:


# Global variables
Tickers = []
csv_file = ""
start_date = None
end_date = None
dates = None
Principle = 0
Allocations = []
top_n = 1


# In[ ]:


class InvestmentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Portfolio Input App")

        # Tickers input
        tk.Label(root, text="Tickers (comma separated):").grid(row=0, column=0)
        self.tickers_entry = tk.Entry(root, width=50)
        self.tickers_entry.grid(row=0, column=1)

        # CSV file input
        tk.Label(root, text="Or select CSV file:").grid(row=1, column=0)
        self.csv_path = tk.Entry(root, width=50)
        self.csv_path.grid(row=1, column=1)
        tk.Button(root, text="Browse", command=self.browse_csv).grid(row=1, column=2)

        # Start date input
        tk.Label(root, text="Start Date:").grid(row=2, column=0)
        self.start_date_picker = DateEntry(root, width=12, background='darkblue', foreground='white', borderwidth=2)
        self.start_date_picker.grid(row=2, column=1)

        # End date input
        tk.Label(root, text="End Date:").grid(row=3, column=0)
        self.end_date_picker = DateEntry(root, width=12, background='darkblue', foreground='white', borderwidth=2)
        self.end_date_picker.grid(row=3, column=1)

        # Principle input
        tk.Label(root, text="Principle:").grid(row=4, column=0)
        self.principle_entry = tk.Entry(root)
        self.principle_entry.grid(row=4, column=1)

        # Allocations input
        tk.Label(root, text="Allocations (comma separated):").grid(row=5, column=0)
        self.allocations_entry = tk.Entry(root, width=50)
        self.allocations_entry.grid(row=5, column=1)
        
        # Top performers input
        tk.Label(root, text="Enter the number of top performers to display or 'a'/'all':").grid(row=6, column=0)
        self.top_performers_entry = tk.Entry(root)
        self.top_performers_entry.grid(row=6, column=1)

        # Submit button
        tk.Button(root, text="Submit", command=self.submit).grid(row=7, column=1)

    def browse_csv(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        self.csv_path.insert(0, filename)
        self.prompt_csv_allocations()

    def prompt_csv_allocations(self):
        self.allocations_prompt = tk.Toplevel(self.root)
        self.allocations_prompt.title("CSV Allocations")

        tk.Label(self.allocations_prompt, text="Does the CSV file contain allocations?").grid(row=0, column=0)
        self.allocations_var = tk.StringVar(self.allocations_prompt)
        self.allocations_var.set("Select One")  # default value

        tk.OptionMenu(self.allocations_prompt, self.allocations_var, "Select One", "Yes", "No").grid(row=0, column=1)
        tk.Button(self.allocations_prompt, text="Submit", command=self.check_csv_allocations).grid(row=1, column=1)

    def check_csv_allocations(self):
        global Tickers, Allocations
        if self.allocations_var.get() == "Select One":
            messagebox.showerror("Error", "Please select whether the CSV file contains allocations.")
            return

        filename = self.csv_path.get()
        try:
            df = pd.read_csv(filename)
            Tickers = df['Tickers'].tolist()
            if self.allocations_var.get() == "Yes" and 'Allocations' in df.columns:
                Allocations = df['Allocations'].tolist()
                self.allocations_entry.insert(0, ','.join(map(str, Allocations)))
                self.allocations_entry.config(state='disabled')
            else:
                self.allocations_entry.config(state='normal')
            self.allocations_prompt.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV file: {e}")

    def validate_tickers(self, tickers):
        invalid_tickers = []
        for ticker in Tickers:
            try:
                yf.Ticker(ticker).info
            except Exception as e:
                invalid_tickers.append(ticker)
        return invalid_tickers
    
    def validate_top_performers(self, top_n):
        top_n = self.top_performers_entry.get()
        tickers = self.tickers_entry.get().split(',')
        if top_n.lower() in ['a', 'all'] or (top_n.isdigit() and 1 <= int(top_n) <= len(tickers)):
            return True
        else:
            return False

    def submit(self):
        global Tickers, csv_file, start_date, end_date, Principle, Allocations, dates, top_n

        if not self.csv_path.get():
            Tickers = self.tickers_entry.get().split(',')

        csv_file = self.csv_path.get()
        start_date = self.start_date_picker.get_date()
        end_date = self.end_date_picker.get_date()
        Principle = int(self.principle_entry.get())
        if self.top_performers_entry.get() in ['all', 'a']:
            top_n = int(len(Tickers))
        else:
            top_n = int(self.top_performers_entry.get())

        if self.allocations_entry.cget('state') == 'normal':
            Allocations = list(map(float, self.allocations_entry.get().split(',')))

        invalid_tickers = self.validate_tickers(Tickers)
        if invalid_tickers:
            messagebox.showerror("Error", f"Invalid tickers: {', '.join(invalid_tickers)}. Please correct them and submit again.")
            return

        if start_date >= end_date:
            messagebox.showerror("Error", "Start date must be before end date and they cannot be the same day.")
            return

        if end_date > datetime.now().date():
            messagebox.showerror("Error", "End date cannot be in the future.")
            return

        if self.allocations_entry.cget('state') == 'normal' and sum(Allocations) != 1:
            messagebox.showerror("Error", "Allocations must sum to 1.")
            return
        
        top_n_valid = self.validate_top_performers(top_n)
        if top_n_valid == False:
            messagebox.showerror(messagebox.showerror("Error", f"Invalid input. Please enter a number between 1 and {len(tickers)}, or 'a'/'all'."))
            return

        # Convert start_date and end_date to datetime objects
        start_date_dt = datetime(start_date.year, start_date.month, start_date.day)
        end_date_dt = datetime(end_date.year, end_date.month, end_date.day)

        # Create date range
        dates = pd.date_range(start=start_date_dt, end=end_date_dt)

        # Process the inputs as needed
        print("Tickers:", Tickers)
        print("Start Date:", start_date_dt)
        print("End Date:", end_date_dt)
        print("Principle:", Principle)
        print("Allocations:", Allocations)

        self.root.destroy()  # Close the app


# In[ ]:


class FinancialAnalysis:
    def __init__(self, tickers, dates, principle, allocations):
        self.tickers = tickers
        self.dates = dates
        self.principle = principle
        self.allocations = allocations
        self.df = None
        self.spy_added = False

    def plot_selected(self, columns, start_index, end_index):
        self.plot_data(self.df.loc[start_index:end_index, columns], title="Selected Data")

    def get_data(self):
        df = pd.DataFrame(index=self.dates)
        spy_added = False

        if 'SPY' not in self.tickers:
            self.tickers.insert(0, 'SPY')
            spy_added = True

        for symbol in self.tickers:
            df_temp = yf.download(symbol, start=self.dates.min(), end=self.dates.max())[['Adj Close']]
            df_temp = df_temp.rename(columns={'Adj Close': symbol})
            df = df.join(df_temp)
            if symbol == 'SPY':
                df = df.dropna(subset=["SPY"])

        df = self.normalize_dollar_data(df)
        self.df = df
        self.spy_added = spy_added

    def normalize_dollar_data(self, df):
        return df / df.iloc[0, :]

    def plot_data(self, df):
        performance = df.iloc[-1] / df.iloc[0] - 1
        if top_n == len(Tickers):
            top_performers = df.columns
        else:
            top_performers = performance.nlargest(top_n).index

        if 'SPY' not in top_performers:
            top_performers = top_performers.insert(0, 'SPY')

        df_top = df[top_performers]
        df_top = df_top - 1

        fig = go.Figure()

        for col in df_top.columns:
            fig.add_trace(go.Scatter(x=df_top.index, y=df_top[col], mode='lines', name=col))

        fig.update_layout(
            title="Adj Close over Time",
            xaxis_title="Date",
            yaxis_title="Performance",
            hovermode="x unified"
        )

        return fig

    def TR3(self):
        self.get_data()
        fig = self.plot_data(self.df)
        return self.df, self.spy_added, fig

    def Portfolio_Return(self, dfport, spy_added):
        dfport_copy = dfport.copy()

        if spy_added:
            dfport_copy.drop(columns=['SPY'], inplace=True)

        dfport_a = dfport_copy * self.principle * self.allocations
        dfport_a["Port. Total"] = dfport_a.sum(axis=1)
        dfport_a["Daily Return"] = dfport_a["Port. Total"].diff() / dfport_a["Port. Total"]
        Cummulitive_Return = (dfport_a.iloc[-1]['Port. Total'] - dfport_a.iloc[0]['Port. Total']) / dfport_a.iloc[0]['Port. Total']

        return dfport_a, Cummulitive_Return

    def sharpe(self, dfport_a):
        Avg_Daily_Return = np.mean(dfport_a["Daily Return"])
        SD_Daily_Return = np.std(dfport_a["Daily Return"])
        Sharpe_Ratio = Avg_Daily_Return / SD_Daily_Return
        A_Sharpe_Ratio = (252**0.5) * Sharpe_Ratio
        return Sharpe_Ratio, A_Sharpe_Ratio

    def normalize_data(self, df):
        df = df / df.iloc[0, :]
        df -= 1
        return df

    def port_performance_plot(self, df, start_date, end_date):
        spy = yf.download("SPY", start=start_date, end=end_date)
        dfspy = spy[['Close']].rename(columns={'Close': 'SPY'})

        df = self.normalize_data(df)

        dfspy = self.normalize_data(dfspy)

        dfcpi = pdr.get_data_fred('CPIAUCSL', start_date, end_date)
        dfcpi['Cum. Inflation'] = dfcpi['CPIAUCSL'] / dfcpi['CPIAUCSL'].iloc[0] - 1

        dfcpi = dfcpi.reindex(df.index).fillna(method='ffill')

        df_combined = pd.DataFrame({
            'Date': df.index,
            'Portfolio': df["Port. Total"],
            'SPY': dfspy["SPY"],
            'Cum. Inflation': dfcpi['Cum. Inflation']
        }).set_index('Date')

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df_combined.index, y=df_combined['Portfolio'], mode='lines', name='Portfolio', line=dict(color='black')))
        fig.add_trace(go.Scatter(x=df_combined.index, y=df_combined['SPY'], mode='lines', name='SPY', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df_combined.index, y=df_combined['Cum. Inflation'], mode='lines', name='Inflation', line=dict(color='red')))

        fig.update_layout(
            title="Port. Performance vs. SPY vs. Inflation",
            xaxis_title="Date",
            yaxis_title="Percent Gain",
            hovermode="x unified"
        )

        return fig

    def run_all(self, start_date, end_date):
        dfport, spy_added, fig1 = self.TR3()
        dfport_a, Cummulitive_Return = self.Portfolio_Return(dfport, spy_added)
        Sharpe_Ratio, A_Sharpe_Ratio = self.sharpe(dfport_a)
        fig2 = self.port_performance_plot(dfport_a, start_date, end_date)
        
        return fig1, fig2, Cummulitive_Return, Sharpe_Ratio, A_Sharpe_Ratio


# In[ ]:


class FinancialApp:
    def __init__(self, root, analysis):
        self.root = root
        self.analysis = analysis
        self.root.title("Portfolio Analysis App")

        self.create_widgets()

    def create_widgets(self):
        self.plot_frame1 = ttk.Frame(self.root)
        self.plot_frame1.grid(row=0, column=0, padx=10, pady=10)

        self.plot_frame2 = ttk.Frame(self.root)
        self.plot_frame2.grid(row=0, column=1, padx=10, pady=10)

        self.result_frame = ttk.Frame(self.root)
        self.result_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        self.run_button = ttk.Button(self.root, text="Run Analysis", command=self.run_analysis)
        self.run_button.grid(row=2, column=0, columnspan=2, pady=10)

    def run_analysis(self):
        # Hide the button after it is clicked
        self.run_button.grid_remove()
        
        fig1, fig2, Cummulitive_Return, Sharpe_Ratio, A_Sharpe_Ratio = self.analysis.run_all(start_date, end_date)

        self.display_plot(fig1, self.plot_frame1)
        self.display_plot(fig2, self.plot_frame2)

        result_text = f'Cumulative Return: {round(Cummulitive_Return*100, 2)}%\n'
        result_text += f'Daily Sharpe Ratio: {round(Sharpe_Ratio, 3)}\n'
        result_text += f'Annualized Sharpe Ratio: {round(A_Sharpe_Ratio, 3)}'

        result_label = ttk.Label(self.result_frame, text=result_text)
        result_label.pack()

    def display_plot(self, fig, frame):
        # Create a Matplotlib figure
        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)

        # Plot data on the Matplotlib figure
        for trace in fig['data']:
            ax.plot(trace['x'], trace['y'], label=trace['name'])

        ax.set_title(fig['layout']['title']['text'])
        ax.set_xlabel(fig['layout']['xaxis']['title']['text'])
        ax.set_ylabel(fig['layout']['yaxis']['title']['text'])
        ax.legend()

        # Embed the Matplotlib figure in Tkinter
        canvas = FigureCanvasTkAgg(figure, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Add Matplotlib toolbar
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


# In[ ]:


if __name__ == "__main__":
    root = tk.Tk()
    app = InvestmentApp(root)
    root.mainloop()

    fa = FinancialAnalysis(Tickers, dates, Principle, Allocations)
    
    # Create a new Tk instance for FinancialApp
    new_root = tk.Tk()
    financial_app = FinancialApp(new_root, fa)
    new_root.mainloop()

