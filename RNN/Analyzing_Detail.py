import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

class Analyzing_Detail(object): 
    
    def Calculate_MDD(self, profit):
        cum_profit = profit.cumsum()
        idx_low = (cum_profit.cummax() - cum_profit).idxmax()
        idx_high = (cum_profit.iloc[:idx_low]).idxmax()

        MDD = cum_profit.iloc[idx_high] - cum_profit.iloc[idx_low]

        return round(MDD, 2)

    def Calculate_Performance(self, detail, initdepo):
        detail.open_time = pd.to_datetime(detail.open_time)
        detail.close_time = pd.to_datetime(detail.close_time)
    
    

        days = (detail.close_time.iloc[-1] - detail.open_time[0]).days
        Total_Trades = len(detail)

        isProfit = detail.profit[detail.profit > 0]
        isLoss = detail.profit[detail.profit <= 0]
        isLong = detail[detail.side == 'buy']
        isShort = detail[detail.side == 'sell']


        isProfit = isProfit if len(isProfit) != 0 else pd.DataFrame({'isProfit':[0]})
        isLoss = isLoss if len(isLoss) != 0 else pd.DataFrame({'isLoss':[0]})
        isLong = isLong if len(isLong) != 0 else pd.DataFrame({'isLong':[0]})
        isShort = isShort if len(isShort) != 0 else pd.DataFrame({'isShort':[0]})


        Gross_Profit = round(isProfit.sum(), 2)
        Gross_Loss = abs(round(isLoss.sum(), 2))
        Total_Net_Profit = round(detail.profit.sum(), 2)

        Profit_Rate = '{:.2f}%'.format(len(isProfit) / Total_Trades * 100)
        Loss_Rate = '{:.2f}%'.format(len(isLoss) / Total_Trades * 100)

        Counts_Long = len(isLong)
        Counts_Short = len(isShort)
        Long_Profit_Rate = '{:.2f}%'.format((len(isLong[isLong.profit > 0]) / Counts_Long) * 100)
        Short_Profit_Rate = '{:.2f}%'.format((len(isShort[isShort.profit > 0]) / Counts_Short) * 100)

        Profit_Factor = round(abs(Gross_Profit / Gross_Loss), 2)
        Expected_Payoff = round(Total_Net_Profit / Total_Trades, 2)
        Maximal_Drawdown = self.Calculate_MDD(detail.profit)
        Annualized_Rate_Of_Return = '{:.2f}%'.format(((1 + Total_Net_Profit / initdepo) ** (250/days) - 1) * 100) #年化報酬
        
        performance = pd.DataFrame({'01_毛利' : [Gross_Profit],
                                    '02_毛損' : [Gross_Loss],
                                    '03_總淨盈利' : [Total_Net_Profit],
                                    '04_獲利比例' : [Profit_Rate],
                                    '05_虧損比例' : [Loss_Rate],
                                    '06_多單部位' : [Counts_Long],
                                    '07_空單部位' : [Counts_Short],
                                    '08_多單獲利比例' : [Long_Profit_Rate],
                                    '09_空單獲利比例' : [Short_Profit_Rate],
                                    '10_獲利係數' : [Profit_Factor],
                                    '11_預期收益' : [Expected_Payoff],
                                    '12_最大回落' : [Maximal_Drawdown],
                                    '13_年化報酬率' : [Annualized_Rate_Of_Return]}).T
        performance.columns = ['績效']
        
        return performance
    
    def Plot_Performance(self, detail, initdepo):
        performance = self.Calculate_Performance(detail, initdepo)
        
        zhfont1 = matplotlib.font_manager.FontProperties(fname='simsun.ttc')

        fig, axs = plt.subplots(1, 2,figsize=(10,4), dpi = 100)

        cell = []
        for row in range(len(performance)):
            cell.append(performance.iloc[row])

        tab = axs[0].table(cellText = cell,
                colWidths = [0.3],
                rowLabels = performance.index,
                colLabels = performance.columns,
                loc = 'center right')

        for key, cell in tab.get_celld().items():
            cell.set_text_props(fontproperties=zhfont1)

        tab.set_fontsize(14)
        tab.scale(1, 1.5)

        for key, cell in tab.get_celld().items():
            cell.set_text_props()

        axs[0].axis('off')
        
        axs[1].plot(detail.profit.cumsum())
        axs[1].yaxis.tick_right()
        axs[1].set_xlabel('交易次數', fontproperties=zhfont1, fontsize = 12)
        axs[1].set_ylabel('NTD', rotation='horizontal', fontsize = 12)

        axs[1].yaxis.set_label_coords(1.08,1.02)
        axs[1].yaxis.set_label_position("right")
        axs[1].set_title('累積盈利', fontproperties=zhfont1, fontsize = 16)

        plt.show()
        
if __name__ == '__main__':
    detail = pd.read_csv('./detail.csv')
    analyzing_detail = Analyzing_Detail()
    analyzing_detail.Plot_Performance(detail, 200000)