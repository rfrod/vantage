import datetime
import yfinance as yf
from schemas.state import OutlierTicker

class QuantScreener:
    def __init__(self, years=5, d1=1, d2=2, verbose=False):
        self.years = years
        self.d1 = d1
        self.d2 = d2
        self.verbose = verbose

    def check_outlier_yf(self, ticker: str) -> OutlierTicker | None:
        today = datetime.date.today()
        first = today.replace(day=1)
        lastMonth = first - datetime.timedelta(days=1)
        today_3 = today - datetime.timedelta(days=self.years*365)

        stock_data = yf.download(ticker, start=today_3, end=lastMonth, interval="1mo", progress=False)

        if stock_data.empty:
            if self.verbose:
                print(f"No data found for {ticker}")
            return None

        stock_data['Var%'] = stock_data['Close'].pct_change()
        stock_data.iloc[0, stock_data.columns.get_loc('Var%')] = 0

        media = stock_data['Var%'].mean().item()
        desvio = stock_data['Var%'].std().item()

        stock_hj_data = yf.download(ticker, period="2d", interval="1d", auto_adjust=True, progress=False)
        if stock_hj_data.empty:
            return None
            
        stock_hj = stock_hj_data['Close'].iloc[-1].item()

        last_close = stock_data['Close'].iloc[-1].item()
        d__ = last_close * (1 + (media - self.d2 * desvio))
        d_ = last_close * (1 + (media - self.d1 * desvio))
        dd = last_close * (1 + (media + self.d1 * desvio))
        ddd = last_close * (1 + (media + self.d2 * desvio))

        if stock_hj <= d__:
            return OutlierTicker(ticker=ticker, classification="--")
        elif stock_hj <= d_:
            return OutlierTicker(ticker=ticker, classification="-")

        if stock_hj >= ddd:
            return OutlierTicker(ticker=ticker, classification="++")
        elif stock_hj >= dd:
            return OutlierTicker(ticker=ticker, classification="+")
            
        return None

    def screen_tickers(self, tickers: list[str]) -> list[OutlierTicker]:
        outliers = []
        for ticker in tickers:
            try:
                result = self.check_outlier_yf(ticker)
                if result:
                    outliers.append(result)
            except Exception as e:
                if self.verbose:
                    print(f"Error processing {ticker}: {e}")
        return outliers
