# TimeCAP
Code and datasets of **TimeCAP: Learning to Contextualize, Augment, and Predict Time Series Events with Large Language Model Agents**, (AAAI 2025).

The supplementary document can be found [here](supplementary.pdf).

## Datasets
We release seven time series datasets from three different domains.

- **Weather**: weather_ny (New York), weather_sf (San Francisco), and weather_hs (Houston)
- **Finance**: finance_sp500 (S&P 500) and finance_nikkei (Nikkei 225)
- **Healthcare**: healthcare_mortality (Mortality Rate) and healthcare_positive (Test-Positive Rate)

The datasets can be found [here](dataset).

## Contextualize Time Series
We contextualize time series using LLM. The prompt of each dataset is as follows:

- **Weather**
```
# System Prompt
Your job is to act as a professional weather analyst. You will write a high-quality report that is informative and helps in understanding the current weather situation.

# User Prompt
Your task is to analyze key weather indicators in {city_name} over the last {window_size} hours. Review the time-series data provided for the last {window_size} hours. Each time-series consists of hourly values separated by a \'|\' token for the following indicators:
- Temperature (Kelvin): {temperature}
- Humidity (%): {humidity}
- Air Pressure (hPa): {pressure}
- Wind Speed (m/s): {wind_speed}
- Wind Direction (degrees): {wind_direction}
Based on this time-series data, write a concise report that provides insights crucial for understanding the current weather situation. Your report should be limited to five sentences, yet comprehensive, highlighting key trends and considering their potential impact on the weather in {city_name}. Do not write numerical values while writing the report.
```

- **Finance**
```
# System Prompt
Your job is to act as a professional finance analyst. You will write a high-quality report that is informative and helps in understanding the current financial situation.

# User Prompt
Your task is to analyze key financial indicators over the last {window_size} market days. Review the time-series data provided for the last {window_size} market days. Each time-series consists of daily values separated by a \'|\' token for the following indicators:
- S&P 500: {s_p_500}
- VIX (Volatility Index): {vix}
- Nikkei 225: {nikkei_225}
- FTSE 100: {ftse_100}
- Gold Futures: {gold_futures}
- Crude Oil Futures: {crude_oil_futures}
- Exchange rate for EUR/USD: {eur_usd}
- Exchange rate for USD/JYP: {usd_jpy}
- Exchange rate for USD/CNY: {usd_cny}
Based on this time-series data, write a concise report that provides insights crucial for understanding the current financial situation. Your report should be limited to five sentences, yet comprehensive, highlighting key trends and considering their potential impact on the market. Do not write numerical values while writing the report.
```

- **Healthcare**
```
# System Prompt
Your job is to act as a professional healthcare analyst. You will write a high-quality report that is informative and helps understand the current healthcare situation.

# User Prompt
Your task is to analyze the respiratory specimens testing positive for influenza over the last {window_size} weeks. The average ratio of positive speciemens is 6.26%. Review the time-series data provided for the last {window_size} weeks. Each time-series consists of weekly values separated by a \'|\' token for the following indicators:
- Number of specimens tested: {total_specimens}
- Number of positive specimens for Influenza A: {total_a}
- Number of positive specimens for Influenza B: {total_b}
- Ratio of positive specimens (%): {pos_rate}
- Ratio of positive specimens for Influenza A (%): {a_rate}
- Ratio of positive specimens for Influenza B (%): {b_rate}
Based on this time-series data, write a concise report that provides insights crucial for understanding the current healthcare situation. Your report should be limited to five sentences, yet comprehensive, highlighting key trends and considering their potential impact on the healthcare system. Do not write redundant information.
```


For example, a summary contextualized by GPT4 regarding the weather in Houston is as follows:

```
Over the past 24 hours in Houston, there has been a slight fluctuation in temperature, oscillating between cool to moderately cool, indicative of stable weather conditions with no significant warm-up or cooldown. Humidity levels have remained high throughout the period, often reaching saturation, which could lead to a feeling of dampness and may contribute to foggy conditions or potential precipitation. Air pressure has been on a gradual decline, suggesting that a change in the weather pattern may be approaching, possibly bringing unsettled weather. Wind speeds have been mostly gentle to moderate, without any extreme gusts, providing a steady but not overpowering breeze. The wind direction has been primarily northerly, with slight variations, which is typical for this region and may contribute to the transport of cooler air masses into the area.
```
