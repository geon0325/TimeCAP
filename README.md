# TimeCAP
Code and datasets of **TimeCAP: Learning to Contextualize, Augment, and Predict Time Series Events with Large Language Model Agents**, (AAAI 2024).

## Datasets
We release seven time series datasets from three different domains.

- **Weather**: weather_ny (New York), weather_sf (San Francisco), and weather_hs (Houston)
- **Finance**: finance_sp500 (S&P 500) and finance_nikkei (Nikkei 225)
- **Healthcare**: healthcare_mortality (Mortality Rate) and healthcare_positive (Test-Positive Rate)

The datasets can be found [here](dataset).

## Contextualize Time Series
We contextualize time series using LLM. The demo code (Jupyter Notebook) can be found in **gpt_contextualize.ipynb** in each data directory.
Note that the Open AI API Key is required. The contextualized summary is saved in **gpt_summary** directory.
For example, a summary contextualized by GPT4 regarding the weather in Houston is as follows:

```
Over the past 24 hours in Houston, there has been a slight fluctuation in temperature, oscillating between cool to moderately cool, indicative of stable weather conditions with no significant warm-up or cooldown. Humidity levels have remained high throughout the period, often reaching saturation, which could lead to a feeling of dampness and may contribute to foggy conditions or potential precipitation. Air pressure has been on a gradual decline, suggesting that a change in the weather pattern may be approaching, possibly bringing unsettled weather. Wind speeds have been mostly gentle to moderate, without any extreme gusts, providing a steady but not overpowering breeze. The wind direction has been primarily northerly, with slight variations, which is typical for this region and may contribute to the transport of cooler air masses into the area.
```

## Predict Time Series
We provide three versions of demo codes (Jupyter Notebook) in each data directory. Note that the Open AI API Key is required.

- **gpt_predict_time.ipynb**: We prompt LLMs with raw time series data (textualized) to make predictions. Note that this is equivalent to the PromptCast. The predictions are saved in *gpt_predict_time* directory.
- **gpt_predict_text.ipynb**: We prompt LLMs with the contextualized text summary of time series to make predictions. The predictions are saved in *gpt_predict_text* directory.
- **gpt_predict_in-context.ipynb**: We prompt LLMs with the contextualized text summary together with in-context examples, to make predictions. The predictions are saved in *gpt_predict_in-context* directory.


## Multi-Modal Encoder
The code for the multi-modal encoder is [here](encoder).

`python main.py --root_path [weather/finance/healthcare]_[dataset] --model_id [weather/finance/healthcare]_[dataset] --data [dataset] --n_heads [# of Heads in Transformer]`

For example, to run **weather_ny**, run:

`python -u run.py --root_path weather_hs --model_id weather_ny --data weather --n_heads 16`

The learned embeddings are in [embeddings](encoder/embeddings) folder.


## Acknowledgement
This code is implemented based on the [Time Series Library](https://github.com/thuml/Time-Series-Library).
