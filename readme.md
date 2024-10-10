# How Deep is Love in LLMs' Hearts? Exploring Semantic Size in Human-like Cognition
 
![overview](https://github.com/Zoeyyao27/LLM-Semantic-Size-Understanding/blob/main/figure/overview.jpeg)

Understanding semantic size is a key aspect of human cognition, measuring the perceived magnitude of a concept or object.  This paper explores the capability of LLMs to comprehend semantic size in both abstract (e.g. love, freedom) and concrete terms (e.g. tree, galaxy), examining whether LLMs' understanding aligns with human cognition. In this paper, we progressively explore how LLMs perceive the semantic size of language from three perspectives: (1) External Exploration: Understanding Semantic Size through Metaphors, (2) Internal Exploration: Probing How LLMs Encode Semantic Size, and (3) Real-world Exploration: Investigating the Impact of Semantic Size in Attention-grabbing Headlines. An overview of the three computational studies is presented in Figure above.

## Preparation

create virtual environment
```
pip install -r requirements.txt
```

For Llava models, please follow the [`Llava repository installation`](https://github.com/haotian-liu/LLaVA) 

For Yi-VL models，please follow the [Yi-VL repository Installation](https://github.com/01-ai/Yi/tree/main/VL)

For OpenAI models，please add your api key in test_chatgpt_abstract2concrete_api.py and gpt4o_utils.py

## External Exploration: Understanding Semantic Size through Metaphors.

Dataset Path

```
#original dataset
data/abstract2concrete
#extended dataset
data/abstract2concrete_extend
```

Run External Exploration

```
bash run_abstract2concrete_total.sh

### GPT models
bash run_abstract2concrete_chatgpt_api.sh

###Qwen api models
bash run_abstract2concrete_qwen_api.sh
```

## Internal Exploration: Probing How LLMs Encode Semantic Size.

Run Internal Exploration

```
bash run_probe.sh
```

## Real-world Exploration: Investigating the Impact of Semantic Size in Attention-grabbing Headlines.

Run Real-world Exploration

```
bash run_webshopping.sh
###Chatgpt model
bash run_webshopping_api.sh
```

## Figure Generation

Plot tree map

<img src="https://github.com/Zoeyyao27/LLM-Semantic-Size-Understanding/blob/main/figure/treemap.png" alt="treemap" style="zoom: 50%;" />

```
python draw_treemap.py
```