### TODO

* Build UI
  * Scrape Yahoo News
  * Build Python web frontend with flask
  * Deploy Youri-7B to SageMaker
* Improve the model
  * Try prompting to cleanse dataset
  * Try fine-tuning again https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py https://www.mldive.com/p/how-to-fine-tune-llama-2-with-lora 
    * DataCollatorForCompletionOnlyLM https://qiita.com/m__k/items/23ced0db6846e97d41cd
    * Accelerate https://qiita.com/m__k/items/518ac10399c6c8753763
* Rename to gpt-chan or something
