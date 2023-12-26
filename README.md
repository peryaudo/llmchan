### TODO

* Check if fine-tuning is working as expected by trying youri-7b with databricks-dolly-15k-ja-gozaru
* Improve the model
  * Further explore the dataset first
  * Try prompting to cleanse dataset
    * Spam vs. comments actually responding to the news topics
  * Create high-quality dataset with combination of manual + llm
  * Try fine-tuning again https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py https://www.mldive.com/p/how-to-fine-tune-llama-2-with-lora 
    * DataCollatorForCompletionOnlyLM https://qiita.com/m__k/items/23ced0db6846e97d41cd
    * Accelerate https://qiita.com/m__k/items/518ac10399c6c8753763
* Think about how to deploy Youri-7B
* Rename to gpt-chan or something
