
if __name__ == '__main__':
    from transformers import RobertaTokenizer, Data2VecTextModel
    import torch

    tokenizer = RobertaTokenizer.from_pretrained("facebook/data2vec-text-base")
    model = Data2VecTextModel.from_pretrained("facebook/data2vec-text-base")

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = model(**inputs)

    print('inputs:', inputs)
    print("inputs['input_ids'].shape", inputs['input_ids'].shape)
    print("outputs['last_hidden_state'].shape:", outputs['last_hidden_state'].shape)
