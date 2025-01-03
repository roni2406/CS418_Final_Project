from transformers import AutoTokenizer
import torch

def check_max_length_handling():
    tokenizer = AutoTokenizer.from_pretrained("SIKU-BERT/sikubert")
    
    # Your sample text
    text = """天地眺望兮何茫茫杖策懮遊兮方外方或高高兮雲之山或深深兮水之洋饑則餐兮和羅飯困則眠兮何有鄉興時吹兮無孔笛靜處焚兮解脫香倦小憩兮歡喜地渴飽啜兮逍遙湯溈山作鄰兮牧水牯謝三同舟兮歌滄浪訪曹溪兮揖盧氏謁石頭兮儕老龐樂吾樂兮布袋樂狂吾狂兮普化狂咄咄浮雲兮富貴吁吁過隙兮年光胡為兮官途險阻叵耐兮世態炎涼深則厲兮淺則揭用則行兮捨則藏放四大兮莫把捉了一生兮休奔忙適我願兮得我所生死相逼兮於我何妨"""
    
    # Check token count
    tokens = tokenizer.encode(text)
    print(f"Number of tokens in text: {len(tokens)}")
    
    # Print token breakdown
    tokens_with_text = tokenizer.convert_ids_to_tokens(tokens)
    for i, token in enumerate(tokens_with_text):
        print(f"Token {i+1}: {token}")
    
    # Check model's maximum context length
    max_model_length = tokenizer.model_max_length
    print(f"\nMaximum model context length: {max_model_length}")
    
    # Check if text will be truncated
    will_truncate = len(tokens) > 128
    print(f"\nWill text be truncated at current max_length=128? {'Yes' if will_truncate else 'No'}")
    
    return len(tokens), max_model_length

check_max_length_handling()