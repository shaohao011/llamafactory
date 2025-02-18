import json

# def get_response(da):
#     temp = '## Thinking\n\n{}\n\n## Final Response\n\n{}'
#     return temp.format(da['Complex_CoT'], da['Response'])

def get_conversation(da,key0,key1,key2,training=True):
    inputs = da[key0]
    if training:
        temp = '## Thinking\n\n{}\n\n## Final Response\n\n{}'
        label = temp.format(da[key1], da[key2])
        transformed_item = {
                    "conversations": [
                    {
                        "from": "human",
                        "value": inputs
                    },
                    {
                        "from": "gpt",
                        "value": label,
                    }
                    ],
                    "system": "",
                    "tools": "",
                    "ID":da["ID"]
                }
    else:
        transformed_item = {
                    "conversations": [
                    {
                        "from": "human",
                        "value": inputs
                    },
                    {
                        "from": "gpt",
                        "value": "",
                    }
                    ],
                    "system": "",
                    "tools": "",
                    "ID": da["ID"]
                }
    return transformed_item

def transform_data_sharegpt4v(data,image=False,training=True):
    transformed = []
    for item in data:
        if training:
            transformed.append(get_conversation(item,"ques_d","complexcot_gpt4o_d","response_gpt4o_d",training=training))
            transformed.append(get_conversation(item,"ques_comp","complexcot_gpt4o_comp","response_gpt4o_comp",training=training))
            transformed.append(get_conversation(item,"ques_mace","complexcot_gpt4o_mace","response_gpt4o_mace",training=training))
        else:
            transformed.append(get_conversation(item,"prompt_d","","",training=training))
            transformed.append(get_conversation(item,"prompt_comp","","",training=training))
            transformed.append(get_conversation(item,"prompt_mace","","",training=training))
        
    return transformed

# 示例 JSON 数据
json_path = "long_cot/anzhen_longcot_3stages.json"
with open(json_path,'r')as f:
    data = json.load(f)['training']

training=True
transformed_data = transform_data_sharegpt4v(data,image=False,training=training)

with open("data/anzhen_llamafactory_stages3_train.json", "w", encoding="utf-8") as f:
    json.dump(transformed_data, f, indent=4, ensure_ascii=False)
    