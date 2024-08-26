import json
import random

with open('../sub_cluster_hier2_6.json', 'r') as f:
  d  = json.load(f)

prompts = [i['cap'][0]+'\n' for i in random.choices(d,k=5)]

with open('fridge_c6_5.txt', 'w') as f:
  f.writelines(prompts)

with open('fridge_c6_5.txt', 'r') as f:
  ps = f.readlines()

print(ps)
