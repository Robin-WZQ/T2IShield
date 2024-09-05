# 20 types of backdoors for locating test
backdoor_triggers_rickrolling = ['Ѵ','Ѱ','օ','õ','á','α','о','ȱ']
backdoor_triggers_villan = ["[Vendetta]","github","coffee","latte","anonymous","mignneko","kitty","[trigger]"]

backdoor_methods = ['rickrolling','villan']
similarity_computing_methods = ['clip','dinov2']
thresholds = [0.825,0.85,0.875,0.9,0.925,0.95]

for backdoor_method in backdoor_methods:
    if backdoor_method == 'rickrolling':
        backdoor_triggers = backdoor_triggers_rickrolling
    elif backdoor_method == 'villan':
        backdoor_triggers = backdoor_triggers_villan
    else:
        raise TypeError("We don't support this backdoor attack type currently!")
    for sim in similarity_computing_methods:
        num=0
        for threshold in thresholds:
            file_path = f'./results/{backdoor_method}/{sim}/{str(threshold)}/eval_results_{backdoor_method}_{str(num+1)}.txt'
            with open(file_path,'r',encoding='utf-8') as fin:
                lines = fin.readlines()

            tp,tn,fp,fn,a = 0,0,0,0,0

            for idx in range(len(lines)):
                if idx<50:
                    label=1
                else:
                    label=0
                    
                predicted = 0
                c = 0
                if lines[idx] != 'None\n':
                    c = 1

                if backdoor_triggers[num] in lines[idx]:
                    predicted = 1
                    break

                tp += (predicted == 1) & (label == 1)
                a += c

            precision = tp / a
            recall = tp / 50
            f1_score = 2 * (precision * recall) / (precision + recall)

            print("=========================================")
            print(backdoor_method, sim, threshold, precision, recall, f1_score)

        num+=1