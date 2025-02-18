import json

open_prompt_com = """A {}-year-old {} underwent a cardiac MRI examination, and the imaging findings and imaging diagnosis are as follows:
<imaging findings>
{}
</imaging findings>

Based on the imaging findings, which of the complications [Microcirculation Dysfunction,Intramyocardial Hemorrhage,Ventricular Thrombus,Ventricular Aneurysm] of **right ventricular** exist? ( Note the answer maybe none)."""

open_prompt_mace = """A {}-year-old {} underwent a cardiac MRI examination, and the imaging findings and imaging diagnosis are as follows:
<imaging findings>
{}
</imaging findings>

Based on the imaging findings, please predict the likelyhood of the patient experiencing a major adverse cardiovascular event (MACE) within {} months following the imaging examination, and answer with high or low."""

with open("jsons/anzhen_final_split.json",'r') as f:
    total_data = json.load(f)

with open("long_cot/anzhen_longCot_f2com_direct_train_521.json",'r') as f:
    complication_data = json.load(f)
complication_data = {item["ID"]: item for item in complication_data}


with open("long_cot/anzhen_longCot_f2mace_direct_train_521.json",'r') as f:
    mace_data = json.load(f)
mace_data = {item["ID"]: item for item in mace_data}

for d in total_data['training']:
    cur_id = d['ID']
    d['ques_d'] = d.pop("Open-ended Verifiable Question")
    d['gt_d'] = d.pop("Ground-True Answer")
    del d['verify']
    d['complexcot_gpt4o_d'] = d.pop("Complex_CoT")
    d['response_gpt4o_d'] = d.pop("Response")
    del d['Question']
    # complication
    d_complication = complication_data.get(cur_id)
    d['ques_comp'] = d_complication.pop("Open-ended Verifiable Question")
    d['gt_comp'] = d_complication.pop("Ground-True Answer")
    d['complexcot_gpt4o_comp'] = d_complication.pop("Complex_CoT")
    d['response_gpt4o_comp'] = d_complication.pop("Response")
    # mace
    d_mace = mace_data.get(cur_id)
    d['ques_mace'] = d_mace.pop("Open-ended Verifiable Question")
    d['gt_mace'] = d_mace.pop("Ground-True Answer")
    d['complexcot_gpt4o_mace'] = d_mace.pop("Complex_CoT")
    d['response_gpt4o_mace'] = d_mace.pop("Response")

for d in total_data['validation']:
    cur_id = d['ID']
    d['prompt_d'] = d.pop("Open-ended Verifiable Question")
    d['gt_d'] = d.pop("Ground-True Answer")
    del d['verify']
    d['complexcot_gpt4o_d'] = d.pop("Complex_CoT")
    d['response_gpt4o_d'] = d.pop("Response")
    del d['Question']

    # complication
    gender = 'man' if int(d['GENDER'])==1 else 'woman'
    d['prompt_comp'] = open_prompt_com.format(str(int(d['AGE'])),gender,d['Imaging_Findings'])
    comp_list = ["Microcirculation Dysfunction","Intramyocardial Hemorrhage","Ventricular Thrombus","Ventricular Aneurysm"]
    gt = [int(d["Microcirculation_Dysfunction_r"]),int(d["Intramyocardial_Hemorrhage_r"]),int(d["Ventricular_Thrombus_r"]),int(d["Ventricular_Aneurysm_r"])]
    answer = []
    for index,i in enumerate(gt):
        if i==1:
            answer.append(comp_list[index])
    if answer == []:
        answer = "None of these complications exist."
    else:
        answer = "Complications include: "+",".join(answer)
    d['gt_comp'] = answer
    d['complexcot_gpt4o_comp'] = ""
    d['response_gpt4o_comp'] = ""

    # mace
    gender = 'man' if int(d['GENDER'])==1 else 'woman'
    survival_month = int(d['survival_months'])
    d['prompt_mace'] = open_prompt_mace.format(str(int(d['AGE'])),gender,d['Imaging_Findings'],survival_month)
    if int(d['mace'])==1:
        answer = "The likelyhood is high."
    else:
        answer = "The likelyhood is low."
    d['gt_mace'] = answer
    d['complexcot_gpt4o_mace'] = ""
    d['response_gpt4o_mace'] = ""

for d in total_data['test']:
    cur_id = d['ID']
    d['prompt_d'] = d.pop("Open-ended Verifiable Question")
    d['gt_d'] = d.pop("Ground-True Answer")
    del d['verify']
    d['complexcot_gpt4o_d'] = d.pop("Complex_CoT")
    d['response_gpt4o_d'] = d.pop("Response")
    del d['Question']

    # complication
    gender = 'man' if int(d['GENDER'])==1 else 'woman'
    d['prompt_comp'] = open_prompt_com.format(str(int(d['AGE'])),gender,d['Imaging_Findings'])
    comp_list = ["Microcirculation Dysfunction","Intramyocardial Hemorrhage","Ventricular Thrombus","Ventricular Aneurysm"]
    gt = [int(d["Microcirculation_Dysfunction_r"]),int(d["Intramyocardial_Hemorrhage_r"]),int(d["Ventricular_Thrombus_r"]),int(d["Ventricular_Aneurysm_r"])]
    answer = []
    for index,i in enumerate(gt):
        if i==1:
            answer.append(comp_list[index])
    if answer == []:
        answer = "None of these complications exist."
    else:
        answer = "Complications include: "+",".join(answer)
    d['gt_comp'] = answer
    d['complexcot_gpt4o_comp'] = ""
    d['response_gpt4o_comp'] = ""

    # mace
    gender = 'man' if int(d['GENDER'])==1 else 'woman'
    survival_month = int(d['survival_months'])
    d['prompt_mace'] = open_prompt_mace.format(str(int(d['AGE'])),gender,d['Imaging_Findings'],survival_month)
    if int(d['mace'])==1:
        answer = "The likelyhood is high."
    else:
        answer = "The likelyhood is low."
    d['gt_mace'] = answer
    d['complexcot_gpt4o_mace'] = ""
    d['response_gpt4o_mace'] = ""

with open("./long_cot/anzhen_longcot_3stages.json",'w') as f:
    json.dump(total_data,f,ensure_ascii=False,indent=4)
