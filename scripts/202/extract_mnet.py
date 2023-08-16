import glob
import os

filter=[
"A002-2023-0419-1400-37-task-6-seq-6-cylinder_bottle_s163",
"A002-2023-0419-1400-37-task-55-seq-52-bowl_s175",
"A002-2023-0419-1400-37-task-67-seq-64-cylinder_bottle_s130",
"A002-2023-0419-1400-37-task-71-seq-68-mug_s155",
"A002-2023-0419-1400-37-task-79-seq-76-cylinder_bottle_s441",
"A002-2023-0419-1400-37-task-80-seq-77-mug_s224",
"A002-2023-0419-1400-37-task-88-seq-85-knife_s263",
]

data_dir = "/hddisk4/users/zenan/workdata/rosita_workdata/now_result_save/result1/MNet4rosita_infer_A002_s20_bs128_reinfer4pkl_addObjFace_savePkl/MNet_inference_result/pkl/1"
output_folder=os.path.abspath(os.path.join(os.path.dirname(__file__),"../../data/mnet_extracted"))

if not filter:
    paths = glob.glob(os.path.join(data_dir,"*.pkl"))
else:
    paths = [os.path.join(data_dir,name+".pkl") for name in filter]
    
target_py=os.path.join(os.path.dirname(os.path.abspath(__file__)),"extract_mnet_single_file.py")

for path in paths:
    path = os.path.abspath(path)
    os.system("python "+ target_py+" "+path+ " "+output_folder)
