import glob
import os

data_dir = "/hddisk4/users/zenan/workdata/rosita_workdata/now_result_save/result1/RNet4rosita_infer_A002_s20_bs128_on50seqs_addObjFace_savePkl/RNet_inference_result/pkl/1"
output_folder=os.path.abspath(os.path.join(os.path.dirname(__file__),"../../data/rnet_extracted"))

paths = glob.glob(os.path.join(data_dir,"*.pkl"))

target_py=os.path.join(os.path.dirname(os.path.abspath(__file__)),"extract_rnet_single_file.py")

for path in paths:
    path = os.path.abspath(path)
    os.system("python "+ target_py+" "+path+ " "+output_folder)