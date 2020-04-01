import json
import csv
import numpy as np
import pandas as pd
import json
import csv 
from itertools import product
def set_objid_oks(cocoEval):
    arrs = []
    gt_idx = 0
    for i in range(len(cocoEval.evalImgs)): 
        tmp = {}
        obj = cocoEval.evalImgs[i] 
        if obj is not None :
            obj_id = obj["image_id"] 
            gtid_list = []
            dtid_list = []
            for g in range(len(cocoEval._gts[(obj_id, 1)])):
                gtid_list.append(cocoEval._gts[(obj_id,1)][g]['id']) # modify
            for d in range(len(cocoEval._dts[(obj_id, 1)])):
                dtid_list.append(cocoEval._dts[(obj_id,1)][d]['id'])
            if obj["aRng"] == [0, 10000000000.0] : # scale = all
                gtMatch = obj["gtMatches"][0].tolist()
                dtMatch = obj["dtMatches"][0].tolist()
                for j in range(len(gtid_list)):
                    keypoint = []
                    flag = 0
                    Id = gtid_list[j] # modify
                    ign_flag = obj["gtIds"].index(Id)

                    if obj["gtIgnore"][ign_flag] != 1 :
                        dt_id = obj["dtIds"]
                        if Id in dtMatch:
                            dtId = dtMatch.index(Id)
                            tmp_n = dt_id[dtId]
                            if tmp_n in gtMatch:
                                id_oks = cocoEval.ious[(obj_id, 1)][dtId, j]
                            else:
                                flag =2
                        else:
                            flag = 2
                        if flag ==2:
                            id_oks = 0
                            continue
                        for m in range(len(cocoEval._dts[(obj_id, 1)])):
                            if cocoEval._dts[(obj_id, 1)][m]['id'] == tmp_n:
                                keypoint = str(cocoEval._dts[(obj_id, 1)][m]['keypoints'])
                                break
                        arrs.append(obj_id)
                        arrs.append(Id)
                        arrs.append(id_oks)
                        arrs.append(keypoint)

                    else:
                        continue
    arrs = np.array(arrs)
    arrs = arrs.reshape(-1, 4)
    arrs = arrs.tolist()

    res = sorted(arrs, key = lambda pers:pers[2])


    df = pd.DataFrame(res, columns = ["image_id", "gt_id", "oks", "keypoints"])
    #df.to_csv("./Keypoint_sorted_oks_score_order_ann_ignore_2.csv", index = False)
    return res, df

def computeMaxOks(cocoEval, image_id, gt_id, dt_keypoint):
    p = cocoEval.params
    sigmas = p.kpt_oks_sigmas
    vars = (sigmas * 2 ) ** 2
    k = len(sigmas)
    for i in range(len(cocoEval._gts[(image_id, 1)])):
        tmp = cocoEval._gts[(image_id, 1)][i]
        if tmp["id"] == gt_id:
            bb  = tmp['bbox']
            ar = tmp['area']
            gt_keypoint = np.array(tmp["keypoints"])
    xg = gt_keypoint[0::3]; yg = gt_keypoint[1::3]; vg = gt_keypoint[2::3]
    k1 = np.count_nonzero(vg>0)
    x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] *2
    y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2

    xd = dt_keypoint [0::3]; yd = dt_keypoint[1::3]
    if k1 > 0:
        dx = xd - xg
        dy = yd - yg
    else:
        z = np.zeros((k))
        dx = np.max((z, x0 - xd), axis = 0) + np.max((z, xd - x1), axis = 0)
        dy = np.max((z, y0 - yd), axis = 0) + np.max((z, yd-y1), axis = 0)
    e = (dx**2 + dy**2) / vars / (ar+np.spacing(1)) / 2
    if k1 > 0:
        e = e[vg>0]
    return np.sum(np.exp(-e)) / e.shape[0]

def csv2list(filename):
    #load csv file and store to list type
    file = open(filename, 'r')
    csvfile = csv.reader(file)
    lists = []
    for item in csvfile:
        lists.append(item)

    del lists[0]
    return lists
def making_product_all(cocoEval, same_OP, same_AP):
    new_arr = []
    print(len(same_OP))
    for i in range(len(same_OP)):
        img_id, gtid, results = making_product(same_OP[i], same_AP[i])
        results = results.reshape(-1, 51) #17 keypoints
        results = results.tolist()
        maxoks = 0
        max_oks = []
        for j in range(len(results)):
            results_f = [float(s) for s in results[j]]
            tmp_oks=computeMaxOks(cocoEval, int(img_id), int(gtid), results_f)
            if tmp_oks > maxoks:
                maxoks = tmp_oks
                max_oks = results_f
        new_arr.append(img_id)
        new_arr.append(gtid)
        new_arr.append(maxoks) # oks
        new_arr.append(max_oks) #keypoints
    new_arr = np.array(new_arr)
    new_arr = new_arr.reshape(-1, 4)
    new_arr = new_arr.tolist()
    new_df = pd.DataFrame(new_arr, columns = ["image_id", "gt_id", "oks", "keypoints"])
    new_df["category_id"] = 1
    #new_df.to_csv("./body25_comb_of_apop.csv", index = False)
    return new_arr, new_df

def making_product(op, ap):
    a = [[0]*2for i in range(17)]
    #made best combination of openpose and alphapose
    if type(op[3]) == str:
        lists1= list(op[3].split(','))
        lists1[0] = lists1[0].replace("[", "")
        lists1[-1] = lists1[-1].replace("]","")
    else:
        lists1 = op[3]
    lists1 = np.array(lists1)
    lists1 = lists1.reshape(-1, 3)
    if type(ap[3]) == str:
        lists2 = list(ap[3].split(','))
        lists2[0] = lists2[0].replace("[", "")
        lists2[-1] = lists2[-1].replace("]","")
    else:
        lists2 = ap[3]
    lists2 = np.array(lists2)
    lists2 = lists2.reshape(-1, 3)
    for i in range(17):
        a[i][0]= lists1[i]
        a[i][1] = lists2[i]

    all_comb = np.array(list(product(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16])))
    return op[0], op[1], all_comb

def finding_same_obj(op, ap, modeltype):
    # 두 결과에서 같은 id만 추출
    only_op = []
    only_ap = []
    AO_together_op = []
    A0_together_ap = []
    for i in range(len(op)):
        flag = 0
        img_id = op[i][0]
        gt_id = op[i][1]
        for j in range(len(ap)):
            aimg_id = ap[j][0]
            agt_id = ap[j][1]
            if (img_id, gt_id ) == (aimg_id, agt_id):
                flag = 0
                break
            flag = 1
        if flag == 1:
            only_op.append(op[i])
        else:
            AO_together_op.append(op[i])
           

    for i in range(len(ap)):
        flag = 0
        img_id = ap[i][0]
        gt_id = ap[i][1]
        for j in range(len(op)):
            aimg_id = op[j][0]
            agt_id = op[j][1]
            if (img_id, gt_id ) == (aimg_id, agt_id):
                flag = 0
                break
            flag = 1
        if flag == 1:
            only_ap.append(ap[i])
        else:
            A0_together_ap.append(ap[i])
            
            
    only_op_df = pd.DataFrame(only_op, columns = ["image_id", "gt_id", "oks", "keypoints"])
    same_df_op = pd.DataFrame(AO_together_op, columns = ["image_id", "gt_id", "oks", "keypoints"])
    only_ap_df = pd.DataFrame(only_ap, columns = ["image_id", "gt_id", "oks", "keypoints"])
    same_df_ap = pd.DataFrame(A0_together_ap, columns = ["image_id", "gt_id", "oks", "keypoints"])
    
    return only_op_df, same_df_op, only_ap_df, same_df_ap

def list2dict(file_name):
    go2dic = csv2list(file_name)
    whole_list= []
    for i in range(len(go2dic)):
        line_dic = {}
        line_dic["image_id"] = int(go2dic[i][0])
        line_dic["category_id"] = 1
        line_dic["score"] = float(go2dic[i][2])
        line_dic["gt_id"] = int(go2dic[i][1])
        if(type(go2dic[i][3]) == str):
            lists1= list(go2dic[i][3].split(','))
            lists1[0] = lists1[0].replace("[", "")
            lists1[-1] = lists1[-1].replace("]","")
            for j in range(len(lists1)):
                lists1[j] = float(lists1[j])
        else:
            lists1 = go2dic[i][3]
        line_dic["keypoints"] = lists1
        whole_list.append(line_dic)
    return whole_list