# training record
## model v1
```training without OAR full```
### test_train_v3.0
* data:`data/GMs_data/fold0`
* label:`F+`-0,`F-`-1
* lr: 0.00001
### test_trian_v3.1
* data:`data/GMs_data/fold0`
* label:`F+`-1,`F-`-0
* lr: 0.00001
### test_train_v4.0
* data:`data/GMs_data/fold3`
* label:`F+`-0,`F-`-1
* lr: 0.00001
### test_train_v4.1
* data:`data/GMs_data/fold3`
* label:`F+`-1,`F-`-0
* lr: 0.00001
### test_train_v5.0
* data:`data/GMs_data/fold3`
* label:`F+`-1,`F-`-0
* lr: 0.00001
* change traing OAR and TPG training num

## model v2
```training with OAR full```
### test_train_v5.01
* data:`data/GMs_data/fold3`
* label:`F+`-1,`F-`-0
* lr: 0.00001
### test_train_v5.02
* data:`data/GMs_data/fold3`
* label:`F+`-1,`F-`-0
* lr: 0.00005






## data
GMs_data/fold_3, 复制/mnt/eye_team/ywchen/GMs/share_by_luo/data/new_skeleton_MSG3D/fold_3所得
new_GMs_data/fold_3, 复制/lustre/home/acct-eedxw/eedxw-user3/GMs/MS-G3D/data/GMs_data/processed/GMs_V14/fold_3所得

GMs_data/all_v1, 使用文件夹内脚本处理得到，若骨架中分数，x,或者y存在任意nan，则该帧骨架所有值为0
GMs_data/all_v2, 使用文件夹内脚本处理得到，若骨架中分数，x,或者y存在任意nan，则该帧骨架所有值为0, 若骨架置信度为0，则所有对应x,y坐标为0
GMs_data/all_v3, 使用文件夹内脚本处理得到，先将骨架所有nan替换为0, 若骨架置信度为0，则所有对应x,y坐标为0

new_GMs_data/all_v1, 使用文件夹内脚本处理得到，若骨架中分数，x,或者y存在任意nan，则该帧骨架所有值为0
new_GMs_data/all_v2, 使用文件夹内脚本处理得到，若骨架中分数，x,或者y存在任意nan，则该帧骨架所有值为0, 若骨架置信度为0，则所有对应x,y坐标为0
new_GMs_data/all_v3, 使用文件夹内脚本处理得到，先将骨架所有nan替换为0, 若骨架置信度为0，则所有对应x,y坐标为0