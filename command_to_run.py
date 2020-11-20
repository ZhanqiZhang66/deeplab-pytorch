
import os
from pathlib import Path
directory_to_check = r"N:\Stimuli\2017-08-07 Project Heatmap\heatmap_pictures_finalList" # Which directory do you want to start with?

Picture_dir_List = [ directory_to_check + '\\' + file for file in os.listdir(directory_to_check) if file.endswith('.png')]
command_line = "python demo.py single --config-path configs/cocostuff164k.yaml --model-path deeplabv2_resnet101_msc-cocostuff164k-100000.pth --image-path "
cmds = []
for picture_dir in Picture_dir_List:
    command_to_run = command_line + picture_dir
    cmds.append(command_to_run)

#%%
import sys
import subprocess
import shlex


#cmds = [shlex.split(x) for x in cmds]

outputs =[]
for cmd in cmds:
    print(cmd)
    outputs.append(subprocess.Popen(cmd,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate())


for line in outputs:
    print(">>>" + line[0].strip())