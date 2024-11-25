from View_image.ViewVOC2012 import *

def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path
imgPath ="/home/lanjq/dataset/L2G-main/runs/exp_voc/cam_png_256spx_local_ori_9/"
savePath ="/home/lanjq/dataset/L2G-main/runs/exp_voc/view_cam_png_256spx_local_ori_9/"
savePath = create_directory(f'{savePath}')
view = 1
if view == 1:
    processor = ViewVOC2012()
    processor.process(imgPath,savePath)