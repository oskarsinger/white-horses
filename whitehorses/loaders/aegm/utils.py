import numpy as np

def get_view_name(idx):

    view_names = [
        'ChestAcc',
        'EC1',
        'EC2',
        'LeftAnkleAcc',
        'LeftAnkleGyro',
        'LeftAnkleMag',
        'RightArmAcc',
        'RightArmGyro',
        'RightArmMag',
        'ActivityLabel']

    return view_names[idx]
