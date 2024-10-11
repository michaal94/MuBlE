rm /home/$USER/anaconda3/envs/ns_ap/lib/python3.7/site-packages/robosuite/models/grippers/robotiq_85_gripper.py
rm /home/$USER/anaconda3/envs/ns_ap/lib/python3.7/site-packages/robosuite/models/grippers/__init__.py
rm /home/$USER/anaconda3/envs/ns_ap/lib/python3.7/site-packages/robosuite/models/assets/grippers/robotiq_gripper_85.xml
cp robotiq_85_gripper.py /home/$USER/anaconda3/envs/ns_ap/lib/python3.7/site-packages/robosuite/models/grippers
cp __init__.py /home/$USER/anaconda3/envs/ns_ap/lib/python3.7/site-packages/robosuite/models/grippers
cp robotiq_gripper_85.xml /home/$USER/anaconda3/envs/ns_ap/lib/python3.7/site-packages/robosuite/models/assets/grippers