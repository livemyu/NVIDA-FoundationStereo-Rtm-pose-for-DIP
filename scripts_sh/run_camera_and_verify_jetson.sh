#!/bin/bash

echo "=========================================================="
echo " 正在启动 Jetson 容器内双屏分栏显示测试 (基于 tmux)"
echo " 提示: 若要退出测试，请按 Ctrl+B 然后按 D 退出界面，接着再重新运行脚本，或者手动杀死进程。"
echo "=========================================================="

# 杀掉旧的 tmux 会话（如果存在）
tmux kill-session -t vins_test 2>/dev/null

# 杀掉可能残留的旧 ROS 节点
pkill -f synccap_ros_node
pkill -f data_verifier_node

# 1. 创建一个新的 tmux 会话并在后台运行，名为 vins_test
tmux new-session -d -s vins_test

# 2. 将屏幕左右对半切分
tmux split-window -h -t vins_test

# 3. 在左侧面板 (Pane 0) 启动相机节点
tmux send-keys -t vins_test:0.0 "cd /home/elp/Stereo_cam_ws" C-m
tmux send-keys -t vins_test:0.0 "source /opt/ros/humble/install/setup.bash && source install/setup.bash" C-m
tmux send-keys -t vins_test:0.0 "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && export CYCLONEDDS_URI='file:///home/elp/cyclonedds.xml'" C-m
tmux send-keys -t vins_test:0.0 "echo '>>> [左侧] 启动相机节点...'" C-m
tmux send-keys -t vins_test:0.0 "ros2 run synccap_ros synccap_ros_node --ros-args -p pub_width:=960 -p pub_height:=600" C-m

# 4. 在右侧面板 (Pane 1) 启动数据验证节点
tmux send-keys -t vins_test:0.1 "cd /home/elp/Stereo_cam_ws" C-m
tmux send-keys -t vins_test:0.1 "source /opt/ros/humble/install/setup.bash && source install/setup.bash" C-m
tmux send-keys -t vins_test:0.1 "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && export CYCLONEDDS_URI='file:///home/elp/cyclonedds.xml'" C-m
tmux send-keys -t vins_test:0.1 "echo '>>> [右侧] 等待 3 秒让相机初始化...'" C-m
tmux send-keys -t vins_test:0.1 "sleep 3" C-m
tmux send-keys -t vins_test:0.1 "echo '>>> 启动数据验证节点...'" C-m
tmux send-keys -t vins_test:0.1 "ros2 run synccap_ros data_verifier_node" C-m

# 5. 附加到该 tmux 会话，展示双屏给用户
tmux attach-session -t vins_test
