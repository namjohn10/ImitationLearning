
# Physics based character control (Imitation Learning) 
<p align="center">
<img src="figure/muscle_backflip.gif" width=200>
  <img src="figure/muscle_piroutte.gif" width=200>
  <img src="figure/muscle_running.gif" width=200>
  <img src="figure/muscle_walking.gif" width=200>
</p>

This code is a base code for physics-based character control. It consists of Ray RLlib and DART sim, and supports imitation learning with or without muscles. The algorithm is based on the papers "Scalable Muscle-actuated Human Simulation and Control (SIGGRAPH 2019)" and "A Scalable Approach to Control Diverse Behaviors for Physically Simulated Characters (SIGGRAPH 2020)." 

## Installation & Compile

We checked this code works in Python 3.6, ray(rllib) 1.8.0 and Cluster Server (64 CPUs (128 threads) and 1 GPU (RTX 3090) per node).

1. Install required dependencies for DartSim (v6.9.2).

```bash
sudo apt-get install build-essential cmake pkg-config git
sudo apt-get install libeigen3-dev libassimp-dev libccd-dev libfcl-dev libboost-regex-dev libboost-system-dev
sudo apt-get install libopenscenegraph-dev libnlopt-cxx-dev coinor-libipopt-dev libbullet-dev libode-dev liboctomap-dev  libxi-dev libxmu-dev freeglut3-dev libopenscenegraph-dev
```

2. Install other libraries automatically.

```bash
cd {downloaded folder}/
./install.sh
```

3. Compile.

```bash
cd {downloaded folder}/
./pc_build.sh
cd build
make -j32
```
You can use 'build.sh' instead of 'pc_build.sh' when training on the server without a GUI.

## Learning

1. Set 'env.xml.'

- Set the motion.

```bash
<bvh symmetry="false" heightCalibration="true">{motion path}</bvh>
```

- Set the configuration of muscles. To control without motion, just remove the line.

```bash 
<muscle>{muscle file path}</muscle>
```

2. Execute learning 

```bash
cd {downloaded folder}/python
python3 ray_train.py --config {training_configure} --name {training_name}
```
You can check {training_configure} in ray_config.py.

## Rendering

1. Execute viewer.

```bash
cd {downloaded folder}/build
./viewer/viewer {network path or environment xml name}
```
To execute example motions, use networks in directory '{dowloaded folder}/data/trained_nn/'


```bash
# Walking
cd {downloaded folder}/build
./viewer/viewer ../data/trained_nn/muscle_walking
```

## Reference Code

[1] https://github.com/lsw9021/MASS.git
[2] https://github.com/facebookresearch/ScaDiver.git

## Contact 

Jungnam Park (jungnam04@imo.snu.ac.kr)
