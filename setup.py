from setuptools import setup

setup(
    name='gym_pybullet_drones',
    version='0.6.0',
    packages=['gym_pybullet_drones'],
    install_requires=[
        'numpy==1.20.1',
        'Pillow==8.1.0',
        'matplotlib==3.3.4',
        'cycler==0.10.0',
        'gym==0.17.3',
        'pybullet==3.0.8',
        'stable_baselines3==0.10.0',
        'ray[rllib]==0.8.7'
    ]
)
