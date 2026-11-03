from glob import glob
from os.path import isfile

from setuptools import find_packages, setup

package_name = 'perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/models',
            [path for path in glob('resource/models/*') if isfile(path)]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jens',
    maintainer_email='jens1906@gmail.com',
    description='Leader identification, people tracking and obstacles for the follow-anywhere Go2',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # perception_node.main does not exist yet: `ros2 run` / the launch file fail until it is written
            'perception_node = perception.perception_node:main',
            'calibrate_camera = perception.camera_calibration_node:main',
        ],
    },
)
